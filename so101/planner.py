"""Planner: search a skill's action space with TabPFN as the world model.

A TabPFN predict call has a fixed cost (~0.6 s here) that barely depends on how many
rows it scores, so the search uses a few *huge* batches instead of many small ones:

  round 0  ~2048 candidate actions -> full predictive distributions (1 call per model)
  round 1  ~1024 actions resampled around the elites (cross-entropy method)
  joint    finalists re-scored with the chained joint model b | (features, a)

Every candidate gets P(success) = P(lands in target) x P(good classifiers) x
(1 - P(bad classifiers)), plus P(leaves the workspace), expected miss distance and an
uncertainty radius. "Explore" candidates are where TabPFN is least sure.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .cerebellum import Cerebellum, qcol, sample_from_quantiles
from .config import CFG
from .robot.base import wrap


class Planner:
    def __init__(self, cb: Cerebellum, reach_fn, seed: int = 0, action_ok=None):
        self.cb, self.spec = cb, cb.spec
        self.reach_fn = reach_fn
        self.action_ok = action_ok        # optional robot feasibility check (vectorised)
        self.rng = np.random.default_rng(seed)
        self.last_cloud = None

    # ---------------------------------------------------------------- proposals
    def _refit_proposal(self, elites: pd.DataFrame, n):
        out = {}
        for k, (lo, hi) in self.spec.actions.items():
            if k == "angle":
                mu = np.angle(np.exp(1j * elites.angle).mean())
                sd = max(0.03, float(np.std(wrap(elites.angle - mu))))
                out[k] = mu + self.rng.normal(0, sd, n)
            else:
                sd = max(0.02 * (hi - lo), float(elites[k].std()))
                out[k] = self.rng.normal(float(elites[k].mean()), sd, n)
        return self.spec.clip(out)

    # ---------------------------------------------------------------- scoring
    def score(self, state, acts: dict, qd=None, probs=None, joint=False):
        """Returns (scored DataFrame, quantile dict, class probs). One TabPFN call per model."""
        sp, cb = self.spec, self.cb
        a, b = sp.chain
        X = sp.featurize(state, acts)
        qd = qd if qd is not None else cb.quantiles(X, [a, b])
        probs = probs if probs is not None else {c: cb.proba(X, c) for c in sp.classifiers}
        if joint:
            sa, sb = cb.joint_samples(X, CFG.mc_samples, self.rng, qd)
        else:
            sa = sample_from_quantiles(qd[a], CFG.mc_samples, self.rng)
            sb = sample_from_quantiles(qd[b], CFG.mc_samples, self.rng)
        fx, fy = sp.landing(state, acts, sa, sb)
        dist = np.hypot(fx - state["target_x"], fy - state["target_y"])

        df = pd.DataFrame({k: np.asarray(v) for k, v in acts.items()})
        df["pred_x"], df["pred_y"] = np.median(fx, 1), np.median(fy, 1)
        df["p_land"] = (dist < state["target_radius"]).mean(1)
        p_ok = np.ones(len(df))
        for c, kind in sp.classifiers.items():
            df[f"p_{c}"] = probs[c]
            p_ok *= probs[c] if kind == "good" else (1 - probs[c])
        df["p_hit"] = df.p_land * p_ok
        df["p_off_table"] = (~self.reach_fn(fx, fy)).mean(1)
        df["exp_dist"] = dist.mean(1)
        w_a = qd[a][:, qcol(0.9)] - qd[a][:, qcol(0.1)]
        w_b = qd[b][:, qcol(0.9)] - qd[b][:, qcol(0.1)]
        df["unc"] = np.hypot(w_a, w_b) / 2                # ~80% landing radius
        for t in qd:
            for name, lvl in (("q02", 0.02), ("q10", 0.1), ("q50", 0.5), ("q90", 0.9), ("q98", 0.98)):
                df[f"{t}_{name}"] = qd[t][:, qcol(lvl)]
        r = state["target_radius"]
        bad = sum(df[f"p_{c}"] for c, k in sp.classifiers.items() if k == "bad")
        df["score"] = df.p_hit + 0.15 * np.exp(-df.exp_dist / (3 * r)) - 0.5 * bad - 1.0 * df.p_off_table
        if self.action_ok is not None:
            df.loc[~self.action_ok(state, acts), "score"] = -9.0
        if joint:
            self.last_cloud = (fx, fy)
        return df, qd, probs

    # ---------------------------------------------------------------- main entry
    def plan(self, state: dict) -> dict:
        t0, calls0 = time.time(), self.cb.predict_calls
        n = CFG.n_candidates
        acts = self.spec.random_actions(self.rng, state, n)
        near = self.spec.experience_proposals(self.rng, state, getattr(self.cb, "_reg", None), n // 4)
        if near is not None:     # a quarter of round 0 sits around the most relevant past actions
            acts = {k: np.concatenate([acts[k][: n - n // 4], near[k]]) for k in acts}
        pools, qds, pss = [], [], []
        for it in range(max(1, CFG.cem_iters)):
            sc, qd, ps = self.score(state, acts)
            sc["iter"] = it
            pools.append(sc), qds.append(qd), pss.append(ps)
            elites = sc[sc.score > -9].nlargest(max(16, len(sc) // 20), "score")
            if len(elites) < 4:
                elites = sc.nlargest(16, "score")
            acts = self._refit_proposal(elites, max(256, n // 2))
        pool = pd.concat(pools, ignore_index=True)
        qall = {t: np.concatenate([q[t] for q in qds]) for t in qds[0]}
        pall = {c: np.concatenate([p[c] for p in pss]) for c in pss[0]}

        keys = list(self.spec.actions)
        fin = pool[pool.score > -9].nlargest(CFG.top_k * 4, "score")
        fin = fin.loc[~fin[keys[0]].round(3).duplicated()].head(CFG.top_k * 2)
        if fin.empty:
            fin = pool.nlargest(CFG.top_k, "score")
        idx = fin.index.to_numpy()
        joint, _, _ = self.score(state, {k: fin[k].to_numpy() for k in keys},
                                 qd={t: v[idx] for t, v in qall.items()},
                                 probs={c: v[idx] for c, v in pall.items()}, joint=True)
        cloud_x, cloud_y = self.last_cloud
        joint["_row"] = np.arange(len(joint))
        exploit = joint.nlargest(CFG.top_k, "score").reset_index(drop=True)

        # explore: most uncertain feasible actions that don't make things worse
        cur = state["dist_to_target"]
        bad = sum(pool[f"p_{c}"] for c, k in self.spec.classifiers.items() if k == "bad")
        cand = pool[(pool.score > -9) & (pool.exp_dist < cur + 0.03) & (pool.p_off_table < 0.05) & (bad < 0.6)]
        cand = cand if len(cand) >= 3 else pool[pool.score > -9]
        cand = cand if len(cand) >= 3 else pool
        info = cand.unc / (cand.unc.max() + 1e-9)
        explore = cand.assign(info=info).nlargest(3, "info").reset_index(drop=True)

        sk = self.spec.name
        cands = [_pack(r, f"{sk}:E{i+1}", "exploit", keys, self.spec) for i, r in exploit.iterrows()]
        cands += [_pack(r, f"{sk}:X{i+1}", "explore", keys, self.spec) for i, r in explore.iterrows()]
        clouds = {f"{sk}:E{i+1}": (cloud_x[int(r._row)].round(4).tolist(), cloud_y[int(r._row)].round(4).tolist())
                  for i, r in exploit.iterrows()}
        return {
            "skill": sk,
            "candidates": cands,
            "clouds": clouds,
            "n_evaluated": int(len(pool)),
            "tabpfn_calls": int(self.cb.predict_calls - calls0),
            "seconds": round(time.time() - t0, 2),
            "memory_rows": self.cb.n_rows,
            "cold_start": not self.cb.ready,
        }



def _pack(row, cid, mode, keys, spec):
    out = {"id": cid, "skill": spec.name, "mode": mode,
           "action": {k: round(float(row[k]), 4) for k in keys}}
    for k in ("pred_x", "pred_y"):
        out[k] = round(float(row[k]), 4)
    for k in ["p_hit", "p_land", "p_off_table"] + [f"p_{c}" for c in spec.classifiers]:
        out[k] = round(float(row[k]), 3)
    out["exp_miss_cm"] = round(100 * float(row.exp_dist), 2)
    out["feasible"] = bool(row.score > -9)
    out["uncertainty_cm"] = round(100 * float(row.unc), 2)
    out["pred"] = {k: round(float(row[k]), 4) for k in row.index if k[-4:] in ("_q02", "_q10", "_q50", "_q90", "_q98")}
    return out
