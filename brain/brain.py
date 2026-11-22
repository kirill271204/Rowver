"""The brain loop, robot-agnostic: sense -> recall -> predict -> decide -> act -> remember.

  Brain(nerve).episode(skill, goal, mode)   one episode: reset, then tries until done or the step limit
  Brain(nerve).practice(skill, goal)        one practice round: the reference as it is, then the
                                            brain's corrections, both for the same goal
  modes:  "practice"  random corrections for the first random_tries rows, then the best guess
          "play"      always the best guess
          "baseline"  zero corrections: the recorded reference as it is (no model involved)
Every try becomes one row in the shared experience table, with the prediction made before it.

Provenance of every row: source "sim" (what the robot actually did) or "counterfactual" (in
simulation only: an alternative it did NOT choose, rolled out from a snapshot of the same moment,
then the world restored), phase ("practice" / "mission", set by the caller in self.phase) and the
propensity of the chosen action (1 for the argmax, the softmax probability in practice, the
uniform density for a random try; NaN for a counterfactual, which was never chosen).
self.counterfactual = K > 0 rolls out K alternatives per decision when the body can
(body.counterfactual); self.temperature samples the choice by softmax instead of the argmax.

Why practice pairs each goal with a reference episode: the zero-correction rows show the world
model what the recording does on its own, so it has a contrast to judge corrections against, and
the planner moves away from zero only where experience says a correction does better.
"""
from __future__ import annotations

import time

import numpy as np

from .hippocampus import Hippocampus
from .prefrontal import Prefrontal
from .provenance import ROOT, Config
from . import fit_cache
from .refresh import Refresh
from .world_model import WorldModel


class Brain:
    def __init__(self, nerve, cfg: Config | None = None, fresh=False, seed=0, verbose=True,
                 mem: Hippocampus | None = None):
        """mem: share one Hippocampus between several robots in one process (each robot's new
        rows are then context for the others at once). Default: this brain opens its own."""
        self.nerve, self.cfg = nerve, cfg or Config()
        d = nerve.describe()
        self.robot_id = d["robot_id"]
        self.desc = {k: float(v.value) for k, v in d["descriptors"].items()}
        self.skills = d["skills"]
        self.mem = mem if mem is not None else Hippocampus(ROOT / self.cfg("memory.dir"), self.cfg("memory.fail_lookahead"),
                                                           fresh_for=self.robot_id if fresh else None)
        self.rng = np.random.default_rng(seed)
        self.pfc = {k: Prefrontal(s, self.cfg, self.rng) for k, s in self.skills.items()}
        self.wm = {k: WorldModel(s.features + list(s.actions) + list(self.desc), s.targets, s.fail, family=s.family)
                   for k, s in self.skills.items()}
        fit_dir = getattr(self.mem, "fit_dir", None) or self.mem.dir   # a rover's own table may share a fit cache
        if fit_cache._STORE is None or fit_cache._STORE.path != fit_dir / "fits.json":
            fit_cache.set_store(fit_dir / "fits.json")             # fits reused across rovers and runs
        self.sched = {k: Refresh(self.robot_id, k, self.wm[k].kinds, self.cfg, self.mem.dir / "refresh_state.json",
                                 lookahead=self.mem.lookahead) for k in self.skills}
        self.refits: list = []
        self.on_refit = None                               # callback(event dict) for logs
        self.ep = self.mem.next_episode(self.robot_id)
        self.verbose = verbose
        self.phase = "mission"
        self.counterfactual = 0
        self.temperature = None
        self._last: dict = {}
        self.on_decide = None                              # callback(event dict) per decision: viewers, logs

    # ---------------------------------------------------------------- decide
    def _recall(self, skill):
        s = self.skills[skill]
        cur = self.cfg.opt("memory.curate", None)
        cols = s.features + list(s.actions) + list(self.desc)
        return self.mem.recall(self.robot_id, skill, s.family, self.desc, self.cfg("memory.recall_cap"),
                               curate=cur, columns=cols)

    def _refresh(self, skill, rows):
        """TabPFN's context for `skill`, model by model (brain/refresh.py):
          1. a model with no fit yet is restored from the rows up to a watermark (this rover's
             own, from before a restart, or the newest fit any rover made of the same rows), so
             brain/fit_cache finds the same rows and no request is sent;
          2. every new row of the skill family (any rover) is judged once per model: news?
          3. only the models with news (or a stale context) are refitted, on all recalled rows.
        A refit never trains anything: it reloads the rows as context (the server's KV cache)."""
        wm, sch, s = self.wm[skill], self.sched[skill], self.skills[skill]
        if "event_seq" in rows:
            for name in wm.kinds:
                if wm.has(name):
                    continue
                W, own = sch.resumed_at(name), True
                if W is None:
                    W, own = fit_cache.latest_seq(wm.lineage(name)), False
                if W is None:
                    continue
                tw = rows[rows.event_seq <= W]
                if not wm.can_fit(name, tw):
                    continue
                key = wm.fit_model(name, tw)
                if not own:
                    sch.adopt(name, W, key)
                self._event(skill, name, "restored", f"from the fit up to event {W}", key)
        df = self.mem.df
        sch.evaluate(wm, df[df.family == s.family] if len(df) and "family" in df else df)
        for name, why in sch.due(wm, rows).items():
            key = wm.fit_model(name, rows)
            sch.fitted(name, rows, key)
            if key is not None:
                self._event(skill, name, "refit", why, key)
        return wm

    def _event(self, skill, name, what, why, key):
        ev = {"robot_id": self.robot_id, "skill": skill, "model": name, "what": what, "why": why,
              "key": (key or "")[:10], "rows": len(self.mem)}
        self.refits.append(ev)
        self.mem.log({"kind": "refit", **ev})
        if self.on_refit is not None:
            self.on_refit(ev)

    def decide(self, skill: str, ctx: dict, mode: str) -> tuple[dict, dict, str]:
        pfc = self.pfc[skill]
        self._last = {"propensity": 1.0, "alts": []}
        if mode == "baseline":
            return pfc.zero_action(), {}, "reference"
        rows = self._recall(skill)
        if mode == "practice" and len(rows) < self.cfg("planner.random_tries"):
            self._last["propensity"] = pfc.density()
            return pfc.random_action(), {}, "random"           # deliberate exploration
        wm = self._refresh(skill, rows)
        if mode == "practice" and not wm.ready:
            self._last["propensity"] = pfc.density()
            return pfc.random_action(), {}, "no_model"         # no fit ready yet (not exploration)
        best = None
        if len(rows) and "score" in rows:
            ok = rows[rows.failed == 0].nlargest(16, "score")
            best = ok[pfc.names].to_numpy(float) if len(ok) else None
        plan = pfc.plan(wm, ctx, self.desc, best, temperature=self.temperature, n_alts=int(self.counterfactual))
        keys = {f"pred_fit_key_{n}": k[:16] for n, k in wm.keys.items() if k}
        own = int((rows.robot_id == self.robot_id).sum()) if len(rows) and "robot_id" in rows else 0
        self._last = {"propensity": plan["propensity"],
                      "alts": [{**a, "pred": a["pred"] | keys} for a in plan["alts"]],
                      "eval": plan.get("eval"), "rows": int(len(rows)), "own_rows": own}
        return plan["action"], plan["pred"] | keys, "play"

    # ---------------------------------------------------------------- one episode
    def episode(self, skill: str, goal: dict, mode: str = "practice", max_steps: int | None = None,
                on_step=None) -> dict:
        s = self.skills[skill]
        max_steps = max_steps or self.cfg("practice.episode_steps")
        self.nerve.reset()
        ep, steps, t0 = self.ep, [], time.time()
        self.ep += 1
        carry, n_commit = None, 0
        commit = getattr(self.nerve, "commit", None)
        for k in range(max_steps):
            if carry is not None:                               # the same arc on: no new look, no new plan
                ctx, action, pred, how = carry
                t_decide, self._last = 0.0, {"propensity": 1.0, "alts": []}
            else:
                ctx = self.nerve.context(skill, goal)
                td = time.time()
                action, pred, how = self.decide(skill, ctx, mode)
                t_decide = time.time() - td
                n_commit = 0
            if self.on_decide is not None:
                self.on_decide({"robot_id": self.robot_id, "skill": skill, "how": how, "action": dict(action),
                                "pred": dict(pred or {}), "decide_s": t_decide, "step": k, "n_commit": n_commit,
                                "eval": self._last.get("eval"), "rows": self._last.get("rows"),
                                "own_rows": self._last.get("own_rows")})
            alts = self._last.get("alts") or []
            seen = self._seen(s, ctx, action)                   # features as seen at the decision, before anything moves
            seen_alt = [self._seen(s, ctx, a["action"]) for a in alts]
            cf = getattr(self.nerve, "counterfactual", None)
            outs = cf(skill, [a["action"] for a in alts]) if alts and cf is not None else []
            out = self.nerve.act(skill, action)                 # the counterfactuals ran from the same moment
            row = self._row(s, skill, goal, seen, action, out, pred, ep, k, mode, how, "sim",
                            self._last.get("propensity", 1.0), t_decide)
            self.mem.remember(row)
            steps.append(row)
            cf_rows = [self._row(s, skill, goal, c, a["action"], o, a["pred"], ep, k, mode, "alt", "counterfactual",
                                 float("nan"), 0.0) for a, c, o in zip(alts, seen_alt, outs)]
            if cf_rows:
                self.mem.remember_many(cf_rows)
            if on_step:
                on_step(row)
                for r in cf_rows:
                    on_step(r)
            if out["done"]:
                break
            carry = None
            if commit is not None and how in ("play", "commit") and not alts:
                c = commit(skill, action, pred, out, n_commit)
                if c is not None:
                    carry, n_commit = (c, action, pred, "commit"), n_commit + 1
        self.mem.flush()
        n_ok = sum(1 for r in steps if not r["failed"])
        summ = {"episode": ep, "mode": mode, "goal": goal, "steps": len(steps), "survived": n_ok,
                "fell": bool(steps and steps[-1]["failed"]),
                "true_dist": float(sum(r.get("rep_true_dx", 0.0) for r in steps)),
                "walk_time": float(sum(r.get("rep_duration", 0.0) for r in steps)),
                "seconds": time.time() - t0, "memory_rows": len(self.mem)}
        summ["mean_speed"] = summ["true_dist"] / max(summ["walk_time"], 1e-6)
        self.mem.log({"kind": "episode", "robot_id": self.robot_id, "skill": skill, **summ})
        return summ

    @staticmethod
    def _seen(s, ctx, action):
        """The context with this candidate's own features (computed now, before it is driven)."""
        if "_per_action" not in ctx:
            return ctx
        Xa, _ = ctx["_per_action"](np.array([[action[n] for n in s.actions]], float))
        return {**ctx, **{k_: float(v) for k_, v in Xa.iloc[0].items()}}

    def _row(self, s, skill, goal, ctx, action, out, pred, ep, k, mode, how, source, propensity, t_decide):
        row = {"robot_id": self.robot_id, "skill": skill, "family": s.family, "episode": ep, "step": k,
               "mode": mode, "how": how, "source": source, "phase": self.phase, "propensity": propensity,
               "time": time.time(), "decide_s": t_decide,
               **{f"goal_{g}": v for g, v in goal.items()},
               **{f: ctx[f] for f in s.features}, **{f"ctx_{k_}": v for k_, v in ctx.items() if k_ not in s.features and not k_.startswith("_")},
               **action, **self.desc,
               **{t: out[t] for t in s.targets}, "failed": int(out["failed"]), "fail": int(out["failed"]),
               **pred,
               **{f"rep_{k_}": v for k_, v in out.items() if k_ not in s.targets and k_ not in ("failed", "done", "exec")},
               **{f"exec_{k_}": v for k_, v in out.get("exec", {}).items()}}
        row["score"] = self._realised_score(s, ctx, out, action)
        return row

    def practice(self, skill: str, goal: dict, on_step=None) -> tuple[dict, dict]:
        """One practice round: the reference unchanged, then the brain's corrections, same goal."""
        ref = self.episode(skill, goal, mode="baseline", on_step=on_step)
        return ref, self.episode(skill, goal, mode="practice", on_step=on_step)

    def _realised_score(self, s, ctx, out, action):
        """The objective computed on what actually happened (used to rank past actions)."""
        c = self.cfg
        sc =-c("objective.w_fail") * float(out["failed"])
        for t, g in s.goal.items():
            sc -= c("objective.w_goal") * abs(out[t] - ctx[g])
        for t in s.minimize:
            sc -= c("objective.w_tilt") * out[t]
        names = list(s.actions)
        half = np.array([max(abs(s.actions[k].value[0]), abs(s.actions[k].value[1])) for k in names])
        a = np.array([action[k] for k in names])
        return float(sc - c("objective.w_reference") * ((a / half) ** 2).mean())
