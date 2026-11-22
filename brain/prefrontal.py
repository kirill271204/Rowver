"""Prefrontal cortex: choose the action for a situation and a goal, with TabPFN as the world model.

Search (cross-entropy method, one batched TabPFN pass per round):
  round 0   half uniform within the action bounds, a quarter around 0 (= the recorded
            reference, no correction), a quarter around the best past actions
  round 1+  resampled around the elites of the previous round
Score of a candidate (weights from brain.toml [objective]):
  - w_goal      * |E[target] - goal|          for each goal target
  - w_tilt      * E[target]                   for each target to minimise
  - w_fail      * P(fail)
  - w_reference * mean((action / bound)^2)    stay close to the reference
objective.risk = "lcb" (env CORTEX_RISK) scores the pessimistic side of TabPFN's predicted
distribution instead of its mean: max(|q10 - goal|, |q90 - goal|) for a goal, q90 for a target to
minimise. "mean" is the control.
Amygdala: candidates with P(fail) above veto_p_fail are rejected whenever any candidate is below it.

Choice: argmax of the score (propensity 1), or, in practice, a softmax over the candidates still
allowed (temperature T) with the chosen one's probability logged as its propensity (given the
sampled candidate set), so a later off-policy evaluation can reweight the rows.
Counterfactual alternatives (n_alts): the allowed candidates farthest from the choice in action
space among the better half by score, plus one at random, each with its own prediction.

Per-candidate features (optional): when the body's context carries ctx["_per_action"], a function
A -> (DataFrame of features that depend on the action, extra(q, p_fail) -> score added), those
columns replace the broadcast context per candidate (a rover's arcs each cross different ground)
and the body adds its own goal term (progress toward the goal, a geometric safety veto).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nerves import SkillSchema
from .provenance import Config
from .world_model import WorldModel


class Prefrontal:
    def __init__(self, schema: SkillSchema, cfg: Config, rng):
        self.s, self.cfg, self.rng = schema, cfg, rng
        self.names = list(schema.actions)
        self.lo = np.array([schema.actions[k].value[0] for k in self.names], float)
        self.hi = np.array([schema.actions[k].value[1] for k in self.names], float)
        self.half = np.maximum(np.abs(self.lo), np.abs(self.hi))

    def _clip(self, A):
        return np.clip(A, self.lo, self.hi)

    def random_action(self) -> dict:
        return dict(zip(self.names, self.rng.uniform(self.lo, self.hi).tolist()))

    def zero_action(self) -> dict:
        return dict(zip(self.names, self._clip(np.zeros(len(self.names))).tolist()))

    def _score(self, wm: WorldModel, ctx: dict, desc: dict, A: np.ndarray):
        X = pd.DataFrame(A, columns=self.names)
        per = ctx.get("_per_action")
        Xa, extra = per(A) if per is not None else (None, None)
        for k in self.s.features:                       # a sensor this body lacks: NaN (TabPFN: missing)
            X[k] = Xa[k].to_numpy() if Xa is not None and k in Xa else ctx.get(k, np.nan)
        for k, v in desc.items():
            X[k] = v
        q, p_fail = wm.predict(X)
        c = self.cfg
        lcb = self.risk() == "lcb"
        score = -c("objective.w_fail") * p_fail
        for t, g in self.s.goal.items():
            if t not in q:                              # no model for it yet: no term
                continue
            if lcb:
                miss = np.maximum(np.abs(np.quantile(q[t], 0.1, axis=1) - ctx[g]),
                                  np.abs(np.quantile(q[t], 0.9, axis=1) - ctx[g]))
            else:
                miss = np.abs(q[t].mean(1) - ctx[g])
            score -= c("objective.w_goal") * miss
        for t in self.s.minimize:
            if t not in q:
                continue
            score -= c("objective.w_tilt") * (np.quantile(q[t], 0.9, axis=1) if lcb else q[t].mean(1))
        score -= c("objective.w_reference") * ((A / self.half) ** 2).mean(1)
        if extra is not None:
            score = score + extra(q, p_fail)
        veto = c.opt("planner.veto_p_fail", 0.9, env="CORTEX_VETO_P_FAIL")
        if (p_fail <= veto).any():
            score = np.where(p_fail <= veto, score, -np.inf)
        return score, q, p_fail

    def risk(self):
        return str(self.cfg.opt("objective.risk", "mean", env="CORTEX_RISK"))

    def density(self):
        """Propensity of a uniform random action: 1 / volume of the action box."""
        return float(1.0 / np.prod(np.maximum(self.hi - self.lo, 1e-12)))

    @staticmethod
    def _pred(q, p, score, i):
        pred = {f"pred_{t}": float(np.median(q[t][i])) for t in q}
        pred |= {f"pred_{t}_q10": float(np.quantile(q[t][i], 0.1)) for t in q}
        pred |= {f"pred_{t}_q90": float(np.quantile(q[t][i], 0.9)) for t in q}
        pred["pred_p_fail"] = float(p[i])
        pred["pred_score"] = float(score[i])
        return pred

    def _alts(self, A, score, i, k):
        ok = np.flatnonzero(np.isfinite(score) & (score > -1e5))
        ok = ok[ok != i]
        if not len(ok) or k <= 0:
            return []
        span = np.maximum(self.hi - self.lo, 1e-9)
        better = ok[score[ok] >= np.median(score[ok])]
        out = []
        for _ in range(min(max(k - 1, 1), len(better))):          # far from the choice (and each other)
            ref = np.vstack([A[i]] + [A[j] for j in out])
            d = np.abs((A[better][:, None] - ref[None]) / span).sum(2).min(1)
            d[np.isin(better, out)] = -1
            out.append(int(better[int(np.argmax(d))]))
        rest = ok[~np.isin(ok, out)]
        if len(out) < k and len(rest):
            out.append(int(rest[self.rng.integers(len(rest))]))
        return out[:k]

    def plan(self, wm: WorldModel, ctx: dict, desc: dict, best_past: np.ndarray | None,
             temperature: float | None = None, n_alts: int = 0) -> dict:
        n, sd = self.cfg("planner.candidates"), self.cfg("planner.explore_sd") * (self.hi - self.lo)
        k4 = n // 4
        parts = [self.rng.uniform(self.lo, self.hi, (n - 2 * k4, len(self.names))),
                 self.rng.normal(0.0, sd, (k4, len(self.names)))]
        if best_past is not None and len(best_past):
            pick = best_past[self.rng.integers(len(best_past), size=k4)]
            parts.append(pick + self.rng.normal(0.0, sd, pick.shape))
        else:
            parts.append(self.rng.normal(0.0, sd, (k4, len(self.names))))
        A = self._clip(np.vstack(parts))
        for r in range(self.cfg("planner.cem_rounds")):
            score, q, p = self._score(wm, ctx, desc, A)
            if r == self.cfg("planner.cem_rounds") - 1:
                break
            elite = A[np.argsort(-score)[:max(16, len(A) // 20)]]
            mu, es = elite.mean(0), np.maximum(elite.std(0), 0.02 * (self.hi - self.lo))
            A = self._clip(np.vstack([elite, self.rng.normal(mu, es, (n // 2, len(self.names)))]))
        i, prop = int(np.argmax(score)), 1.0
        allowed = np.flatnonzero(np.isfinite(score) & (score > -1e5))
        if temperature and len(allowed) > 1:                     # practice: sample, log the propensity
            z = (score[allowed] - score[allowed].max()) / float(temperature)
            pr = np.exp(z) / np.exp(z).sum()
            j = int(self.rng.choice(len(allowed), p=pr))
            i, prop = int(allowed[j]), float(pr[j])
        alts = [{"action": dict(zip(self.names, A[j].tolist())), "pred": self._pred(q, p, score, j)}
                for j in self._alts(A, score, i, n_alts)]
        return {"action": dict(zip(self.names, A[i].tolist())), "pred": self._pred(q, p, score, i),
                "n_scored": int(len(A)), "propensity": prop, "alts": alts, "eval": self._eval(A, score, q, p, i)}

    def _eval(self, A, score, q, p, i) -> dict:
        """The whole scored candidate set of this decision (for viewers and logs; arrays it already has)."""
        ev = {"names": list(self.names), "A": A.copy(), "score": np.asarray(score, float).copy(),
              "p_fail": np.asarray(p, float).copy(), "chosen": int(i), "n_scored": int(len(A))}
        for t, Q in q.items():
            ev[f"{t}_q10"] = np.quantile(Q, 0.1, axis=1)
            ev[f"{t}_q50"] = np.median(Q, axis=1)
            ev[f"{t}_q90"] = np.quantile(Q, 0.9, axis=1)
        return ev
