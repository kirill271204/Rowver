"""Controller: the decision loop. TabPFN is the only model; there is no LLM.

Each attempt:
  1. perceive,
  2. plan every skill (TabPFN scores ~3000 candidate actions per skill),
  3. pick the skill + action with the best optimism-under-uncertainty score
         score = P(success) + beta * uncertainty / max_uncertainty - penalties
     with beta = 1 / sqrt(rows of experience for this skill and object + 1),
     so the robot explores a new object or skill and exploits once it knows it,
  4. execute, record the row (it becomes TabPFN context immediately), repeat.
"""
from __future__ import annotations

import numpy as np

from .brain import Brain

SKILL_COST = {"push": 0.0, "pick": 0.03, "throw": 0.0}     # pick-and-place takes longer; prefer push on a tie
MIN_TRIES = 2    # try every skill at least this often on each object before trusting the scores


class Controller:
    def __init__(self, brain: Brain, verbose=True):
        self.brain, self.verbose = brain, verbose

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def choose(self, plans: dict) -> tuple[dict, str]:
        b = self.brain
        obj = plans[next(iter(plans))]["state"]["type"]
        rows = []
        for sk, plan in plans.items():
            n = b.rows_for(sk, obj)
            beta = 1.0 / np.sqrt(n + 1)
            cands = plan["candidates"]
            umax = max(c["uncertainty_cm"] for c in cands) + 1e-9
            feas = [c for c in cands if c.get("feasible", True)] or cands
            for c in feas:
                bad = sum(c[f"p_{k}"] for k, kind in b.spec[sk].classifiers.items() if kind == "bad")
                score = c["p_hit"] + beta * c["uncertainty_cm"] / umax - 0.5 * bad - c["p_off_table"] - SKILL_COST.get(sk, 0.0)
                rows.append((score, beta, n, c))
        untried = [sk for sk in plans if b.rows_for(sk, obj) < MIN_TRIES]
        pool = [r for r in rows if r[3]["skill"] in untried] if untried else rows
        pool = sorted(pool, key=lambda r: -r[0])
        self._fallbacks = [r[3] for r in pool[1:4]]
        score, beta, n, c = pool[0]
        best_other = max((r for r in rows if r[3]["skill"] != c["skill"]), key=lambda r: r[3]["p_hit"], default=None)
        detail = f"P(hit)={c['p_hit']:.2f}"
        if c["skill"] == "pick":
            detail += f" [P(grasp)={c['p_grasped']:.2f} x P(land)={c['p_land']:.2f}]"
        if c["skill"] == "throw":
            st = plans["throw"]["state"]
            rng_cm = 100 * np.hypot(c["pred_x"] - st["origin_x"], c["pred_y"] - st["origin_y"])
            a = c["action"]
            detail = (f"P(in cup)={c['p_hit']:.2f} [P(hold)={c['p_grasped']:.2f} x P(over the cup)={c['p_land']:.2f}], "
                      f"aim {np.degrees(a['aim']):+.1f} deg, speed {a['speed']:.2f}, release {a['release']:.2f}, "
                      f"snap {a.get('snap', 0):.2f} "
                      f"-> predicted range {rng_cm:.1f} cm (cup at {100 * st['cup_dist']:.0f} cm)")
        why = (f"{c['mode']} {c['id']}: {detail}, uncertainty {c['uncertainty_cm']:.1f} cm, "
               f"beta={beta:.2f} ({n} rows for this {c['skill']}/object)")
        if untried:
            why += f"; trying {c['skill']} (fewer than {MIN_TRIES} rows on this object)"
        if best_other is not None:
            why += f"; best {best_other[3]['skill']} P(hit)={best_other[3]['p_hit']:.2f}"
        return c, why

    def step(self) -> dict:
        b = self.brain
        state = b.robot.get_state()
        plans = {}
        for sk in b.skills:
            plans[sk] = b.plan(sk, state)
            self._say(f"  [tabpfn:{sk}] {plans[sk]['n_evaluated']} actions in {plans[sk]['seconds']}s, "
                      f"{plans[sk]['tabpfn_calls']} calls -> best P(hit)={plans[sk]['candidates'][0]['p_hit']}")
        c, why = self.choose(plans)
        self.last = {"choice": c, "why": why,                  # for viewers: what TabPFN weighed, and why
                     "plans": {sk: {"n": p["n_evaluated"], "seconds": p["seconds"], "calls": p["tabpfn_calls"]}
                               for sk, p in plans.items()}}
        self._say(f"  [decide] {why}")
        out = b.execute(candidate_id=c["id"], reasoning=why)
        for alt in self._fallbacks:              # perception noise can push a borderline action out of reach
            if out.get("error") != "out of reach":
                break
            self._say(f"  [motor] {c['id']} out of reach -> trying {alt['id']}")
            b.plans = plans | {alt["skill"]: plans[alt["skill"]]}
            c = alt
            out = b.execute(candidate_id=alt["id"], reasoning=why + f" (fallback {alt['id']})")
        if "error" in out:
            self._say(f"  [motor] {out['error']}")
            return out
        tag = "HIT" if out["hit"] else ("TOPPLED" if out["toppled"] else "miss")
        if out["skill"] == "throw":
            tag = "IN THE CUP" if out["hit"] else "miss"
        if out["skill"] in ("pick", "throw") and not out.get("grasped"):
            tag += " (grasp failed)"
        pe = out.get("prediction_error_cm")
        pe = f"prediction error {pe} cm" if pe is not None else "landing model n/a (classifiers scored it)"
        self._say(f"  [motor] {out['skill']} {tag}: {out['miss_cm']} cm from target, {pe}"
                  + (" (SURPRISE)" if out.get("surprise") else ""))
        return out

    def reflect(self):
        for sk in self.brain.skills:
            r = self.brain.reflect(sk)
            if "strongest_effects" not in r:
                continue
            text = f"TabPFN on {sk} ({r['object']}, {r['memory_rows']} rows): " + "; ".join(r["strongest_effects"][:3])
            self._say(f"  [reflection] {text}")
            self.brain.mem[sk].log_event({"kind": "reflection", "skill": sk, "text": text,
                                          "effects": r["strongest_effects"], "pd": r["partial_dependence"]})
