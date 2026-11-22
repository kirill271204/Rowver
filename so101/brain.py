"""The brain: perception -> TabPFN cerebellum (one per skill) -> motor -> memory.

There is no LLM. Every public method returns plain JSON-able dicts for the controller
and the dashboard.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .cerebellum import Cerebellum, backend_name
from .config import CFG
from .memory import Memory
from .planner import Planner, _pack
from .skills import SkillSpec, pick_spec, push_spec, throw_spec
from .robot.base import Robot


def make_robot(kind: str, gui=False, seed=0, realtime=1.0, **kw) -> Robot:
    if kind == "so101":
        from .robot.so101_mujoco import SO101Sim
        return SO101Sim(gui=gui, seed=seed, realtime=realtime)
    if kind == "finger":
        from .robot.finger_pybullet import FingerSim
        return FingerSim(gui=gui, seed=seed, realtime=realtime)
    if kind == "so101-throw":
        from .robot.so101_throw import SO101Thrower
        return SO101Thrower(gui=gui, seed=seed, realtime=realtime, **kw)
    if kind == "so101-real":
        from .robot.so101_real import SO101Real
        return SO101Real()
    raise ValueError(kind)


class Brain:
    def __init__(self, robot: Robot, skills=("push", "pick"), fresh=False, seed=0, obj_type=0):
        self.robot = robot
        self.skills = [s for s in skills if s in robot.skills]
        specs = {"push": push_spec(getattr(robot, "push_dist_max", 0.12)), "pick": pick_spec(),
                 "throw": throw_spec()}
        self.spec: dict[str, SkillSpec] = {s: specs[s] for s in self.skills}
        CFG.run_dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            CFG.events_jsonl.unlink(missing_ok=True)
        self.mem = {s: Memory(CFG.memory_csv(s), CFG.events_jsonl, fresh=fresh) for s in self.skills}
        self.cb = {s: Cerebellum(self.spec[s]) for s in self.skills}
        ok = getattr(robot, "action_ok", None)
        land_ok = getattr(robot, "landing_ok", robot.reachable)   # a throw lands beyond the arm's reach
        self.planner = {s: Planner(self.cb[s], land_ok, seed=seed + i,
                                   action_ok=(lambda st, a, s=s: ok(s, st, a)) if ok else None)
                        for i, s in enumerate(self.skills)}
        self._dirty = {s: True for s in self.skills}
        self.cal_scale = {}
        for s in self.skills:
            df = self.mem[s].df
            has = len(df) and "cal_scale" in df and df.cal_scale.notna().any()
            self.cal_scale[s] = float(df.cal_scale.dropna().iloc[-1]) if has else 1.0
        self.obj_type = obj_type
        self.episode = max([int(m.df.episode.max()) + 1 for m in self.mem.values() if len(m)] or [0])
        self.plans: dict[str, dict] = {}
        self.rng = np.random.default_rng(seed)
        self.robot.spawn_near_target(obj_type)

    # ================================================================ helpers
    @property
    def n_rows(self) -> int:
        return sum(len(m) for m in self.mem.values())

    def rows_for(self, skill: str, obj_type: int) -> int:
        df = self.mem[skill].df
        return int((df.obj_type == obj_type).sum()) if len(df) else 0

    def perceive(self) -> dict:
        s = self.robot.get_state()
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}

    def _refit(self, skill):
        if self._dirty[skill]:
            self.cb[skill].fit(self.mem[skill].df)
            self._dirty[skill] = False

    def _set_scale(self, skill, s):
        """Conformal scale x novelty: unseen object geometry -> wider intervals."""
        df = self.mem[skill].df
        n_geom = int(((df.obj_height == s["height"]) & (df.obj_size == s["size"])).sum()) if len(df) else 0
        self.cb[skill].scale = self.cal_scale[skill] * (1.6 if n_geom < 3 else 1.0)

    # ================================================================ prediction
    def plan(self, skill: str, state: dict | None = None) -> dict:
        """Refit TabPFN on this skill's memory and search ~3000 candidate actions."""
        self._refit(skill)
        s = state or self.robot.get_state()
        self._set_scale(skill, s)
        plan = self.planner[skill].plan(s)
        plan["state"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
        plan["calibration_scale"] = round(self.cb[skill].scale, 3)
        plan["backend"] = backend_name()
        self.plans[skill] = plan
        return plan

    def predict(self, skill: str, act: dict, state: dict) -> dict:
        """TabPFN's prediction for one specific action (what-if)."""
        self._refit(skill)
        self._set_scale(skill, state)
        acts = {k: np.array([v], float) for k, v in act.items()}
        df, _, _ = self.planner[skill].score(state, acts, joint=True)
        out = _pack(df.iloc[0], f"{skill}:manual", "manual", list(self.spec[skill].actions), self.spec[skill])
        out["cloud"] = [c[0].round(4).tolist() for c in self.planner[skill].last_cloud]
        return out

    # ================================================================ motor + memory
    def execute(self, candidate_id: str | None = None, skill: str | None = None, action: dict | None = None,
                mode: str | None = None, reasoning: str = "") -> dict:
        """Run a planned candidate ('push:E1', 'pick:X2') or an explicit action; record it."""
        before = self.robot.get_state()
        cand, cloud = None, None
        if candidate_id:
            skill = candidate_id.split(":")[0]
            plan = self.plans.get(skill)
            if plan is None:
                return {"error": f"no current {skill} plan: plan first"}
            cand = next((c for c in plan["candidates"] if c["id"] == candidate_id), None)
            if cand is None:
                return {"error": f"unknown candidate {candidate_id}"}
            action, mode = cand["action"], mode or cand["mode"]
            cloud = plan["clouds"].get(candidate_id)
        sp = self.spec[skill]
        action = {k: float(np.asarray(v).reshape(-1)[0])
                  for k, v in sp.clip({k: np.array([action[k]], float) for k in sp.actions}).items()}
        mode = mode or "manual"
        pred = None
        if mode != "seed":
            pred = cand if cand is not None else self.predict(skill, action, before)
            cloud = cloud or pred.get("cloud")

        res = sp.execute(self.robot, before, action)
        if res.get("error"):                     # e.g. the camera did not find the ball or the cup
            self.plans.pop(skill, None)
            return {"error": res["error"], "skill": skill}
        if res.get("out_of_reach"):
            self.plans.pop(skill, None)
            return {"error": "out of reach", "skill": skill}
        after = self.robot.get_state()
        tg = sp.targets_from(before, after, action, res)
        if "land_x" in res:      # thrown: the outcome is where it landed, not where the ball respawned
            after = after | {"x": res["land_x"], "y": res["land_y"],
                             "dist_to_target": float(np.hypot(res["land_x"] - before["target_x"],
                                                              res["land_y"] - before["target_y"]))}
        if "in_cup" in res:
            hit = bool(res["in_cup"])
        else:
            hit = bool(after["dist_to_target"] < before["target_radius"] and not res["toppled"] and after["on_table"])
        feat = sp.featurize(before, {k: np.array([v]) for k, v in action.items()}).iloc[0].to_dict()
        row = {
            "attempt": self.n_rows, "episode": self.episode, "time": time.time(), "mode": mode, "skill": skill,
            "obj_type": before["type"], "type_name": before["type_name"], **feat, **action, **tg,
            "x0": before["x"], "y0": before["y"], "theta0": before["theta"],
            "target_x": before["target_x"], "target_y": before["target_y"],
            "x1": after["x"], "y1": after["y"], "dist_before": before["dist_to_target"],
            "dist_after": after["dist_to_target"], "off_table": int(res["out_of_bounds"]), "hit": int(hit),
        }
        out = {"skill": skill, "hit": hit, "toppled": bool(res["toppled"]), "left_workspace": bool(res["out_of_bounds"]),
               "end": {"x": round(after["x"], 4), "y": round(after["y"], 4)},
               "miss_cm": round(100 * after["dist_to_target"], 2)}
        if skill in ("pick", "throw"):
            out |= {"grasped": bool(res.get("grasped")), "dropped": bool(res.get("dropped"))}
        if skill == "throw":
            out |= {"in_cup": bool(res["in_cup"]), "release_speed": round(float(res.get("release_speed", 0)), 2)}
        if pred is not None:
            a, b = sp.chain
            P = pred["pred"]
            regress_ok = bool(pd.DataFrame([tg]).eval(sp.reg_rows).iloc[0])
            # landing error only means something when the landing model applies (grasp held, upright)
            err = 100 * np.hypot(after["x"] - pred["pred_x"], after["y"] - pred["pred_y"]) if regress_ok else np.nan
            row |= {"pred_x": pred["pred_x"], "pred_y": pred["pred_y"], "pred_p_hit": pred["p_hit"],
                    "pred_err_cm": err, "pred_unc_cm": pred["uncertainty_cm"]}
            for c in sp.classifiers:
                row[f"pred_p_{c}"] = pred[f"p_{c}"]
            surprise = False
            if regress_ok:
                ins = [P[f"{a}_q10"] <= tg[a] <= P[f"{a}_q90"], P[f"{b}_q10"] <= tg[b] <= P[f"{b}_q90"]]
                surprise = not (P[f"{a}_q02"] <= tg[a] <= P[f"{a}_q98"] and P[f"{b}_q02"] <= tg[b] <= P[f"{b}_q98"])
                for inside in ins:   # adaptive conformal inference: settles at 20% outside the 80% band
                    self.cal_scale[skill] *= 0.972 if inside else 1.12
                self.cal_scale[skill] = float(np.clip(self.cal_scale[skill], 0.6, 4.0))
                row["in_80"] = sum(ins) / 2
            for c in sp.classifiers:   # a confident classifier that was wrong is a surprise too
                p = pred[f"p_{c}"]
                surprise |= (tg[c] == 1 and p < 0.1) or (tg[c] == 0 and p > 0.9)
            row |= {"surprise": int(surprise), "cal_scale": self.cal_scale[skill]}
            out |= {"predicted_end": {"x": round(pred["pred_x"], 4), "y": round(pred["pred_y"], 4)},
                    "prediction_error_cm": None if np.isnan(err) else round(err, 2),
                    "predicted_p_hit": round(pred["p_hit"], 3),
                    "surprise": bool(surprise)}
        self.mem[skill].append(row)
        self._dirty[skill] = True
        meta = self.plans.get(skill, {})
        self.mem[skill].log_event({"kind": "action", "skill": skill, "row": row, "reasoning": reasoning,
                                   "candidate": cand, "cloud": cloud,
                                   "plan_meta": {k: meta.get(k) for k in ("n_evaluated", "tabpfn_calls", "seconds")}
                                   if mode != "seed" else None})
        self.plans = {}
        self.new_attempt()
        out["memory_rows"] = self.n_rows
        return _native(out)

    def new_attempt(self, obj_type: int | None = None):
        if obj_type is not None:
            self.obj_type = obj_type
        self.episode += 1
        self.robot.spawn_near_target(self.obj_type)

    def switch_object(self, obj_type: int) -> dict:
        self.new_attempt(obj_type)
        return self.perceive()

    def random_action(self, skill: str) -> dict:
        s = self.robot.get_state()
        sp = self.spec[skill]
        ok = getattr(self.robot, "action_ok", None)
        for _ in range(50):
            a = sp.random_actions(self.rng, s, 1)
            if ok is None or ok(skill, s, a)[0]:
                break
        return self.execute(skill=skill, action={k: float(v[0]) for k, v in a.items()}, mode="seed")

    def memory_stats(self) -> dict:
        st = {s: self.mem[s].stats() for s in self.skills}
        st["tabpfn"] = {"backend": backend_name(),
                        "predict_calls_total": sum(c.predict_calls for c in self.cb.values()),
                        "rows_predicted_total": sum(c.predicted_rows for c in self.cb.values())}
        return st

    # ================================================================ reflection
    def reflect(self, skill: str) -> dict:
        """Partial dependence: ask TabPFN how each action parameter changes the outcome."""
        self._refit(skill)
        cb, sp = self.cb[skill], self.spec[skill]
        if not cb.ready:
            return {"note": "not enough experience to reflect yet"}
        s = self.robot.get_state()
        base = {k: (lo + hi) / 2 for k, (lo, hi) in sp.actions.items()}
        if skill == "push":
            base["angle"] = s["theta"]
        keys = [k for k in sp.actions if k not in ("angle", "aim_dx", "aim_dy", "aim")]
        frames, labels = [], []
        for k in keys:
            lo, hi = sp.actions[k]
            a = {kk: np.full(9, v) for kk, v in base.items()}
            a[k] = np.linspace(lo, hi, 9)
            frames.append(sp.featurize(s, a))
            labels.extend([k] * 9)
        X = pd.concat(frames, ignore_index=True)
        q = cb.quantiles(X, list(sp.chain))
        probs = {c: cb.proba(X, c) for c in sp.classifiers}
        labels = np.array(labels)
        res, effects = {}, []
        names = {"d_par": "cm forward", "d_perp": "cm sideways", "e_par": "cm overshoot", "e_perp": "cm sideways",
                 "r_par": "cm throw range", "r_perp": "cm sideways"}
        for k in keys:
            m = labels == k
            lo, hi = sp.actions[k]
            d = {"grid": np.round(np.linspace(lo, hi, 9), 3).tolist()}
            for t in sp.chain:
                d[t] = np.round(100 * q[t][m, 12], 1).tolist()
                effects.append((abs(d[t][-1] - d[t][0]), f"{k} {lo:.2g}->{hi:.2g}: {names[t]} {d[t][0]} -> {d[t][-1]}"))
            for c in sp.classifiers:
                pc = np.round(probs[c][m].astype(float), 2).tolist()
                d[f"p_{c}"] = pc
                effects.append((10 * abs(pc[-1] - pc[0]), f"{k} {lo:.2g}->{hi:.2g}: P({c}) {pc[0]} -> {pc[-1]}"))
            res[k] = d
        effects.sort(reverse=True)
        return {"skill": skill, "object": s["type_name"], "memory_rows": cb.n_rows,
                "strongest_effects": [e[1] for e in effects[:5]], "partial_dependence": res}


def _native(o):
    """numpy scalars -> plain Python, so results are JSON-safe."""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o
