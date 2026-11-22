"""Skill specs: what TabPFN sees (features), what it predicts (targets) and how a
prediction turns into where the object ends up.

PUSH  targets in the push frame (along / across the push): d_par, d_perp, dtheta
      classifier: toppled
PICK  targets in the carry frame (along / across the carry direction, relative to the
      commanded place point): e_par, e_perp
      classifiers: grasped, toppled
THROW targets in the throw frame (where the ball crosses the cup's rim plane, along /
      across the aim, measured from the base's rotation axis): r_par, r_perp
      classifier: grasped (ball still held when the swing starts)
Expressing outcomes in the frame of the motion makes the physics translation- and
rotation-invariant, so every past row is relevant to every new decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CFG
from .robot.base import wrap


def rot(angle, a, b):
    """Rotate frame-local (a, b) into the world by angle."""
    c, s = np.cos(angle), np.sin(angle)
    return c * a - s * b, s * a + c * b


def unrot(angle, dx, dy):
    c, s = np.cos(angle), np.sin(angle)
    return c * dx + s * dy, -s * dx + c * dy


GEOM = ["obj_shape", "obj_size", "obj_height"]


@dataclass
class SkillSpec:
    name: str
    actions: dict[str, tuple[float, float]]
    features: list[str]
    targets: list[str]                   # all regression targets (first two are used for landing)
    chain: tuple[str, str]               # (a, b): b is also modelled as b | (features, a)
    classifiers: dict[str, str]          # column -> "good" (want 1) or "bad" (want 0)
    reg_rows: str                        # pandas query selecting rows for the regressors
    prior: dict[str, tuple[float, float]] = field(default_factory=dict)   # cold-start mean/std per target
    prior_p: dict[str, float] = field(default_factory=dict)              # cold-start classifier probs

    # ------------------------------------------------------------------ features
    def featurize(self, state: dict, acts: dict) -> pd.DataFrame:
        n = len(np.atleast_1d(next(iter(acts.values()))))
        a = {k: np.broadcast_to(np.asarray(v, float), n).copy() for k, v in acts.items()}
        df = pd.DataFrame({"obj_shape": np.full(n, float(state["shape"])),
                           "obj_size": np.full(n, float(state["size"])),
                           "obj_height": np.full(n, float(state["height"])),
                           "obj_x": np.full(n, state["x"]), "obj_y": np.full(n, state["y"])})
        if self.name == "push":
            rel = wrap(a["angle"] - state["theta"])
            df["rel_cos4"], df["rel_sin4"] = np.cos(4 * rel), np.sin(4 * rel)
            df["rel_cos"], df["rel_sin"] = np.cos(rel), np.sin(rel)
            df["push_speed"], df["push_offset"], df["push_dist"] = a["speed"], a["offset"], a["push_dist"]
        elif self.name == "throw":   # rotation invariant: the aim itself is not a feature
            df["throw_speed"], df["throw_release"] = a["speed"], a["release"]
            df["throw_snap"] = a["snap"] if "snap" in a else np.zeros(n)
        else:
            g = a["grasp_yaw"]
            place_x = state["target_x"] + a["aim_dx"]
            place_y = state["target_y"] + a["aim_dy"]
            carry = np.arctan2(place_y - state["y"], place_x - state["x"])
            crel = wrap(carry - (state["theta"] + g))
            df["g_cos4"], df["g_sin4"] = np.cos(4 * g), np.sin(4 * g)
            df["g_cos"], df["g_sin"] = np.cos(g), np.sin(g)
            df["grasp_along"], df["grasp_across"] = a["grasp_along"], a["grasp_across"]
            df["grip_force"], df["lift_speed"], df["release_height"] = a["grip_force"], a["lift_speed"], a["release_height"]
            df["grasp_height"] = a["grasp_height"]
            df["carry_dist"] = np.hypot(place_x - state["x"], place_y - state["y"])
            df["carry_rel_cos"], df["carry_rel_sin"] = np.cos(crel), np.sin(crel)
        return df

    # ------------------------------------------------------------------ geometry
    def landing(self, state: dict, acts: dict, t0: np.ndarray, t1: np.ndarray):
        """Predicted (x, y) from samples of the first two targets. t0/t1 shape (n, S)."""
        if self.name == "push":
            ang = np.asarray(acts["angle"])[:, None]
            dx, dy = rot(ang, t0, t1)
            return state["x"] + dx, state["y"] + dy
        if self.name == "throw":
            aim = np.asarray(acts["aim"])[:, None]
            dx, dy = rot(aim, t0, t1)
            return state.get("origin_x", 0.0) + dx, state.get("origin_y", 0.0) + dy
        px = state["target_x"] + np.asarray(acts["aim_dx"])[:, None]
        py = state["target_y"] + np.asarray(acts["aim_dy"])[:, None]
        carry = np.arctan2(py - state["y"], px - state["x"])
        ex, ey = rot(carry, t0, t1)
        return px + ex, py + ey

    def targets_from(self, before: dict, after: dict, act: dict, res: dict) -> dict:
        dx, dy = after["x"] - before["x"], after["y"] - before["y"]
        if self.name == "push":
            a, b = unrot(act["angle"], dx, dy)
            return {"d_par": a, "d_perp": b, "dtheta": float(wrap(after["theta"] - before["theta"])),
                    "toppled": int(res["toppled"])}
        if self.name == "throw":
            a, b = unrot(act["aim"], res["land_x"] - before["origin_x"], res["land_y"] - before["origin_y"])
            return {"r_par": a, "r_perp": b, "grasped": int(res["grasped"]), "toppled": 0,
                    "in_cup": int(res["in_cup"]), "land_x": res["land_x"], "land_y": res["land_y"],
                    "cup_x": before["cup_x"], "cup_y": before["cup_y"], "cup_dist": before["cup_dist"],
                    "cup_angle": before["cup_angle"],
                    # vision diagnostics (not TabPFN features): detector confidence, error vs the truth
                    **{k: before[k] for k in ("det_conf", "ball_err_mm", "cup_err_mm") if k in before}}
        px, py = before["target_x"] + act["aim_dx"], before["target_y"] + act["aim_dy"]
        carry = np.arctan2(py - before["y"], px - before["x"])
        a, b = unrot(carry, after["x"] - px, after["y"] - py)
        return {"e_par": a, "e_perp": b, "grasped": int(res.get("grasped", False)),
                "toppled": int(res["toppled"]), "place_x": px, "place_y": py}

    def execute(self, robot, state: dict, act: dict) -> dict:
        if self.name == "push":
            return robot.push(act["angle"], act["speed"], act["offset"], act["push_dist"])
        if self.name == "throw":
            return robot.throw(act["aim"], act["speed"], act["release"], act.get("snap", 0.0))
        return robot.pick_place(grasp_along=act["grasp_along"], grasp_across=act["grasp_across"],
                                grasp_yaw=act["grasp_yaw"], grip_force=act["grip_force"],
                                lift_speed=act["lift_speed"], release_height=act["release_height"],
                                grasp_height=act["grasp_height"],
                                place_x=state["target_x"] + act["aim_dx"], place_y=state["target_y"] + act["aim_dy"])

    def random_actions(self, rng, state: dict, n: int) -> dict:
        out = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in self.actions.items()}
        if self.name == "push":   # half aimed roughly at the target; no other physics hard-coded
            bearing = np.arctan2(state["target_y"] - state["y"], state["target_x"] - state["x"])
            k = n // 2
            out["angle"][:k] = wrap(bearing + rng.normal(0, 0.5, k))
        elif self.name == "throw":   # half aimed roughly at the cup; the range is left to TabPFN
            bearing = np.arctan2(state["target_y"] - state.get("origin_y", 0.0),
                                 state["target_x"] - state.get("origin_x", 0.0))
            k = n // 2
            out["aim"][:k] = bearing + rng.normal(0, 0.06, k)
        else:   # proposal prior: half the grasps near the object's center (TabPFN still scores them all)
            k = n // 2
            out["grasp_along"][:k] = rng.normal(0, 0.2, k)
            out["grasp_across"][:k] = rng.normal(0, 0.2, k)
            out["aim_dx"] = rng.normal(0, 0.008, n)
            out["aim_dy"] = rng.normal(0, 0.008, n)
        return self.clip(out)

    def experience_proposals(self, rng, state: dict, reg: pd.DataFrame, n: int):
        """Trust-region proposals: perturb past actions whose outcome came closest to what is
        needed now. Only the *proposals* use this; TabPFN still scores every candidate."""
        if self.name != "throw" or reg is None or len(reg) < 3:
            return None
        ox, oy = state.get("origin_x", 0.0), state.get("origin_y", 0.0)
        D = float(np.hypot(state["target_x"] - ox, state["target_y"] - oy))
        bearing = float(np.arctan2(state["target_y"] - oy, state["target_x"] - ox))
        near = reg.assign(gap=(reg.r_par - D).abs()).nsmallest(6, "gap")
        pick = near.iloc[rng.integers(len(near), size=n)]
        out = {"speed": pick.speed.to_numpy() + rng.normal(0, 0.03, n),
               "release": pick.release.to_numpy() + rng.normal(0, 0.06, n),
               "snap": (pick.snap.to_numpy() if "snap" in pick else rng.uniform(0, 1, n)) + rng.normal(0, 0.06, n),
               "aim": bearing - np.arctan2(pick.r_perp.to_numpy(), D) + rng.normal(0, 0.02, n)}
        return self.clip(out)

    def clip(self, a: dict) -> dict:
        for k, (lo, hi) in self.actions.items():
            a[k] = wrap(a[k]) if k == "angle" else np.clip(a[k], lo, hi)
        return a

    @property
    def feature_cols(self):
        return self.features + (["obj_x", "obj_y"] if CFG.use_position_features else [])


def push_spec(push_dist_max=0.12) -> SkillSpec:
    return SkillSpec(
        name="push",
        actions={"angle": (-np.pi, np.pi), "speed": (0.0, 1.0), "offset": (-1.0, 1.0),
                 "push_dist": (0.0, push_dist_max)},
        features=GEOM + ["rel_cos4", "rel_sin4", "rel_cos", "rel_sin", "push_speed", "push_offset", "push_dist"],
        targets=["d_par", "d_perp", "dtheta"],
        chain=("d_par", "d_perp"),
        classifiers={"toppled": "bad"},
        reg_rows="toppled == 0",
        prior={"d_par": (push_dist_max * 0.6 + 0.02, 0.05), "d_perp": (0.0, 0.03), "dtheta": (0.0, 0.8)},
        prior_p={"toppled": 0.05},
    )


def pick_spec() -> SkillSpec:
    return SkillSpec(
        name="pick",
        actions={"grasp_along": (-0.7, 0.7), "grasp_across": (-0.7, 0.7), "grasp_yaw": (-np.pi / 2, np.pi / 2),
                 "grip_force": (0.05, 1.0), "lift_speed": (0.0, 1.0), "release_height": (0.0, 0.04),
                 "grasp_height": (0.15, 0.7),
                 "aim_dx": (-0.03, 0.03), "aim_dy": (-0.03, 0.03)},
        features=GEOM + ["g_cos4", "g_sin4", "g_cos", "g_sin", "grasp_along", "grasp_across", "grip_force",
                         "lift_speed", "release_height", "grasp_height", "carry_dist", "carry_rel_cos", "carry_rel_sin"],
        targets=["e_par", "e_perp"],
        chain=("e_par", "e_perp"),
        classifiers={"grasped": "good", "toppled": "bad"},
        reg_rows="grasped == 1 and toppled == 0",
        prior={"e_par": (0.0, 0.02), "e_perp": (0.0, 0.02)},
        prior_p={"grasped": 0.5, "toppled": 0.1},
    )


def throw_spec() -> SkillSpec:
    return SkillSpec(
        name="throw",
        actions={"aim": (-1.4, 1.4), "speed": (0.0, 1.0), "release": (0.0, 1.0), "snap": (0.0, 1.0)},
        features=GEOM + ["throw_speed", "throw_release", "throw_snap"],
        targets=["r_par", "r_perp"],
        chain=("r_par", "r_perp"),
        classifiers={"grasped": "good"},
        reg_rows="grasped == 1",
        prior={"r_par": (0.6, 0.4), "r_perp": (0.0, 0.03)},
        prior_p={"grasped": 0.9},
    )
