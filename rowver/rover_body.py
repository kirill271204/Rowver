"""A rover as a brain/ Body: the learned skill "drive_arc" (family rover_arc), AutoNav-style.

There is no site map and no path planner. Every second the rover looks with its own navcam
(navcam.py), and TabPFN chooses the arc to drive next:
  action      d_speed (fraction of the nominal speed, <= 0: slow down) and omega (turn rate, rad/s),
              held for one segment
  features    per candidate arc: what the navcam saw along it (DINOv2 codes, colour, texture,
              slope, step, roughness, how much of it was seen), plus the ground under the rover
              (seen earlier), the IMU, payload, site
  targets     progress_n  distance made good along the arc's chord / commanded distance (1 - slip)
              drift       sideways distance / commanded distance
              tilt_max    peak chassis tilt (rad)
  fail        stuck (progress_n < stuck_progress) or tipped (tilt > tip_deg)
The body adds the goal term to TabPFN's prediction: score += w_dir * E[progress] * speed * (how much
the arc's end is closer to the goal). Scripted reflexes, not decisions: a point turn toward the
goal when it is far off the nose, a geometric veto of arcs whose step exceeds the wheels'
clearance or whose drop is too deep (like AutoNav's hazard check), a veto of arcs whose
predicted q90 tilt is past the rover's tip limit, and an escape turn when every arc is vetoed.
The safety shield (safety.py) watches every segment while it is driven and stops it (rep_shield).
Simulation only: counterfactual(actions) rolls out arcs NOT chosen from a snapshot of this moment
and restores the world after each (practice rows with source "counterfactual").
No terrain label and no physics number enters the table. All rovers write to one Hippocampus.
"""
from __future__ import annotations

import copy

import numpy as np

from brain.nerves import SkillSchema
from brain.provenance import Tagged

from .rover import wrap
from .safety import Shield

SKILL = "drive_arc"
HW = "hardware:rowver/mars_rover.py"


class YieldTimeout(Exception):
    """Raised by a traffic guard when another rover has blocked the way too long."""


class Blocked(Exception):
    """Every candidate arc is unsafe even after looking around: never drive one of them."""


class RoverBody:
    def __init__(self, rover, navcam, cfg, use_site=True):
        self.rv, self.cam, self.cfg = rover, navcam, cfg
        self.robot_id = rover.rid
        m = rover.w.m
        k = rover.kin
        self.descriptors = {
            "rover_mass": Tagged(float(m.body_subtreemass[rover.ch]), HW),
            "wheel_radius": Tagged(float(np.mean(k.R)), HW),
            "wheel_base_r": Tagged(float(np.linalg.norm(k.r, axis=1).mean()), HW),
            "track": Tagged(float(np.ptp(k.r[:, 1])), HW),
        }
        T = "user:" + cfg.path.name + "#lekiwi.drive."
        feats = [f for f in navcam.features if use_site or f != "site"]
        self.v_nom = Tagged(cfg("lekiwi.drive.speed_frac") * k.v_max, HW + " (top speed) x " + T + "speed_frac")
        wlim = cfg("lekiwi.drive.omega_frac") * k.w_max
        self.skills = {SKILL: SkillSchema(
            name=SKILL, family="rover_arc",
            actions={"d_speed": Tagged(tuple(cfg("lekiwi.drive.d_speed")), T + "d_speed"),
                     "omega": Tagged((-wlim, wlim), HW + " (top turn rate) x " + T + "omega_frac")},
            features=feats, targets=["progress_n", "drift", "tilt_max"],
            goal={"progress_n": "goal_progress"}, minimize=["drift", "tilt_max"], fail="fail",
            info={"segment_s": cfg.tag("lekiwi.drive.segment_s"), "v_nominal": self.v_nom,
                  "v_max": Tagged(float(k.v_max), HW), "w_max": Tagged(float(k.w_max), HW)})}
        self.goal = None
        self.tol = cfg("lekiwi.drive.arrive_tol")
        self.last = None
        self.stuck_run = 0
        self.aborted = self.stalled = False
        self.dead_ends: list = []                    # where this rover got stuck: arcs through them cost more
        self.on_segment = None                       # callback(dict) for printing / overlays
        self.fan = None                              # the last decision's candidate arcs (viewers only)
        self.guard = None                            # hive traffic: guard(body, heading) may wait or raise YieldTimeout
        self.shield = Shield(rover, navcam, cfg)     # its limits exist either way; it watches only when enabled
        self.shield_on = bool(cfg.opt("safety.shield", True, env="CORTEX_SHIELD"))
        self.lcb = cfg.opt("objective.risk", "mean", env="CORTEX_RISK") == "lcb"
        self.vetoes = bool(cfg.opt("safety.vetoes", True, env="CORTEX_VETOES"))     # drop + q90 tilt (V1: off)
        self.shield_trip = None

    # ------------------------------------------------------------ goal
    def set_goal(self, xy, tol=None):
        self.goal = np.asarray(xy[:2], float)
        self.tol = tol if tol is not None else self.cfg("lekiwi.drive.arrive_tol")

    def to_go(self, x, y):
        return float(np.hypot(*(self.goal - [x, y])))

    def speeds(self, A, to_go):
        T = self.cfg("lekiwi.drive.segment_s")
        cap = (to_go + 0.06) / (self.v_nom.value * T)          # no overshoot on the last metres
        return self.v_nom.value * np.clip(np.minimum(1.0 + A[:, 0], cap), 0.15, 1.0)

    # ------------------------------------------------------------ Body contract
    def reset(self):
        self.stuck_run, self.aborted, self.stalled, self.last = 0, False, False, None
        self.shield_trip = None
        self._best, self._since = np.inf, 0
        return {}

    def context(self, skill, goal) -> dict:
        cfg = self.cfg
        x, y, yaw = self.rv.pose()
        bearing = float(np.arctan2(*(self.goal - [x, y])[::-1]))
        if self.guard is not None:
            self.guard(self, bearing)
        if abs(float(wrap(bearing - yaw))) > cfg("lekiwi.drive.turn_first"):
            self.rv.turn_to(bearing, tol=0.15, monitor=self._tilt_watch())   # scripted point turn, as AutoNav does
        for k in range(3):                                   # look; if every arc is unsafe, turn and look again
            self.cam.update()
            x, y, yaw = self.rv.pose()
            to_go = self.to_go(x, y)
            fan = np.column_stack([np.zeros(15), np.linspace(*self.skills[skill].actions["omega"].value, 15)])
            _, ex = self.cam.arcs(self.speeds(fan, to_go), fan[:, 1], x, y, yaw, self._horizon(to_go))
            if (self._safe(ex)).any():
                break
            if k == 2:
                raise Blocked(f"every arc is blocked (smallest step {float(ex['step'].min()):.2f} m)")
            side = 1.0 if float(wrap(bearing - yaw)) >= 0 else -1.0
            self.rv.turn_to(yaw + side * np.radians(50), tol=0.15)
        return self._build_ctx()

    def _build_ctx(self):
        """The context from the local map as it is now (no new frame): base readings + the per-arc
        features and scorer the planner calls for its candidates."""
        cfg = self.cfg
        x, y, yaw = self.rv.pose()
        to_go = self.to_go(x, y)
        bearing = float(np.arctan2(*(self.goal - [x, y])[::-1]))
        pitch, roll = self.rv.imu_tilt()
        base = self.cam.under(x, y) | {"imu_pitch": pitch, "imu_roll": roll, "payload": float(self.rv.carrying),
                                       "site": float(self.rv.w.site_code)}
        h = self._horizon(to_go)

        def per_action(A):
            A = np.asarray(A, float)
            v = self.speeds(A, to_go)
            F, ex = self.cam.arcs(v, A[:, 1], x, y, yaw, h)
            gain = (to_go - np.hypot(*(ex["end"] - self.goal).T)) / ex["L"]       # closer to the goal, per m
            pen = np.zeros(len(A))
            for dx, dy in self.dead_ends:
                sx, sy = ex["samples"]
                near = np.hypot(sx - dx, sy - dy).min(1) < 0.35
                pen -= cfg("navcam.w_dead") * near
            safe = self._safe(ex)
            fan = None
            if len(A) > 1:                                   # the planner's candidates (not one arc's features)
                fan = self.fan = {"A": A.copy(), "samples": (ex["samples"][0].copy(), ex["samples"][1].copy()),
                                  "safe": safe.copy(), "v": v.copy(), "gain": np.asarray(gain, float).copy(),
                                  "pen": pen.copy(), "veto": self._veto_reasons(ex), "pose": (x, y, yaw),
                                  "to_go": to_go, "goal": tuple(self.goal), "w_dir": cfg("navcam.w_dir"),
                                  "v_nom": self.v_nom.value, "tilt_limit": float(self.shield.tilt_limit)}

            def extra(q, p_fail):
                qp = np.quantile(q["progress_n"], 0.1, axis=1) if self.lcb else q["progress_n"].mean(1)
                prog = np.clip(qp, 0.0, 1.2)
                s = cfg("navcam.w_dir") * prog * (v / self.v_nom.value) * gain + pen
                tilt_ok = (np.quantile(q["tilt_max"], 0.9, axis=1) <= self.shield.tilt_limit) | (not self.vetoes)
                ok = safe & tilt_ok
                if fan is not None:
                    fan["tilt_veto"] = safe & ~tilt_ok       # the map allowed it; TabPFN's tilt q90 did not
                return np.where(ok, s, -1e6)                 # unsafe arcs lose to every safe one
            for k_, v_ in base.items():
                F[k_] = v_
            return F, extra

        self.last = {"x": x, "y": y, "yaw": yaw, "to_go": to_go, "bearing": bearing}
        return base | {"_per_action": per_action, "goal_progress": 1.0, "x": x, "y": y, "to_go": to_go,
                       "heading": bearing}

    def commit(self, skill, action, pred, out, n_done):
        """After a segment: drive the same arc on, without a new frame or a new plan, if that is
        safe and the brain was sure -> the context for the next segment, else None (decide anew).
        The arc's ground was judged out to the horizon when it was chosen; a commit covers at most
        commit_max more segments of it, well inside that horizon, and the shield watches each."""
        cfg = self.cfg
        if n_done >= cfg("lekiwi.drive.commit_max") or out["failed"] or out["shield"] or out["done"]:
            return None
        if self.stuck_run or not pred or "pred_p_fail" not in pred:
            return None
        q10, q90 = pred.get("pred_progress_n_q10", 0.0), pred.get("pred_progress_n_q90", 1.0)
        sure = (pred["pred_p_fail"] < cfg("lekiwi.drive.commit_p_fail") and q10 >= cfg("lekiwi.drive.commit_q10")
                and q90 - q10 <= cfg("lekiwi.drive.commit_spread")
                and abs(out["progress_n"] - pred.get("pred_progress_n", 0.0)) <= cfg("lekiwi.drive.commit_surprise")
                and pred.get("pred_tilt_max_q90", 0.0) <= self.shield.tilt_limit)
        if not sure:
            return None
        x, y, yaw = self.rv.pose()
        to_go = self.to_go(x, y)
        bearing = float(np.arctan2(*(self.goal - [x, y])[::-1]))
        if to_go < cfg("lekiwi.drive.commit_min_to_go") or abs(float(wrap(bearing - yaw))) > cfg("lekiwi.drive.turn_first"):
            return None
        A = np.array([[action["d_speed"], action["omega"]]])
        _, ex = self.cam.arcs(self.speeds(A, to_go), A[:, 1], x, y, yaw, self._horizon(to_go))
        if not self._safe(ex)[0]:
            return None
        return self._build_ctx()

    def _horizon(self, to_go):
        return float(min(self.cfg("navcam.horizon"), to_go + 0.10))

    def _veto_reasons(self, ex) -> dict:
        """Per arc, which map check would veto it (the same checks as _safe; for viewers)."""
        c = self.cfg
        out = {"step": ex["step"] >= c("navcam.step_veto"), "seen": ex["seen"] < c("navcam.min_seen"),
               "hit": np.asarray(ex["hit"], bool)}
        if self.vetoes:
            out["drop"] = ex["drop"] >= c("navcam.drop_veto")
            out["grade"] = ex["grade"] >= np.tan(np.radians(c("navcam.grade_veto_deg")))
        return {k: np.asarray(v, bool).copy() for k, v in out.items()}

    def _safe(self, ex):
        drop_ok = (ex["drop"] < self.cfg("navcam.drop_veto")) if self.vetoes else True
        if self.vetoes:                                   # never down (or up) a slope it could not climb back
            drop_ok = drop_ok & (ex["grade"] < np.tan(np.radians(self.cfg("navcam.grade_veto_deg"))))
        return ((ex["step"] < self.cfg("navcam.step_veto")) & drop_ok
                & (ex["seen"] >= self.cfg("navcam.min_seen")) & ~ex["hit"])

    def leave_structure(self):
        """Reflex before a new leg: if a plinth is right in front (after working at a station),
        back away from it first so the arcs have room."""
        x, y, yaw = self.rv.pose()
        for ox, oy, r in self.cam.infra:
            d = np.hypot(ox - x, oy - y)
            if d < r + 0.55 and np.cos(np.arctan2(oy - y, ox - x) - yaw) > 0.3:
                self.rv.drive(-0.7 * self.v_nom.value, 0.0, 0.0, 3.0)
                self.rv.brake(True)
                return True
        return False

    def _segment(self, action, camera=True) -> dict:
        """Drive one arc for segment_s and measure it: progress along the commanded chord, drift
        across it, peak tilt, failed (stuck, tipped or stopped by the shield)."""
        L = self.last
        T = self.cfg("lekiwi.drive.segment_s")
        A = np.array([[action["d_speed"], action["omega"]]])
        v = float(self.speeds(A, L["to_go"])[0])
        w = float(action["omega"])
        x0, y0, yaw0 = self.rv.true_pose()[:3]
        if abs(w) < 1e-6:                                    # the arc's chord after T seconds (world frame)
            ch = np.array([v * T * np.cos(yaw0), v * T * np.sin(yaw0)])
        else:
            r = v / w
            ch = np.array([r * (np.sin(yaw0 + w * T) - np.sin(yaw0)), -r * (np.cos(yaw0 + w * T) - np.cos(yaw0))])
        mon = self.shield.monitor(v, w, self._horizon(L["to_go"]), camera=camera) if self.shield_on else None
        o = self.rv.drive(v, 0.0, w, T, monitor=mon)
        disp = np.array([o["x"] - x0, o["y"] - y0])
        cmd_d = max(float(np.hypot(*ch)), 1e-6)
        u = ch / cmd_d
        prog = float(disp @ u / cmd_d)
        drift = float(abs(disp[0] * u[1] - disp[1] * u[0]) / cmd_d)
        failed = (prog < self.cfg("lekiwi.drive.stuck_progress") or o["tilt_max"] > np.radians(self.cfg("lekiwi.drive.tip_deg"))
                  or bool(o["shield"]))
        return {"progress_n": prog, "drift": drift, "tilt_max": float(o["tilt_max"]), "failed": bool(failed),
                "true_dx": float(disp @ u), "duration": float(o["elapsed"]), "speed": v, "shield": o["shield"] or "",
                "exec": {"vx": v, "vy": 0.0, "wz": w}}

    def counterfactual(self, skill, actions) -> list:
        """Simulation only: the outcome of each arc in `actions` driven from THIS moment, each from
        the same snapshot; the world, the local map, the pose noise and the steering are restored
        after each, so the arc chosen next is driven exactly as if none of these had run."""
        w = self.rv.w
        if not actions or w.sched is not None or self.last is None:
            return []
        snap = w.snapshot()
        keep = (copy.deepcopy(self.cam.map), self.rv.rng.bit_generator.state, self.rv.kin._delta.copy(),
                self.rv.braked, self.cam.frames)

        def back():
            w.restore(snap)
            self.cam.map = copy.deepcopy(keep[0])
            self.rv.rng.bit_generator.state = keep[1]
            self.rv.kin._delta = keep[2].copy()
            self.rv.braked = keep[3]
            self.cam.frames = keep[4]
        outs = []
        w.quiet = True
        try:
            for a in actions:
                back()
                o = self._segment(a)
                xe, ye, _ = self.rv.pose()
                o |= {"to_go": self.to_go(xe, ye), "done": False, "arrived": False, "aborted": False}
                outs.append(o)
        finally:
            back()
            w.quiet = False
        return outs

    def act(self, skill, action) -> dict:
        L = self.last
        seg = self._segment(action)
        prog, failed = seg["progress_n"], seg["failed"]
        if seg["shield"]:
            self.shield_trip = seg["shield"]
            self.shield.trips.append({"xy": self.rv.true_pose()[:2], "why": seg["shield"]})
        self.stuck_run = self.stuck_run + 1 if failed else 0
        xe, ye, _ = self.rv.pose()
        to_go = self.to_go(xe, ye)
        arrived = to_go < self.tol
        self.aborted = self.stuck_run >= self.cfg("lekiwi.drive.stuck_segments")
        if to_go < self._best - 0.05:
            self._best, self._since = to_go, 0
        else:
            self._since += 1
        self.stalled = self._since >= self.cfg("lekiwi.drive.stall_segments")
        out = seg | {"done": bool(arrived or self.aborted or self.stalled or self.shield_trip),
                     "arrived": bool(arrived), "aborted": bool(self.aborted), "to_go": to_go}
        if self.on_segment:
            self.on_segment(self, L, action, out)
        return out

    # ------------------------------------------------------------ scripted recovery
    def _tilt_watch(self, rise=np.radians(3.0)):
        """A tilt watcher for reflex moves (backing off, point turns): stop when the tilt is over the
        shield's limit AND rising (3 deg above where the move began). A rover already tilted past the
        limit (resting on a rock) may still move to get off it; it may not tip further."""
        if not self.shield_on:
            return None
        sh = self.shield
        start = sh.tilt()

        def f(t):
            tilt = sh.tilt()
            if tilt > sh.tilt_limit and tilt > start + rise:
                return f"tilt {np.degrees(tilt):.0f} deg > limit {np.degrees(sh.tilt_limit):.0f} (rising)"
            return None
        return f

    def back_out(self, seconds=3.0):
        """Reverse straight back the way it came (a reflex after a stuck abort; not a learned row)."""
        v = 0.7 * self.v_nom.value
        self.rv.drive(-v, 0.0, 0.0, seconds, monitor=self._tilt_watch())
        self.rv.brake(True)

    def _fan(self, n=15):
        """Look, then the candidate fan from here: (omegas, safe mask, extras)."""
        self.cam.update()
        x, y, yaw = self.rv.pose()
        w = np.linspace(*self.skills[SKILL].actions["omega"].value, n)
        v = np.full(n, self.v_nom.value)
        _, ex = self.cam.arcs(v, w, x, y, yaw, self.cfg("navcam.horizon"))
        return w, self._safe(ex), ex

    def escape(self, side=None):
        """A stall or a blocked view: back off, then turn 60 deg to the side whose ground (judged on
        the local map from here, before turning: a pivot onto a rim can tip a tall rover) is safe and
        gentlest, and drive on there with the shield watching. Turns are watched for tilt; with no
        safe side it backs off further instead (an escape is never driven blind)."""
        self.back_out(2.0)
        self.cam.update()
        x, y, yaw = self.rv.pose()
        v = self.v_nom.value
        h = self.cfg("navcam.horizon")
        sides = (side,) if side is not None else (1.0, -1.0)
        options = []
        for s in sides:
            _, ex = self.cam.arcs(np.array([v]), np.array([0.0]), x, y, yaw + s * np.radians(60), h)
            if self._safe(ex)[0]:
                options.append((float(ex["grade"][0]) + float(ex["step"][0]), s))
        if not options:
            self.back_out(2.0)
            return
        if len(options) == 2 and abs(options[0][0] - options[1][0]) < 0.02:   # a tie: alternate
            side = 1.0 if len(self.dead_ends) % 2 == 0 else -1.0
        else:
            side = min(options)[1]
        if not self.rv.turn_to(yaw + side * np.radians(60), tol=0.15, monitor=self._tilt_watch()):
            self.back_out(1.0)                              # the turn tipped it: undo by backing off
            return
        w, safe, _ = self._fan()
        if safe[len(w) // 2]:
            mon = self.shield.monitor(v, 0.0, h) if self.shield_on else None
            self.rv.drive(v, 0.0, 0.0, 3.0, monitor=mon)
        self.rv.brake(True)


def heading_error(a, b):
    return float(abs(wrap(a - b)))
