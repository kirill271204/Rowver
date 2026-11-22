"""The fleet: one Mars world, one experience table, one brain per rover.

Driving (brain/: Brain + Hippocampus), AutoNav-style: each rover sees with its own navcam and
TabPFN picks every 1 s arc (rover_body.py); no site map. Every rover's drive arcs go into ONE Hippocampus
(runs/memory.csv). recall() gives a rover its own rows first, then the other rovers' rows of
the same family, so a patch one rover learned is in every rover's context at once.
  --isolated  each rover recalls only its own rows (no sharing: the baseline)
  --frozen    the context is fixed to the rows that existed at start (no adaptation)
Arm skill (so101/: the lab's TabPFN cerebellum and planner, unchanged): the pick table in
runs/, every row tagged with robot_id and a `site` feature (earth 0, mars 1, moon 2).
  --seed-from-lab  copy the lab's Earth pick rows in (robot_id "lab", site 0)
  --no-site        drop the site feature (the ablation)
"""
from __future__ import annotations

import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from brain.brain import Brain as DriveBrain
from brain.hippocampus import Hippocampus
from brain.nerves import LocalNerve
from brain.provenance import ROOT, Config

from . import world as W
from .dock import DockServer, RoverMemory
from .navcam import NavCam
from .navigation import approach_point
from .rover import Rover
from .rover_body import SKILL, Blocked, RoverBody, YieldTimeout

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "rowver.toml"
LAB = {"pick": ROOT / "so101" / "runs" / "memory_pick.csv"}


class FleetMemory(Hippocampus):
    def __init__(self, run_dir, fail_lookahead=1, isolated=False, frozen=False):
        super().__init__(run_dir, fail_lookahead)
        self.isolated, self.frozen = isolated, frozen
        self.n0 = len(self.df)

    def recall(self, robot_id, skill, family, descriptors, cap, **kw):
        if self.frozen:
            full, self.df = self.df, self.df.iloc[:self.n0]
            try:
                return self._recall(robot_id, skill, family, descriptors, cap, **kw)
            finally:
                self.df = full
        return self._recall(robot_id, skill, family, descriptors, cap, **kw)

    def _recall(self, robot_id, skill, family, descriptors, cap, **kw):
        if self.isolated:
            if self.df.empty:
                return self.df
            return self.df[(self.df.robot_id == robot_id) & (self.df.skill == skill)].tail(cap)
        return super().recall(robot_id, skill, family, descriptors, cap, **kw)


class Fleet:
    def __init__(self, rovers=("r1", "r2"), gravity="mars", gui=False, realtime=1.0, fresh=False,
                 isolated=False, frozen=False, use_site=True, seed_from_lab=False, seed=0, verbose=True,
                 mem_cls=None, zone=0, start_at=None, designs=None, run_name="fleet", panel=True):
        """designs: {rid: "scout" | "scientist" | "sampler"} -> the heterogeneous fleet: each rover
        has its own onboard table and TabPFN, and shares rows only through the dock (dock.py)."""
        self.cfg = Config(CONFIG)
        self.fleet_mode = designs is not None
        self.dir = ROOT / self.cfg("memory.dir") / (run_name if self.fleet_mode else "")
        if fresh and self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.use_site, self.verbose, self.seed = use_site, verbose, seed
        if seed_from_lab:
            self._seed_from_lab()
        self.w = W.build(rovers, gravity=gravity, gui=gui, realtime=realtime, seed=seed, zone=zone, start_at=start_at,
                         designs=designs)
        self.rv = {r: Rover(self.w, r, seed=seed + i) for i, r in enumerate(rovers)}
        self.cam = {r: NavCam(self.w, r, self.cfg, self.rv[r]) for r in rovers}    # each rover's own eyes
        if self.fleet_mode:
            self.dock = DockServer(self.dir, self.cfg, self.w)
            self.mem = self.dock.master                  # reports read what was uploaded
            rmem = {r: RoverMemory(self.dir / "rovers" / r, r, self.cfg("memory.fail_lookahead"), fit_dir=self.dir)
                    for r in rovers}
        else:
            self.dock = None
            self.mem = (mem_cls or FleetMemory)(self.dir, self.cfg("memory.fail_lookahead"), isolated=isolated,
                                                frozen=frozen)
            rmem = {r: self.mem for r in rovers}
        self.sched = None                        # hive.shard.Shard when the rovers run in parallel
        self.body = {r: RoverBody(self.rv[r], self.cam[r], self.cfg, use_site=use_site) for r in rovers}
        self.brain = {r: DriveBrain(LocalNerve(self.body[r]), self.cfg, seed=seed + i, verbose=False, mem=rmem[r])
                      for i, r in enumerate(rovers)}
        for b in self.brain.values():                # TabPFN context reloads, with the reason (news / staleness)
            b.on_refit = lambda ev: self.say(f"  [{ev['robot_id']} brain] {ev['what']} {ev['model']}: {ev['why']}")
        self.arm: dict = {}
        self.listeners: list = []                # callbacks(event: dict) for the viewer / logs
        self.legs = 0
        self.panel = None                        # the viewer's TabPFN decision panel (panel.py), --gui only
        if gui and panel and self.w.viewer is not None:
            from .panel import DecisionPanel
            self.panel = DecisionPanel(self)

    # ------------------------------------------------------------ helpers
    def say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def emit(self, ev: dict):
        for f in self.listeners:
            f(ev)

    def drive_rows(self, executed=False, rid=None):
        """The drive table; executed=True: only the arcs actually driven (no counterfactuals).
        rid (fleet mode): that rover's onboard table instead of the dock's."""
        df = self.brain[rid].mem.df if (rid is not None and self.fleet_mode) else self.mem.df
        d = df[df.skill == SKILL] if len(df) and "skill" in df else df.iloc[0:0]
        if executed and "source" in d:
            d = d[d.source.fillna("sim") != "counterfactual"]
        return d

    def set_phase(self, rid, phase, counterfactual=0, temperature=None):
        """practice: rollouts of K arcs not chosen per decision (simulation only) and a softmax
        choice with its propensity logged; mission: the argmax, no rollouts."""
        b = self.brain[rid]
        b.phase, b.counterfactual, b.temperature = phase, int(counterfactual), temperature

    # ------------------------------------------------------------ driving
    def drive_to(self, rid, park, label="", max_tries=6, tol=None) -> bool:
        """Drive to a waypoint (yaw None) or into the rover's garage bay (its park pose) by the
        rover's own decisions: navcam frames, TabPFN picks each 1 s arc. A rover in its bay first
        backs straight out onto the apron and turns outward; to dock it drives by arcs to its bay's
        apron point, then docks scripted: line up on the bay's axis, creep in, square up.
        Emits {"kind": "dest"} with the final destination (the panel's 3D marker), and xy None when done."""
        self.emit({"kind": "dest", "rid": rid, "xy": (float(park[0]), float(park[1])), "label": label})
        try:
            return self._drive_to(rid, park, label, max_tries, tol)
        finally:
            self.emit({"kind": "dest", "rid": rid, "xy": None, "label": label})

    def _drive_to(self, rid, park, label, max_tries, tol) -> bool:
        rv = self.rv[rid]
        self.leave_bay(rid)
        g = self.w.garage
        if g is not None and park[2] is not None and rid in g.park and np.hypot(*(np.subtract(park[:2], g.park[rid][:2]))) < 0.05:
            for wx, wy in self._route(rid, g.apron[rid])[:-1]:
                self._drive_leg(rid, (wx, wy, None), f"{label} (via the orbital route)", tol=0.35, max_tries=3)
            if not self._drive_leg(rid, (*g.apron[rid], None), "the apron of " + label, tol=0.3, max_tries=max_tries):
                return False
            line = float(np.hypot(*(np.subtract(g.apron[rid], park[:2]))))
            ok = rv.goto(*park, line_up=line)
            self.say(f"[{rid}] docked in {label}" + ("" if ok else " (docking timed out)"))
            return ok
        self.body[rid].leave_structure()
        if park[2] is None:                          # a waypoint: around the craters by the orbital route
            for wx, wy in self._route(rid, park[:2])[:-1]:
                self._drive_leg(rid, (wx, wy, None), f"{label} (via the orbital route)", tol=0.35, max_tries=3)
        return self._drive_leg(rid, park, label, tol=tol, max_tries=max_tries)

    def _route(self, rid, goal):
        """Waypoints to goal from the orbital DTM (navigation.orbital_route: 0.5 m blocks, walls
        steeper than the planner's grade veto avoided): the ground team's route, which the rover's own
        arcs then drive. A short leg goes straight."""
        from .navigation import orbital_route
        x, y, _ = self.rv[rid].pose()
        if np.hypot(goal[0] - x, goal[1] - y) < 1.5:
            return [tuple(goal)]
        pts = orbital_route(self.w.site, (x, y), goal, cell=0.5, spacing=1.0,
                            max_slope=self.cfg("navcam.grade_veto_deg"))
        if len(pts) > 1:
            self.say(f"[{rid}] orbital route: {len(pts) - 1} waypoints, "
                     + " -> ".join(f"({p[0]:.1f}, {p[1]:.1f})" for p in pts))
        return pts

    def in_bay(self, rid) -> bool:
        g = self.w.garage
        if g is None or rid not in g.park:
            return False
        from .garage import OUT
        lx, ly = g.to_local(self.rv[rid].pose()[:2])
        return bool(lx < OUT - 0.25 and abs(ly - g.bay_y(list(g.park).index(rid))) < 0.6)

    def leave_bay(self, rid):
        """Undock: reverse straight out of the bay onto its apron point, then turn to face out."""
        if not self.in_bay(rid):
            return False
        g, rv = self.w.garage, self.rv[rid]
        x, y, yaw = g.park[rid]
        rv.brake(False)
        rv.back_out(x, y, yaw, float(np.hypot(*(np.subtract(g.apron[rid], (x, y))))))
        rv.turn_to(g.yaw, tol=0.05)
        rv.brake(True)
        self.say(f"[{rid}] out of its bay, on the apron")
        return True

    def _open_ground(self, p, **kw) -> bool:
        from .survey import open_ground
        return open_ground(self.w.site, p, **kw)

    def _drive_leg(self, rid, park, label, tol=None, max_tries=6) -> bool:
        rv, body, brain = self.rv[rid], self.body[rid], self.brain[rid]
        if len(park) == 2:
            park = (park[0], park[1], None)
        goal = approach_point(park, self.cfg)
        body.set_goal(goal, tol)
        for attempt in range(max_tries):
            x, y, _ = rv.pose()
            if attempt == 0:
                rows = self.drive_rows(rid=rid)
                own = int((rows.robot_id == rid).sum()) if len(rows) else 0
                self.say(f"[{rid}] driving to {label}: {np.hypot(goal[0] - x, goal[1] - y):.1f} m, "
                         f"{len(rows)} drive rows in {'its own' if self.fleet_mode else 'the'} table ({own} own)")
            self.legs += 1
            try:
                brain.episode(SKILL, {"leg": self.legs}, mode="practice",
                              max_steps=self.cfg("practice.episode_steps"), on_step=self._seg_printer(rid))
            except YieldTimeout as e:
                self.say(f"[{rid}] {e} has blocked the way too long: turning away and trying again")
                body.escape()
                continue
            except Blocked as e:
                self.say(f"[{rid}] {e}: backing off and turning toward open ground")
                body.dead_ends.append(rv.pose()[:2])
                body.escape()
                continue
            if body.shield_trip:
                self.say(f"[{rid}] SHIELD stop ({body.shield_trip}): backing off and turning toward open ground")
                self.emit({"kind": "shield", "rid": rid, "xy": rv.true_pose()[:2], "why": body.shield_trip})
                body.dead_ends.append(rv.pose()[:2])
                body.escape()
                continue
            if body.aborted:
                xy = rv.true_pose()[:2]
                self.say(f"[{rid}] STUCK: backing out; the new rows change every rover's next arcs")
                self.emit({"kind": "stuck", "rid": rid, "xy": xy})
                body.dead_ends.append(rv.pose()[:2])
                body.back_out()
                continue
            if body.stalled:
                self.say(f"[{rid}] no progress toward {label} for a while: escape turn")
                body.escape()
                continue
            if body.to_go(*rv.pose()[:2]) < body.tol + 0.03:
                if park[2] is None:                    # a waypoint, not a station: no docking
                    rv.brake(True)
                    return True
                ok = rv.goto(*park)                    # docking (scripted, last ~55 cm)
                self.say(f"[{rid}] docked at {label}" + ("" if ok else " (docking timed out)"))
                return ok
        rv.brake(True)
        self.say(f"[{rid}] gave up driving to {label}")
        return False

    def _seg_printer(self, rid):
        def f(row):
            if row.get("source") == "counterfactual":
                self.say(f"      [{rid} not taken] turn {row['omega']:+.2f} speed {row['rep_speed']:.3f}: pred "
                         f"{row.get('pred_progress_n', float('nan')):4.2f} -> progress {row['progress_n']:5.2f}"
                         + ("  FAIL" if row["failed"] else ""))
                return
            pp, pf = row.get("pred_progress_n"), row.get("pred_p_fail")
            pred = f"pred {pp:4.2f} P(stuck) {pf:4.2f}" if pp is not None and not pd.isna(pp) else f"({row['how']})"
            tag = "arc, committed" if row.get("how") == "commit" else "arc"
            self.say(f"  [{rid} {tag}] {pred} -> progress {row['progress_n']:5.2f}"
                     + ("  FAIL" if row["failed"] else "") + (f" (shield: {row['rep_shield']})" if row.get("rep_shield") else "")
                     + f"  speed {row['rep_speed']:.3f} m/s turn "
                     f"{row['omega']:+.2f} rad/s  step {row['step']:.3f} seen {row['seen']:.2f}  to go {row['rep_to_go']:.2f} m")
            self.emit({"kind": "segment", "rid": rid, "row": row})
        return f

    # ------------------------------------------------------------ arm brains
    def _seed_from_lab(self):
        for skill, src in LAB.items():
            dst = self.dir / f"memory_{skill}.csv"
            if dst.exists() and dst.stat().st_size:
                old = pd.read_csv(dst)
                if "robot_id" in old and (old.robot_id == "lab").any():
                    continue
            if not src.exists():
                print(f"(no lab {skill} rows at {src})")
                continue
            lab = pd.read_csv(src).assign(robot_id="lab", site=0)
            old = pd.read_csv(dst) if dst.exists() and dst.stat().st_size else pd.DataFrame()
            pd.concat([lab, old], ignore_index=True).to_csv(dst, index=False)
            self.say(f"seeded {len(lab)} lab {skill} rows (robot_id=lab, site=0 Earth) into {dst.name}")

    def arm_brain(self, rid, kind="pick", frozen=False):
        """The SO-101 on rover rid (MountedPicker), with the lab's so101 Brain for `kind` (pick)."""
        if (rid, kind) in self.arm:
            return self.arm[(rid, kind)]
        from .arm import MountedPicker
        from .arm_brain import ArmBrain
        robot = MountedPicker(self.w, rid, seed=self.seed)
        b = ArmBrain(robot, kind, self.dir, use_site=self.use_site, frozen=frozen, seed=self.seed)
        self.arm[(rid, kind)] = b
        return b

    def close(self):
        for c in self.cam.values():
            c.close()
        self.w.close()
