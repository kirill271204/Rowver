"""The fleet's sorties: undock with a briefing, do the job, dock, upload.

  scout_sortie      Pathfinder drives to candidate drill sites first and runs a trafficability probe
                    at each (a short straight drive: slip from odometry vs motion, sinkage from its
                    wheel camera) -> site rows with sinkage_mm, slip
  survey_sortie     Spectra parks at vantage points and reads the candidate sites (and the ground
                    around them) with its mast spectrometer, 0.5-4 m away -> photochem_score, mineral
  sample_sortie     Hauler: briefing 1 (site rows over the zone) -> its own TabPFN picks the drill
                    sites -> briefing 2 (drive rows along the route to them) -> drives there, drills
                    (drill_torque_nm, sample_fail) -> picks a loose rock its navcam sees near the
                    last hole with the arm (the lab's TabPFN grasp), folds holding it, drives home and
                    sets it in the tray of its bay. --no-brief: no packs; with no photochem model it
                    takes the nearest candidates (a seeded tie-break).
Every sortie starts by backing out of its garage bay and ends docked in it, where the dock takes
its new rows. Sim-side truth (geology, terrain class) only labels the events for the report; no
rover reads it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geology import Geology
from .world import HOLE_MARKERS
from .rover_body import SKILL as DRIVE
from .survey import FEATURES, SKILL as SITE, TARGETS, FAIL, Orbital, SiteSurvey, drill_candidates, site_row


CORE_DEPTH = 0.04          # m: how deep the drill cuts
DRILL_SPIN = 25.0          # rad/s: the bit's speed while cutting


class FleetOps:
    """Shared per-run state for the sorties (geology, orbital data, candidates, site models)."""

    def __init__(self, fleet, seed=0, brief=True):
        self.f, self.brief = fleet, brief
        self.geo = Geology(fleet.w, seed=seed)
        self.orb = Orbital(fleet.w)
        self.cands = drill_candidates(fleet)
        self.site = {r: SiteSurvey(r, fleet.brain[r].mem, fleet.cfg) for r in fleet.rv}
        self.rng = np.random.default_rng(seed)
        self.stucks = {r: 0 for r in fleet.rv}
        fleet.listeners.append(self._on_event)
        self.sorties = {r: 0 for r in fleet.rv}
        self.holes, self.slots = {}, {}                # drilled holes and filled carousel slots per rover

    def _on_event(self, ev):
        if ev.get("kind") == "stuck":
            self.stucks[ev["rid"]] = self.stucks.get(ev["rid"], 0) + 1

    def mem(self, rid):
        return self.f.brain[rid].mem

    def _next_step(self, rid):
        df = self.mem(rid).df
        own = df[(df.robot_id == rid) & (df.skill == SITE)] if len(df) and "skill" in df else df
        return len(own)

    # ---------------------------------------------------------------- dock round trip
    def undock(self, rid, mission):
        """Briefing (or none) -> the rover's onboard table for this sortie."""
        f = self.f
        if not self.brief:
            self.mem(rid).load_pack(None, keep_own=True)
            self.mem(rid).pack_info = {"rows": 0}
            f.say(f"[{rid}] undocking without a briefing (own rows only: {len(self.mem(rid).df)})")
            return
        b = f.brain[rid]
        pack = f.dock.brief(rid, b.desc, mission)
        self.mem(rid).load_pack(pack, keep_own=False)
        info = f.dock.log[-1]
        self.mem(rid).pack_info = info
        f.say(f"[{rid}] DOCK briefing: {info.get('rows', 0)} rows (cap {info.get('cap')}) | "
              + " | ".join(f"{k}: {v.get('rows', 0)} rows, {v.get('in_corridor', 0)} in corridor, "
                           f"{v.get('rare', 0)} rare, from {v.get('by_robot', {})}"
                           for k, v in info.items() if isinstance(v, dict)))

    def dock_home(self, rid, drive=True, ok=True):
        """Drive into its bay (drive=False: already there, e.g. a rock was carried home) and upload."""
        f = self.f
        if drive:
            ok = f.drive_to(rid, f.w.start[rid], "its garage bay")
        f.rv[rid].brake(True)
        n = f.dock.upload(rid, f.rv[rid], self.mem(rid))
        f.say(f"[{rid}] {'docked' if ok else 'NOT in its bay'}: uploaded {n} rows to the dock "
              f"(master table {len(f.dock.master.df)} rows)")
        return ok, n

    def mission(self, rid, route, skills, targets=None):
        b = self.f.brain[rid]
        cols = {DRIVE: list(b.wm[DRIVE].columns), SITE: FEATURES}
        return {"route": route, "skills": skills, "targets": targets or {}, "columns": cols}

    def drive_mission(self, rid, waypoints, label="sortie"):
        """Briefing for a drive: drive rows along bay -> waypoints -> bay."""
        bay = self.f.w.start[rid][:2]
        return self.mission(rid, [bay] + [tuple(p) for p in waypoints] + [bay], {DRIVE: 1.0})

    # ---------------------------------------------------------------- the instruments
    def probe(self, rid, secs=1.5):
        """Trafficability probe (scout): a short straight drive; slip from odometry vs motion
        (visual odometry on a real rover), sinkage from the wheel camera."""
        f = self.f
        rv = f.rv[rid]
        v = 0.6 * rv.kin.v_max
        rv.drive(v, 0.0, 0.0, 0.6)                          # up to speed first (the drive's ramp)
        o = rv.drive(v, 0.0, 0.0, secs)
        rv.brake(True)
        slip = float(np.clip(1.0 - np.hypot(o["dx"], o.get("dy", 0.0)) / (v * secs), 0.0, 1.0))
        m, d = f.w.m, f.w.d
        sink = []
        for g in f.w.wheels[rid]:
            x, y, z = d.geom_xpos[g]
            sink.append(float(m.geom_size[g][0]) - (z - f.w.ground_z(x, y)))
        x, y, _ = rv.pose()
        return x, y, {"slip": slip, "sinkage_mm": 1000.0 * max(0.0, float(np.mean(sink)))}

    def drill(self, rid):
        """Core drill (sampler), the way a coring drill works:
        1. deploy: lower the head until the bit's tip touches the ground (its own contact, no map);
        2. spin up the bit; 3. feed CORE_DEPTH into the ground, slower the harder it cuts (the torque);
        4. stop, break the core off; 5. retract (the core shows in the bit if it held together),
        turn the carousel one slot to store it; the hole stays marked on the ground."""
        from .mars_rover import CAROUSEL_SLOTS, DRILL_TRAVEL
        f, rv = self.f, self.f.rv[rid]
        w = f.w
        m, d = w.m, w.d
        act = {n: m.actuator(f"{rid}/{n}").id for n in ("drill", "drill_spin", "carousel")}
        qa = m.joint(f"{rid}/drill_slide").qposadr[0]
        tip, core = m.geom(f"{rid}/drill_tip").id, m.geom(f"{rid}/drill_core").id
        dt = m.opt.timestep
        rv.brake(True)

        def touching():
            n = d.ncon
            g = np.column_stack([d.contact.geom1[:n], d.contact.geom2[:n]]) if n else np.empty((0, 2), int)
            return bool((g == tip).any())

        def ramp(a, to, seconds, stop=None):
            """Move actuator a's setpoint to `to` over `seconds` (10-step chunks); stop() -> True ends it."""
            c0, n = float(d.ctrl[a]), max(1, int(seconds / dt))
            for k in range(0, n, 10):
                d.ctrl[a] = c0 + (to - c0) * min(1.0, (k + 10) / n)
                w.step(10)
                if stop is not None and stop():
                    return True
            return False

        def tip_contact(on):
            """The tip's contact switch (the geom's masks and its body's, which the broad phase checks)."""
            b = m.geom_bodyid[tip]
            m.geom_contype[tip] = m.geom_conaffinity[tip] = int(on)
            m.body_contype[b] = m.body_conaffinity[b] = int(on)

        # 1. deploy: down at 4 cm/s until the tip touches
        tip_contact(True)
        t0 = w.sim_time
        hit = ramp(act["drill"], -DRILL_TRAVEL, DRILL_TRAVEL / 0.04, stop=touching)
        q_touch = float(d.qpos[qa])
        d.ctrl[act["drill"]] = q_touch
        tip_contact(False)
        x, y, zt = map(float, d.geom_xpos[tip])
        z_surf = zt - float(m.geom_size[tip][0])
        r = self.geo.drill(x, y)
        tq = r["drill_torque_nm"]
        ok = bool(r["sample_success"]) and hit
        # 2. spin up; 3. feed: the harder the ground, the slower the bit advances
        ramp(act["drill_spin"], DRILL_SPIN, 0.6)
        feed_s = float(np.clip(1.0 + 1.6 * tq, 2.0, 8.0))
        depth = min(CORE_DEPTH, DRILL_TRAVEL + q_touch) if hit else 0.0
        if depth > 0:
            ramp(act["drill"], q_touch - depth, feed_s)
        # 4. stop the bit, break the core off
        ramp(act["drill_spin"], 0.0, 0.5)
        w.step(int(0.4 / dt))
        if ok:
            m.geom_rgba[core][3] = 1.0
        # 5. retract; store the core in the next carousel slot; mark the hole
        ramp(act["drill"], 0.0, 3.0)
        if hit:
            k = self.holes.get(rid, 0)
            self.holes[rid] = k + 1
            mid = m.body_mocapid[m.body(f"hole_{rid}_{k % HOLE_MARKERS}").id]
            d.mocap_pos[mid] = [x, y, z_surf]
        slot = self.slots.get(rid, 0)
        if ok:
            slot += 1
            self.slots[rid] = slot
            ramp(act["carousel"], slot * 2 * np.pi / CAROUSEL_SLOTS, 1.5)
            m.geom_rgba[core][3] = 0.0
        w.step(int(0.3 / dt))
        f.say(f"[{rid} drill] " + (f"tip on the ground at z {z_surf:.3f} m, " if hit else "NO ground contact, ")
              + f"cut {depth:.3f} m in {feed_s:.1f} s at {tq:.2f} N m, core {'INTACT' if ok else 'collapsed'}"
              + (f", stored in carousel slot {slot}/{CAROUSEL_SLOTS}" if ok else "")
              + f" ({w.sim_time - t0:.1f} s)")
        return x, y, {"drill_torque_nm": tq, FAIL: 1.0 - float(ok), "sample_success": float(ok)}

    def record(self, rid, x, y, **readings):
        row = site_row(rid, self.sorties[rid], self._next_step(rid), x, y, self.orb, **readings)
        self.mem(rid).remember(row)
        return row


# ===================================================================== sorties
def scout_sortie(ops: FleetOps, rid, n=6):
    f = ops.f
    ops.sorties[rid] += 1
    done = ops.mem(rid).df
    seen = done[done.skill == SITE][["ctx_x", "ctx_y"]].to_numpy(float) if len(done) and "skill" in done else np.empty((0, 2))
    pts, avoid = [], list(seen)
    x, y, _ = f.rv[rid].pose()
    for _ in range(n):                                   # farthest-first over the candidate sites
        A = np.array(avoid + [[x, y]] + pts)
        d = np.hypot(ops.cands[:, None, 0] - A[None, :, 0], ops.cands[:, None, 1] - A[None, :, 1]).min(1)
        pts.append(list(ops.cands[int(np.argmax(d))]))
    ops.undock(rid, ops.drive_mission(rid, pts))
    t0 = f.w.sim_time
    for k, (tx, ty) in enumerate(pts):
        ok = f.drive_to(rid, (tx, ty, None), f"scouting site {k + 1}/{n}", tol=0.3)
        px, py, r = ops.probe(rid)
        ops.record(rid, px, py, **r)
        gt = ops.geo.truth(px, py)
        f.say(f"[{rid}] probe at ({px:.2f}, {py:.2f}): slip {r['slip']:.2f}, sinkage {r['sinkage_mm']:.1f} mm "
              f"(truth, report only: {gt['ground']})")
        yield {"rid": rid, "job": "scout", "event": "probe", "ok": ok, **r, "ground": gt["ground"]}
    ok, n_up = ops.dock_home(rid)
    yield {"rid": rid, "job": "scout", "event": "sortie", "ok": ok, "uploaded": n_up, "seconds": f.w.sim_time - t0,
           "stucks": ops.stucks[rid]}


def survey_sortie(ops: FleetOps, rid, n_vantage=2, per_vantage=8):
    f = ops.f
    ops.sorties[rid] += 1
    C = ops.cands
    lo, hi = 1.0, 3.5                                       # read inside the spectrometer's best range
    # vantage points: open candidates of the grid with the most unread candidate sites in range
    x0, x1, y0, y1 = f.w.zone
    gx, gy = np.meshgrid(np.linspace(x0 + 0.6, x1 - 0.6, 15), np.linspace(y0 + 0.6, y1 - 0.6, 11))
    V = np.column_stack([gx.ravel(), gy.ravel()])
    V = V[[f._open_ground(p) for p in V]]
    read = np.zeros(len(C), bool)
    vps = []
    for _ in range(n_vantage):
        D = np.hypot(V[:, None, 0] - C[None, :, 0], V[:, None, 1] - C[None, :, 1])
        gain = ((D > lo) & (D < hi) & ~read[None]).sum(1)
        i = int(np.argmax(gain))
        vps.append(V[i])
        read |= (D[i] > lo) & (D[i] < hi)
    ops.undock(rid, ops.drive_mission(rid, vps))
    t0 = f.w.sim_time
    for k, vp in enumerate(vps):
        ok = f.drive_to(rid, (float(vp[0]), float(vp[1]), None), f"vantage point {k + 1}/{n_vantage}", tol=0.3)
        x, y, _ = f.rv[rid].pose()
        D = np.hypot(C[:, 0] - x, C[:, 1] - y)
        tg = [tuple(C[i]) for i in np.argsort(D) if lo < D[i] < hi][:per_vantage]
        a = ops.rng.uniform(0, 2 * np.pi, 4)                   # and the ground around (the field between sites)
        r = ops.rng.uniform(lo, hi, 4)
        tg += [(x + rr * np.cos(aa), y + rr * np.sin(aa)) for aa, rr in zip(a, r)]
        got = 0
        for tx, ty in tg:
            s = ops.geo.spectrometer((x, y), (tx, ty))
            if s is None:
                continue
            ops.record(rid, tx, ty, photochem_score=s["photochem_score"], mineral_class=s["mineral_class"],
                       spec_range=s["spec_range"])
            got += 1
        f.say(f"[{rid}] spectra from vantage {k + 1}: {got} targets read 0.5-4 m away")
        yield {"rid": rid, "job": "survey", "event": "vantage", "ok": ok, "spectra": got}
    ok, n_up = ops.dock_home(rid)
    yield {"rid": rid, "job": "survey", "event": "sortie", "ok": ok, "uploaded": n_up, "seconds": f.w.sim_time - t0,
           "stucks": ops.stucks[rid]}


def apron_calibration(ops: FleetOps, rid, dist=0.8):
    """The sampler's first undock: a few arcs on the garage apron, so its own body is in its table."""
    f = ops.f
    g = f.w.garage
    ax, ay = g.apron[rid]
    tx, ty = ax + dist * np.cos(g.yaw), ay + dist * np.sin(g.yaw)
    f.say(f"[{rid}] apron calibration: a few arcs in front of its bay")
    f.drive_to(rid, (tx, ty, None), "the garage apron", tol=0.25)


# ===================================================================== rocks
def find_rock(ops: FleetOps, rid, near, radius=1.2, looks=4):
    """Look for a loose rock with the navcam: a rock counts as seen when enough of it is in the
    frame (its pixels in the camera's object mask); turn 90 deg and look again if none is.
    Returns the seen rock nearest the rover within `radius` of `near` (the last drill hole)."""
    f, w = ops.f, ops.f.w
    rv = f.rv[rid]
    by_body = {it["body"]: it for it in w.rocks if it.get("state") == "ground"}
    for k in range(looks):
        seg = f.cam[rid].cam.segmentation()                 # the navcam's own frame (its renderer)
        ids = seg[seg >= 0]
        bodies, counts = np.unique(w.m.geom_bodyid[ids], return_counts=True)
        x, y, _ = rv.pose()
        seen = [by_body[b] for b, c in zip(bodies, counts) if b in by_body and c >= 20
                and np.hypot(*(w.item_pos(by_body[b])[:2] - near)) < radius]
        if seen:
            return min(seen, key=lambda it: np.hypot(*(w.item_pos(it)[:2] - [x, y])))
        if k < looks - 1:
            rv.turn_to(rv.pose()[2] + np.pi / 2, tol=0.05)
    return None


PICK_TRIES = 3                                              # grasp attempts on one rock before giving up


def pick_rock_home(ops: FleetOps, rid, near):
    """Pick a rock near the last hole, carry it home folded, set it in the tray of the bay.
    -> (event dict, docked)"""
    from .arm import MountedPicker  # noqa: F401  (the arm brain builds it)
    from .world import PICK_LOCAL, probe
    from so101.controller import Controller
    f, w = ops.f, ops.f.w
    rv = f.rv[rid]
    rock = find_rock(ops, rid, np.asarray(near, float))
    if rock is None:
        f.say(f"[{rid}] no loose rock in view near the hole")
        return {"rid": rid, "job": "sample", "event": "rock", "ok": False, "why": "none seen"}, None
    ab = f.arm_brain(rid, "pick")
    robot = ab.robot
    docked = {}

    def carry():
        rv.carrying = True
        f.say(f"[{rid}] holding the rock folded: driving home")
        docked["ok"] = f.drive_to(rid, w.start[rid], "its garage bay")
        rv.carrying = False
        return w.garage.tray if docked["ok"] else None     # not home: it sets the rock down where it is

    reach = probe(w.designs[rid]).arm_base_xy[0] + PICK_LOCAL[0]
    out, tries = {}, 0
    for tries in range(1, PICK_TRIES + 1):                  # a failed grasp is a row; the rock is still there
        p = w.item_pos(rock)
        x, y, _ = rv.pose()
        u = (p[:2] - [x, y]) / max(np.hypot(*(p[:2] - [x, y])), 1e-6)
        st = p[:2] - reach * u                              # stand so the rock is at the arm's work point
        if np.hypot(*(st - [x, y])) < 0.45:                 # too close to line up: back off first
            rv.drive(-0.5 * rv.kin.v_max, 0.0, 0.0, 0.5 / rv.kin.v_max)
        rv.goto(float(st[0]), float(st[1]), float(np.arctan2(u[1], u[0])), tol=0.01)
        p = w.item_pos(rock)
        robot.select(rock)
        robot.set_target(PICK_LOCAL)
        robot.set_work_plane([p[0], p[1], w.work_z(p[:2])])
        robot.carry = carry
        f.say(f"[{rid} arm] pick {rock['name']} ({rock['tone']}), try {tries}/{PICK_TRIES} | rows by source "
              f"{ab.rows_by_source()}")
        ctl = Controller(ab, verbose=f.verbose)
        try:
            out = ctl.step()
        finally:
            robot.carry = None
            robot.set_work_plane(None)
        c = (getattr(ctl, "last", None) or {}).get("choice") or {}
        f.emit({"kind": "pick", "rid": rid, "try": tries, "tries_max": PICK_TRIES, "rock": rock["name"],
                "skill": c.get("skill"), "p_grasp": c.get("p_grasped"), "p_hit": c.get("p_hit"),
                "uncertainty_cm": c.get("uncertainty_cm"), "grasped": bool(out.get("grasped")),
                "rows": ab.rows_by_source(), "plans": (getattr(ctl, "last", None) or {}).get("plans")})
        if out.get("grasped"):
            break
        f.say(f"[{rid} arm] grasp failed: the row goes into its table; looking again")
    rock["state"] = "taken"
    ok = w.in_tray(rock)
    f.say(f"[{rid}] rock {'IN THE TRAY' if ok else 'not in the tray'}"
          + ("" if out.get("grasped") else " (grasp failed)"))
    ev = {"rid": rid, "job": "sample", "event": "rock", "ok": ok, "grasped": bool(out.get("grasped")),
          "tries": tries, "tone": rock["tone"]}
    return ev, docked.get("ok")


def sample_sortie(ops: FleetOps, rid, n_sites=2):
    f, w = ops.f, ops.f.w
    ops.sorties[rid] += 1
    first = ops.sorties[rid] == 1
    C = ops.cands
    # briefing 1: the site rows over the whole candidate area (what is where)
    if ops.brief:
        m1 = ops.mission(rid, [tuple(p) for p in C], {SITE: 1.0}, targets={SITE: TARGETS + [FAIL]})
        ops.undock(rid, m1)
        pack1 = ops.mem(rid).df.copy()
    else:
        ops.undock(rid, None)
    sm = ops.site[rid].refit()
    X = pd.DataFrame({k: v for k, v in ops.orb.at(C[:, 0], C[:, 1]).items()})
    score, det = sm.score(X)
    x, y, _ = f.rv[rid].pose()
    dist = np.hypot(C[:, 0] - x, C[:, 1] - y)
    if score is None:                                     # nothing to choose by: the nearest sites
        order = np.lexsort((ops.rng.random(len(C)), np.round(dist, 1)))
        how = f"no photochem model ({sm.n_rows} site rows; heads {sm.heads() or 'none'}): nearest candidates"
    else:
        order = np.argsort(-score, kind="stable")
        how = f"its TabPFN site model ({sm.n_rows} site rows; heads {sm.heads()}): best score"
    picks = [C[i] for i in order[:n_sites]]
    f.emit({"kind": "sites", "rid": rid, "n_cands": int(len(C)), "rows": int(sm.n_rows), "how": how,
            "by_tabpfn": score is not None, "picks": [tuple(map(float, p)) for p in picks],
            "pred_mid": None if score is None else [float(det["photochem_mid"][i]) for i in order[:n_sites]],
            "pred_lo": None if score is None else [float(det["photochem_lo"][i]) for i in order[:n_sites]]})
    f.say(f"[{rid}] drill sites by {how}: " + ", ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in picks))
    # briefing 2: drive rows along the route to the chosen sites
    if ops.brief:
        pack2 = f.dock.brief(rid, f.brain[rid].desc, ops.drive_mission(rid, picks))
        info = f.dock.log[-1]
        ops.mem(rid).load_pack(pd.concat([pack1, pack2], ignore_index=True), keep_own=False)
        f.say(f"[{rid}] DOCK route briefing: {info.get('rows', 0)} drive rows "
              f"({info.get(DRIVE, {}).get('in_corridor', 0)} in the corridor, from {info.get(DRIVE, {}).get('by_robot', {})})")
    if first:
        apron_calibration(ops, rid)
    t0, s0 = w.sim_time, ops.stucks[rid]
    m, d = w.m, w.d                                        # the drill hangs behind the chassis centre
    ch = f.rv[rid].ch
    back = float(-(d.xmat[ch].reshape(3, 3).T @ (d.xpos[m.body(f"{rid}/drill").id] - d.xpos[ch]))[0])
    for k, (tx, ty) in enumerate(picks):
        x, y, _ = f.rv[rid].pose()                         # overshoot along the approach: the drill ends over it
        u = np.array([tx - x, ty - y]) / max(np.hypot(tx - x, ty - y), 1e-6)
        ax, ay = np.array([tx, ty]) + back * u
        ok = f.drive_to(rid, (float(ax), float(ay), None), f"drill site {k + 1}/{n_sites}", tol=0.2)
        px, py, r = ops.drill(rid)
        ops.record(rid, px, py, drill_torque_nm=r["drill_torque_nm"], **{FAIL: r[FAIL]})
        gt = ops.geo.truth(px, py)
        f.say(f"[{rid}] drill {np.hypot(px - tx, py - ty):.2f} m from the chosen site")
        f.say(f"[{rid}] DRILLED at ({px:.2f}, {py:.2f}): torque {r['drill_torque_nm']:.2f} N m, core "
              f"{'INTACT' if r['sample_success'] else 'collapsed/stalled'} | truth (report only): photochem "
              f"{gt['photochem']:.2f}, {gt['ground']}, {gt['mineral']}")
        yield {"rid": rid, "job": "sample", "event": "drill", "ok": ok, "reached": ok, "photochem": gt["photochem"],
               "success": r["sample_success"], "ground": gt["ground"], "x": px, "y": py,
               "pred": None if score is None else float(det["photochem_mid"][order[k]])}
    ev, docked = pick_rock_home(ops, rid, (px, py))
    yield ev
    ok, n_up = ops.dock_home(rid, drive=not docked, ok=bool(docked))
    yield {"rid": rid, "job": "sample", "event": "sortie", "ok": ok, "uploaded": n_up, "seconds": w.sim_time - t0,
           "stucks": ops.stucks[rid] - s0, "brief": ops.brief,
           "pack": ops.mem(rid).pack_info.get("rows", 0) if ops.brief else 0}
