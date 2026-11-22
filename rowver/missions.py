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
from .sitemap import SiteMap
from .survey import FEATURES, SKILL as SITE, TARGETS, FAIL, Orbital, RockModel, SiteSurvey, drill_candidates, \
    site_features, site_row


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
        self.rockm = {r: RockModel(fleet.brain[r].mem, fleet.cfg) for r in fleet.rv}
        self.collected: set = set()                    # rocks a sampler already took (a second sortie takes another)
        self.chosen: list = []                         # (rock name, how) per sampler sortie, for the summary
        self.rock_home = {it["name"]: np.asarray(fleet.w.item_pos(it)[:2], float) for it in fleet.w.rocks}
        # each rover's own map of the site (its stereo navcam), and the dock's merged map
        site = fleet.w.site
        cell = float(fleet.cfg.opt("map.cell", 0.10))
        b = (site.x_min, site.x_min + 2 * site.half, site.y_max - 2 * site.half, site.y_max) if site is not None \
            else fleet.w.zone
        for r in fleet.rv:
            fleet.cam[r].sitemap = SiteMap(*b, cell=cell)
        fleet.dock.map = SiteMap(*b, cell=cell)

    def smap(self, rid):
        return self.f.cam[rid].sitemap

    def X(self, rid, x, y) -> pd.DataFrame:
        """Model features of spots for rover rid: orbital data + its site map."""
        return site_features(self.orb, self.smap(rid), x, y)

    def rock_truth(self, name):
        """Sim side: a rock's hidden truth, from where it lay at the start (it may be in a tray now)."""
        it = next(i for i in self.f.w.rocks if i["name"] == name)
        return self.geo.rock_truth(name, *self.rock_home[name], lith=it.get("tone"))

    PASSIVE_EVERY = 3                                  # a passive trafficability row every N drive segments

    def _on_event(self, ev):
        if ev.get("kind") == "stuck":
            self.stucks[ev["rid"]] = self.stucks.get(ev["rid"], 0) + 1
        if ev.get("kind") == "segment":
            self._passive(ev["rid"], ev["row"])

    def _passive(self, rid, row):
        """Every rover, while it drives: a trafficability site row at the segment's midpoint: slip from wheel
        odometry against the motion (1 - progress along the arc), rolling resistance from the wheel motors,
        vibration from the IMU. Only straight-ish, unblocked segments (a turn is not a slip reading)."""
        if row.get("source") == "counterfactual" or row.get("rep_shield"):
            return
        self._seg = getattr(self, "_seg", {})
        self._seg[rid] = self._seg.get(rid, 0) + 1
        if self._seg[rid] % self.PASSIVE_EVERY:
            return
        sens = getattr(self.f.rv[rid], "last_sensing", None)
        pn = row.get("progress_n")
        if sens is None or pn is None or not np.isfinite(pn) or abs(float(row.get("omega", 0.0))) > 0.5:
            return
        x, y = (sens["x0"] + sens["x1"]) / 2, (sens["y0"] + sens["y1"]) / 2
        vo = getattr(self.f.cam[rid], "last_vo", None)          # stereo visual odometry vs the wheels
        fresh = vo is not None and self.f.w.sim_time - vo["t"] < 3.0
        slip = vo["slip_vo"] if fresh else float(np.clip(1.0 - pn, 0, 1))
        self.record(rid, x, y, slip=slip, rolling_res=sens["rolling_res"], vib=sens["vib"], probe=0.0,
                    slip_vo=vo["slip_vo"] if fresh else None)

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
        self.smap(rid).load(f.dock.map)                     # the briefing includes the dock's map cells
        f.say(f"[{rid}] DOCK map: {f.dock.map.cells} cells inherited")
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
        f.dock.map.add_delta(self.smap(rid).upload_delta())    # its new map cells go to the dock too
        self.map_log = getattr(self, "map_log", []) + [(rid, self.smap(rid).cells, f.dock.map.cells)]
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
        return x, y, {"slip": slip, "sinkage_mm": 1000.0 * max(0.0, float(np.mean(sink))),
                      "rolling_res": o.get("rolling_res"), "vib": o.get("vib"), "probe": 1.0}

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
        row = site_row(rid, self.sorties[rid], self._next_step(rid), x, y, self.orb, self.smap(rid), **readings)
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
        f.say(f"[{rid}] probe at ({px:.2f}, {py:.2f}): slip {r['slip']:.2f}, sinkage {r['sinkage_mm']:.1f} mm, "
              f"rolling resistance {r['rolling_res']:.3f}, vibration {r['vib']:.3f} m/s2 "
              f"(truth, report only: {gt['ground']})")
        yield {"rid": rid, "job": "scout", "event": "probe", "ok": ok, **r, "ground": gt["ground"]}
    ok, n_up = ops.dock_home(rid)
    yield {"rid": rid, "job": "scout", "event": "sortie", "ok": ok, "uploaded": n_up, "seconds": f.w.sim_time - t0,
           "stucks": ops.stucks[rid]}


def survey_sortie(ops: FleetOps, rid, n_vantage=2, per_vantage=8):
    """The scientist. 1. Briefing: the scout's trafficability rows over the candidate area. 2. Vantage points:
    the most unread candidate sites in range, minus the risk its own TabPFN predicts there from those rows
    (q90 sinkage and slip); risky vantage points are skipped. 3. Briefing 2: drive rows along the route.
    4. At each vantage: ground spectra, then a look around (4 navcam frames) for loose rocks, and every rock
    in range is read with the RMI camera, LIBS, Raman and VISIR -> one row per rock (rock_id)."""
    f = ops.f
    ops.sorties[rid] += 1
    C = ops.cands
    lo, hi = 1.0, 3.5                                       # read inside the spectrometer's best range
    x0, x1, y0, y1 = f.w.zone
    gx, gy = np.meshgrid(np.linspace(x0 + 0.6, x1 - 0.6, 15), np.linspace(y0 + 0.6, y1 - 0.6, 11))
    V = np.column_stack([gx.ravel(), gy.ravel()])
    V = V[[f._open_ground(p) for p in V]]
    pack1 = None
    if ops.brief:                                           # 1. the scout's experience over the candidate area
        ops.undock(rid, ops.mission(rid, [tuple(p) for p in C], {SITE: 1.0},
                                    targets={SITE: ["sinkage_mm", "slip", "rolling_res", "vib"]}))
        pack1 = ops.mem(rid).df.copy()
    else:
        ops.undock(rid, None)
    sm = ops.site[rid].refit()
    risk = sm.risk(ops.X(rid, V[:, 0], V[:, 1]))
    w_risk = float(f.cfg.opt("survey.w_risk_vantage", 2.0))
    max_risk = float(f.cfg.opt("survey.max_risk_vantage", 1.2))
    if risk is not None and len(risk):
        keep = risk <= max(max_risk, float(np.median(risk)))       # skip the risky ones, never all of them
        f.say(f"[{rid}] vantage risk from {sm.n_rows} site rows (TabPFN, q90 sinkage + slip; median "
              f"{float(np.median(risk)):.2f}): "
              f"{int((~keep).sum())} of {len(V)} vantage points skipped as risky")
        V, risk = V[keep], risk[keep]
    read = np.zeros(len(C), bool)
    vps = []
    for _ in range(n_vantage):                              # 2. coverage minus risk
        D = np.hypot(V[:, None, 0] - C[None, :, 0], V[:, None, 1] - C[None, :, 1])
        gain = ((D > lo) & (D < hi) & ~read[None]).sum(1).astype(float)
        if risk is not None:
            gain = gain - w_risk * risk
        i = int(np.argmax(gain))
        vps.append(V[i])
        read |= (D[i] > lo) & (D[i] < hi)
    if ops.brief:                                           # 3. drive rows along the route
        pack2 = f.dock.brief(rid, f.brain[rid].desc, ops.drive_mission(rid, vps))
        ops.mem(rid).load_pack(pd.concat([pack1, pack2], ignore_index=True), keep_own=False)
        info = f.dock.log[-1]
        f.say(f"[{rid}] DOCK route briefing: {info.get('rows', 0)} drive rows "
              f"(from {info.get(DRIVE, {}).get('by_robot', {})})")
    t0 = f.w.sim_time
    n_rocks = 0
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
            s_ = ops.geo.spectrometer((x, y), (tx, ty))
            if s_ is None:
                continue
            ops.record(rid, tx, ty, photochem_score=s_["photochem_score"], mineral_class=s_["mineral_class"],
                       spec_range=s_["spec_range"])
            got += 1
        f.say(f"[{rid}] spectra from vantage {k + 1}: {got} targets read 0.5-4 m away")
        rocks = read_rocks(ops, rid)                           # 4. the rocks it can see, read one by one
        n_rocks += len(rocks)
        yield {"rid": rid, "job": "survey", "event": "vantage", "ok": ok, "spectra": got, "rocks": len(rocks)}
    f.say(f"[{rid}] {n_rocks} rocks read in all")
    ok, n_up = ops.dock_home(rid)
    yield {"rid": rid, "job": "survey", "event": "sortie", "ok": ok, "uploaded": n_up, "seconds": f.w.sim_time - t0,
           "stucks": ops.stucks[rid], "rocks_read": n_rocks}


def drillcam_look(ops: FleetOps, rid) -> dict:
    """The sampler's drill camera: brightness and texture of the ground under the bit, before it drills."""
    import cv2
    from so101.perception.camera import RGBDCamera
    try:
        c = ops._drillcam = getattr(ops, "_drillcam", None) or RGBDCamera(ops.f.w.m, ops.f.w.d, f"{rid}/drillcam", 96, 72)
    except KeyError:
        return {}
    rgb, _ = c.capture()
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].astype(float) / 255.0
    return {"drill_val": float(v.mean()), "drill_tex": float(v.std())}


def apron_calibration(ops: FleetOps, rid, dist=0.8):
    """The sampler's first undock: a few arcs on the garage apron, so its own body is in its table."""
    f = ops.f
    g = f.w.garage
    ax, ay = g.apron[rid]
    tx, ty = ax + dist * np.cos(g.yaw), ay + dist * np.sin(g.yaw)
    f.say(f"[{rid}] apron calibration: a few arcs in front of its bay")
    f.drive_to(rid, (tx, ty, None), "the garage apron", tol=0.25)


# ===================================================================== rocks
def visible_rocks(ops: FleetOps, rid, looks=4, min_px=8, max_range=None):
    """Look around with the navcam: a loose rock counts as seen when at least min_px of its pixels are in a
    frame (the camera's object mask); turn 90 deg between looks. -> {name: (item, xy)}: where the rover's
    stereo puts it (a few mm of error per metre of range), within max_range."""
    f, w = ops.f, ops.f.w
    rv = f.rv[rid]
    by_body = {it["body"]: it for it in w.rocks if it.get("state") == "ground"}
    out = {}
    import cv2
    for k in range(looks):
        seg = f.cam[rid].cam.segmentation()                 # the navcam's own frame (its renderer)
        rgb, _ = f.cam[rid].cam.capture()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        body_px = np.where(seg >= 0, w.m.geom_bodyid[np.maximum(seg, 0)], -1)
        ids = seg[seg >= 0]
        bodies, counts = np.unique(w.m.geom_bodyid[ids], return_counts=True)
        x, y, _ = rv.pose()
        for b, c in zip(bodies, counts):
            if b not in by_body or c < min_px:
                continue
            it = by_body[b]
            p = np.asarray(w.item_pos(it)[:2], float)
            d = float(np.hypot(*(p - [x, y])))
            if max_range is not None and d > max_range:
                continue
            px = body_px == b                                # the rock's own pixels: what it looks like
            v = hsv[..., 2][px] / 255.0
            look = {"rock_val": float(v.mean()), "rock_sat": float(hsv[..., 1][px].mean() / 255.0),
                    "rock_tex": float(v.std())}
            out[it["name"]] = (it, p + ops.rng.normal(0, 0.004 * max(d, 0.3), 2), look)
        if k < looks - 1:
            rv.turn_to(rv.pose()[2] + np.pi / 2, tol=0.05)
    return out


def _has_cam(w, name) -> bool:
    try:
        w.m.camera(name)
        return True
    except KeyError:
        return False


def read_rocks(ops: FleetOps, rid):
    """The scientist reads every rock it can see in its instruments' range -> one site row per rock."""
    from .geology import SPEC_RANGE
    f = ops.f
    x, y, _ = f.rv[rid].pose()
    seen = visible_rocks(ops, rid, max_range=SPEC_RANGE[1])
    done = []
    from .mast import ZOOM, Mast, instrument_pos
    mast = ops.mast = getattr(ops, "mast", None) or Mast(f.w)
    w = f.w
    head = instrument_pos(w, rid, "rmi") if _has_cam(w, f"{rid}/rmi") else instrument_pos(w, rid)
    for name, (it, p, look) in seen.items():
        truth = ops.rock_truth(name)
        r = ops.geo.read_rock((x, y), p, truth, it.get("size_m", 0.03))
        if r is None:
            continue
        tgt = np.asarray(w.item_pos(it), float)           # where the mast points: the rock the navcam found
        look = {**look, **mast.rmi(head, it, tgt), **mast.pair(head, it, tgt, ZOOM, "zoom")}
        rid_num = float(int("".join(ch for ch in name if ch.isdigit())))
        ops.record(rid, float(p[0]), float(p[1]), rock_id=rid_num, spec_range=r["spec_range"],
                   **{k: v for k, v in r.items() if k.startswith("rock_")}, **look)
        done.append((name, r))
    if done:
        from .geology import LITHOLOGY
        best = max(done, key=lambda t: t[1]["rock_organics"])
        f.say(f"[{rid}] read {len(done)} rocks (RMI, LIBS, Raman, VISIR): " + ", ".join(
            f"R{n[5:]} {LITHOLOGY[int(r['rock_class'])]} org {r['rock_organics']:.2f}" for n, r in done[:6])
              + (" ..." if len(done) > 6 else ""))
        f.emit({"kind": "rocks", "rid": rid, "n": len(done), "best": best[0],
                "lith": LITHOLOGY[int(best[1]["rock_class"])], "organics": best[1]["rock_organics"],
                "sio2": best[1]["rock_sio2"], "hydration": best[1]["rock_hydration"]})
    else:
        f.say(f"[{rid}] no loose rock in range to read")
    return done


def choose_rock(ops: FleetOps, rid):
    """The sampler's rock: the best read rock it has not taken yet, by its own TabPFN rock model (lower
    quantile of organics), minus the trafficability risk its site model predicts at the rock.
    -> (rock item, how, details) or (None, why, {})."""
    w = ops.f.w
    rm = ops.rockm[rid].refit()
    R = rm.rank(float(ops.f.cfg.opt("survey.organics_q", 0.2)))
    if R is None or R.empty:
        return None, "no rock readings in its table", {}
    names = {int("".join(ch for ch in it["name"] if ch.isdigit())): it for it in w.rocks}
    R = R[[int(i) in names and names[int(i)]["name"] not in ops.collected and names[int(i)].get("state") == "ground"
           for i in R.rock_id]]
    if R.empty:
        return None, "every read rock is already taken", {}
    risk = ops.site[rid].refit().risk(ops.X(rid, R.ctx_x.to_numpy(), R.ctx_y.to_numpy()))
    score = R.organics_lo.to_numpy() - (0.1 * risk if risk is not None else 0.0)
    i = int(np.argmax(score))
    row = R.iloc[i]
    it = names[int(row.rock_id)]
    by = "TabPFN rock model" if "rock_organics" in rm.wm.reg else "raw readings"
    det = {"rock": it["name"], "n_read": int(len(R) + len([n for n in ops.collected])), "n_left": int(len(R)),
           "organics_mid": float(row.organics_mid), "organics_lo": float(row.organics_lo), "by": by,
           "lith": it.get("tone")}
    return it, f"best of {len(R)} read rocks by its {by}", det


def find_rock(ops: FleetOps, rid, near, radius=1.2, looks=4):
    """Fallback (no rock readings): the seen rock nearest the rover within `radius` of `near` (the last hole)."""
    f, w = ops.f, ops.f.w
    x, y, _ = f.rv[rid].pose()
    seen = [it for it, p, _ in visible_rocks(ops, rid, looks=looks, min_px=20).values()
            if np.hypot(*(w.item_pos(it)[:2] - near)) < radius and it["name"] not in ops.collected]
    return min(seen, key=lambda it: np.hypot(*(w.item_pos(it)[:2] - [x, y]))) if seen else None


PICK_TRIES = 3                                              # grasp attempts on one rock before giving up


def pick_rock_home(ops: FleetOps, rid, near):
    """Pick the rock the science chose (choose_rock: the scientist's readings, its own TabPFN), or with no
    readings the rock nearest the last hole; carry it home folded, set it in the tray of the bay.
    -> (event dict, docked)"""
    from .arm import MountedPicker  # noqa: F401  (the arm brain builds it)
    from .world import PICK_LOCAL, probe
    from so101.controller import Controller
    f, w = ops.f, ops.f.w
    rv = f.rv[rid]
    rock, how, det = choose_rock(ops, rid)
    if rock is not None:
        f.say(f"[{rid}] rock: {rock['name']} ({rock['tone']}), {how}: organics {det['organics_mid']:.2f} "
              f"(at worst {det['organics_lo']:.2f})")
        f.emit({"kind": "rock_pick", "rid": rid, **det, "how": how})
        p = np.asarray(w.item_pos(rock)[:2], float)
        x, y, _ = rv.pose()
        if np.hypot(*(p - [x, y])) > 0.7:                   # drive over to it (its own arcs), stop short
            u = (p - [x, y]) / max(np.hypot(*(p - [x, y])), 1e-6)
            st = p - 0.55 * u
            f.drive_to(rid, (float(st[0]), float(st[1]), None), f"rock {rock['name'][5:]}", tol=0.2)
        from .mast import PANO, Mast, instrument_pos
        mast = ops.mast = getattr(ops, "mast", None) or Mast(w)
        pm = mast.pair(instrument_pos(w, rid), rock, np.asarray(w.item_pos(rock), float), PANO, "pano")
        if "pano_size_cm" in pm:
            f.say(f"[{rid}] panoramic stereo: {rock['name']} at {pm['pano_range']:.2f} m, {pm['pano_size_cm']:.1f} cm")
            q = w.item_pos(rock)
            ops.record(rid, float(q[0]), float(q[1]), rock_id=float(int(rock["name"][5:])), **pm)
    else:
        f.say(f"[{rid}] {how}: the nearest rock to the hole")
        f.emit({"kind": "rock_pick", "rid": rid, "rock": None, "how": how})
        rock = find_rock(ops, rid, np.asarray(near, float))
    ops.chosen.append((rock["name"] if rock else None, how))
    if rock is None:
        f.say(f"[{rid}] no loose rock in view near the hole")
        return {"rid": rid, "job": "sample", "event": "rock", "ok": False, "why": "none seen"}, None
    ops.collected.add(rock["name"])
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
          "tries": tries, "tone": rock["tone"], "rock": rock["name"], "how": how,
          "organics": ops.rock_truth(rock["name"])["organics"]}
    return ev, docked.get("ok")


def sample_sortie(ops: FleetOps, rid, n_sites=2):
    f, w = ops.f, ops.f.w
    ops.sorties[rid] += 1
    first = ops.sorties[rid] == 1
    C = ops.cands
    # briefing 1: the site rows over the whole candidate area (what is where)
    if ops.brief:
        m1 = ops.mission(rid, [tuple(p) for p in C], {SITE: 1.0}, targets={SITE: TARGETS + [FAIL, "rock_organics"]})
        ops.undock(rid, m1)
        pack1 = ops.mem(rid).df.copy()
    else:
        ops.undock(rid, None)
    sm = ops.site[rid].refit()
    X = ops.X(rid, C[:, 0], C[:, 1])
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
        cam_ = drillcam_look(ops, rid)
        px, py, r = ops.drill(rid)
        ops.record(rid, px, py, drill_torque_nm=r["drill_torque_nm"], **{FAIL: r[FAIL]}, **cam_)
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
