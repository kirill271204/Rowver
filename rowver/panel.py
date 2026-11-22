"""The decision panel: what TabPFN weighed for each rover's decision, live in the MuJoCo viewer.

A read-only observer of decisions already made (the panel never changes one):

  sidebar  (panel_card.py) the decision in plain words, the arcs it weighed seen from above (about 24
           representative arcs, one per band of curvature, trimmed after a 120 deg turn), prediction vs
           reality, the track record and the fleet
  3D       only one marker per rover: its drive's FINAL destination (not the route's waypoints), gone on
           arrival; the arcs live in the sidebar. The camera follows the deciding rover

Hotkeys (every letter and 0-5 are the viewer's own): 9 shows / hides the panel, 7 switches the rover it
follows (auto = the rover that decided last), 8 turns the camera follow on / off.
Callbacks run in the rovers' threads and only copy small arrays; drawing happens in World._advance's
viewer sync. Counterfactual rollouts (world.quiet) never reach it: they call no decision hooks.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import mujoco
import numpy as np

from .rover_body import SKILL

NICK = {"scout": "Pathfinder", "scientist": "Spectra", "sampler": "Hauler"}
ROLE = {"scout": "scout · probes the ground", "scientist": "scientist · reads spectra",
        "sampler": "sampler · drills and collects"}
ROVER_RGB = [(46, 204, 182), (178, 140, 255), (244, 162, 97)]
KEY_TOGGLE, KEY_CYCLE, KEY_CAMERA = ord("9"), ord("7"), ord("8")
HIST, N_BANDS, MAX_TURN = 40, 24, np.radians(120)


def _rgb01(c, a=1.0):
    return (c[0] / 255, c[1] / 255, c[2] / 255, a)


def _cut(cx, cy):
    """Per arc: how many samples until it has turned MAX_TURN (a slow rover's sharp arcs curl up)."""
    dx, dy = np.diff(cx, axis=1), np.diff(cy, axis=1)
    h = np.unwrap(np.arctan2(dy, dx), axis=1)
    turned = np.abs(h - h[:, :1]) > MAX_TURN
    first = np.where(turned.any(1), turned.argmax(1), dx.shape[1])
    return np.maximum(first + 1, 3)


def _fmt_pct(p):
    return "-" if p is None or not np.isfinite(p) else f"{p:.0%}"


class DecisionPanel:
    def __init__(self, fleet):
        self.f, self.w = fleet, fleet.w
        self.lock = threading.Lock()
        self.rovers = list(fleet.rv)
        self.color = {r: ROVER_RGB[k % len(ROVER_RGB)] for k, r in enumerate(self.rovers)}
        self.st = {r: {"decisions": 0, "commits": 0, "tabpfn_s": 0.0, "tabpfn_n": 0, "overrides": 0, "compared": 0,
                       "stucks": 0, "shields": 0, "rows": 0, "own_rows": 0, "dec": None, "out": None,
                       "site": None, "pick": None, "how": "", "t_last": 0.0, "status": "IDLE",
                       "hist": deque(maxlen=HIST), "dest": None} for r in self.rovers}
        self.sci, self.rpick = None, None               # the science chain: last rocks read, last rock chosen
        self.active, self.pinned, self.visible, self.follow_cam = None, None, True, True
        self._dirty, self._ver, self._drawn = True, 0, (None, None)
        self._t_img, self._shown, self._img = 0.0, False, None
        for r in self.rovers:
            fleet.brain[r].on_decide = self._decider(r)
        fleet.listeners.append(self._on_event)
        self.w.on_sync.append(self.draw_scene)
        self.w.on_overlay.append(self.draw_overlay)
        self.w.key_handlers.append(self._key)

    # ------------------------------------------------------------ inputs (rover threads)
    def _changed(self):
        self._dirty = True
        self._ver += 1

    def _decider(self, rid):
        def f(ev):
            if ev.get("skill") != SKILL:
                return
            with self.lock:
                s = self.st[rid]
                self.active = rid
                s["how"], s["t_last"] = ev["how"], time.time()
                if ev["how"] == "commit":
                    s["commits"] += 1
                    s["status"] = "COMMITTED"
                    if s["dec"] is not None:
                        s["dec"]["n_commit"] = ev["n_commit"]
                    self._changed()
                    return
                s["decisions"] += 1
                s["status"] = "DRIVING" if ev["how"] == "play" else "LEARNING"
                if ev.get("rows") is not None:
                    s["rows"], s["own_rows"] = ev["rows"], ev.get("own_rows") or 0
                if ev["how"] == "play":
                    s["tabpfn_s"] += ev["decide_s"]
                    s["tabpfn_n"] += 1
                s["dec"] = self.summarize(rid, ev)
                d = s["dec"]
                if d.get("geo") is not None:
                    s["compared"] += 1
                    s["overrides"] += int(d["override"])
                self._changed()
        return f

    def summarize(self, rid, ev) -> dict:
        """One decision -> what the panel shows: categories per arc, the geometry-only pick, the
        representative arcs (rover frame and world, from the rover's TRUE pose: its estimate may have
        drifted), the goal."""
        body, rv = self.f.body[rid], self.f.rv[rid]
        pfc = self.f.brain[rid].pfc[SKILL]
        d = {"how": ev["how"], "action": ev["action"], "pred": ev["pred"], "decide_s": ev["decide_s"],
             "n_commit": 0, "t": time.time()}
        k_w = list(pfc.names).index("omega")
        d["w_max"] = float(max(abs(pfc.lo[k_w]), abs(pfc.hi[k_w])))
        e, fan = ev.get("eval"), body.fan
        if e is None or fan is None or fan["A"].shape != e["A"].shape or not np.allclose(fan["A"], e["A"]):
            return d                                       # random / no model yet: nothing scored
        score, i = e["score"], e["chosen"]
        n = len(score)
        map_v = ~fan["safe"]
        tilt_v = np.asarray(fan.get("tilt_veto", np.zeros(n, bool))) & ~map_v
        pfail_v = ~np.isfinite(score) & ~map_v & ~tilt_v
        open_ = ~(map_v | tilt_v | pfail_v)
        cat = np.where(map_v, 0, np.where(tilt_v | pfail_v, 1, 2))
        geo = fan["w_dir"] * (fan["v"] / fan["v_nom"]) * fan["gain"] + fan["pen"]
        geo = np.where(map_v, -np.inf, geo)
        g = int(np.argmax(geo)) if np.isfinite(geo).any() else None
        span = np.maximum(np.asarray(pfc.hi) - np.asarray(pfc.lo), 1e-9)
        override = g is not None and float(np.abs((e["A"][g] - e["A"][i]) / span).sum()) > 0.15
        col = np.full(n, np.nan)
        if open_.any():
            s_ = score[open_]
            col[open_] = (s_ - s_.min()) / max(float(np.ptp(s_)), 1e-9)
        # representatives, per category: one arc per band of curvature (open: the best-scored one), so
        # the fan reads cleanly and every kind of arc that was there still shows
        kap = e["A"][:, list(e["names"]).index("omega")] / np.maximum(fan["v"], 1e-3)
        reps = []
        for mask, nb in ((open_, N_BANDS), (cat == 1, N_BANDS // 2), (cat == 0, N_BANDS // 2)):
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            nb = min(nb, len(idx))
            edges = np.quantile(kap[idx], np.linspace(0, 1, nb + 1))
            band = np.clip(np.searchsorted(edges, kap[idx], side="right") - 1, 0, nb - 1)
            for b in np.unique(band):
                mem = idx[band == b]
                reps.append(int(mem[np.argmax(np.nan_to_num(col[mem], nan=0.0))] if mask is open_
                                else mem[len(mem) // 2]))
        reps = sorted(set(reps))
        # the rover frame (forward x, left y), from the pose the plan was made in
        xe, ye, yaw_e = fan["pose"]
        cx, cy = fan["samples"]
        cx, cy = np.column_stack([np.full(n, xe), cx]), np.column_stack([np.full(n, ye), cy])
        ncut = _cut(cx, cy)
        c, s = np.cos(-yaw_e), np.sin(-yaw_e)
        lx, ly = c * (cx - xe) - s * (cy - ye), s * (cx - xe) + c * (cy - ye)
        Lc = np.stack([lx, ly], -1)
        gx, gy = fan["goal"]
        gl = (c * (gx - xe) - s * (gy - ye), s * (gx - xe) + c * (gy - ye))
        P = lambda k: e[k] if k in e else np.full(n, np.nan)  # noqa: E731
        d |= {"n": n, "n_open": int(open_.sum()), "n_map": int(map_v.sum()), "n_tilt": int(tilt_v.sum()),
              "n_pfail": int(pfail_v.sum()), "chosen": i, "geo": g, "override": bool(override),
              "cat": cat, "col": col, "reps": reps, "ncut": ncut, "local": Lc,
              "goal_local": gl, "reach": float(np.max(np.hypot(lx, ly))),
              "p_fail": e["p_fail"], "prog10": P("progress_n_q10"), "prog50": P("progress_n_q50"),
              "prog90": P("progress_n_q90"), "tilt90": P("tilt_max_q90"),
              "A": e["A"], "names": e["names"], "tilt_limit": fan["tilt_limit"], "to_go": fan["to_go"]}
        return d

    def _on_event(self, ev):
        rid = ev.get("rid")
        if rid not in self.st:
            return
        with self.lock:
            s, k = self.st[rid], ev.get("kind")
            if k == "segment":
                row = ev["row"]
                s["out"] = row
                s["hist"].append((row.get("pred_progress_n_q10"), row.get("pred_progress_n"),
                                  row.get("pred_progress_n_q90"), row.get("progress_n"), row.get("pred_p_fail"),
                                  int(row.get("failed", 0))))
                if row.get("failed"):
                    s["status"] = "SHIELD STOP" if row.get("rep_shield") else "STUCK"
            elif k == "dest":
                s["dest"] = ev["xy"]
                if ev["xy"] is not None:
                    x, y = self.f.rv[rid].true_pose()[:2]
                    s["dest_label"], s["dest_t0"] = ev.get("label") or "its destination", time.time()
                    s["dest_d0"] = float(np.hypot(ev["xy"][0] - x, ev["xy"][1] - y))
            elif k == "stuck":
                s["stucks"] += 1
                s["status"] = "STUCK"
            elif k == "shield":
                s["shields"] += 1
                s["status"] = "SHIELD STOP"
            elif k == "sites":
                s["site"] = ev
                s["status"] = "CHOOSING SITES"
                self.active = rid
            elif k == "rocks":
                self.sci = ev
            elif k == "rock_pick":
                self.rpick = ev
            elif k == "pick":
                s["pick"] = ev
                s["status"] = "PICKING"
                self.active = rid
            self._changed()

    def _key(self, keycode):
        if keycode == KEY_TOGGLE:
            self.visible = not self.visible
        elif keycode == KEY_CYCLE:
            order = [None] + self.rovers
            self.pinned = order[(order.index(self.pinned) + 1) % len(order)]
        elif keycode == KEY_CAMERA:
            self.follow_cam = not self.follow_cam
        else:
            return
        self._changed()
        self._t_img = 0.0

    def follow(self):
        return self.pinned or self.active

    # ------------------------------------------------------------ 3D (viewer lock held)
    def draw_scene(self, viewer):
        rid = self.follow()
        if self.follow_cam and rid is not None and viewer.cam is not None:     # glide toward the deciding rover
            x, y = self.f.rv[rid].true_pose()[:2]
            tgt = np.array([x, y, self.w.ground_z(x, y) + 0.15])
            viewer.cam.lookat[:] = viewer.cam.lookat + 0.06 * (tgt - viewer.cam.lookat)
        if not self._dirty:
            return
        scn = viewer.user_scn
        if scn is None:
            return
        with self.lock:
            self._dirty = False
            scn.ngeom = 0
            if not self.visible or rid is None:
                return
            for r in self.rovers:                    # one marker per rover: its drive's final destination
                xy = self.st[r]["dest"]
                if xy is None:
                    continue
                gx, gy = xy
                gz = self.w.ground_z(gx, gy)
                a = 1.0 if r == rid else 0.55
                self._geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.13, 0.004, 0.0), (gx, gy, gz + 0.01),
                           _rgb01(self.color[r], 0.30 * a))
                self._connector(scn, (gx, gy, gz), (gx, gy, gz + 0.45), _rgb01(self.color[r], 0.55 * a), 0.006)
                self._sphere(scn, (gx, gy, gz + 0.47), 0.03, _rgb01(self.color[r], 0.9 * a))

    @staticmethod
    def _geom(scn, kind, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return False
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return True

    def _sphere(self, scn, p, r, rgba):
        return self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (r, r, r), p, rgba)

    @staticmethod
    def _connector(scn, a, b, rgba, width):
        if scn.ngeom >= scn.maxgeom:
            return False
        kind = mujoco.mjtGeom.mjGEOM_CAPSULE
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, kind, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1
        return True

    # ------------------------------------------------------------ the sidebar (outside the lock)
    def draw_overlay(self, viewer):
        """The sidebar image. MuJoCo 4.0.1's viewer draws user images only while a user text is set
        (image-only overlays are silently skipped), so a small caption goes with it."""
        now = time.time()
        if now - self._t_img < 0.2:
            return
        self._t_img = now
        if not self.visible:
            if self._shown:
                viewer.clear_images()
                viewer.clear_texts()
                self._shown = False
            return
        vp = viewer.viewport
        if vp is None or vp.width < 600:
            return
        W, H = int(np.clip(0.24 * vp.width, 430, 700)), int(vp.height)
        if self._drawn != (self._ver, (W, H)) or self._img is None:
            from .panel_card import render_sidebar
            with self.lock:
                v = self.view()
                ver = self._ver
            self._img = render_sidebar(v, W, H)
            self._drawn = (ver, (W, H))
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Rowver", "TabPFN panel  (9 hides)"))
        viewer.set_images((mujoco.MjrRect(0, 0, W, H), self._img))
        self._shown = True

    def name(self, rid):
        dsg = self.w.designs.get(rid)
        n = getattr(dsg, "name", "") if dsg is not None else ""
        return NICK.get(n, n) or rid, f"{rid} · {ROLE.get(n, n)}"

    # ------------------------------------------------------------ plain words
    @staticmethod
    def _direction(omega, w_max):
        a = omega / max(w_max, 1e-6)
        side = "left" if a > 0 else "right"
        if abs(a) < 0.08:
            return "to drive straight ahead"
        if abs(a) < 0.45:
            return f"a gentle {side} curve"
        return f"a sharp {side} turn"

    def explain(self, rid) -> tuple[list, tuple | None]:
        """-> (sentences, (result sentence, +1 good / 0 neutral / -1 bad) or None)."""
        s = self.st[rid]
        d = s["dec"]
        if d is None:
            return ["Waiting for its first decision."], None
        if d["how"] == "random":
            out = [f"Still learning: with only {s['rows']} memories it tries a random arc, to gather experience."]
        elif d["how"] == "no_model":
            out = ["Exploring while TabPFN's first model loads."]
        else:
            a, p = d["action"], d["pred"]
            spd = 1 + a.get("d_speed", 0.0)
            slow = f", slowing to {spd:.0%} speed" if spd < 0.9 else ""
            out = [f"TabPFN chose {self._direction(a.get('omega', 0.0), d['w_max'])}{slow}."]
            if "pred_progress_n" in p:
                out.append(f"It expects to cover {p['pred_progress_n']:.0%} of the planned distance, with a "
                           f"{p.get('pred_p_fail', np.nan):.0%} chance of getting stuck.")
            g = d.get("geo")
            if g is not None:
                gdir = self._direction(d["A"][g][list(d["names"]).index("omega")], d["w_max"])
                if not d["override"]:
                    out.append("A geometry-only planner would pick the same arc: here TabPFN agrees with it.")
                elif d["cat"][g] == 1:
                    out.append(f"It overruled the geometry-only choice ({gdir}): TabPFN predicted that arc would "
                               f"tilt the rover too far or get it stuck.")
                elif d["p_fail"][g] > p.get("pred_p_fail", 0) + 0.04:
                    out.append(f"It overruled the geometry-only choice ({gdir}): TabPFN rated that arc "
                               f"{d['p_fail'][g]:.0%} likely to get stuck.")
                else:
                    out.append(f"It overruled the geometry-only choice ({gdir}): TabPFN expected less progress "
                               f"there ({_fmt_pct(d['prog50'][g])}).")
            if s["how"] == "commit":
                out.append(f"Confident in that arc, it keeps driving it without re-thinking "
                           f"(commit {d['n_commit']} of 2).")
        o = s["out"]
        res = None
        if o is not None and o.get("time", 0) >= d["t"] - 0.01:
            pr = o.get("progress_n", np.nan)
            if o.get("rep_shield"):
                res = (f"The safety shield stopped it ({o['rep_shield']}): a lesson row for TabPFN.", -1)
            elif o.get("failed"):
                res = ("It got stuck: that becomes a lesson row in its memory.", -1)
            elif "pred_progress_n" in d["pred"]:
                diff = pr - d["pred"]["pred_progress_n"]
                if abs(diff) < 0.15:
                    res = (f"Result: it covered {pr:.0%}, as TabPFN predicted.", 1)
                elif diff > 0:
                    res = (f"Result: it covered {pr:.0%}, better than predicted.", 1)
                else:
                    res = (f"Result: only {pr:.0%}, worse than predicted: the surprise is now a memory.", 0)
            else:
                res = (f"Result: it covered {pr:.0%} of the arc.", 0)
        return out, res

    def view(self) -> dict:
        """Everything the sidebar draws (panel_card.render_sidebar), as plain data."""
        rid = self.follow()
        fleet = []
        for r in self.rovers:
            s = self.st[r]
            h = [x for x in s["hist"] if x[4] is not None and np.isfinite(x[4])]
            stat = s["status"] if time.time() - s["t_last"] < 20 or s["status"] in ("STUCK",) else "IDLE"
            fleet.append({"rid": r, "name": f"{r} {self.name(r)[0]}", "color": self.color[r], "active": r == rid,
                          "status": stat, "decisions": s["decisions"], "commits": s["commits"], "stucks": s["stucks"],
                          "rows": s["rows"], "pred": _fmt_pct(np.mean([x[4] for x in h]) if h else None),
                          "real": _fmt_pct(np.mean([x[5] for x in s["hist"]]) if s["hist"] else None),
                          "overrides": f"{s['overrides']}/{s['compared']}", "memories": s["rows"]})
        keys = ("9 hide panel   ·   7 switch rover"
                + (f" ({'pinned' if self.pinned else 'auto'})") + f"   ·   8 camera follow ({'on' if self.follow_cam else 'off'})")
        if rid is None:
            return {"name": "Rowver fleet", "role": "TabPFN decisions", "color": (156, 165, 184), "status": "IDLE",
                    "memory": "No decision yet.", "fleet": fleet, "keys": keys}
        s = self.st[rid]
        d = s["dec"]
        nm, role = self.name(rid)
        v = {"name": nm, "role": role, "color": self.color[rid], "status": s["status"], "fleet": fleet, "keys": keys,
             "memory": f"TabPFN brain  ·  {s['rows']} memories ({s['own_rows']} its own, "
                       f"{s['rows'] - s['own_rows']} from other rovers)"}
        v["headline"], v["result"] = self.explain(rid)
        if s["dest"] is not None:
            x, y = self.f.rv[rid].true_pose()[:2]
            to_go = float(np.hypot(s["dest"][0] - x, s["dest"][1] - y))
            v["mission"] = {"label": s["dest_label"], "to_go": to_go, "d0": max(s["dest_d0"], to_go),
                            "elapsed": time.time() - s["dest_t0"]}
        if d is not None and "local" in d:
            L, nc, cat, col = d["local"], d["ncut"], d["cat"], d["col"]
            reps = [j for j in d["reps"] if j not in (d["chosen"],)]
            v["fan"] = {"arcs": [(L[j, :nc[j]], int(cat[j]), float(col[j]) if np.isfinite(col[j]) else float("nan"))
                                 for j in reps if j != d["geo"] or not d["override"]],
                        "chosen": L[d["chosen"], :nc[d["chosen"]]],
                        "geo": L[d["geo"], :nc[d["geo"]]] if d["override"] and d["geo"] is not None else None,
                        "goal": d["goal_local"], "reach": max(float(np.max([np.hypot(*L[j, nc[j] - 1]) for j in d["reps"]])),
                                                              0.3)}
            v["census"] = {"n": d["n"], "open": d["n_open"], "tab": d["n_tilt"] + d["n_pfail"], "map": d["n_map"]}
            ts = s["tabpfn_s"] / s["tabpfn_n"] if s["tabpfn_n"] else d["decide_s"]
            v["think"] = (f"TabPFN scored {d['n']} arcs in {d['decide_s']:.2f} s (avg {ts:.2f} s)  ·  overrode "
                          f"geometry in {s['overrides']} of {s['compared']} decisions")
        if d is not None and "pred_progress_n" in d["pred"]:
            p = d["pred"]
            v["pred"] = {"prog10": p.get("pred_progress_n_q10", np.nan), "prog50": p["pred_progress_n"],
                         "prog90": p.get("pred_progress_n_q90", np.nan), "pfail": p.get("pred_p_fail", 0.0),
                         "tilt90": p.get("pred_tilt_max_q90"), "tilt_limit": d.get("tilt_limit")}
            o = s["out"]
            if o is not None and o.get("time", 0) >= d["t"] - 0.01:
                v["actual"] = {"prog": o.get("progress_n", np.nan), "tilt": o.get("tilt_max"), "failed": o.get("failed")}
        v["hist"] = list(s["hist"])
        ex = []
        if s["site"] is not None:
            e = s["site"]
            if e["by_tabpfn"] and e["pred_mid"]:
                ex.append(((178, 140, 255), f"Drill site: its TabPFN site model ({e['rows']} rows) ranked "
                           f"{e['n_cands']} candidates; the best should score {e['pred_mid'][0]:.2f} photochem "
                           f"(at worst {e['pred_lo'][0]:.2f})."))
            else:
                ex.append(((156, 165, 184), f"Drill site: no photochem model yet ({e['rows']} rows), so the "
                           f"nearest of {e['n_cands']} candidates."))
        if s["pick"] is not None:
            e = s["pick"]
            p_ = e.get("p_grasp")
            ex.append(((120, 200, 255), f"Arm grasp, try {e['try']} of {e['tries_max']}: TabPFN gave "
                       f"{_fmt_pct(p_)} for a grasp -> {'grasped it' if e['grasped'] else 'missed; that row teaches the next try'}."))
        cam = self.f.cam.get(rid) if hasattr(self.f, "cam") else None
        if cam is not None:                              # the rover's eyes
            st_ = getattr(cam, "stereo", None)
            try:
                sv = cam.map.seen
                seen = float(np.mean(sv() if callable(sv) else sv))
            except Exception:
                seen = float("nan")
            sm_ = getattr(cam, "sitemap", None)
            how = (f"stereo depth, {st_.valid:.0%} of pixels matched" if st_ is not None and np.isfinite(st_.valid)
                   else "rendered depth")
            ex.append(((120, 200, 255), (f"Eyes: {how} · {seen:.0%} of its 6 m map seen" if np.isfinite(seen)
                                         else f"Eyes: {how}")
                       + (f" · site map {sm_.cells} cells" if sm_ is not None else "")))
        if self.sci is not None:
            e = self.sci
            ex.append(((178, 140, 255), f"Scientist read R{e['best'][5:]}: {e['lith']} · organics {e['organics']:.2f} · "
                       f"SiO₂ {e['sio2']:.0f}% · hydration {e['hydration']:.2f} ({e['n']} rocks at this vantage)"))
        if self.rpick is not None:
            e = self.rpick
            if e.get("rock"):
                ex.append(((244, 162, 97), f"Sampler: R{e['rock'][5:]} ({e.get('lith')}), {e['how']}; organics "
                           f"{e['organics_mid']:.2f}, at worst {e['organics_lo']:.2f}"))
            else:
                ex.append(((156, 165, 184), f"Sampler: nearest rock ({e['how']})"))
        v["extras"] = ex
        return v
