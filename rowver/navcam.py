"""The rover's own eyes: the navigation camera on its mast -> a local map -> features of every
candidate drive arc (the way Perseverance's AutoNav drives: no site map, only what it sees).

  NavCam(world, rid)            one per rover; RGB-D frames from "<rid>/navcam" (336 x 252, 70 deg)
  cam.update()                  one frame -> the rolling LocalMap (6 m x 6 m, 5 cm cells, world frame,
                                registered with the rover's own pose ESTIMATE)
  cam.arcs(A, ...)              per-candidate features of the ground each arc would cross

Per cell the map keeps what the camera measured: mean and max height of the depth points, colour
(val, sat), texture (std of the brightness) and the 8 frozen DINOv2 codes of the image patch the
cell was seen in (vision.py). Cells are kept from earlier frames, so the ground under the rover is
known from when it was ahead. The rover masks its own body (it knows its own shape, SELF_BOX) and
its own arm and whatever the arm holds (it knows its joint angles: a bounding sphere per arm link,
from the arm's current pose), so a sample carried in front of the nose is never seen as a rock.

An arc = (speed, turn rate) held for the next segment. Its features are taken along its path out to
`horizon` (0.9 m, or less near the goal) over the rover's width:
  e0..e7, val, sat, tex   mean appearance of that ground
  slope, xslope           a plane fitted to the heights: rise per metre along the arc, and across
  step, rough             tallest point above that plane, and the residual std (rocks, plinths, ruts)
  drop                    deepest point below that plane (a hole, a ledge going down)
  seen                    the fraction of that ground the camera has actually measured
No terrain label and no simulator ground truth is read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from so101.perception.camera import RGBDCamera

from . import mars_rover as P
from .vision import N_CODES, PATCH

W_PX, H_PX = 24 * PATCH, 18 * PATCH        # 336 x 252: whole DINOv2 patches
HAZ_W, HAZ_H = 160, 120                    # the hazcam: depth only (geometry of the near ground)
CODES = [f"e{k}" for k in range(N_CODES)]
ARC_FEATS = CODES + ["val", "sat", "tex", "slope", "xslope", "step", "drop", "rough", "seen"]
BODY_FEATS = ["h_val", "h_tex", "imu_pitch", "imu_roll", "payload", "site"]
N_ALONG, N_ACROSS = 18, 7


def infrastructure(world) -> np.ndarray:
    """(N, 3) x, y, radius of the base's own fixed structures (the garage's walls and dividers, the
    sample tray): the rover knows where its own garage stands. A long wall is a row of small
    circles along it, so the planner's model stays as tight as the wall."""
    import mujoco
    m, d = world.m, world.d
    mujoco.mj_forward(m, d)
    out = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if m.body_weldid[b] != 0 or m.geom_contype[g] == 0:     # static colliding geoms only
            continue
        n = m.geom(g).name
        if n == "ground" or n.startswith("rock"):
            continue
        c = d.geom_xpos[g]
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            out.append((float(c[0]), float(c[1]), float(m.geom_rbound[g])))
            continue
        s, R = m.geom_size[g], d.geom_xmat[g].reshape(3, 3)
        k = int(np.argmax(s[:2]))                                # the box's long horizontal axis
        u = R[:2, k] / max(np.linalg.norm(R[:2, k]), 1e-9)
        half_w = float(s[1 - k])
        step = max(2 * half_w, 0.08)
        n_c = max(1, int(np.ceil(2 * s[k] / step)))
        r = float(np.hypot(half_w, s[k] / n_c))
        for t in (np.arange(n_c) + 0.5) / n_c * 2 - 1:
            p = c[:2] + u * t * s[k]
            out.append((float(p[0]), float(p[1]), r))
    return np.array(out).reshape(-1, 3)


class LocalMap:
    """A square window of 5 cm cells in the world frame that follows the rover."""

    def __init__(self, x, y, size=6.0, cell=0.05):
        self.cell, self.n = cell, int(round(size / cell))
        self.size = self.n * cell
        n = self.n
        self.z = np.full((n, n), np.nan)
        self.zmax = np.full((n, n), np.nan)
        self.val = np.full((n, n), np.nan)
        self.sat = np.full((n, n), np.nan)
        self.tex = np.full((n, n), np.nan)
        self.E = np.full((n, n, N_CODES), np.nan)
        self.rng = np.full((n, n), np.inf)
        self.t = np.full((n, n), -np.inf)
        self.x0 = self.y0 = 0.0
        self._origin(x, y)

    def _origin(self, x, y):
        c = self.cell
        self.x0 = np.floor((x - self.size / 2) / c) * c
        self.y0 = np.floor((y - self.size / 2) / c) * c

    def follow(self, x, y, slack=1.0):
        """Re-centre on (x, y) when the rover has moved more than `slack` from the centre."""
        cx, cy = self.x0 + self.size / 2, self.y0 + self.size / 2
        if abs(x - cx) < slack and abs(y - cy) < slack:
            return
        ox, oy = self.x0, self.y0
        self._origin(x, y)
        sx, sy = int(round((self.x0 - ox) / self.cell)), int(round((self.y0 - oy) / self.cell))
        n = self.n
        for name in ("z", "zmax", "val", "sat", "tex", "E", "rng", "t"):
            a = getattr(self, name)
            fill = np.inf if name == "rng" else (-np.inf if name == "t" else np.nan)
            b = np.full_like(a, fill)
            if abs(sx) < n and abs(sy) < n:                  # new[i] = old[i + shift]
                b[max(-sy, 0):n - max(sy, 0), max(-sx, 0):n - max(sx, 0)] = \
                    a[max(sy, 0):n - max(-sy, 0), max(sx, 0):n - max(-sx, 0)]
            setattr(self, name, b)

    def index(self, x, y):
        ix = np.floor((np.asarray(x) - self.x0) / self.cell).astype(int)
        iy = np.floor((np.asarray(y) - self.y0) / self.cell).astype(int)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        return np.clip(iy, 0, self.n - 1), np.clip(ix, 0, self.n - 1), ok

    @property
    def seen(self):
        return np.isfinite(self.z)


class NavCam:
    def __init__(self, world, rid, cfg, rover, encoder=None):
        self.w, self.rid, self.cfg, self.rv = world, rid, cfg, rover
        self.cam = RGBDCamera(world.m, world.d, f"{rid}/navcam", W_PX, H_PX)
        self.haz = RGBDCamera(world.m, world.d, f"{rid}/hazcam", HAZ_W, HAZ_H)
        # depth: "stereo" = block matching on the stereo pair (stereo.py), "render" = the renderer's perfect depth
        import os
        self.depth_mode = os.environ.get("ROWVER_DEPTH") or str(cfg.opt("navcam.depth", "render"))
        self.stereo = self.haz_stereo = None
        if self.depth_mode == "stereo":
            from .mars_rover import HAZ_BASELINE, NAV_BASELINE
            from .stereo import StereoDepth
            if mujoco_has_camera(world.m, f"{rid}/navcam_r"):
                self.stereo = StereoDepth(world, f"{rid}/navcam", f"{rid}/navcam_r", W_PX, H_PX, NAV_BASELINE,
                                          left_cam=self.cam)
                self.haz_stereo = StereoDepth(world, f"{rid}/hazcam", f"{rid}/hazcam_r", HAZ_W, HAZ_H, HAZ_BASELINE,
                                              max_range=2.0, min_range=0.08, left_cam=self.haz)
        self.use_codes = bool(cfg("navcam.codes")) and rover.design.dino     # DINOv2 only on rovers that run it
        if self.use_codes and encoder is None:
            from .vision import encoder as _enc
            encoder = _enc()
        self.enc = encoder
        x, y, _ = rover.pose()
        self.map = LocalMap(x, y, cfg("navcam.window"), cfg("navcam.cell"))
        self.range = cfg("navcam.range")
        v, u = np.mgrid[0:H_PX, 0:W_PX]
        self._patch = ((v // PATCH) * (W_PX // PATCH) + u // PATCH).ravel()
        self.last_rgb = None
        self.frames = 0
        self.infra = infrastructure(world)
        m = world.m
        self._arm = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"{rid}/arm")], int)
        self._holds = [(i, int(m.eq_obj2id[i])) for i in range(m.neq) if m.equality(i).name.startswith(f"hold_{rid}_")]
        self._body_geoms = {}

    @property
    def features(self):
        return (ARC_FEATS if self.use_codes else [f for f in ARC_FEATS if f not in CODES]) + BODY_FEATS

    def close(self):
        self.cam.close()
        self.haz.close()
        for s_ in (self.stereo, self.haz_stereo):
            if s_ is not None:
                s_.close()

    # ------------------------------------------------------------ frames -> the local map
    def haz_update(self):
        """One hazcam depth frame only (the safety shield, while an arc is being driven)."""
        xe, ye, _ = self.rv.pose()
        self.map.follow(xe, ye)
        return self._fuse(self.haz, self._haz_depth(), self.cfg("navcam.haz_range"), None)

    def _haz_depth(self):
        if self.haz_stereo is not None:
            import mujoco
            mujoco.mj_forward(self.w.m, self.w.d)
            return self.haz_stereo.depth()
        return _depth(self.haz)

    def update(self):
        """One navcam frame (geometry + appearance) and one hazcam depth frame (geometry of the
        near ground the deck hides from the navcam) -> the local map. -> cells updated."""
        rgb, depth = self.cam.capture()
        if self.stereo is not None:                          # depth from the stereo pair, not the renderer
            depth = self.stereo.depth(rgb)
        self.last_rgb = rgb
        self._visual_odometry(rgb, depth)
        xe, ye, _ = self.rv.pose()
        self.map.follow(xe, ye)
        n = self._fuse(self.cam, depth, self.range, rgb)
        n += self._fuse(self.haz, self._haz_depth(), self.cfg("navcam.haz_range"), None)
        self.frames += 1
        if getattr(self, "sitemap", None) is not None:       # the whole-run map (sitemap.py)
            self.sitemap.add_local(self.map, self.map.t == self.w.sim_time)
        return n

    # ------------------------------------------------------------ stereo visual odometry
    def _wheel_m(self):
        """Distance the wheels say the rover rolled (mean wheel angle x radius), cumulative and signed."""
        if not hasattr(self, "_wq"):
            m = self.w.m
            self._wq = [m.jnt_qposadr[m.joint(f"{self.rid}/wheel_{w}").id] for w in self.rv.design.wheels]
        return float(np.mean(self.w.d.qpos[self._wq])) * self.rv.design.wheel_r

    def _visual_odometry(self, rgb, depth):
        """How far the camera really moved since the last navcam frame: ORB features matched between the two
        frames, 3D points from the last frame's (stereo) depth, solvePnPRansac -> the camera's translation.
        Compared with the wheels' rolled distance -> slip_vo = 1 - visual / wheel (self.last_vo)."""
        import cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if not hasattr(self, "_orb"):
            self._orb = cv2.ORB_create(800)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._vo_prev = None
            self.last_vo = None
        kp, des = self._orb.detectAndCompute(gray, None)
        wheel = self._wheel_m()
        prev, self._vo_prev = self._vo_prev, (kp, des, depth.copy(), wheel)
        if prev is None or des is None or prev[1] is None:
            return
        kp0, des0, d0, w0 = prev
        ms = self._bf.match(des0, des)
        f, H, W = self.cam.f, d0.shape[0], d0.shape[1]
        obj, img = [], []
        for mt in ms:
            u, v = kp0[mt.queryIdx].pt
            z = d0[int(v), int(u)]
            if np.isfinite(z) and 0.2 < z < 4.0:
                obj.append(((u + 0.5 - W / 2) * z / f, (v + 0.5 - H / 2) * z / f, z))
                img.append(kp[mt.trainIdx].pt)
        wheel_d = abs(wheel - w0)
        if len(obj) < 12 or wheel_d < 0.03:
            return
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(np.array(obj, np.float32), np.array(img, np.float32), K, None,
                                                 reprojectionError=2.0, iterationsCount=200)
        if not ok or inl is None or len(inl) < 10:
            return
        vo = float(np.linalg.norm(tvec))
        self.last_vo = {"slip_vo": float(np.clip(1.0 - vo / wheel_d, 0.0, 1.0)), "vo_m": vo, "wheel_m": wheel_d,
                        "inliers": int(len(inl)), "t": self.w.sim_time}

    def _fuse(self, cam, depth, max_range, rgb):
        Pw = cam.points(depth).reshape(-1, 3)
        cpos, _ = cam.pose
        xt, yt, _, R = self.rv.true_pose()
        xe, ye, _ = self.rv.pose()
        loc = (Pw - self.w.d.xpos[self.rv.ch]) @ R              # the rover's own frame
        x0, x1, y0, y1, zu = self.rv.design.self_box
        own = (loc[:, 0] > x0) & (loc[:, 0] < x1) & (loc[:, 1] > y0) & (loc[:, 1] < y1) & (loc[:, 2] > zu)
        C, Rr = self._own_spheres()
        if len(C):
            for c, r in zip(C, Rr):
                own |= ((Pw - c) ** 2).sum(1) < r * r
        rng = np.linalg.norm(Pw - cpos, axis=1)
        ok = ~own & (rng < max_range) & np.isfinite(rng)
        Pw = Pw + [xe - xt, ye - yt, 0.0]                       # registered with the pose ESTIMATE
        M = self.map
        iy, ix, inside = M.index(Pw[:, 0], Pw[:, 1])
        ok &= inside
        n = M.n
        idx = np.where(ok, iy * n + ix, n * n)
        N = n * n + 1

        def S(w):
            return np.bincount(idx, weights=w, minlength=N)[:-1]

        cnt = S(np.ones(len(idx)))
        c1 = np.maximum(cnt, 1)
        z = S(Pw[:, 2]) / c1
        zmax = np.full(N, -np.inf)
        np.maximum.at(zmax, idx, Pw[:, 2])
        zmax = zmax[:-1]
        r = S(rng) / c1
        good = cnt >= self.cfg("navcam.min_px")
        old_r, old_t = M.rng.ravel(), M.t.ravel()
        now = self.w.sim_time
        # a new look replaces the old one unless the old one was clearly closer and recent
        take = good & ((r <= old_r + 0.5) | (now - old_t > 15.0))
        sh = (n, n)
        fields = [("z", z), ("zmax", zmax), ("rng", r)]
        if rgb is not None:                                     # appearance: the navcam only
            hsv = _hsv(rgb)
            v = hsv[:, 2]
            val = S(v) / c1
            fields += [("val", val), ("sat", S(hsv[:, 1]) / c1), ("tex", np.sqrt(np.maximum(S(v * v) / c1 - val ** 2, 0)))]
        for name, a in fields:
            cur = getattr(M, name).ravel()
            cur[take] = a[take]
            setattr(M, name, cur.reshape(sh))
        M.t = np.where(take.reshape(sh), now, M.t)
        if rgb is not None and self.use_codes:
            codes = self.enc.codes(rgb).reshape(-1, N_CODES)[self._patch]       # per pixel
            E = np.stack([S(np.where(ok, codes[:, k], 0.0)) for k in range(N_CODES)], -1) / c1[:, None]
            cur = M.E.reshape(-1, N_CODES)
            cur[take] = E[take]
            M.E = cur.reshape(n, n, N_CODES)
        return int(take.sum())

    def _own_spheres(self):
        """Bounding spheres (centres, radii) of this rover's arm links and of what it holds now."""
        m, d = self.w.m, self.w.d
        G = list(self._arm)
        for eq, body in self._holds:
            if d.eq_active[eq]:
                if body not in self._body_geoms:
                    self._body_geoms[body] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
                G += self._body_geoms[body]
        if not G:
            return np.zeros((0, 3)), np.zeros(0)
        G = np.asarray(G, int)
        return d.geom_xpos[G], m.geom_rbound[G] + 0.02

    # ------------------------------------------------------------ the ground under the rover
    def under(self, x, y):
        M = self.map
        g = np.linspace(-0.15, 0.15, 3)
        X, Y = np.meshgrid(x + g, y + g)
        iy, ix, ok = M.index(X.ravel(), Y.ravel())
        val, tex = M.val[iy, ix][ok], M.tex[iy, ix][ok]
        fv = lambda a, d: float(np.nanmean(a)) if np.isfinite(a).any() else d          # noqa: E731
        return {"h_val": fv(val, np.nan), "h_tex": fv(tex, np.nan)}

    # ------------------------------------------------------------ candidate arcs
    def arcs(self, speeds, omegas, x, y, yaw, horizon, near=None) -> tuple[pd.DataFrame, dict]:
        """Features of N arcs (speed m/s, turn rate rad/s) from the pose estimate (x, y, yaw), each
        followed out to `horizon` m. -> (features DataFrame, extras: end xy, step, seen, samples).
        near: also step_near / drop_near / hit_near over the samples within `near` m along the arc
        (the safety shield: the plane is still fitted over the whole arc)."""
        M = self.map
        sw = self.cfg("navcam.swath")
        L = np.linspace(0.05, max(horizon, 0.25), N_ALONG)
        T = np.linspace(-sw, sw, N_ACROSS)
        kap = np.asarray(omegas, float) / np.maximum(np.asarray(speeds, float), 1e-3)          # curvature
        kap = kap[:, None]
        th = yaw + kap * L[None]
        small = np.abs(kap) < 1e-3
        ks = np.where(small, 1.0, kap)
        cx = np.where(small, x + L[None] * np.cos(yaw), x + (np.sin(th) - np.sin(yaw)) / ks)
        cy = np.where(small, y + L[None] * np.sin(yaw), y - (np.cos(th) - np.cos(yaw)) / ks)
        nx, ny = -np.sin(th), np.cos(th)
        X = cx[..., None] + nx[..., None] * T                   # (N, along, across)
        Y = cy[..., None] + ny[..., None] * T
        iy, ix, ok = M.index(X, Y)
        z = np.where(ok, M.z[iy, ix], np.nan)
        zmax = np.where(ok, M.zmax[iy, ix], np.nan)
        seen = np.isfinite(z)
        Nn = len(kap)
        Z, Zm, Sn = z.reshape(Nn, -1), zmax.reshape(Nn, -1), seen.reshape(Nn, -1)
        frac = Sn.mean(1)
        with np.errstate(invalid="ignore"):
            zfill = np.nanmean(np.where(Sn, Z, np.nan), 1)
        zfill = np.where(np.isfinite(zfill), zfill, 0.0)
        Zf = np.where(Sn, Z, zfill[:, None])
        Zmf = np.where(Sn, Zm, Zf)
        Ls, Ts = np.meshgrid(L, T, indexing="ij")
        D = np.column_stack([Ls.ravel(), Ts.ravel(), np.ones(Ls.size)])
        coef = Zf @ np.linalg.pinv(D).T                         # (N, 3): along, across, offset
        plane = coef @ D.T
        step = np.where(Sn, Zmf - plane, -np.inf).max(1)
        step = np.where(np.isfinite(step), step, 0.0)
        drop = np.where(Sn, plane - Zf, -np.inf).max(1)
        drop = np.where(np.isfinite(drop), drop, 0.0)
        rough = np.sqrt(((Zf - plane) ** 2 * Sn).sum(1) / np.maximum(Sn.sum(1), 1))
        # grade: the steepest slope along the arc over ~0.2 m, up or down, on any line along it (a
        # wheel's track: averaging across the swath would flatten a small crater's wall). The plane
        # fit hides a smooth wall (a crater's bowl is a tilted plane locally), so the step and drop
        # alone let a rover drive down a slope it can never climb back up. Unseen cells don't count.
        z3 = np.where(Sn, Z, np.nan).reshape(Nn, len(L), len(T))
        kk = max(1, int(np.ceil(0.2 / max(L[1] - L[0], 1e-6))))
        if len(L) > kk:
            with np.errstate(invalid="ignore"):
                g = np.abs((z3[:, kk:] - z3[:, :-kk]) / (L[kk:] - L[:-kk])[None, :, None])
            grade = np.where(np.isfinite(g), g, 0.0).max((1, 2))
        else:
            grade = np.zeros(Nn)

        def mean(a):
            a = np.where(ok, a, np.nan).reshape(Nn, -1)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(a, 1) if a.size else np.zeros(Nn)
            return np.where(np.isfinite(m), m, np.nan)

        F = {"val": mean(M.val[iy, ix]), "sat": mean(M.sat[iy, ix]), "tex": mean(M.tex[iy, ix]),
             "slope": coef[:, 0], "xslope": coef[:, 1], "step": step, "drop": drop, "rough": rough, "seen": frac}
        if self.use_codes:
            Ek = M.E[iy, ix]                                    # (N, along, across, 8)
            for k in range(N_CODES):
                F[f"e{k}"] = mean(Ek[..., k])
        ends = np.column_stack([cx[:, -1], cy[:, -1]])
        hit = np.zeros(Nn, bool)                                # through one of the outpost's structures
        for ox, oy, r in self.infra:
            hit |= (np.hypot(X - ox, Y - oy) < r + 0.04).reshape(Nn, -1).any(1)
        ex = {}
        if near is not None:
            nm = (Ls.ravel() <= near)[None]
            ex["step_near"] = np.where(Sn & nm, Zmf - plane, 0.0).max(1)
            ex["drop_near"] = np.where(Sn & nm, plane - Zf, 0.0).max(1)
            hn = np.zeros(Nn, bool)
            close = (L <= near)
            for ox, oy, r in self.infra:
                hn |= (np.hypot(X[:, close] - ox, Y[:, close] - oy) < r).reshape(Nn, -1).any(1)   # the planner keeps 4 cm more
            ex["hit_near"] = hn
        return pd.DataFrame(F), ex | {"end": ends, "step": step, "drop": drop, "grade": grade, "seen": frac,
                                      "samples": (cx, cy), "L": float(L[-1]),
                                 "hit": hit}


def mujoco_has_camera(m, name) -> bool:
    try:
        m.camera(name)
        return True
    except KeyError:
        return False


def _depth(cam):
    """Depth only (one render, no colour pass)."""
    r = cam._r
    r.enable_depth_rendering()
    r.update_scene(cam.d, camera=cam.id)
    d = r.render().copy()
    r.disable_depth_rendering()
    return d


def _hsv(rgb):
    import cv2
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    return np.column_stack([h[:, 0] * 2, h[:, 1] / 255, h[:, 2] / 255])
