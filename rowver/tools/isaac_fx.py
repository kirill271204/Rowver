"""Mars look for the Isaac renders (live mirror and offline video): scenery and dust. Render side only:
nothing here touches the MuJoCo sim, the rovers' logic or what the rovers sense.

  backdrop        the HiRISE 600 m ring and the CTX 10 km ring around the site (our mars/world
                  scene_backdrop.usdc: same site, same z datum, Mars curvature), so the map runs to the horizon
  haze            distance haze in Mars dust colour (Z-up height fog off, distance fog from ~80 m)
  dusty()         a dust film on the rovers and the garage: colours pulled towards the regolith, matte
  Motes           fine dust drifting in the wind around the camera target (one PointInstancer)
  WheelDust       dust kicked up behind each wheel while a rover drives (one PointInstancer, pooled puffs)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

SCALE = 5.0
DUST = (0.66, 0.46, 0.31)                 # regolith dust (linear-ish sRGB)
WIND = np.array([1.6, 0.7, 0.0])          # m/s, full scale: a light breeze from the WSW


def backdrop(stage, path: str):
    if not Path(path).exists():
        return False
    stage.DefinePrim("/World/Backdrop", "Xform").GetReferences().AddReference(path)
    return True


def haze(settings, start=150.0, end=9000.0, density=0.3, color=(0.60, 0.44, 0.31)):
    s = settings
    for k, v in (("/rtx/raytracing/fractionalCutoutOpacity", True),      # see-through dust puffs
                 ("/rtx/translucency/enabled", True),
                 ("/rtx/fog/enabled", density > 0), ("/rtx/fog/fogColor", list(color)),
                 ("/rtx/fog/fogColorIntensity", 1.0), ("/rtx/fog/fogZup/enabled", True),
                 ("/rtx/fog/fogStartHeight", 0.0), ("/rtx/fog/fogHeightDensity", 0.0),
                 ("/rtx/fog/fogStartDist", float(start)), ("/rtx/fog/fogEndDist", float(end)),
                 ("/rtx/fog/fogDistanceDensity", float(density))):
        try:
            s.set(k, v)
        except Exception:
            pass


def dusty(stage, root="/World/Mars/Fleet", amount=0.28, min_rough=0.7):
    """Pull every UsdPreviewSurface colour under root towards the dust colour and make it matte."""
    n = 0
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return 0
    from pxr import Usd
    it = iter(Usd.PrimRange(prim))
    for p in it:
        nm = p.GetName()
        # natural rocks keep their own colour (mujoco.usd puts materials in _materials/Material_<geom>)
        if nm.startswith(("Mesh_Xform_rocks_", "Mesh_Xform_srock", "Material_rocks_", "Material_srock")):
            it.PruneChildren()
            continue
        if p.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(p)
        if sh.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        dc = sh.GetInput("diffuseColor")
        if dc and not dc.HasConnectedSource():
            c = dc.Get()
            if c is not None:
                c = np.array(c, float)
                dc.Set(Gf.Vec3f(*((1 - amount) * c + amount * np.array(DUST))))
        r = sh.GetInput("roughness")
        if r and not r.HasConnectedSource():
            v = r.Get()
            r.Set(float(max(v if v is not None else 0.5, min_rough)))
        else:
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(min_rough))
        n += 1
    return n


def _material(stage, path, color, opacity, rough=1.0):
    mtl = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.01)   # cutout mode: RTX fractional cutout = see-through
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mtl


def _instancer(stage, path, radius, mtl, n):
    inst = UsdGeom.PointInstancer.Define(stage, path)
    proto = UsdGeom.Sphere.Define(stage, path + "/proto")
    proto.CreateRadiusAttr(float(radius))
    UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(mtl)
    inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
    inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.zeros(n, np.int32)))
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.zeros((n, 3), np.float32)))
    inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.zeros((n, 3), np.float32)))
    return inst


class Motes:
    """Fine dust drifting in the wind in a box around the target (wraps around, so it never runs out)."""

    def __init__(self, stage, n=1200, box=(30.0, 30.0, 3.0), seed=3):
        self.n, self.box = n, np.array(box)
        rng = np.random.default_rng(seed)
        self.p = (rng.random((n, 3)) - [0.5, 0.5, 0.0]) * self.box
        self.size = 0.5 + rng.random(n) * 1.2
        self.phase = rng.random((n, 3)) * 6.28
        mtl = _material(stage, "/World/FX/Looks/mote", (0.85, 0.66, 0.48), 0.3)
        self.inst = _instancer(stage, "/World/FX/Motes", 0.004, mtl, n)
        self.t = 0.0

    def step(self, target, dt):
        self.t += dt
        turb = 0.25 * np.sin(self.t * 0.7 + self.phase)
        self.p += (WIND + turb) * dt
        half = self.box * [0.5, 0.5, 1.0]
        rel = self.p.copy()
        rel[:, :2] = (rel[:, :2] + half[:2]) % self.box[:2] - half[:2]
        rel[:, 2] = rel[:, 2] % self.box[2]
        self.p = rel
        pos = rel + [target[0], target[1], target[2] - 0.5]
        self.inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
        self.inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(self.size[:, None], 3, 1).astype(np.float32)))


class Ground:
    """Terrain height at any (x, y), full scale, from the terrain mesh on the stage (401 x 401 grid, row 0 south)."""

    def __init__(self, stage, path="/World/Mars/Terrain/ground"):
        self.ok = False
        m = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        pts = m.GetPointsAttr().Get() if m else None
        if not pts:
            return
        P = np.array(pts, float)
        n = int(round(math.sqrt(len(P))))
        if n * n != len(P):
            return
        G = P.reshape(n, n, 3) * SCALE
        self.x0, self.y0 = G[0, 0, 0], G[0, 0, 1]
        self.dx, self.dy = G[0, 1, 0] - G[0, 0, 0], G[1, 0, 1] - G[0, 0, 1]
        self.Z, self.n, self.ok = G[:, :, 2], n, True

    def z(self, x, y, default=0.0):
        if not self.ok:
            return np.full(np.shape(x), default)
        fx = np.clip((np.asarray(x) - self.x0) / self.dx, 0, self.n - 1.001)
        fy = np.clip((np.asarray(y) - self.y0) / self.dy, 0, self.n - 1.001)
        i, j = np.floor(fy).astype(int), np.floor(fx).astype(int)
        ty, tx = fy - i, fx - j
        Z = self.Z
        return ((1 - ty) * ((1 - tx) * Z[i, j] + tx * Z[i, j + 1]) + ty * ((1 - tx) * Z[i + 1, j] + tx * Z[i + 1, j + 1]))


class WheelDust:
    """Regolith kicked up by the wheels of every moving rover, as a plume of small particles (RTX real-time
    does not blend translucent clouds): coarse grains thrown in an arc that fall back under Mars gravity, and
    fine dust that hangs, drags and drifts downwind for a few seconds. Plus wheel tracks: darker, compressed
    regolith laid on the terrain under every wheel, kept for the whole run.
    wheels: {design: {wheels: [[x, y]], wheel_r}} at sim scale in the chassis frame (mars_rover.py)."""

    G = 3.71

    def __init__(self, stage, wheels: dict, designs: list, n=9000, seed=5, tracks=False, n_tracks=12000):
        self.wheels = [np.array(wheels[d]["wheels"]) * SCALE if d in wheels else np.zeros((0, 2)) for d in designs]
        self.wr = [float(wheels.get(d, {}).get("wheel_r", 0.06)) * SCALE for d in designs]
        self.ground = Ground(stage)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.age = np.full(n, 1e9)
        self.life = np.ones(n)
        self.rad = np.zeros(n)
        self.drag = np.zeros(n)
        self.k = 0
        self.last = None
        self.acc = np.zeros(len(designs))
        mtl = _material(stage, "/World/FX/Looks/dust", (0.80, 0.62, 0.46), 1.0)
        self.inst = _instancer(stage, "/World/FX/WheelDust", 1.0, mtl, n)
        self.tracks = None
        if tracks:
            self.nt, self.kt = n_tracks, 0
            self.tpos = np.zeros((n_tracks, 3), np.float32)
            self.tscale = np.zeros((n_tracks, 3), np.float32)
            self.tyaw = np.zeros(n_tracks)
            self.last_mark = {}
            tm = _material(stage, "/World/FX/Looks/track", (0.30, 0.19, 0.12), 1.0, rough=0.95)
            ti = UsdGeom.PointInstancer.Define(stage, "/World/FX/Tracks")
            cube = UsdGeom.Cube.Define(stage, "/World/FX/Tracks/proto")
            cube.CreateSizeAttr(1.0)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(tm)
            ti.CreatePrototypesRel().SetTargets([cube.GetPath()])
            ti.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.zeros(n_tracks, np.int32)))
            ti.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(self.tpos))
            ti.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(self.tscale))
            ti.CreateOrientationsAttr([Gf.Quath(1, 0, 0, 0)] * n_tracks)
            self.tracks, self.tdirty = ti, False

    def _emit(self, p, v, r, life, drag):
        i = self.k % self.n
        self.k += 1
        self.pos[i], self.vel[i], self.rad[i], self.age[i], self.life[i], self.drag[i] = p, v, r, 0.0, life, drag

    def _track(self, j, w, p, yaw, width):
        key = (j, w)
        q = self.last_mark.get(key)
        if q is not None and np.hypot(p[0] - q[0], p[1] - q[1]) < 0.11:
            return
        self.last_mark[key] = p.copy()
        i = self.kt % self.nt
        self.kt += 1
        self.tpos[i] = (p[0], p[1], p[2] + 0.004)
        self.tscale[i] = (0.13, width, 0.03)
        self.tyaw[i] = yaw
        self.tdirty = True

    def step(self, rovers, dt):
        """rovers: [(x, y, z, yaw)] at sim scale, in the packets' rover order."""
        cur = np.array([[r[0], r[1], r[2], r[3]] for r in rovers], float)
        # packets do not arrive every render frame: measure speed between pose changes, hold it briefly
        if self.last is None or len(self.last) != len(cur):
            self.last, self.since, self.speed = cur, 0.0, np.zeros(len(cur))
        elif not np.allclose(cur, self.last, atol=1e-6):
            d = (cur[:, :2] - self.last[:, :2]) * SCALE
            self.speed = np.hypot(d[:, 0], d[:, 1]) / max(self.since + dt, 1e-3)   # full-scale m/s
            self.last, self.since = cur, 0.0
        else:
            self.since += dt
            if self.since > 0.5:                                    # nothing new for a while: the rover is still
                self.speed = np.zeros(len(cur))
        speed = self.speed
        if dt > 0:
            for j, (x, y, z, yaw) in enumerate(cur):
                if not len(self.wheels[j]):
                    continue
                c, s_ = math.cos(yaw), math.sin(yaw)
                fwd = np.array([c, s_, 0.0])
                wx, wy = self.wheels[j][:, 0], self.wheels[j][:, 1]
                px, py = x * SCALE + c * wx - s_ * wy, y * SCALE + s_ * wx + c * wy
                pz = self.ground.z(px, py, default=z * SCALE)
                if self.tracks is not None and speed[j] > 0.02:
                    for w in range(len(px)):
                        self._track(j, w, np.array([px[w], py[w], pz[w]]), yaw, 1.15 * self.wr[j])
                if speed[j] < 0.05 or speed[j] > 5.0:
                    continue
                self.acc[j] += dt * min(speed[j], 1.5) / 0.07          # bursts per wheel: ~14 / s at 1 m/s
                while self.acc[j] >= 1.0:
                    self.acc[j] -= 1.0
                    for w in range(len(px)):
                        base = np.array([px[w], py[w], pz[w] + 0.02]) - 0.8 * self.wr[j] * fwd   # behind the contact
                        for _ in range(6):                             # coarse grains, ballistic
                            v = -(0.3 + 0.6 * self.rng.random()) * min(speed[j], 1.5) * fwd \
                                + [0, 0, 0.5 + 1.0 * self.rng.random()] + self.rng.normal(0, 0.25, 3) * [1, 1, 0]
                            self._emit(base + [*self.rng.normal(0, 0.05, 2), 0], v,
                                       0.012 + 0.02 * self.rng.random(), 1.2, 0.0)
                        for _ in range(10):                            # fine dust, hangs and drifts
                            v = -0.25 * min(speed[j], 1.5) * fwd + [0, 0, 0.25 + 0.35 * self.rng.random()] \
                                + self.rng.normal(0, 0.2, 3)
                            self._emit(base + [*self.rng.normal(0, 0.12, 2), 0.05 * self.rng.random()], v,
                                       0.008 + 0.012 * self.rng.random(), 2.0 + 1.5 * self.rng.random(), 1.6)
        a = self.age < self.life
        self.age[a] += dt
        heavy = a & (self.drag == 0)
        fine = a & (self.drag > 0)
        self.vel[heavy, 2] -= self.G * dt
        k = np.exp(-self.drag[fine] * dt)[:, None]
        self.vel[fine] = self.vel[fine] * k + (WIND * 0.8 + [0, 0, -0.12]) * (1 - k)
        self.pos[a] += self.vel[a] * dt
        if self.ground.ok and heavy.any():                              # grains stop where they land
            gz = self.ground.z(self.pos[heavy, 0], self.pos[heavy, 1])
            landed = np.flatnonzero(heavy)[self.pos[heavy, 2] < gz]
            self.age[landed] = 1e9
        f = np.clip(self.age / self.life, 0, 1)
        r = np.where(self.age < self.life, self.rad * np.clip((1 - f) / 0.3, 0, 1), 0.0)
        if self.tracks is not None and self.tdirty:
            self.tracks.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self.tpos))
            self.tracks.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(self.tscale))
            self.tracks.GetOrientationsAttr().Set([Gf.Quath(float(math.cos(t / 2)), 0.0, 0.0, float(math.sin(t / 2)))
                                                   for t in self.tyaw])
            self.tdirty = False
        self.inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self.pos.astype(np.float32)))
        self.inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(r[:, None], 3, 1).astype(np.float32)))
