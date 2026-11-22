"""Shared Isaac Sim stage for the live mirror (isaac_live.py) and the offline render (isaac_render.py).

Runs inside Isaac Sim's Python (/isaac-sim/python.sh); needs only pxr and carb. Render side only.

  /World/Mars            scale 5: the sim is 1:5, the render is full size
  /World/Mars/Terrain    reference -> terrain.usdc (the compiled hfield, craters included)
  /World/Mars/Fleet      reference -> the MuJoCo geoms (scene.usdc live, fleet.usdc animated)
  /World/Lights/Sun      DistantLight, azimuth 267.4 deg (from north, clockwise), elevation 30 deg, 0.35 deg disc
  /World/Lights/Sky      DomeLight, butterscotch Mars sky (latlong HDR) or a flat butterscotch colour
  /World/Cameras/Chase   follows the deciding rover;  /World/Cameras/Overview   high, over the work zone
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

SCALE = 5.0
SUN_AZ, SUN_EL = 267.37, 30.0          # world.json "sun" (HiRISE label azimuth; the low warm sun)
SUN_COLOR = (1.0, 0.90, 0.78)
SKY_COLOR = (0.80, 0.62, 0.45)         # butterscotch
FOG_COLOR = (0.80, 0.62, 0.45)


def look_at(eye, target, up=(0, 0, 1)) -> Gf.Matrix4d:
    """Camera-to-parent transform for a USD camera (looks down -Z, +Y up)."""
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(*up)).GetInverse()


def camera(stage, path, hfov_deg=60.0, aspect=16 / 9, near=0.1, far=20000.0):
    cam = UsdGeom.Camera.Define(stage, path)
    ha = 20.955
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha / aspect)
    cam.CreateFocalLengthAttr(float(ha / 2 / math.tan(math.radians(hfov_deg) / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(near, far))
    op = UsdGeom.Xformable(cam).AddTransformOp()
    op.Set(look_at((30, -30, 20), (0, 0, 0)))
    return cam, op


def build(stage: Usd.Stage, base: Path, scene_rel: str, terrain_rel: str = "terrain.usdc", sky_hdr: str | None = None,
          sun_intensity=2500.0, sky_intensity=600.0, fps: float | None = None):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if fps:
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mars = UsdGeom.Xform.Define(stage, "/World/Mars")
    mars.AddScaleOp().Set(Gf.Vec3f(SCALE, SCALE, SCALE))
    terr = stage.DefinePrim("/World/Mars/Terrain", "Xform")
    terr.GetReferences().AddReference(str(Path(base) / terrain_rel))
    fleet = stage.DefinePrim("/World/Mars/Fleet", "Xform")
    fleet.GetReferences().AddReference(str(Path(base) / scene_rel))

    az, el = math.radians(SUN_AZ), math.radians(SUN_EL)
    sd = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])   # towards the sun
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateAngleAttr(0.35)
    sun.CreateIntensityAttr(float(sun_intensity))
    sun.CreateColorAttr(Gf.Vec3f(*SUN_COLOR))
    rot = look_at(sd, (0, 0, 0))                           # shines down its -Z, away from the sun
    rot.SetTranslateOnly(Gf.Vec3d(0, 0, 0))
    UsdGeom.Xformable(sun).AddTransformOp().Set(rot)
    sky = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    sky.CreateIntensityAttr(float(sky_intensity))
    if sky_hdr and Path(sky_hdr).exists():
        sky.CreateTextureFileAttr(Sdf.AssetPath(str(sky_hdr)))
        sky.CreateTextureFormatAttr("latlong")
    else:
        sky.CreateColorAttr(Gf.Vec3f(*SKY_COLOR))

    cams = {"chase": camera(stage, "/World/Cameras/Chase", hfov_deg=55.0),
            "overview": camera(stage, "/World/Cameras/Overview", hfov_deg=50.0),
            "top": camera(stage, "/World/Cameras/Top", hfov_deg=60.0),
            # the rover's own NavCam: 70 deg vertical FOV (mars_rover.py), 4:3
            "navcam": camera(stage, "/World/Cameras/NavCam", aspect=4 / 3, near=0.05,
                             hfov_deg=math.degrees(2 * math.atan(math.tan(math.radians(35.0)) * 4 / 3)))}
    return cams


def render_settings(settings, film_iso=200.0, fog=0.0015, renderer="RaytracedLighting"):
    s = settings
    s.set("/rtx/rendermode", renderer)
    s.set("/rtx/post/tonemap/op", 4)                       # ACES
    s.set("/rtx/post/tonemap/filmIso", float(film_iso))
    s.set("/rtx/post/tonemap/whitepoint", 6500.0)
    s.set("/rtx/post/tonemap/enabled", True)
    s.set("/rtx/post/aa/op", 3)                            # TAA
    s.set("/rtx/directLighting/sampledLighting/enabled", True)
    if fog > 0:                                            # Mars dust haze
        for k, v in (("/rtx/fog/enabled", True), ("/rtx/fog/fogColor", list(FOG_COLOR)),
                     ("/rtx/fog/fogColorIntensity", 1.0), ("/rtx/fog/fogStartDist", 40.0),
                     ("/rtx/fog/fogEndDist", 3000.0), ("/rtx/fog/fogDistanceDensity", float(fog))):
            try:
                s.set(k, v)
            except Exception:
                pass


class Chase:
    """A smoothed chase camera behind the deciding rover (world metres, render scale)."""

    def __init__(self, back=7.0, up=3.2, look_up=0.6, tau=0.9):
        self.back, self.up, self.look_up, self.tau = back, up, look_up, tau
        self.eye = self.tgt = None

    def step(self, x, y, z, yaw, dt):
        p = np.array([x, y, z]) * SCALE
        tgt = p + [0, 0, self.look_up]
        eye = p + [-self.back * math.cos(yaw), -self.back * math.sin(yaw), self.up]
        if self.eye is None:
            self.eye, self.tgt = eye, tgt
        a = 1.0 - math.exp(-max(dt, 1e-3) / self.tau)
        self.eye = self.eye + a * (eye - self.eye)
        self.tgt = self.tgt + min(1.0, 1.6 * a) * (tgt - self.tgt)     # the target leads the eye a little
        return look_at(self.eye, self.tgt)


def overview(rover_xyz: np.ndarray):
    """High three-quarter view over the rovers (sim-scale positions in)."""
    c = np.asarray(rover_xyz, float).mean(0) * SCALE
    eye = c + [-22.0, -26.0, 18.0]
    return look_at(eye, c + [0, 0, 1.0])


def top(x, y, z, height=32.0):
    """Straight down over the rover, north up (sim-scale position in)."""
    p = np.array([x, y, z]) * SCALE
    return look_at(p + [0, 0, height], p, up=(0, 1, 0))


def navcam(x, y, z, yaw, mount):
    """The rover's NavCam: mount = {pos (chassis frame, sim scale), pitch (rad, down)} from mars_rover.py.
    MuJoCo's camera looks along -z with +y up, as USD's does: x axis (0,-1,0), y axis (sin p, 0, cos p)."""
    sp, cp = math.sin(mount["pitch"]), math.cos(mount["pitch"])
    Rc = np.column_stack([[0, -1, 0], [sp, 0, cp], np.cross([0, -1, 0], [sp, 0, cp])])
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Rc
    t = (np.array([x, y, z]) + Rz @ np.asarray(mount["pos"], float)) * SCALE
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return Gf.Matrix4d(M.T.tolist())


def matrix(pos, mat9) -> Gf.Matrix4d:
    """MuJoCo geom pose -> USD row-vector transform (as mujoco.usd's exporter writes it)."""
    M = np.eye(4)
    M[:3, :3] = np.asarray(mat9, float).reshape(3, 3)
    M[:3, 3] = pos
    return Gf.Matrix4d(M.T.tolist())
