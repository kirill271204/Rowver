"""Pointing a rover's mast instruments at a target, the way a real mast does (pan and tilt).

The cameras that only see what is straight ahead (rover.xml cameras) cannot look at a rock off to the side;
a real mast turns its head. Mast renders from an instrument's real position on the rover, aimed at a target
point, with that instrument's field of view, and measures the target in the image:

  rmi(rid, item, target)   scientist: the RMI telescope (8 deg): rock colour / texture and its size from its
                           pixel extent at the measured range
  pair(rid, item, target, ...)  a stereo pair (two parallel cameras a baseline apart, aimed together):
                           block-matched depth (stereo.py's matcher) at the rock's pixels -> its range and its
                           size from the 3D extent of its points. The scientist's colour zoom pair (9 cm, 25 deg)
                           and the sampler's panoramic bar (17 cm, 70 deg)

The target's pixels come from the renderer's object mask, the same way the navcam finds rocks (missions.
visible_rocks); every number written to the tables is measured from the images.
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

ZOOM = {"fovy": 25.0, "baseline": 0.09, "size": (192, 144)}          # scientist's colour zoom pair
PANO = {"fovy": 70.0, "baseline": 0.17, "size": (240, 180)}          # sampler's panoramic camera bar
RMI = {"fovy": 8.0, "size": (128, 128)}


class Mast:
    def __init__(self, world):
        self.w = world
        self._r = {}

    def _renderer(self, W, H):
        if (W, H) not in self._r:
            self._r[(W, H)] = mujoco.Renderer(self.w.m, H, W)
        return self._r[(W, H)]

    def _render(self, eye, target, fovy, W, H, seg=False):
        """A free camera at `eye` looking at `target` with vertical FOV `fovy` (deg) -> rgb or geom-id mask."""
        m, d = self.w.m, self.w.d
        v = np.asarray(target, float) - np.asarray(eye, float)
        dist = float(np.linalg.norm(v))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        cam.elevation = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
        old = float(m.vis.global_.fovy)
        m.vis.global_.fovy = fovy                         # the instrument's field of view (render-time only)
        try:
            r = self._renderer(W, H)
            if seg:
                r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam)
            out = r.render().copy()
            if seg:
                r.disable_segmentation_rendering()
                out = np.where(out[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, out[..., 0], -1)
        finally:
            m.vis.global_.fovy = old
        return out

    def _mask(self, eye, target, fovy, W, H, body):
        seg = self._render(eye, target, fovy, W, H, seg=True)
        return (seg >= 0) & (self.w.m.geom_bodyid[np.maximum(seg, 0)] == body)

    # ---------------------------------------------------------------- instruments
    def rmi(self, eye, item, target) -> dict:
        W, H = RMI["size"]
        rgb = self._render(eye, target, RMI["fovy"], W, H)
        px = self._mask(eye, target, RMI["fovy"], W, H, item["body"])
        if px.sum() < 6:
            return {"rmi_px": float(px.sum())}
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        v = hsv[..., 2][px] / 255.0
        ys, xs = np.nonzero(px)
        rng = float(np.linalg.norm(np.asarray(target) - np.asarray(eye)))
        m_per_px = 2 * rng * np.tan(np.radians(RMI["fovy"]) / 2) / H
        return {"rmi_px": float(px.sum()), "rmi_val": float(v.mean()), "rmi_sat": float(hsv[..., 1][px].mean() / 255),
                "rmi_tex": float(v.std()), "rmi_size_cm": float(100 * m_per_px * max(np.ptp(xs) + 1, np.ptp(ys) + 1))}

    def pair(self, eye, item, target, spec, prefix) -> dict:
        """A parallel stereo pair centred at `eye`, aimed at `target`: SGBM depth at the target's pixels."""
        W, H = spec["size"]
        fovy, B = spec["fovy"], spec["baseline"]
        v = np.asarray(target, float) - np.asarray(eye, float)
        right = np.cross(v, [0, 0, 1.0])
        right /= max(np.linalg.norm(right), 1e-9)
        eL, eR = np.asarray(eye) - right * B / 2, np.asarray(eye) + right * B / 2
        gl = cv2.cvtColor(self._render(eL, np.asarray(target) - right * B / 2, fovy, W, H), cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(self._render(eR, np.asarray(target) + right * B / 2, fovy, W, H), cv2.COLOR_RGB2GRAY)
        px = self._mask(eL, np.asarray(target) - right * B / 2, fovy, W, H, item["body"])
        f = (H / 2) / np.tan(np.radians(fovy) / 2)
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=int(np.ceil(f * B / 0.4 / 16) * 16), blockSize=5,
                                     P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=8, speckleWindowSize=40,
                                     speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
        ok = px & (disp > 0.5)
        if ok.sum() < 5:
            return {f"{prefix}_px": float(px.sum())}
        z = f * B / disp[ok]
        ys, xs = np.nonzero(ok)
        X = (xs + 0.5 - W / 2) * z / f
        Y = (ys + 0.5 - H / 2) * z / f
        size = float(max(np.ptp(X), np.ptp(Y)))
        return {f"{prefix}_px": float(px.sum()), f"{prefix}_range": float(np.median(z)),
                f"{prefix}_size_cm": 100 * size}

    def close(self):
        for r in self._r.values():
            r.close()
        self._r = {}


def instrument_pos(world, rid, cam_name="navcam") -> np.ndarray:
    """World position of one of the rover's cameras (the mast head)."""
    return world.d.cam_xpos[world.m.camera(f"{rid}/{cam_name}").id].copy()
