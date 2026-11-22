"""A simulated RGB-D camera: MuJoCo renders the frames, the rest is ordinary pinhole geometry.

The camera's pose (extrinsics) and field of view (intrinsics) are known, as they would be
for a calibrated real camera. Every pixel with its depth can then be turned into a point
on the table, in metres, in the robot's world frame.
"""
from __future__ import annotations

import mujoco
import numpy as np

# One renderer (one OpenGL context) per model and frame size, shared by every camera of that size:
# each context uploads the whole scene (meshes, textures), so a renderer per camera multiplies the
# scene's memory by the number of cameras. Frames are rendered one at a time, so sharing changes no
# frame. Reference-counted: the context closes with its last camera.
_SHARED: dict = {}


def shared_renderer(model, height, width):
    key = (id(model), int(height), int(width))
    if key not in _SHARED:
        _SHARED[key] = [mujoco.Renderer(model, int(height), int(width)), 0]
    _SHARED[key][1] += 1
    return _SHARED[key][0], key


def release_renderer(key):
    entry = _SHARED.get(key)
    if entry is None:
        return
    entry[1] -= 1
    if entry[1] <= 0:
        entry[0].close()
        del _SHARED[key]


class RGBDCamera:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str, width: int, height: int):
        self.m, self.d = model, data
        self.id = model.camera(name).id
        self.w, self.h = int(width), int(height)
        self._r, self._key = shared_renderer(model, self.h, self.w)
        self.f = (self.h / 2) / np.tan(np.radians(model.cam_fovy[self.id]) / 2)   # focal length in pixels
        u = np.arange(self.w) + 0.5 - self.w / 2
        v = np.arange(self.h) + 0.5 - self.h / 2
        self._rays = np.stack([*np.meshgrid(u / self.f, -v / self.f), -np.ones((self.h, self.w))], -1)

    # ---------------------------------------------------------------- frames
    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB (h, w, 3) uint8 and depth (h, w) float metres along the optical axis."""
        mujoco.mj_forward(self.m, self.d)
        self._r.update_scene(self.d, camera=self.id)
        rgb = self._r.render().copy()
        self._r.enable_depth_rendering()
        self._r.update_scene(self.d, camera=self.id)
        depth = self._r.render().copy()
        self._r.disable_depth_rendering()
        return rgb, depth

    def segmentation(self) -> np.ndarray:
        """Geom id per pixel (-1 = none). Ground truth for testing the detector only; never used to detect."""
        self._r.enable_segmentation_rendering()
        self._r.update_scene(self.d, camera=self.id)
        seg = self._r.render().copy()
        self._r.disable_segmentation_rendering()
        return np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)

    # ---------------------------------------------------------------- geometry
    @property
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.cam_xpos[self.id].copy(), self.d.cam_xmat[self.id].reshape(3, 3).copy()

    def points(self, depth: np.ndarray) -> np.ndarray:
        """World point (h, w, 3) of every pixel."""
        c, R = self.pose
        return c + (self._rays * depth[..., None]) @ R.T

    def project(self, p) -> tuple[float, float]:
        """World point -> pixel (u, v), same convention as the image arrays (column, row)."""
        c, R = self.pose
        pc = R.T @ (np.asarray(p, float) - c)
        return (self.w / 2 + self.f * pc[0] / -pc[2] - 0.5, self.h / 2 - self.f * pc[1] / -pc[2] - 0.5)

    def px_per_m(self, p) -> float:
        """Pixels per metre at the depth of world point p."""
        c, R = self.pose
        return self.f / max(-(R.T @ (np.asarray(p, float) - c))[2], 1e-6)

    def close(self):
        if self._key is not None:
            release_renderer(self._key)
            self._key = None
