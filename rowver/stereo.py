"""Stereo depth, as a real rover gets it: two cameras a baseline apart, block matching, triangulation.

  StereoDepth(world, left, right, W, H, baseline).depth(rgb_left=None) -> (H, W) depth in metres along the
  optical axis (the convention of RGBDCamera.capture), NaN where the pair found no match.

The left camera is the rover's navcam / hazcam itself (its colour frame and DINOv2 codes are unchanged); the
right one is its stereo partner (mars_rover.py: navcam_r, hazcam_r). Both are rendered, turned grey, and
matched with OpenCV's semi-global block matcher (cv2.StereoSGBM, 3-way, sub-pixel), with its uniqueness and
left-right consistency checks and a speckle filter; depth = f * B / disparity. The error grows with range
(disparity is quantised at 1/16 px), and textureless or occluded ground stays NaN: the map counts it unseen.
"""
from __future__ import annotations

import cv2
import numpy as np
from so101.perception.camera import RGBDCamera


class StereoDepth:
    def __init__(self, world, left: str, right: str, W: int, H: int, baseline: float, max_range=6.0,
                 min_range=0.15, left_cam: RGBDCamera | None = None):
        m, d = world.m, world.d
        self.L = left_cam or RGBDCamera(m, d, left, W, H)
        self.R = RGBDCamera(m, d, right, W, H)
        self.f, self.B = float(self.L.f), float(baseline)
        self.min_range, self.max_range = float(min_range), float(max_range)
        max_disp = self.f * self.B / self.min_range
        nd = int(min(np.ceil(max_disp / 16.0) * 16, (W // 2) // 16 * 16))
        bs = 5
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max(nd, 16), blockSize=bs,
                                          P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=8,
                                          speckleWindowSize=60, speckleRange=2, preFilterCap=31,
                                          mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.valid = float("nan")

    def _rgb(self, cam):
        r = cam._r
        r.update_scene(cam.d, camera=cam.id)
        return r.render()

    def depth(self, rgb_left=None) -> np.ndarray:
        gl = cv2.cvtColor(rgb_left if rgb_left is not None else self._rgb(self.L), cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(self._rgb(self.R), cv2.COLOR_RGB2GRAY)
        disp = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.f * self.B / disp
        z[(disp <= 0.5) | ~np.isfinite(z) | (z > self.max_range) | (z < self.min_range)] = np.nan
        self.valid = float(np.isfinite(z).mean())
        return z

    def close(self):
        self.R.close()
