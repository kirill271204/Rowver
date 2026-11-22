"""The safety shield: an independent reflex that watches every drive segment while it runs.

It never consults TabPFN (this module imports nothing from brain/). Its limits come from the
rover's own geometry, not from experience:
  tilt_limit  the static tip-over angle (CoM height over the nearest edge of the wheels' support
              polygon) x safety.tilt_margin
  stop_dist   v^2 / (2 a), a from the drive's acceleration ramp, + safety.margin, counted from
              the rover's nose
During a segment (Rover.drive(monitor=...), every 20 ms):
  1. the chassis tilt (IMU) above tilt_limit -> stop
  2. every safety.recheck_s: a fresh hazcam depth frame goes into the local map, and the ground
     the wheels reach within stop_dist is checked again: a step above safety.step_limit, a drop
     below safety.drop_limit, a station plinth -> stop
A stop ends the segment at once (wheels stop, brake). The body records it (rep_shield, failed = 1),
so TabPFN learns which arcs end in a stop, and the driver escapes as from a blocked view.
The planner's own vetoes (navcam.step_veto, drop_veto) are stricter than the shield's limits: the
shield is the backstop for what appears or is measured better while driving.
"""
from __future__ import annotations

import mujoco
import numpy as np



def _hull(P):
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    P = sorted(map(tuple, P))
    if len(P) < 3:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def tip_angle(rover) -> tuple[float, float, float]:
    """Static tip-over angle (rad) of the rover as built: atan(distance from the CoM's ground
    projection to the nearest support-polygon edge / CoM height above the wheels' contact plane).
    -> (angle, that distance, CoM height)."""
    w, k = rover.w, rover.kin
    mujoco.mj_forward(w.m, w.d)
    x, y, _, R = rover.true_pose()
    pc = w.d.xpos[rover.ch]
    com = (R.T @ (w.d.subtree_com[rover.ch] - pc))                  # chassis frame
    wz = np.array([(R.T @ (w.d.xpos[w.m.body(f"{rover.rid}/wheel_{n}").id] - pc))[2]
                   for n in k.names])
    h = float(com[2] - (wz.mean() - k.R))                           # above the contact plane
    H = _hull(k.r)
    c = com[:2]
    dist = []
    for a, b in zip(H, np.roll(H, -1, 0)):
        e = b - a
        dist.append(abs(e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0])) / np.hypot(*e))
    d = float(min(dist))
    return float(np.arctan2(d, h)), d, h


class Shield:
    def __init__(self, rover, cam, cfg):
        self.rv, self.cam, self.cfg = rover, cam, cfg
        self.tip, self.edge, self.com_h = tip_angle(rover)
        self.tilt_limit = cfg("safety.tilt_margin") * self.tip
        self.a = rover.kin.v_max / rover.RAMP_S                       # the drive's own acceleration limit
        self.front = max(float(np.max(rover.kin.r[:, 0])), rover.design.self_box[1])  # the nose (body or wheels)
        self.trips: list = []

    def tilt(self):
        """Chassis tilt against gravity (what the IMU accelerometer measures at rest)."""
        g = np.asarray(self.rv.w.m.opt.gravity, float)
        up = -g / max(float(np.linalg.norm(g)), 1e-9)
        z = self.rv.true_pose()[3][:, 2]
        return float(np.arccos(np.clip(z @ up, -1, 1)))

    def stop_dist(self, v):
        return float(v) ** 2 / (2 * self.a) + self.cfg("safety.margin")

    def monitor(self, v, w, horizon, camera=True):
        """A per-segment watcher for Rover.drive: f(t) -> None (fine) or the reason to stop."""
        cfg, cam, rv = self.cfg, self.cam, self.rv
        last = [-np.inf]
        near = self.front + self.stop_dist(v)
        start = self.tilt()                # a rover already resting past the limit may drive off, not tip further

        def f(t):
            tilt = self.tilt()
            if tilt > self.tilt_limit and tilt > start + np.radians(3.0):
                return f"tilt {np.degrees(tilt):.0f} deg > limit {np.degrees(self.tilt_limit):.0f}"
            if not camera or t - last[0] < cfg("safety.recheck_s"):
                return None
            last[0] = t
            cam.haz_update()
            x, y, yaw = rv.pose()
            _, ex = cam.arcs(np.array([max(v, 1e-3)]), np.array([w]), x, y, yaw, horizon, near=near)
            if ex["hit_near"][0]:
                return "a structure within the stopping distance"
            if ex["step_near"][0] > cfg("safety.step_limit"):
                return f"step {ex['step_near'][0]:.3f} m ahead"
            if ex["drop_near"][0] > cfg("safety.drop_limit"):
                return f"drop {ex['drop_near'][0]:.3f} m ahead"
            return None
        return f
