"""The SO-101 on a rover: the lab's arm code (so101/robot), unchanged, running in the arm's own frame.

Everything the lab skills compute (IK targets, object poses, the landing point) is in the arm's
WORK frame: the arm base frame shifted down its z axis onto the work plane, so the plane is z = 0
exactly like the lab table. The sampler's arm mounts low on the body's front face and the plane is
the ground under the rock it picks (or the tray's floor it places on): set_work_plane(). Only the
places where the simulator is read or commanded convert between that frame and the world:
  _ik_once / _tool_R   IK target position and tool orientation -> world
  _tool_pos / _current_yaw / true_pose / ball_pos              world -> arm frame
  _step                one shared world steps for every rover
So a pick or a throw on Mars runs the same motion code as in the lab, while the rover is braked.

While the rover drives, the arm is folded across the body's front (mars_rover.STOW_Q), out of both
cameras' view. It unfolds to the lab's home pose before a pick and folds back after it; between the
lift and the place, it folds with the rock held (the transport hold, World.hold) while the rover
drives home to its bay, then unfolds to exactly where it lifted the rock and places it in the tray.
"""
from __future__ import annotations

import mujoco
import numpy as np

from so101.robot.base import wrap
from so101.robot.so101_mujoco import ARM_JOINTS, GRIP_CLOSED, OBJ_TYPES, SO101Sim

from .world import PICK_LOCAL, ROVER


class _Mounted:
    """Mixin: wire an SO101Sim subclass to rover `rid`'s arm inside a shared World."""

    def _mount(self, world, rid, seed=0, obs_noise=0.002):
        self.w, self.rid = world, rid
        p = f"{rid}/arm_"
        m = self.m = world.m
        self.d = world.d
        self.dk = mujoco.MjData(m)                       # scratch data for IK (whole world, base included)
        self.rng = np.random.default_rng(seed)
        self.obs_noise, self.realtime, self.viewer, self._last_sync = obs_noise, 0.0, None, 0.0
        self.plane = 0.0                                 # the work plane along the base's z (m): 0 = the base
        self.site = m.site(p + "gripperframe").id
        self.grip_body = m.body(p + "gripper").id
        self.base_body = m.body(p + "base").id
        self.jq = np.array([m.joint(p + j).qposadr[0] for j in ARM_JOINTS])
        self.jdof = np.array([m.joint(p + j).dofadr[0] for j in ARM_JOINTS])
        self.act = {m.actuator(i).name[len(p):]: i for i in range(m.nu) if m.actuator(i).name.startswith(p)}
        ids = [m.joint(p + j).id for j in ARM_JOINTS]
        self.lo, self.hi = m.jnt_range[ids, 0].copy(), m.jnt_range[ids, 1].copy()
        self.grip_force_max = float(m.actuator_forcerange[self.act["gripper"], 1])
        self._gq = int(m.joint(p + "gripper").qposadr[0])
        self._pan = m.joint(p + "shoulder_pan").id
        mujoco.mj_forward(m, self.d)
        self.home_q = self.ik([0.15, 0.0, 0.14], yaw=np.pi / 2)[0]
        self._set_q(np.asarray(ROVER.STOW_Q, float), GRIP_CLOSED, hard=True)    # folded: the rover may drive
        self._reach = self._measure_reach()

    # ------------------------------------------------------------ folding for driving
    def _joint_move(self, q, seconds=1.5):
        """Smooth joint-space motion to pose q (folding and unfolding: not a lab skill motion)."""
        q0, q = self.d.qpos[self.jq].copy(), np.asarray(q, float)
        grip = self.d.ctrl[self.act["gripper"]]
        n = 30
        steps = max(1, int(seconds / n / self.m.opt.timestep))
        for i in range(1, n + 1):
            self._set_q(q0 + (0.5 - 0.5 * np.cos(np.pi * i / n)) * (q - q0), grip)
            self._step(steps)
        self._step(100)                                  # let the servos settle

    def stow(self):
        self._joint_move(ROVER.STOW_Q)

    def unstow(self):
        self._joint_move(self.home_q)

    # ------------------------------------------------------------ frames
    def base_frame(self, d=None):
        d = self.d if d is None else d
        return d.xpos[self.base_body].copy(), d.xmat[self.base_body].reshape(3, 3).copy()

    def to_world(self, p):
        """Work frame -> world."""
        pb, Rb = self.base_frame()
        return pb + Rb @ (np.asarray(p, float) + [0.0, 0.0, self.plane])

    def to_local(self, p):
        """World -> work frame (the arm base frame with z measured from the work plane)."""
        pb, Rb = self.base_frame()
        return Rb.T @ (np.asarray(p, float) - pb) - [0.0, 0.0, self.plane]

    def set_work_plane(self, world_xyz=None):
        """The work plane passes through world point world_xyz (the ground under a rock, the tray's
        floor); None: back to the base's own height."""
        if world_xyz is None:
            self.plane = 0.0
            self._remeasure()
            return
        pb, Rb = self.base_frame()
        self.plane = float((Rb.T @ (np.asarray(world_xyz, float) - pb))[2])
        self._remeasure()

    def _remeasure(self):
        """The reach band on the current work plane (cached per 5 mm of plane height)."""
        cache = self.__dict__.setdefault("_reach_cache", {})
        k = round(self.plane / 0.005)
        if k not in cache:
            cache[k] = self._measure_reach()
        self._reach = cache[k]

    def base_yaw(self):
        R = self.base_frame()[1]
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------ the frame-sensitive lab hooks
    def _ik_once(self, pos, yaw, q0, iters, tol):
        return super()._ik_once(self.to_world(pos), yaw, q0, iters, tol)

    def _tool_R(self, yaw):
        return self.base_frame()[1] @ super()._tool_R(yaw)

    def _tool_pos(self):
        return self.to_local(self.d.site_xpos[self.site])

    def _current_yaw(self):
        R = self.base_frame()[1].T @ self.d.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(R[1, 2], R[0, 2]))

    def _grip_qadr(self):
        return self._gq

    def _step(self, n=1):
        self.w.step(n)

    def close(self):
        pass                                             # the World owns the viewer


class MountedPicker(_Mounted, SO101Sim):
    """Picks the selected rock off the ground; `carry` (set by the mission) drives home between the
    lift and the place, and returns the tray's world point."""
    name = "so101-lekiwi-pick"
    skills = ("pick",)

    def __init__(self, world, rid, seed=0, obs_noise=0.002):
        self._mount(world, rid, seed, obs_noise)
        self.target_radius = 0.03                        # the lab's place tolerance (the hopper is wider)
        self.set_target(PICK_LOCAL)
        self.carry = None
        self._shift = np.zeros(2)
        if world.rocks:
            self.select(world.rocks[0])

    def set_target(self, xy):
        self.target = np.asarray(xy, float)

    def select(self, sample: dict):
        self._shift = np.zeros(2)
        self.sample = sample
        self.obj_type, self.obj = sample["type"], sample["body"]
        self.obj_ids = {sample["type"]: sample}

    def pick_place(self, **params) -> dict:
        """The lab's pick-and-place, unfolded from the driving pose before and folded back after."""
        self.unstow()
        try:
            return super().pick_place(**params)
        finally:
            self.stow()

    def spawn(self, obj_type=None, pose=None):
        self._settle()                                   # samples lie where they lie: nothing is placed

    def spawn_near_target(self, obj_type=None, **_):
        self.spawn()

    def true_pose(self):
        q = self.d.qpos[self.sample["qadr"]:self.sample["qadr"] + 7]
        p = self.to_local(q[:3])
        w, x, y, z = q[3:]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - self.base_yaw()
        up = 1 - 2 * (x * x + y * y)
        return float(p[0]), float(p[1]), float(p[2]), float(wrap(yaw)), float(up)

    def get_state(self):
        st = super().get_state() | {"site": self.w.site_code}
        if np.any(self._shift):                         # after a carry: positions relative to the world target
            st["x"], st["y"] = st["x"] - float(self._shift[0]), st["y"] - float(self._shift[1])
            st["dist_to_target"] = float(np.hypot(st["x"] - self.target[0], st["y"] - self.target[1]))
        return st

    def _carry(self):
        """The job's carry: it drives to the place station and returns the place point in the WORLD.
        Returns the shift of the place point in the arm frame (docking is never perfect), and keeps
        that shift so the outcome is still measured against the world point (see get_state)."""
        if self.carry is None:
            return None
        q_lift = self.d.qpos[self.jq].copy()
        self.w.hold(self.rid, self.sample, True)         # the transport hold, then fold back over the deck
        self.stow()
        try:
            world_xy = self.carry()
        finally:
            self._joint_move(q_lift)                     # unfold to where the object was lifted
            self.w.hold(self.rid, self.sample, False)
        if world_xy is None:
            return None
        self._shift = self.place_local(world_xy) - self.target
        return self._shift

    def place_local(self, world_xy):
        """A world point on the place surface (the tray's floor) in this arm's work frame, where the
        rover stands now; the work plane moves onto that surface."""
        z = self.w.work_z(world_xy)
        self.set_work_plane([world_xy[0], world_xy[1], z])
        return np.asarray(self.to_local(np.array([world_xy[0], world_xy[1], z]))[:2], float)
