"""Real SO-101 backend (STUB: written against LeRobot's API, NOT yet tested on hardware).

Design: the MuJoCo model is used as a *digital twin* for IK and motion planning. This
class subclasses SO101Sim and overrides only the three things that touch the world:
  _set_q   -> send joint targets to the real arm (LeRobot SO101Follower)
  _step    -> wait in real time and mirror the measured joints into the twin
  get_state/true_pose -> overhead camera + ArUco markers (cortex/perception/aruco.py)
So push() and pick_place() are exactly the simulator's code paths.

Setup (once):
  pip install "lerobot[feetech]" opencv-contrib-python
  calibrate the arm with LeRobot (lerobot-calibrate ...), then fill assets/real_calibration.json
  (see REAL_CALIBRATION_TEMPLATE below) and set SO101_PORT / CAMERA_INDEX in .env.

Things you MUST verify on your arm before running autonomously (go slow, hand on the e-stop):
  * joint sign/offset between LeRobot degrees and the MJCF "new calib" zero (both put zero
    at the middle of the range, so offsets should be ~0, but signs may differ),
  * the gripper mapping (sim radians -> LeRobot 0..100),
  * that the camera homography puts the arm base at (0, 0) with x pointing forward.
"""
from __future__ import annotations

import json
import os
import time

import mujoco
import numpy as np

from .so101_mujoco import ARM_JOINTS, GRIP_CLOSED, GRIP_OPEN, OBJ_TYPES, ROOT, SO101Sim

CALIB = ROOT / "assets" / "real_calibration.json"
REAL_CALIBRATION_TEMPLATE = {
    "port": "COM5",
    "joint_sign": {j: 1.0 for j in ARM_JOINTS},
    "joint_offset_deg": {j: 0.0 for j in ARM_JOINTS},
    "gripper_range_0_100": [0.0, 100.0],
    "camera_index": 0,
    "aruco_dict": "DICT_4X4_50",
    "corner_markers": {"0": [0.10, -0.20], "1": [0.32, -0.20], "2": [0.32, 0.20], "3": [0.10, 0.20]},
    "object_markers": {"10": 0, "11": 1, "12": 2, "13": 3},
    "target": [0.22, 0.07],
}


def _load_lerobot():
    errs = []
    for mod in ("lerobot.robots.so101_follower", "lerobot.robots.so_follower"):
        try:
            m = __import__(mod, fromlist=["SO101Follower", "SO101FollowerConfig"])
            return m.SO101Follower, m.SO101FollowerConfig
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    raise ImportError("LeRobot SO101Follower not found. Install with  pip install \"lerobot[feetech]\"\n" + "\n".join(errs))


class SO101Real(SO101Sim):
    name = "so101-real"

    def __init__(self, calib_path=CALIB):
        if not calib_path.exists():
            calib_path.write_text(json.dumps(REAL_CALIBRATION_TEMPLATE, indent=2))
            raise RuntimeError(f"wrote a calibration template to {calib_path}: fill it in for your arm, then rerun")
        self.cal = json.loads(calib_path.read_text())
        super().__init__(gui=False, target=tuple(self.cal["target"]), realtime=1.0)   # the twin
        from ..perception.aruco import ArucoTable
        self.cam = ArucoTable(self.cal)
        Follower, Config = _load_lerobot()
        port = os.getenv("SO101_PORT", self.cal["port"])
        arm = Follower(Config(port=port, id="cortex_follower", use_degrees=True))
        arm.connect()
        self.arm = arm
        self._sync_twin()

    # ------------------------------------------------------------ joint mapping
    def _to_robot(self, q, grip) -> dict:
        act = {}
        for j, name in enumerate(ARM_JOINTS):
            deg = np.degrees(q[j]) * self.cal["joint_sign"][name] + self.cal["joint_offset_deg"][name]
            act[f"{name}.pos"] = float(deg)
        lo, hi = self.cal["gripper_range_0_100"]
        act["gripper.pos"] = float(lo + (grip - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED) * (hi - lo))
        return act

    def _sync_twin(self):
        obs = self.arm.get_observation()
        for j, name in enumerate(ARM_JOINTS):
            deg = (obs[f"{name}.pos"] - self.cal["joint_offset_deg"][name]) * self.cal["joint_sign"][name]
            self.d.qpos[self.jq[j]] = np.radians(deg)
        mujoco.mj_kinematics(self.m, self.d)

    # ------------------------------------------------------------ overrides that touch the world
    @property
    def _live(self):
        return self.__dict__.get("arm") is not None   # False while the twin is being built

    def _set_q(self, q, grip, hard=False):
        super()._set_q(q, grip, hard=hard and not self._live)
        if self._live:
            self.arm.send_action(self._to_robot(q, grip))

    def _grip(self, target, force=1.0, steps=250):
        # the real gripper has no force setpoint in this stub: force is approximated by how far we close
        lo, hi = GRIP_CLOSED, GRIP_OPEN
        tgt = target if target > lo + 1e-3 else lo + (1 - force) * 0.35 * (hi - lo)
        self.d.ctrl[self.act["gripper"]] = tgt
        self._set_q(self.d.qpos[self.jq], tgt)
        self._step(steps)

    def _step(self, n=1):
        if not self._live:
            return super()._step(n)
        time.sleep(n * self.m.opt.timestep)
        self._sync_twin()

    def _settle(self, max_steps=0):
        if not self._live:
            return super()._settle(max_steps)
        time.sleep(0.8)

    # ------------------------------------------------------------ perception
    def true_pose(self):
        if not self._live:
            return super().true_pose()
        x, y, th, up = self.cam.object_pose()
        return x, y, OBJ_TYPES[self.obj_type]["height"] / 2, th, up

    def get_state(self) -> dict:
        if self._live:
            self.obj_type = self.cam.object_type()
        return super().get_state()

    # ------------------------------------------------------------ resets need a human
    def spawn(self, obj_type=None, pose=None):
        if not self._live:
            return super().spawn(obj_type if obj_type is not None else 0, pose)
        input(f"Place the {'object' if obj_type is None else OBJ_TYPES[obj_type]['name']} in the workspace, "
              "then press Enter...")
        self._home()

    def spawn_near_target(self, obj_type=None):
        self.spawn(obj_type)

    def close(self):
        try:
            if self._live:
                self.arm.disconnect()
        finally:
            if "cam" in self.__dict__:
                self.cam.close()
