"""Fast fallback robot: a floating fingertip pusher in PyBullet (push skill only).

The "robot" is a fingertip (a small cylinder dragged by a constraint). It pushes one
object at a time. Each object type has *hidden* physics (mass, friction) the brain has
to discover from experience; only its visible geometry (shape, size, height) is
reported by perception.
"""
from __future__ import annotations

import time

import numpy as np
import pybullet as p
import pybullet_data

from .base import Robot

DT = 1 / 240
WORKSPACE = 0.40          # the table is +-0.40 m; leaving it = "fell off the table"
PUSHER_R = 0.01
TRAVEL_Z = 0.20

# type_id -> visible geometry + hidden physics
#   shape: 0 = box, 1 = cylinder; size = half-width / radius; height = full height
OBJ_TYPES: dict[int, dict] = {
    0: dict(name="small box", shape=0, size=0.025, height=0.05, mass=0.20, mu=0.50, rgba=[0.9, 0.4, 0.1, 1]),
    1: dict(name="puck", shape=1, size=0.030, height=0.04, mass=0.15, mu=0.25, rgba=[0.95, 0.8, 0.1, 1]),
    2: dict(name="heavy box", shape=0, size=0.035, height=0.07, mass=0.60, mu=0.75, rgba=[0.6, 0.3, 0.8, 1]),
    3: dict(name="bottle", shape=1, size=0.022, height=0.13, mass=0.28, mu=0.65, rgba=[0.2, 0.7, 0.9, 1]),
}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class FingerSim(Robot):
    name = "finger-pybullet"
    skills = ("push",)
    push_dist_max = 0.12

    def __init__(self, gui: bool = False, obs_noise: float = 0.002, seed: int = 0,
                 realtime: float = 1.0, target=(0.15, 0.10), target_radius: float = 0.035):
        self.rng = np.random.default_rng(seed)
        self.gui, self.obs_noise, self.realtime = gui, obs_noise, realtime
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            p.resetDebugVisualizerCamera(0.75, 0, -60, [0.05, 0.0, 0.0], physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(DT, physicsClientId=self.cid)
        plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.changeDynamics(plane, -1, lateralFriction=0.5, physicsClientId=self.cid)
        self._draw_table_edge()

        self.target = np.array(target, float)
        self.target_radius = target_radius
        self._target_body = None
        self.set_target(self.target)

        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=PUSHER_R, height=0.05, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=PUSHER_R, length=0.05,
                                  rgbaColor=[0.2, 0.2, 0.9, 1], physicsClientId=self.cid)
        self.pivot = np.array([0.0, -0.35, TRAVEL_Z])
        self.pusher = p.createMultiBody(1.0, col, vis, self.pivot.tolist(), physicsClientId=self.cid)
        self.cst = p.createConstraint(self.pusher, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.pivot.tolist(), physicsClientId=self.cid)
        self.push_z = 0.03
        self.obj = None
        self.obj_type = 0
        self.spawn(0)

    # ------------------------------------------------------------------ world
    def _draw_table_edge(self):
        if not self.gui:
            return
        w = WORKSPACE
        pts = [(-w, -w), (w, -w), (w, w), (-w, w), (-w, -w)]
        for a, b in zip(pts, pts[1:]):
            p.addUserDebugLine([*a, 0.002], [*b, 0.002], [0.8, 0.1, 0.1], 2, physicsClientId=self.cid)

    def set_target(self, xy):
        self.target = np.asarray(xy, float)
        if self._target_body is not None:
            p.removeBody(self._target_body, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.target_radius, length=0.001,
                                  rgbaColor=[0, 1, 0, 0.45], physicsClientId=self.cid)
        self._target_body = p.createMultiBody(0, -1, vis, [*self.target, 0.0005], physicsClientId=self.cid)

    def _step(self):
        p.stepSimulation(physicsClientId=self.cid)
        if self.gui and self.realtime > 0:
            time.sleep(DT / self.realtime)

    def _settle(self, max_steps=720):
        for _ in range(max_steps):
            self._step()
            v, w = p.getBaseVelocity(self.obj, physicsClientId=self.cid)
            if np.linalg.norm(v) < 1e-3 and np.linalg.norm(w) < 1e-2:
                break

    def spawn(self, obj_type: int | None = None, pose=None):
        """Place an object. pose = (x, y, theta); random if None."""
        if self.obj is not None:
            p.removeBody(self.obj, physicsClientId=self.cid)
        if obj_type is None:
            obj_type = int(self.rng.integers(len(OBJ_TYPES)))
        o = OBJ_TYPES[obj_type]
        s, h = o["size"], o["height"]
        if o["shape"] == 0:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s, s, h / 2], physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s, s, h / 2], rgbaColor=o["rgba"], physicsClientId=self.cid)
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=s, height=h, physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=s, length=h, rgbaColor=o["rgba"], physicsClientId=self.cid)
        if pose is None:
            pose = (*self.rng.uniform(-0.25, 0.25, 2), self.rng.uniform(-np.pi, np.pi))
        x, y, th = pose
        self.obj = p.createMultiBody(o["mass"], col, vis, [x, y, h / 2 + 1e-4],
                                     p.getQuaternionFromEuler([0, 0, th]), physicsClientId=self.cid)
        p.changeDynamics(self.obj, -1, lateralFriction=o["mu"], spinningFriction=0.001,
                         rollingFriction=0.0005, physicsClientId=self.cid)
        self.obj_type = obj_type
        self._settle()

    def spawn_near_target(self, obj_type=None, r_min=0.06, r_max=0.16):
        """Spawn an object at a random distance/bearing from the target (one-push reach)."""
        for _ in range(100):
            r, b = self.rng.uniform(r_min, r_max), self.rng.uniform(-np.pi, np.pi)
            x, y = self.target + r * np.array([np.cos(b), np.sin(b)])
            if abs(x) < WORKSPACE - 0.08 and abs(y) < WORKSPACE - 0.08:
                break
        self.spawn(obj_type, (x, y, self.rng.uniform(-np.pi, np.pi)))

    # ------------------------------------------------------------------ perception
    def true_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.obj, physicsClientId=self.cid)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        return pos[0], pos[1], yaw, roll, pitch

    def get_state(self) -> dict:
        """What the camera sees: noisy pose + visible geometry (not mass/friction)."""
        x, y, th, roll, pitch = self.true_pose()
        n = self.rng.normal(0, self.obs_noise, 3)
        o = OBJ_TYPES[self.obj_type]
        x, y, th = x + n[0], y + n[1], float(wrap(th + 5 * n[2]))
        return {
            "x": float(x), "y": float(y), "theta": th,
            "type": self.obj_type, "type_name": o["name"],
            "shape": o["shape"], "size": o["size"], "height": o["height"],
            "upright": bool(abs(roll) < 0.5 and abs(pitch) < 0.5),
            "target_x": float(self.target[0]), "target_y": float(self.target[1]),
            "target_radius": self.target_radius,
            "dist_to_target": float(np.hypot(x - self.target[0], y - self.target[1])),
            "on_table": bool(abs(x) < WORKSPACE and abs(y) < WORKSPACE),
        }

    # ------------------------------------------------------------------ motor
    def _move_to(self, goal, speed):
        start, goal = self.pivot.copy(), np.asarray(goal, float)
        n = max(1, int(np.linalg.norm(goal - start) / (speed * DT)))
        for i in range(1, n + 1):
            pt = start + (goal - start) * i / n
            p.changeConstraint(self.cst, jointChildPivot=pt.tolist(), maxForce=300, physicsClientId=self.cid)
            self._step()
        self.pivot = goal

    def push(self, angle: float, speed: float, offset: float, push_dist: float = 0.08) -> dict:
        """Execute one push (runs locally; the controller never sits in this loop).

        angle: push direction in world frame (rad); speed: 0..1 (-> 0.05..0.60 m/s);
        offset: -1..1, lateral contact offset in units of object size (causes spin);
        push_dist: how far past the object's center the fingertip travels (m).
        """
        x, y, *_ = self.true_pose()
        o = OBJ_TYPES[self.obj_type]
        d = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-d[1], d[0]])
        c = np.array([x, y])
        side = perp * float(np.clip(offset, -1, 1)) * o["size"]
        start = c - d * (o["size"] * 1.5 + 0.04) + side
        end = c + d * float(push_dist) + side

        self._move_to([*start, TRAVEL_Z], 1.0)
        self._move_to([*start, self.push_z], 0.6)
        self._move_to([*end, self.push_z], 0.05 + 0.55 * float(np.clip(speed, 0, 1)))
        self._move_to([*end, TRAVEL_Z], 0.6)
        self._settle()
        x, y, th, roll, pitch = self.true_pose()
        return {
            "toppled": bool(abs(roll) > 0.5 or abs(pitch) > 0.5),
            "out_of_bounds": bool(abs(x) > WORKSPACE or abs(y) > WORKSPACE),
            "out_of_reach": False,
        }

    def reachable(self, x, y):
        return (np.abs(x) < WORKSPACE) & (np.abs(y) < WORKSPACE)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
