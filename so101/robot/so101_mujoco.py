"""SO-101 arm (LeRobot / TheRobotStudio) with its gripper, simulated in MuJoCo.

The arm is the official MJCF (assets/so101, shipped with the project).
Changes made at load time, all in code:
  * the gripper's convex-hull mesh collisions are replaced by box finger pads
    (convex hulls would fill the gap between the jaws and make grasping impossible),
  * other arm meshes are visual-only (the tool never goes below the table),
  * a floor, a target zone (visual only) and four objects with hidden mass/friction.

Motion: Cartesian waypoints -> damped-least-squares IK (tool pointing down, jaw yaw
chosen per skill) -> position actuators. Only the joint targets would differ on the
real arm, and those come from the same IK (see so101_real.py).
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from .base import Robot, wrap

ROOT = Path(__file__).resolve().parents[1]   # so101/
MJCF = ROOT / "assets" / "so101" / "so101_new_calib.xml"

# visible geometry (shape 0 box / 1 cylinder, size = half-width or radius, height) + hidden physics
OBJ_TYPES: dict[int, dict] = {
    0: dict(name="small cube", shape=0, size=0.015, height=0.030, mass=0.05, mu=0.55, rgba=[0.95, 0.45, 0.1, 1]),
    1: dict(name="puck", shape=1, size=0.017, height=0.020, mass=0.04, mu=0.25, rgba=[0.95, 0.8, 0.1, 1]),
    2: dict(name="heavy cube", shape=0, size=0.0175, height=0.035, mass=0.25, mu=0.85, rgba=[0.6, 0.3, 0.85, 1]),
    3: dict(name="bottle", shape=1, size=0.015, height=0.075, mass=0.08, mu=0.7, rgba=[0.2, 0.7, 0.95, 1]),
}

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.17           # jaw angle (rad): ~7.5 cm gap vs closed
STATIC_PAD = dict(pos=[-0.0174, 0.0, -0.0842], size=[0.0095, 0.0087, 0.0203])      # gripper frame
JAW_PAD = dict(pos=[-0.00335, -0.06085, 0.01895], size=[0.00895, 0.02115, 0.00865])  # jaw frame
TRAVEL_Z = 0.11      # above the tallest object; the tool may tilt a little up here (wrist-flex limit)
APPROACH_Z = 0.06    # tool exactly vertical from here down
DT_VIEW = 1 / 60


class SO101Sim(Robot):
    name = "so101-mujoco"
    skills = ("push", "pick")
    push_dist_max = 0.06

    def __init__(self, gui=False, seed=0, obs_noise=0.002, realtime=1.0,
                 target=(0.22, 0.07), target_radius=0.03):
        if not MJCF.exists():
            raise FileNotFoundError(f"{MJCF} missing: the SO-101 model ships in so101/assets/so101")
        self.rng = np.random.default_rng(seed)
        self.obs_noise, self.realtime = obs_noise, realtime
        self.target_radius = target_radius
        self.m, self.obj_ids = self._build(target)
        self.d = mujoco.MjData(self.m)
        self.dk = mujoco.MjData(self.m)          # scratch data for IK
        self.site = self.m.site("gripperframe").id
        self.grip_body = self.m.body("gripper").id
        self.jq = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.jdof = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.lo = self.m.jnt_range[[self.m.joint(j).id for j in ARM_JOINTS], 0]
        self.hi = self.m.jnt_range[[self.m.joint(j).id for j in ARM_JOINTS], 1]
        self.grip_force_max = float(self.m.actuator_forcerange[self.act["gripper"], 1])
        self.target_geom = self.m.geom("target_zone").id
        self.viewer = None
        if gui:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.lookat[:] = [0.2, 0.0, 0.02]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 0.75, 200, -35
        self._last_sync = time.time()
        self.home_q = self.ik([0.15, 0.0, 0.14], yaw=np.pi / 2)[0]
        self._set_q(self.home_q, GRIP_CLOSED, hard=True)
        self._reach = self._measure_reach()
        self.set_target(target)
        self.obj = None
        self.obj_type = 0
        self.spawn(0)

    # ================================================================ model
    def _build(self, target):
        spec = mujoco.MjSpec.from_file(str(MJCF))
        spec.option.timestep = 0.002
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        for g in spec.geoms:                       # arm meshes: visual only
            g.contype, g.conaffinity = 0, 0
        pad = dict(type=mujoco.mjtGeom.mjGEOM_BOX, friction=[1.6, 0.02, 0.002], condim=4,
                   rgba=[0.15, 0.15, 0.15, 1], contype=1, conaffinity=1, solref=[0.004, 1])
        spec.body("gripper").add_geom(name="static_pad", pos=STATIC_PAD["pos"], size=STATIC_PAD["size"], **pad)
        spec.body("moving_jaw_so101_v1").add_geom(name="jaw_pad", pos=JAW_PAD["pos"], size=JAW_PAD["size"], **pad)

        wb = spec.worldbody
        wb.add_light(pos=[0.2, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.8])
        tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=[0.22, 0.24, 0.27], rgb2=[0.17, 0.19, 0.21], width=300, height=300)
        spec.add_material(name="grid", textures=["", "grid"], texrepeat=[8, 8], texuniform=True)
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], material="grid",
                    friction=[0.6, 0.005, 0.0001], contype=1, conaffinity=1, condim=4)
        wb.add_geom(name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[self.target_radius, 0.0004, 0],
                    pos=[target[0], target[1], 0.0005], rgba=[0.05, 0.75, 0.2, 0.55], contype=0, conaffinity=0)
        ids = {}
        for k, o in OBJ_TYPES.items():
            b = wb.add_body(name=f"obj{k}", pos=[1.0 + 0.2 * k, 1.0, o["height"] / 2])
            b.add_freejoint(name=f"obj{k}_free")
            gtype = mujoco.mjtGeom.mjGEOM_BOX if o["shape"] == 0 else mujoco.mjtGeom.mjGEOM_CYLINDER
            size = [o["size"], o["size"], o["height"] / 2] if o["shape"] == 0 else [o["size"], o["height"] / 2, 0]
            b.add_geom(name=f"obj{k}_geom", type=gtype, size=size, mass=o["mass"], rgba=o["rgba"],
                       friction=[o["mu"], 0.01, 0.001], condim=4, contype=1, conaffinity=1)
        m = spec.compile()
        for k in OBJ_TYPES:
            ids[k] = dict(body=m.body(f"obj{k}").id, qadr=m.joint(f"obj{k}_free").qposadr[0],
                          vadr=m.joint(f"obj{k}_free").dofadr[0])
        return m, ids

    # ================================================================ kinematics
    def _tool_R(self, yaw):
        """Desired site frame: x (approach) straight down, z (jaw closing axis) at `yaw`."""
        x = np.array([0.0, 0.0, -1.0])
        z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return np.column_stack([x, np.cross(z, x), z])

    SEEDS = [np.array(v, float) for v in (
        [0, 0, 0, 0, 0], [0, -0.8, 0.8, 0.8, 0], [0, 0.4, 0.4, 1.0, 0], [0, -0.4, 1.2, 0.4, 0],
        [0, 0.8, -0.4, 1.2, 0], [0, -1.2, 1.4, 0.6, 0], [0, 0.2, 1.0, 1.4, 0])]

    def ik(self, pos, yaw, q0=None, iters=120, tol=5e-4):
        """Damped least squares with multi-start. Returns (arm q[5], pos err m, ori err rad).
        With q0 given (motion tracking) it only refines from q0."""
        if q0 is not None:
            return self._ik_once(pos, yaw, q0, iters, tol)
        best = None
        pan = np.arctan2(pos[1], pos[0])
        for sd in self.SEEDS:
            sd = sd.copy(); sd[0] = pan
            r = self._ik_once(pos, yaw, sd, iters, tol)
            if best is None or r[1] + 0.02 * r[2] < best[1] + 0.02 * best[2]:
                best = r
            if r[1] < 1e-3 and r[2] < 0.03:
                break
        return best

    def _ik_once(self, pos, yaw, q0, iters, tol):
        m, d = self.m, self.dk
        d.qpos[:] = self.d.qpos
        d.qpos[self.jq] = q0
        Rd = self._tool_R(yaw)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        w_ori = 0.25
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            ep = np.asarray(pos) - p
            eo = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
            if np.linalg.norm(ep) < tol and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.jdof], w_ori * jacr[:, self.jdof]])
            e = np.concatenate([ep, w_ori * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[self.jq] = np.clip(d.qpos[self.jq] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[self.site]
        R = d.site_xmat[self.site].reshape(3, 3)
        eo = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
        return d.qpos[self.jq].copy(), float(np.linalg.norm(np.asarray(pos) - p)), float(np.linalg.norm(eo))

    def _measure_reach(self):
        """Radial band where the tool can point straight down at table height."""
        ok = []
        for r in np.arange(0.06, 0.42, 0.01):
            _, ep, eo = self.ik([r, 0.0, 0.015], yaw=np.pi / 2)
            _, ep2, eo2 = self.ik([r, 0.0, APPROACH_Z], yaw=np.pi / 2)
            ok.append((r, ep < 0.003 and eo < 0.15 and ep2 < 0.003 and eo2 < 0.15))
        good = [r for r, g in ok if g]
        return (min(good) + 0.015, max(good) - 0.015) if good else (0.12, 0.3)

    def reachable(self, x, y, margin=0.0):
        r = np.hypot(x, y)
        ang = np.abs(np.arctan2(y, x))
        return (r > self._reach[0] + margin) & (r < self._reach[1] - margin) & (ang < np.radians(95))

    def action_ok(self, skill, state, acts) -> np.ndarray:
        """Vectorised feasibility: would every tool waypoint of this action be reachable?"""
        o = OBJ_TYPES[state["type"]]
        x, y, th = state["x"], state["y"], state["theta"]
        if skill == "push":
            a = np.asarray(acts["angle"])
            dv = np.stack([np.cos(a), np.sin(a)], -1)
            perp = np.stack([-dv[:, 1], dv[:, 0]], -1)
            side = perp * (np.asarray(acts["offset"]) * o["size"])[:, None]
            c = np.array([x, y])
            start = c - dv * (o["size"] * 1.5 + 0.03) + side
            end = c + dv * np.asarray(acts["push_dist"])[:, None] + side
            return self.reachable(*start.T, margin=0.008) & self.reachable(*end.T, margin=0.008)
        g = np.asarray(acts["grasp_yaw"])
        yaw = th + g
        zc = np.stack([np.cos(yaw), np.sin(yaw)], -1)
        hw = o["size"] * (np.abs(np.cos(g)) + np.abs(np.sin(g))) if o["shape"] == 0 else np.full(len(g), o["size"])
        Ro = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        gp = np.array([x, y]) + (np.stack([acts["grasp_along"], acts["grasp_across"]], -1) * o["size"]) @ Ro.T
        tcp = gp - zc * (hw + 0.004)[:, None]
        place = np.stack([state["target_x"] + np.asarray(acts["aim_dx"]),
                          state["target_y"] + np.asarray(acts["aim_dy"])], -1)
        tcp_place = place + (tcp - np.array([x, y]))
        m = 0.008   # perception noise margin
        return self.reachable(*tcp.T, margin=m) & self.reachable(*tcp_place.T, margin=m) & self.reachable(*place.T, margin=m)

    # ================================================================ low-level control
    def _set_q(self, q, grip, hard=False):
        for j, name in enumerate(ARM_JOINTS):
            self.d.ctrl[self.act[name]] = q[j]
        self.d.ctrl[self.act["gripper"]] = grip
        if hard:
            self.d.qpos[self.jq] = q
            self.d.qpos[self._grip_qadr()] = grip
            mujoco.mj_forward(self.m, self.d)

    # hooks for a moving base (rowver/arm.py overrides them: positions in the arm's own frame)
    def _grip_qadr(self) -> int:
        return int(self.m.joint("gripper").qposadr[0])

    def _tool_pos(self) -> np.ndarray:
        """Tool point position (world frame here: the base is fixed at the origin)."""
        return self.d.site_xpos[self.site].copy()

    def _carry(self):
        """Between lift and place, with the object held at TRAVEL_Z. Nothing here (the base is fixed);
        a mobile base drives to the place station in this hook."""

    def _step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
        if self.viewer is not None:
            now = time.time()
            if now - self._last_sync > DT_VIEW:
                self.viewer.sync()
                self._last_sync = now
            if self.realtime > 0:
                lag = n * self.m.opt.timestep / self.realtime - (time.time() - now)
                if lag > 0:
                    time.sleep(lag)

    def _grip(self, target, force=1.0, steps=250):
        self.m.actuator_forcerange[self.act["gripper"]] = [-force * self.grip_force_max, force * self.grip_force_max]
        self.d.ctrl[self.act["gripper"]] = target
        self._step(steps)

    def _move(self, pos, yaw, speed=0.25, grip=None):
        """Straight-line tool motion at `speed` m/s (IK every 4 mm)."""
        mujoco.mj_kinematics(self.m, self.d)
        start = self._tool_pos()
        pos = np.asarray(pos, float)
        dist = np.linalg.norm(pos - start)
        n = max(1, int(dist / 0.004))
        steps = max(1, int(dist / max(speed, 1e-3) / n / self.m.opt.timestep))
        q = self.d.qpos[self.jq].copy()
        grip = self.d.ctrl[self.act["gripper"]] if grip is None else grip
        worst = 0.0
        for i in range(1, n + 1):
            q, ep, _ = self.ik(start + (pos - start) * i / n, yaw, q0=q, iters=40)
            worst = max(worst, ep)
            self._set_q(q, grip)
            self._step(steps)
        self._step(60)  # let the servos catch up
        return worst

    def _home(self):
        mujoco.mj_kinematics(self.m, self.d)
        p = self._tool_pos()
        yaw = self._current_yaw()
        self._move([p[0], p[1], TRAVEL_Z], yaw, 0.3)
        self._move([0.15, 0.0, 0.14], np.pi / 2, 0.3)

    def _current_yaw(self):
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(R[1, 2], R[0, 2]))

    def _settle(self, max_steps=1000):
        o = self.obj_ids[self.obj_type]
        for _ in range(max_steps // 20):
            self._step(20)
            v = self.d.qvel[o["vadr"]:o["vadr"] + 6]
            if np.linalg.norm(v[:3]) < 2e-3 and np.linalg.norm(v[3:]) < 2e-2:
                break

    # ================================================================ world
    def set_target(self, xy):
        self.target = np.asarray(xy, float)
        self.m.geom_pos[self.target_geom][:2] = self.target

    def spawn(self, obj_type=None, pose=None):
        if obj_type is None:
            obj_type = int(self.rng.integers(len(OBJ_TYPES)))
        for k, o in self.obj_ids.items():   # park everything, then place the chosen one
            self.d.qpos[o["qadr"]:o["qadr"] + 7] = [1.0 + 0.2 * k, 1.0, OBJ_TYPES[k]["height"] / 2, 1, 0, 0, 0]
            self.d.qvel[o["vadr"]:o["vadr"] + 6] = 0
        if pose is None:
            while True:
                x, y = self.rng.uniform(0.1, 0.35), self.rng.uniform(-0.2, 0.2)
                if self.reachable(x, y) and self.reachable(x * 1.12, y * 1.12):
                    break
            pose = (x, y, self.rng.uniform(-np.pi, np.pi))
        x, y, th = pose
        o = self.obj_ids[obj_type]
        self.d.qpos[o["qadr"]:o["qadr"] + 7] = [x, y, OBJ_TYPES[obj_type]["height"] / 2 + 1e-3,
                                                np.cos(th / 2), 0, 0, np.sin(th / 2)]
        self.obj_type = obj_type
        self.obj = o["body"]
        self._set_q(self.home_q, GRIP_CLOSED, hard=True)
        mujoco.mj_forward(self.m, self.d)
        self._settle()

    def spawn_near_target(self, obj_type=None, r_min=0.04, r_max=0.17):
        for _ in range(500):
            r, b = self.rng.uniform(r_min, r_max), self.rng.uniform(-np.pi, np.pi)
            x, y = self.target + r * np.array([np.cos(b), np.sin(b)])
            if self.reachable(x, y) and self.reachable(*(np.array([x, y]) * 1.1)) and self.reachable(*(np.array([x, y]) * 0.9)):
                break
        self.spawn(obj_type, (x, y, self.rng.uniform(-np.pi, np.pi)))

    # ================================================================ perception
    def true_pose(self):
        o = self.obj_ids[self.obj_type]
        q = self.d.qpos[o["qadr"]:o["qadr"] + 7]
        w, x, y, z = q[3:]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        up = 1 - 2 * (x * x + y * y)             # z-axis of the object, world z component
        return float(q[0]), float(q[1]), float(q[2]), float(yaw), float(up)

    def get_state(self) -> dict:
        x, y, z, th, up = self.true_pose()
        n = self.rng.normal(0, self.obs_noise, 3)
        o = OBJ_TYPES[self.obj_type]
        x, y, th = x + n[0], y + n[1], float(wrap(th + 5 * n[2]))
        return {
            "x": float(x), "y": float(y), "theta": th,
            "type": self.obj_type, "type_name": o["name"],
            "shape": o["shape"], "size": o["size"], "height": o["height"],
            "upright": bool(up > 0.85),
            "target_x": float(self.target[0]), "target_y": float(self.target[1]),
            "target_radius": self.target_radius,
            "dist_to_target": float(np.hypot(x - self.target[0], y - self.target[1])),
            "on_table": bool(self.reachable(x, y)),
        }

    def _outcome(self):
        x, y, z, th, up = self.true_pose()
        return {"toppled": bool(up < 0.85), "out_of_bounds": bool(not self.reachable(x, y))}

    # ================================================================ skills
    def push(self, angle, speed, offset, push_dist):
        """Push with closed jaws. Same parameters as the fingertip robot."""
        x, y, *_ = self.true_pose()
        o = OBJ_TYPES[self.obj_type]
        dvec = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-dvec[1], dvec[0]])
        c = np.array([x, y])
        side = perp * float(np.clip(offset, -1, 1)) * o["size"]
        start = c - dvec * (o["size"] * 1.5 + 0.03) + side
        end = c + dvec * float(push_dist) + side
        if not (self.reachable(*start) and self.reachable(*end)):
            return {"toppled": False, "out_of_bounds": False, "out_of_reach": True}
        yaw = float(wrap(angle + np.pi / 2))      # jaws side by side across the push direction
        z = 0.016
        self._grip(GRIP_CLOSED, 1.0, 100)
        self._move([*start, TRAVEL_Z], yaw, 0.3)
        self._move([*start, APPROACH_Z], yaw, 0.2)
        self._move([*start, z], yaw, 0.15)
        self._move([*end, z], yaw, 0.03 + 0.27 * float(np.clip(speed, 0, 1)))
        self._move([*end, APPROACH_Z], yaw, 0.15)
        self._home()
        self._settle()
        return {**self._outcome(), "out_of_reach": False}

    def pick_place(self, grasp_along=0.0, grasp_across=0.0, grasp_yaw=0.0, grip_force=0.5,
                   lift_speed=0.5, place_x=None, place_y=None, release_height=0.01, grasp_height=0.35, **_):
        """Top-down grasp, carry, release.

        grasp_along/across: grasp point offset in object frame (units of object size, -1..1)
        grasp_yaw: jaw closing axis relative to object yaw (rad)
        grip_force: 0..1 of servo torque; lift_speed: 0..1; release_height: m above the table contact
        grasp_height: where the fingertips close, as a fraction of the object's height
        """
        x, y, z0, th, up = self.true_pose()
        o = OBJ_TYPES[self.obj_type]
        s, h = o["size"], o["height"]
        place = np.array([place_x, place_y], float)
        c = np.array([x, y])
        Ro = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        gp = c + Ro @ (np.array([grasp_along, grasp_across]) * s)
        yaw = float(wrap(th + grasp_yaw))
        zc = np.array([np.cos(yaw), np.sin(yaw)])
        phi = grasp_yaw
        hw = s * (abs(np.cos(phi)) + abs(np.sin(phi))) if o["shape"] == 0 else s
        tcp_xy = gp - zc * (hw + 0.004)             # static finger just outside the object
        offset = tcp_xy - c                          # planned object->tool offset
        tcp_place = place + offset
        if not all(self.reachable(*p) for p in (tcp_xy, tcp_place, c, place)):
            return {"grasped": False, "dropped": False, "toppled": False, "out_of_bounds": False, "out_of_reach": True}
        z_grasp = float(np.clip(grasp_height * h + 0.004, 0.009, APPROACH_Z - 0.005))
        lift_v = 0.05 + 0.35 * float(np.clip(lift_speed, 0, 1))

        self._grip(GRIP_OPEN, 1.0, 150)
        self._move([*tcp_xy, TRAVEL_Z], yaw, 0.3)
        self._move([*tcp_xy, APPROACH_Z], yaw, 0.2)
        self._move([*tcp_xy, z_grasp], yaw, 0.12)
        self._grip(GRIP_CLOSED, float(np.clip(grip_force, 0.02, 1.0)), 300)
        self._move([*tcp_xy, TRAVEL_Z], yaw, lift_v)
        grasped = self.true_pose()[2] > z0 + 0.02
        if grasped:
            shift = self._carry()                    # a mobile base may drive somewhere else first ...
            if shift is not None:                    # ... and then the place point moves in the arm frame
                place = place + np.asarray(shift, float)
                tcp_place = place + offset
        self._move([*tcp_place, TRAVEL_Z], yaw, lift_v)
        dropped = grasped and self.true_pose()[2] < z0 + 0.02
        self._move([*tcp_place, APPROACH_Z], yaw, 0.15)
        self._move([*tcp_place, z_grasp + float(np.clip(release_height, 0, 0.04))], yaw, 0.12)
        self._grip(GRIP_OPEN, 1.0, 150)
        self._move([*tcp_place, APPROACH_Z], yaw, 0.2)
        self._home()
        self._grip(GRIP_CLOSED, 1.0, 100)
        self._settle()
        return {**self._outcome(), "grasped": bool(grasped), "dropped": bool(dropped), "out_of_reach": False}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
