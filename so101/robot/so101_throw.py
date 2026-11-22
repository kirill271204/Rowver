"""SO-101 ball throwing in MuJoCo: pick up a ball, throw it into a conical cup on a table.

The scene is placed in polar coordinates around the base's rotation axis (shoulder_pan):
the ball at (ball_dist, ball_angle) and the cup at (cup_dist, cup_angle). Both can change
between throws. With vision=True the robot is not told either position: a fixed RGB-D
camera above the table looks after every scene change and an OpenCV detector finds the
ball and the cup (cortex/perception). The grasp and the aim use only what it sees; the
simulator's true positions only judge the outcome (in the cup or not, where it landed).

The throw is an overhand swing. The arm winds up behind itself, then shoulder_lift,
elbow_flex and wrist_flex rotate forward together at a capped joint speed. The jaws
open at a chosen point of the swing. The four actions TabPFN chooses:
  aim        heading of the throw (rad, + = to the robot's left)
  speed      0..1  peak joint speed, MIN_JOINT_SPEED..MAX_JOINT_SPEED rad/s
  release    0..1  where in the swing the jaws open (REL_LO..REL_HI: nearly all of it)
  snap       0..1  extra forward wrist flick at the end of the swing
The action space is deliberately raw: very early or late releases, slow drops and big
wrist snaps are all allowed. Which of them are chaotic is for TabPFN to find out.

TabPFN predicts where the ball crosses the cup's rim plane, in the throw frame:
r_par (distance from the pan axis along the aim) and r_perp (sideways). Physics is
the same at every aim angle, so every throw teaches every cup position.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .base import wrap
from .so101_mujoco import APPROACH_Z, GRIP_CLOSED, GRIP_OPEN, TRAVEL_Z, OBJ_TYPES, SO101Sim

MAX_JOINT_SPEED = 6.5      # rad/s, about the STS3215 no-load speed at 12 V
MIN_JOINT_SPEED = 1.0
BALL = dict(name="ball", shape=2, size=0.018, height=0.036, mass=0.02, mu=0.8, rgba=[0.95, 0.35, 0.1, 1])
CUP_R_IN, CUP_H, CUP_WALL = 0.045, 0.09, 0.003   # rim opening (inner radius), height, half wall thickness
CUP_R_BOT = 0.030             # conical cup: inner radius at the bottom (the rim opening stays CUP_R_IN)
TEE_H, TEE_R = 0.035, 0.006   # the ball sits on a tee so the jaws can close around its middle
TABLE = dict(x=(-0.30, 1.30), y=(-1.10, 1.10), legs=0.75, rgba=[0.80, 0.78, 0.74, 1])   # tabletop is z = 0
CAM = dict(name="table_cam", pos=(0.50, 0.0, 1.55), fovy=56.0, width=1280, height=960)  # fixed, looking down
GRASP_DEPTH = 0.012           # tool point this far below the ball's center
BALL_TYPE = 4
OBJ_TYPES.setdefault(BALL_TYPE, BALL)

# swing in joint space (shoulder_lift, elbow_flex, wrist_flex); positive = forward/down
WINDUP = np.array([-1.65, -0.55, -0.9])
FOLLOW = np.array([-0.35, 0.25, 0.0])
SNAP_EXTRA = 1.4           # extra wrist rotation at snap=1 (rad)
REL_LO, REL_HI = 0.05, 0.95  # release window: (almost) the whole swing, no hand-picked sweet spot


class SO101Thrower(SO101Sim):
    name = "so101-throw"
    skills = ("throw",)

    def __init__(self, gui=False, seed=0, obs_noise=0.002, realtime=1.0,
                 ball=(0.20, 0.0), cup=(0.45, 0.0), vision=False, frame_dir=None):
        self.ball_polar = tuple(ball)
        self.cup_polar = tuple(cup)
        self.vision, self.cam, self.seen, self.n_looks = bool(vision), None, None, 0
        self.frame_dir = frame_dir           # where look() saves camera_last.png and frames/ (None = don't save)
        super().__init__(gui=gui, seed=seed, obs_noise=obs_noise, realtime=realtime, target=(0.3, 0.0),
                         target_radius=CUP_R_IN - 0.6 * BALL["size"])
        if self.vision:
            from ..perception.camera import RGBDCamera
            self.cam = RGBDCamera(self.m, self.d, CAM["name"], CAM["width"], CAM["height"])
            self.view_q = np.array([0.0, *WINDUP, self.home_q[4]])    # arm cocked behind: clear of the camera's view
        self.origin = self.d.xanchor[self.m.joint("shoulder_pan").id][:2].copy()
        self.ball = self.m.body("ball").id
        bj = self.m.joint("ball_free")
        self.bq, self.bv = bj.qposadr[0], bj.dofadr[0]
        self.cup_mocap = self.m.body_mocapid[self.m.body("cup").id]
        self.tee_mocap = self.m.body_mocapid[self.m.body("tee").id]
        self.set_cup(*cup)
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [0.25, 0.0, 0.1]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.1, 220, -25
        self.spawn_near_target()

    # ================================================================ scene
    def _build(self, target):
        return self._build_throw_scene(target)

    @staticmethod
    def _mjcf_path():
        from .so101_mujoco import MJCF
        return MJCF

    def _build_throw_scene(self, target):
        # rebuild with the base scene + ball + cup (the parent builder returns a compiled model)
        spec = mujoco.MjSpec.from_file(str(self._mjcf_path()))
        spec.option.timestep = 0.002
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.density, spec.option.viscosity = 1.2, 1.8e-5      # air drag
        for g in spec.geoms:
            g.contype, g.conaffinity = 0, 0
        from .so101_mujoco import JAW_PAD, STATIC_PAD
        pad = dict(type=mujoco.mjtGeom.mjGEOM_BOX, friction=[1.6, 0.02, 0.002], condim=4,
                   rgba=[0.15, 0.15, 0.15, 1], contype=1, conaffinity=1, solref=[0.004, 1])
        spec.body("gripper").add_geom(name="static_pad", pos=STATIC_PAD["pos"], size=STATIC_PAD["size"], **pad)
        spec.body("moving_jaw_so101_v1").add_geom(name="jaw_pad", pos=JAW_PAD["pos"], size=JAW_PAD["size"], **pad)
        wb = spec.worldbody
        wb.add_light(pos=[0.3, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.8])
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, CAM["width"])
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, CAM["height"])
        # --- table: the collision plane is the tabletop (z = 0, same physics as before); a pale,
        #     low-saturation laminate so the camera can't mistake it for the orange ball or the red cup
        (x0, x1), (y0, y1) = TABLE["x"], TABLE["y"]
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        spec.add_material(name="tabletop", rgba=TABLE["rgba"], specular=0.1, shininess=0.2)
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[hx, hy, 0.05], pos=[cx, cy, 0],
                    material="tabletop", friction=[0.8, 0.01, 0.002], contype=1, conaffinity=1, condim=6,
                    solref=[0.01, 0.6])
        vis = dict(contype=0, conaffinity=0, group=1)
        wood = [0.42, 0.39, 0.35, 1]
        for i, (px, py, sx, sy) in enumerate([(cx, y0, hx, 0.012), (cx, y1, hx, 0.012),
                                              (x0, cy, 0.012, hy), (x1, cy, 0.012, hy)]):
            wb.add_geom(name=f"table_apron{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.03],
                        pos=[px, py, -0.03], rgba=wood, **vis)
        L = TABLE["legs"]
        for i, (px, py) in enumerate([(x0 + 0.05, y0 + 0.05), (x0 + 0.05, y1 - 0.05),
                                      (x1 - 0.05, y0 + 0.05), (x1 - 0.05, y1 - 0.05)]):
            wb.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, (L - 0.06) / 2],
                        pos=[px, py, -0.06 - (L - 0.06) / 2], rgba=wood, **vis)
        wb.add_geom(name="room_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], pos=[cx, cy, -L],
                    rgba=[0.24, 0.25, 0.27, 1], **vis)
        # --- fixed RGB-D camera above the table, on a stand behind the robot (image width along world y)
        px, py, pz = CAM["pos"]
        post_x = x0 - 0.12
        wb.add_geom(name="cam_post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (pz + 0.07 + L) / 2],
                    pos=[post_x, py, (pz + 0.07 - L) / 2], rgba=[0.2, 0.2, 0.22, 1], **vis)
        wb.add_geom(name="cam_boom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(px - post_x) / 2, 0.015, 0.015],
                    pos=[(px + post_x) / 2, py, pz + 0.07], rgba=[0.2, 0.2, 0.22, 1], **vis)
        wb.add_geom(name="cam_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.05, 0.025],
                    pos=[px, py, pz + 0.03], rgba=[0.08, 0.08, 0.09, 1], **vis)
        wb.add_camera(name=CAM["name"], pos=list(CAM["pos"]), quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
                      fovy=CAM["fovy"])
        wb.add_geom(name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.001, 0.0004, 0],
                    pos=[target[0], target[1], -0.01], rgba=[0, 0, 0, 0], contype=0, conaffinity=0)
        # the four push/pick objects are parked far away (keeps the parent class's indexing)
        ids = {}
        for k, o in {k: v for k, v in OBJ_TYPES.items() if k != BALL_TYPE}.items():
            b = wb.add_body(name=f"obj{k}", pos=[2.0 + 0.2 * k, 2.0, o["height"] / 2])
            b.add_freejoint(name=f"obj{k}_free")
            b.add_geom(name=f"obj{k}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[o["size"], o["size"], o["height"] / 2], mass=o["mass"], contype=0, conaffinity=0)
        ball = wb.add_body(name="ball", pos=[0.2, 0, BALL["size"]])
        ball.add_freejoint(name="ball_free")
        ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL["size"], 0, 0], mass=BALL["mass"],
                      rgba=BALL["rgba"], friction=[BALL["mu"], 0.005, 0.0005], condim=6, contype=1, conaffinity=1,
                      solref=[0.008, 0.5])
        tee = wb.add_body(name="tee", pos=[0.2, 0, 0], mocap=True)
        tee.add_geom(name="tee_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TEE_R, TEE_H / 2, 0],
                     pos=[0, 0, TEE_H / 2], rgba=[0.9, 0.9, 0.85, 1], contype=1, conaffinity=1)
        tee.add_geom(name="tee_cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TEE_R * 1.6, 0.002, 0],
                     pos=[0, 0, TEE_H - 0.002], rgba=[0.9, 0.9, 0.85, 1], contype=1, conaffinity=1)
        cup = wb.add_body(name="cup", pos=[0.45, 0, 0], mocap=True)
        # conical cup: 20 wall slabs tilted outwards, CUP_R_BOT at the bottom, CUP_R_IN at the rim
        n = 20
        tilt = float(np.arctan2(CUP_R_IN - CUP_R_BOT, CUP_H))
        r_mid = (CUP_R_IN + CUP_R_BOT) / 2 + CUP_WALL / np.cos(tilt)
        q_tilt = np.array([np.cos(tilt / 2), 0, np.sin(tilt / 2), 0])           # about the slab's tangent axis
        for i in range(n):
            a = 2 * np.pi * i / n
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), q_tilt)
            cup.add_geom(name=f"cup_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[CUP_WALL, (CUP_R_IN + 2 * CUP_WALL) * np.tan(np.pi / n) + 0.0008, CUP_H / (2 * np.cos(tilt))],
                         pos=[r_mid * np.cos(a), r_mid * np.sin(a), CUP_H / 2], quat=q,
                         rgba=[0.85, 0.1, 0.12, 1], friction=[0.6, 0.005, 0.0005], condim=4, solref=[0.01, 0.7])
        cup.add_geom(name="cup_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_R_BOT + CUP_WALL, 0.002, 0],
                     pos=[0, 0, 0.002], rgba=[0.95, 0.95, 0.95, 1], friction=[1.0, 0.02, 0.005], condim=6,
                     solref=[0.01, 1.2])
        m = spec.compile()
        for k in OBJ_TYPES:
            if k == BALL_TYPE:
                continue
            ids[k] = dict(body=m.body(f"obj{k}").id, qadr=m.joint(f"obj{k}_free").qposadr[0],
                          vadr=m.joint(f"obj{k}_free").dofadr[0])
        return m, ids

    # ================================================================ world (polar inputs)
    def polar(self, dist, angle_deg):
        a = np.radians(angle_deg)
        return self.origin + dist * np.array([np.cos(a), np.sin(a)])

    def set_cup(self, dist, angle_deg=0.0):
        self.cup_polar = (float(dist), float(angle_deg))
        xy = self.polar(dist, angle_deg)
        self.d.mocap_pos[self.cup_mocap] = [xy[0], xy[1], 0.0]
        self.set_target(xy)
        mujoco.mj_forward(self.m, self.d)

    def set_ball(self, dist, angle_deg=0.0):
        xy = self.polar(dist, angle_deg)
        if not self.reachable(*xy, margin=0.005):
            r = np.hypot(*xy)
            raise ValueError(f"ball at {100 * dist:.0f} cm / {angle_deg:.0f} deg is outside the grasp band "
                             f"({100 * (self._reach[0] - self.origin[0]):.0f}-{100 * (self._reach[1] - self.origin[0]):.0f} cm "
                             f"from the base axis, within about +/-90 deg); it is {100 * r:.1f} cm from the base origin")
        self.ball_polar = (float(dist), float(angle_deg))

    def set_target(self, xy):
        self.target = np.asarray(xy, float)

    def spawn(self, obj_type=None, pose=None):
        """Place the ball (pose = (x, y) or None for the configured position + ~5 mm jitter)."""
        if not hasattr(self, "bq"):          # parent constructor: nothing to place yet
            self.obj_type = BALL_TYPE
            return
        if pose is None:
            pose = self.polar(*self.ball_polar) + self.rng.normal(0, 0.005, 2)
        self.d.mocap_pos[self.tee_mocap] = [pose[0], pose[1], 0.0]
        self.d.qpos[self.bq:self.bq + 7] = [pose[0], pose[1], TEE_H + BALL["size"] + 5e-4, 1, 0, 0, 0]
        self.d.qvel[self.bv:self.bv + 6] = 0
        self.obj_type, self.obj = BALL_TYPE, self.ball
        self._set_q(self.home_q, GRIP_CLOSED, hard=True)
        mujoco.mj_forward(self.m, self.d)
        self._settle_ball(300)
        if self.cam is not None:            # the scene changed: look again
            self.look()

    def spawn_near_target(self, obj_type=None, **_):
        self.spawn()

    # ================================================================ perception
    def ball_pos(self):
        return self.d.qpos[self.bq:self.bq + 3].copy()

    def true_pose(self):
        p = self.ball_pos()
        return float(p[0]), float(p[1]), float(p[2]), 0.0, 1.0

    def look(self) -> dict:
        """Park the arm out of the camera's view, take one RGB-D frame, find the ball and the cup.

        Only the detections are used by the robot. The errors against the simulator's truth
        are kept for diagnostics (memory columns ball_err_mm / cup_err_mm), never for acting.
        """
        from ..perception.detect import detect
        self._joint_move(self.view_q, GRIP_CLOSED, 2.5)
        self._step(40)
        rgb, depth = self.cam.capture()
        det = detect(rgb, depth, self.cam, BALL["size"], CUP_H, CUP_R_IN + CUP_WALL)
        self._joint_move(self.home_q, GRIP_CLOSED, 2.5)
        self._settle_ball()
        b, c = det["ball"], det["cup"]
        tb, tc = self.ball_pos(), self.target
        self.seen = {"ok": b is not None and c is not None, "ball": b, "cup": c,
                     "ball_err_mm": round(1000 * float(np.hypot(b["x"] - tb[0], b["y"] - tb[1])), 2) if b else np.nan,
                     "cup_err_mm": round(1000 * float(np.hypot(c["x"] - tc[0], c["y"] - tc[1])), 2) if c else np.nan}
        self.n_looks += 1
        if self.frame_dir is not None:
            import cv2
            from pathlib import Path
            fd = Path(self.frame_dir)
            fd.mkdir(parents=True, exist_ok=True)
            small = cv2.resize(det["frame"], (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(fd / "camera_last.png"), small)
            if self.n_looks % 10 == 1:
                (fd / "frames").mkdir(exist_ok=True)
                cv2.imwrite(str(fd / "frames" / f"look_{self.n_looks:04d}.png"), small)
        def polar(d):
            dx, dy = d["x"] - self.origin[0], d["y"] - self.origin[1]
            return f"{100 * np.hypot(dx, dy):.1f} cm / {np.degrees(np.arctan2(dy, dx)):+.0f} deg"
        print("[camera] " + (f"ball at {polar(b)} (err {self.seen['ball_err_mm']:.1f} mm)" if b else "ball NOT FOUND")
              + " | " + (f"cup at {polar(c)} (err {self.seen['cup_err_mm']:.1f} mm)" if c else "cup NOT FOUND"),
              flush=True)
        return self.seen

    def get_state(self) -> dict:
        s = super().get_state()
        s["theta"] = 0.0
        cx, cy = self.target
        s |= {"cup_x": float(cx), "cup_y": float(cy), "cup_dist": self.cup_polar[0], "cup_angle": self.cup_polar[1],
              "origin_x": float(self.origin[0]), "origin_y": float(self.origin[1]),
              "on_table": True}
        if self.cam is not None:            # vision: the robot knows only what the camera saw
            sn = self.seen or {"ok": False, "ball": None, "cup": None}
            s["vision_ok"] = bool(sn["ok"])
            b, c = sn["ball"], sn["cup"]
            if b is not None:
                s["x"], s["y"] = b["x"], b["y"]
            if c is not None:
                ox, oy = self.origin
                s |= {"target_x": c["x"], "target_y": c["y"], "cup_x": c["x"], "cup_y": c["y"],
                      "cup_dist": float(np.hypot(c["x"] - ox, c["y"] - oy)),
                      "cup_angle": float(np.degrees(np.arctan2(c["y"] - oy, c["x"] - ox)))}
            s["dist_to_target"] = float(np.hypot(s["x"] - s["target_x"], s["y"] - s["target_y"]))
            s |= {"det_conf": min(b["conf"] if b else 0.0, c["conf"] if c else 0.0),
                  "ball_err_mm": sn.get("ball_err_mm", np.nan), "cup_err_mm": sn.get("cup_err_mm", np.nan)}
        return s

    def landing_ok(self, x, y):
        return np.hypot(x - self.origin[0], y - self.origin[1]) < 1.2

    def action_ok(self, skill, state, acts):
        n = len(np.atleast_1d(acts["aim"]))
        return np.ones(n, bool)

    def _settle_ball(self, max_steps=1500):
        for _ in range(max_steps // 20):
            self._step(20)
            v = self.d.qvel[self.bv:self.bv + 6]
            if np.linalg.norm(v[:3]) < 3e-3:
                break

    def _settle(self, max_steps=1000):
        self._settle_ball(max_steps)

    # ================================================================ skill
    def _joint_move(self, q_target, grip, speed=1.5):
        q0 = self.d.qpos[self.jq].copy()
        dq = q_target - q0
        T = max(np.abs(dq).max() / speed, 0.05)
        n = int(T / self.m.opt.timestep)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            self._set_q(q0 + dq * s, grip)
            self._step(1)
        self._step(80)

    def throw(self, aim, speed, release, snap=0.0, **_):
        if self.cam is not None:            # grasp where the camera saw the ball
            if not (self.seen and self.seen["ok"]):
                missing = " and ".join(k for k in ("ball", "cup") if not (self.seen or {}).get(k))
                return {"error": f"the camera did not find the {missing}"}
            b = self.seen["ball"]
            x, y, z0 = b["x"], b["y"], b["z"]
        else:
            x, y, z0, *_ = self.true_pose()
        out = {"grasped": False, "toppled": False, "out_of_bounds": False, "out_of_reach": False,
               "in_cup": False, "land_x": x, "land_y": y, "release_speed": 0.0}
        r_ball = BALL["size"]
        if not self.reachable(x, y, margin=-0.02):
            return out | {"out_of_reach": True}
        bearing = float(np.arctan2(y - self.origin[1], x - self.origin[0]))
        yaw = float(wrap(bearing + np.pi / 2))             # jaws close across the arm's plane
        # --- grasp (scripted: the interesting physics is the throw)
        zc = np.array([np.cos(yaw), np.sin(yaw)])
        tx, ty = np.array([x, y]) - zc * (r_ball + 0.004)  # the tool point is the static finger
        self._grip(GRIP_OPEN, 1.0, 150)
        self._move([tx, ty, TRAVEL_Z], yaw, 0.3)
        self._move([tx, ty, APPROACH_Z], yaw, 0.2)
        self._move([tx, ty, z0 - GRASP_DEPTH], yaw, 0.12)
        self._grip(GRIP_CLOSED, 0.8, 300)
        grip_hold = float(self.d.ctrl[self.act["gripper"]])
        self._move([tx, ty, TRAVEL_Z + 0.04], yaw, 0.25)
        if self.ball_pos()[2] < z0 + 0.02:
            self._home(); self._grip(GRIP_CLOSED, 1.0, 50); self._settle_ball()
            return out | {"land_x": self.ball_pos()[0], "land_y": self.ball_pos()[1]}
        # --- wind up: pan to the aim, arm cocked behind
        q = self.d.qpos[self.jq].copy()
        snap = float(np.clip(snap, 0, 1))
        qw = q.copy(); qw[1:4] = WINDUP
        qw[0] = float(np.clip(-aim, self.lo[0], self.hi[0]))   # the pan joint turns clockwise for +q
        qf = qw.copy(); qf[1:4] = FOLLOW; qf[3] = FOLLOW[2] + SNAP_EXTRA * snap
        qf[1:4] = np.clip(qf[1:4], self.lo[1:4], self.hi[1:4])
        self._joint_move(np.array([qw[0], *q[1:]]), grip_hold, 1.5)
        self._joint_move(qw, grip_hold, 1.5)
        self._step(150)
        if np.linalg.norm(self.ball_pos() - self._tool_pos()) > 0.06:
            self._home(); self._settle_ball()
            return out | {"land_x": self.ball_pos()[0], "land_y": self.ball_pos()[1]}
        out["grasped"] = True
        # --- swing: minimum-jerk joint profile (servos can track it without saturating and
        #     ringing), peak joint speed w; the jaws open at `release`
        # real servos are not perfectly repeatable: ~0.7% speed and ~4 ms release-timing jitter
        w = (MIN_JOINT_SPEED + (MAX_JOINT_SPEED - MIN_JOINT_SPEED) * float(np.clip(speed, 0, 1))) * (1 + self.rng.normal(0, 0.007))
        dq = qf - qw
        L = np.abs(dq).max()
        T = 1.875 * L / w                      # min-jerk peak speed = 1.875 * L / T
        dt = self.m.opt.timestep
        n = int(T / dt)
        rel_s = REL_LO + (REL_HI - REL_LO) * float(np.clip(release, 0, 1)) + self.rng.normal(0, 0.004 * w / max(L, 1e-6))
        released = False
        rim = CUP_H + 0.0
        land = None
        prev = self.ball_pos()
        for i in range(1, n + 1):
            if not released:
                prev = self.ball_pos()
            t = i * dt
            tau = min(t / T, 1.0)
            s = tau ** 3 * (10 - 15 * tau + 6 * tau * tau)
            if not released and s >= rel_s:
                released = True
                self.d.ctrl[self.act["gripper"]] = GRIP_OPEN
                self.m.actuator_forcerange[self.act["gripper"]] = [-self.grip_force_max, self.grip_force_max]
                out["release_speed"] = float(np.linalg.norm(self.d.qvel[self.bv:self.bv + 3]))
                self.last_release = {"t": t, "s_cmd": s, "q": self.d.qpos[self.jq][1:4].copy(),
                                     "q_cmd": (qw + dq * s)[1:4], "qdot": self.d.qvel[self.jdof][1:4].copy(),
                                     "ball_v": self.d.qvel[self.bv:self.bv + 3].copy()}
            self._set_q(qw + dq * s, GRIP_OPEN if released else grip_hold)
            self._step(1)
            if released:                       # the ball may come down while the arm still swings
                p = self.ball_pos()
                if land is None and prev[2] >= rim > p[2] and self.d.qvel[self.bv + 2] < 0:
                    f = (prev[2] - rim) / max(prev[2] - p[2], 1e-9)
                    land = prev[:2] + f * (p[:2] - prev[:2])
                prev = p
        if not released:
            self.d.ctrl[self.act["gripper"]] = GRIP_OPEN
        # --- flight: hold the follow-through, record the downward crossing of the rim plane
        for _ in range(int(2.5 / dt)):
            self._step(1)
            p = self.ball_pos()
            if land is None and prev[2] >= rim > p[2] and self.d.qvel[self.bv + 2] < 0:
                f = (prev[2] - rim) / max(prev[2] - p[2], 1e-9)
                land = prev[:2] + f * (p[:2] - prev[:2])
            prev = p
            if land is not None and np.linalg.norm(self.d.qvel[self.bv:self.bv + 3]) < 3e-3:
                break
        p = self.ball_pos()
        if land is None:                       # never rose above the rim plane: use where it stopped
            land = p[:2]
        cx, cy = self.target
        in_cup = bool(np.hypot(p[0] - cx, p[1] - cy) < CUP_R_IN and p[2] < CUP_H)
        self._joint_move(self.home_q, GRIP_CLOSED, 1.5)
        return out | {"in_cup": in_cup, "land_x": float(land[0]), "land_y": float(land[1]),
                      "final_x": float(p[0]), "final_y": float(p[1]),
                      "out_of_bounds": bool(not self.landing_ok(*land))}

    def push(self, *a, **k):
        raise NotImplementedError("the throwing scene has only the ball")

    def close(self):
        if self.cam is not None:            # free the camera's GL context before the viewer shuts GLFW down
            self.cam.close()
            self.cam = None
        super().close()
