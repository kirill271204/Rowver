"""The Jezero work zone with the fleet's garage and N rovers in ONE MuJoCo model.

The ground is the user's HiRISE DTM of Jezero Crater (jezero.py), scaled 1:5 like the rovers: a
40 m x 40 m heightfield with the orthoimage on it and the dataset's rocks as meshes. Impact craters
(craters.py) are carved into it. The only structure is the fleet's 3-bay garage (garage.py) with a
solar roof, on the work zone's levelest edge; the rovers start parked in it, nose in.

Loose sample rocks (sample_rocks.py: irregular, faceted, free bodies) lie around the zone's candidate
drill sites. The sampler's arm picks them off the ground and brings them home to the tray in its bay.

The world does not tell the robots anything. HIDDEN: gravity and air, and the ground's physics:
every physics chunk each wheel gets the contact of the ground under it (class, slip patch,
sinkage from the dataset, calibrated offline). terrain_at() is the
ground-truth map: used ONLY by check scripts and to label summaries for people, never by a rover.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np


from . import craters as CR
from . import jezero as J
from . import mars_rover as ROVER
from .terrain import TERRAINS, geom_physics

CHASSIS = ROVER.CHASSIS
GRAVITY = {"earth": 9.81, "mars": 3.71, "moon": 1.62}          # m/s^2 (hidden from the robots)
# air density, viscosity. Earth has none: the lab sim (so101/) had no air, and Earth here must match it
AIR = {"earth": (0.0, 0.0), "mars": (0.020, 1.1e-5), "moon": (0.0, 0.0)}
SITE_CODE = {"earth": 0, "mars": 1, "moon": 2}       # what a robot DOES know: where it is (the lab = earth)

PICK_LOCAL = (0.17, 0.0)        # arm-local xy (arm base frame, m) where a rock is picked: ahead of the arm
WHEELS = ROVER.WHEELS
ROLL_CONTACT = 0.0003     # m: the wheel contact's own rolling friction (a hard wheel on ground)
CAM_ABOVE = 5.2                                  # m above a tile's highest ground: one camera sees 7.2 x 5.4 m
SITE = dict(x=(0.0, 1.0), y=(0.0, 1.0))          # the current work zone (set by build); for older callers


def rover_ids(n: int) -> list:
    return [f"r{k + 1}" for k in range(n)]


@dataclass
class Probe:
    """A design's geometry measured once on hard flat ground under Mars gravity (its springs sag as
    they will on the site). Designs without an arm have no arm fields (None)."""
    chassis_z: float               # settled chassis origin height
    arm_base_z: float | None       # arm base height above the ground = the work-plane height
    arm_base_xy: tuple | None      # arm base in the chassis frame
    pan_axis_local: tuple | None   # shoulder_pan axis in the arm base frame


_PROBES: dict = {}


def _base_options(spec: mujoco.MjSpec, gravity="mars"):
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.gravity = [0, 0, -GRAVITY[gravity]]
    spec.option.density, spec.option.viscosity = AIR[gravity]


HOLE_MARKERS = 6                 # per drilling rover: the holes shown on the ground


def _attach_rover(spec, rid, pose, z, design):
    x, y, yaw = pose
    fr = spec.worldbody.add_frame(pos=[x, y, z], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    spec.attach(ROVER.rover_spec(design), frame=fr, prefix=f"{rid}/")


def probe(design=None) -> Probe:
    """The geometry of a design (default: the one with the arm, which sets the stations' work plane)."""
    D = ROVER.design(design)
    if D.name in _PROBES:
        return _PROBES[D.name]
    spec = mujoco.MjSpec()
    _base_options(spec, "mars")
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                            **geom_physics(TERRAINS["bedrock"]))
    _attach_rover(spec, "p", (0, 0, 0), 0.03, D)
    m = spec.compile()
    d = mujoco.MjData(m)
    if D.arm:
        ROVER.set_arm_pose(m, d, "p", ROVER.STOW_Q)
    for _ in range(2500):
        mujoco.mj_step(m, d)
    ch = m.body(f"p/{CHASSIS}").id
    Rc = d.xmat[ch].reshape(3, 3)
    arm = dict(arm_base_z=None, arm_base_xy=None, pan_axis_local=None)
    if D.arm:
        ab = m.body("p/arm_base").id
        Ra = d.xmat[ab].reshape(3, 3)
        pan = m.joint("p/arm_shoulder_pan").id
        arm = dict(arm_base_z=float(d.xpos[ab][2]),
                   arm_base_xy=tuple(float(v) for v in (Rc.T @ (d.xpos[ab] - d.xpos[ch]))[:2].round(5)),
                   pan_axis_local=tuple(float(v) for v in (Ra.T @ (d.xanchor[pan] - d.xpos[ab]))[:2].round(5)))
    _PROBES[D.name] = Probe(chassis_z=float(d.xpos[ch][2]), **arm)
    return _PROBES[D.name]


def local_to_world(park, local_xy, arm_xy):
    """Arm-local point -> world xy for a chassis parked at `park` (arm base offset arm_xy in chassis)."""
    x, y, yaw = park
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = local_xy[0] + arm_xy[0], local_xy[1] + arm_xy[1]
    return np.array([x + c * lx - s * ly, y + s * lx + c * ly])


def park_for(world_xy, yaw, local_xy, arm_xy):
    """The chassis pose that puts arm-local point local_xy on world_xy, facing yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = local_xy[0] + arm_xy[0], local_xy[1] + arm_xy[1]
    return (float(world_xy[0] - (c * lx - s * ly)), float(world_xy[1] - (s * lx + c * ly)), float(yaw))


@dataclass
class World:
    m: mujoco.MjModel
    d: mujoco.MjData
    gravity: str
    rovers: list
    garage: object                                    # garage.Garage (None: the flat test ground)
    probe: Probe                                      # the arm design's geometry
    designs: dict = field(default_factory=dict)       # rid -> mars_rover.RoverDesign
    site: object = None                               # jezero.Site (None: the flat test ground)
    craters: list = field(default_factory=list)       # craters.Crater (sim side: checks and summaries)
    layout: object = None                             # jezero.Layout of this zone
    zone: tuple = (-3.0, 3.0, -3.0, 3.0)              # (x0, x1, y0, y1) of the work zone
    tiles: dict = field(default_factory=dict)         # survey camera name -> its ground rect
    start: dict = field(default_factory=dict)         # rid -> start pose
    items: list = field(default_factory=list)         # every movable object: the loose sample rocks
    drill_sites: object = None                        # (n, 2) candidate drill sites marked from orbit
    sched: object = None                              # hive.shard.Shard when rovers run in parallel
    viewer: object = None
    realtime: float = 1.0
    _last_sync: float = 0.0
    on_step: list = field(default_factory=list)
    on_sync: list = field(default_factory=list)       # callbacks(viewer) under the viewer lock, before each sync
    key_handlers: list = field(default_factory=list)  # callbacks(keycode) from the viewer thread
    on_overlay: list = field(default_factory=list)    # callbacks(viewer) after each sync, outside the lock
    pending_gravity: str | None = None
    on_gravity: list = field(default_factory=list)    # callbacks(name) after a gravity change
    wheels: dict = field(default_factory=dict)        # rid -> wheel geom ids
    contact_override: dict = field(default_factory=dict)   # rid -> [6 contacts] (calibration only)
    _wheel_key: dict = field(default_factory=dict)
    _contacts: dict = field(default_factory=dict)
    quiet: bool = False                               # counterfactual rollout: no viewer, no pacing, no callbacks

    # ------------------------------------------------------------ stepping
    def step(self, n=1):
        """Advance physics n steps. Inside a rover's task (parallel mode) this only WAITS until the
        shard's real-time clock has advanced n steps, while the other rovers act too."""
        if self.sched is not None and self.sched.in_task():
            self.sched.wait_steps(n)
            return
        self._advance(n, pace=True)

    # ------------------------------------------------------------ counterfactual rollouts (simulation only)
    def snapshot(self) -> dict:
        """The whole simulation state, bitwise: MjData (state and derived quantities) and the model
        fields the sim changes while running (wheel contacts under each wheel, weld poses)."""
        snap = mujoco.MjData(self.m)
        mujoco.mj_copyData(snap, self.m, self.d)
        return {"d": snap, "friction": self.m.geom_friction.copy(), "solref": self.m.geom_solref.copy(),
                "solimp": self.m.geom_solimp.copy(), "eq_data": self.m.eq_data.copy(),
                "wkey": self._wkey.copy() if hasattr(self, "_wkey") else None}

    def restore(self, s: dict):
        mujoco.mj_copyData(self.d, self.m, s["d"])          # in place: cameras keep their MjData
        self.m.geom_friction[:] = s["friction"]
        self.m.geom_solref[:] = s["solref"]
        self.m.geom_solimp[:] = s["solimp"]
        self.m.eq_data[:] = s["eq_data"]
        if s["wkey"] is not None:
            self._wkey = s["wkey"].copy()
        elif hasattr(self, "_wkey"):
            del self._wkey

    def _advance(self, n=1, pace=False):
        if self.quiet:                                      # a rollout: physics only, nobody watches
            self._ground_contact()
            for _ in range(n):
                mujoco.mj_step(self.m, self.d)
            return
        if self.pending_gravity is not None:
            g, self.pending_gravity = self.pending_gravity, None
            self.set_gravity(g)
            for f in self.on_gravity:
                f(g)
        self._ground_contact()
        now = time.time()
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
        for f in self.on_step:
            f(self)
        if self.viewer is not None:
            if time.time() - self._last_sync > 1 / 60:
                with self.viewer.lock():
                    for f in self.on_sync:
                        f(self.viewer)
                    self.viewer.sync()
                for f in self.on_overlay:                     # texts / figures: set outside the lock
                    f(self.viewer)
                self._last_sync = time.time()
        if pace and self.viewer is not None and self.realtime > 0:
            lag = n * self.m.opt.timestep / self.realtime - (time.time() - now)
            if lag > 0:
                time.sleep(lag)

    # ------------------------------------------------------------ the hidden ground (sim physics only)
    def _ground_contact(self):
        """Each wheel gets the contact of the ground under it (wheels have contact priority, so
        their parameters decide a wheel-ground contact; rocks have a higher one: hard rock)."""
        if self.site is None or not self.wheels:
            return
        G = np.concatenate([g for r, g in self.wheels.items() if r not in self.contact_override] or [np.zeros(0, int)])
        if not len(G):
            return
        xy = self.d.geom_xpos[G]
        c, slip, sink = self.site.slip_sinkage(xy[:, 0], xy[:, 1])
        key = c.astype(np.int64) * 10 ** 9 + np.rint(slip * 1000).astype(np.int64) * 10 ** 5 + np.rint(sink * 1e4).astype(np.int64)
        if not hasattr(self, "_wkey"):
            self._wkey = np.full(self.m.ngeom, -1, np.int64)
        for i in np.nonzero(key != self._wkey[G])[0]:
            k3 = (int(c[i]), round(float(slip[i]), 3), round(float(sink[i]), 4))
            self._set_contact(G[i], self._contact(k3))
            self._wkey[G[i]] = key[i]

    def _contact(self, key):
        p = self._contacts.get(key)
        if p is None:
            p = self._contacts[key] = J.contact_for(*key)
        return p

    def _set_contact(self, g, p):
        """Wheel geom g on ground p. Traction and softness go on the contact. The ground's resistance
        to rolling (soft ground: compaction, bulldozing) is a torque about the wheel's axle only:
        p["rolling"] (m) x the wheel's static load, as joint friction (a constraint: stable, it stalls the
        wheel at worst). As contact rolling friction it would also resist the wheel scrubbing sideways
        in a skid turn and dig it in."""
        m = self.m
        m.geom_friction[g] = [p["traction"], 0.005, ROLL_CONTACT]
        m.geom_solref[g] = p["solref"]
        m.geom_solimp[g] = p["solimp"]
        b = m.geom_bodyid[g]
        rid = m.body(b).name.split("/")[0]
        load = m.body_subtreemass[m.body_rootid[b]] * GRAVITY[self.gravity] / len(self.wheels[rid])
        k = self.designs[rid].bekker if rid in self.designs else 1.0    # this wheel's load and size on soft ground
        m.dof_frictionloss[m.jnt_dofadr[m.body_jntadr[b]]] = k * p["rolling"] * load

    def set_wheel_contact(self, rid, contacts):
        """Calibration: fix rid's six wheel contacts (None: back to the ground under them)."""
        if contacts is None:
            self.contact_override.pop(rid, None)
            if hasattr(self, "_wkey"):
                self._wkey[self.wheels[rid]] = -1
            return
        self.contact_override[rid] = contacts
        for g, p in zip(self.wheels[rid], contacts):
            self._set_contact(g, p)

    def ground_z(self, x, y) -> float:
        """Terrain height (graded pads included). Sim side: placing things, checks."""
        return 0.0 if self.site is None else float(self.site.height_at(x, y))

    def terrain_at(self, x, y) -> str:
        """Ground truth class at (x, y): ONLY for check scripts and human-facing summaries."""
        if self.site is None:
            return "flat"
        if not (-self.site.half <= x <= self.site.half and -self.site.half <= y <= self.site.half):
            return "off-site"
        return self.site.cls_name(x, y)

    # ------------------------------------------------------------ gravity
    def set_gravity(self, name: str):
        self.gravity = name
        self.m.opt.gravity[:] = [0, 0, -GRAVITY[name]]
        self.m.opt.density, self.m.opt.viscosity = AIR[name]
        if hasattr(self, "_wkey"):                      # the wheels' rolling resistance scales with the load
            self._wkey[:] = -1
        for rid, cs in getattr(self, "contact_override", {}).items():
            for g, p in zip(self.wheels[rid], cs):
                self._set_contact(g, p)

    @property
    def site_code(self) -> int:
        return SITE_CODE[self.gravity]

    @property
    def sim_time(self) -> float:
        return float(self.d.time)

    # ------------------------------------------------------------ welds: brakes and the carry hold
    def _weld(self, name: str, b1: int, b2: int, on: bool):
        """Switch weld `name` on at the bodies' current relative pose (body2 in body1's frame)."""
        i = self.m.equality(name).id
        if on:
            p1, q1 = self.d.xpos[b1].copy(), self.d.xquat[b1].copy()
            p2, q2 = (np.zeros(3), np.array([1.0, 0, 0, 0])) if b2 == 0 else (self.d.xpos[b2].copy(), self.d.xquat[b2].copy())
            qi = np.zeros(4)
            mujoco.mju_negQuat(qi, q1)
            rp, qr = np.zeros(3), np.zeros(4)
            mujoco.mju_rotVecQuat(rp, p2 - p1, qi)
            mujoco.mju_mulQuat(qr, qi, q2)
            self.m.eq_data[i][0:3] = 0
            self.m.eq_data[i][3:6] = rp
            self.m.eq_data[i][6:10] = qr
        self.d.eq_active[i] = 1 if on else 0

    def brake(self, rid: str, on: bool):
        self._weld(f"brake_{rid}", self.m.body(f"{rid}/{CHASSIS}").id, 0, on)

    def hold(self, rid: str, item: dict, on: bool):
        """The gripper's transport hold while a rover DRIVES with an object (servo position hold).
        Only needed because MuJoCo's soft friction cone lets a held object creep (~0.5 mm/s):
        fine for the lab's 2 s carries, fatal over a long drive. The grasp and lift before it and
        the place after it stay fully physical (and learned)."""
        self._weld(f"hold_{rid}_{item['name']}", self.m.body(f"{rid}/arm_gripper").id, item["body"], on)

    # ------------------------------------------------------------ the loose rocks
    @property
    def rocks(self):
        return [i for i in self.items if i["kind"] == "rock"]

    samples = rocks

    def item_pos(self, item):
        return self.d.qpos[item["qadr"]:item["qadr"] + 3].copy()

    def work_z(self, xy) -> float:
        """The work plane at a point: the ground (rocks are picked off it and placed on the tray floor)."""
        if self.garage is not None and self.garage.tray is not None and np.hypot(*(np.asarray(xy) - self.garage.tray)) < 0.15:
            return self.garage.z + 0.004
        return self.ground_z(*xy)

    def in_tray(self, item) -> bool:
        """Sim side: the rock lies in the sample tray of the sampler's bay."""
        from .garage import TRAY
        g = self.garage
        if g is None or g.tray is None:
            return False
        p = self.item_pos(item)
        lx, ly = g.to_local(p[:2]) - g.to_local(g.tray)
        return bool(abs(lx) < TRAY[0] and abs(ly) < TRAY[1] and p[2] < g.z + 0.05)

    def _key(self, keycode):
        """Viewer hotkeys (viewer thread): G cycles gravity earth -> mars -> moon (applied at the next
        step); the rest go to key_handlers (the decision panel)."""
        if keycode == ord("G"):
            order = ["earth", "mars", "moon"]
            self.pending_gravity = order[(order.index(self.gravity) + 1) % 3]
        for f in self.key_handlers:
            f(keycode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ===================================================================== builder
def _quat_yaw(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _look(pos, target):
    """xyaxes of a camera at pos looking at target (image up ~ world up)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


def _in_rects(x, y, rects, m=0.0):
    ok = np.zeros(np.shape(x), bool)
    for x0, x1, y0, y1 in rects:
        ok |= (x >= x0 - m) & (x <= x1 + m) & (y >= y0 - m) & (y <= y1 + m)
    return ok


def _read_obj(path):
    V, F = [], []
    for line in path.read_text(encoding="ascii").splitlines():
        p = line.split()
        if p and p[0] == "v":
            V.append([float(t) for t in p[1:4]])
        elif p and p[0] == "f":
            F.append([int(t) - 1 for t in p[1:4]])
    return np.array(V), np.array(F, int)


def _sky_and_light(spec, sun):
    vis = spec.visual
    vis.global_.offwidth, vis.global_.offheight = 1600, 1200
    vis.quality.shadowsize = 8192
    vis.headlight.diffuse, vis.headlight.ambient = [0.30, 0.26, 0.24], [0.28, 0.22, 0.20]
    vis.map.fogstart, vis.map.fogend = 15.0, 45.0
    vis.map.zfar = 80.0
    vis.rgba.fog = [0.80, 0.58, 0.45, 1]
    vis.rgba.haze = [0.85, 0.62, 0.48, 1]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.86, 0.62, 0.48], rgb2=[0.30, 0.17, 0.12], width=512, height=3072)
    s = spec.worldbody.add_light(name="sun", pos=[0, 0, 30], dir=sun, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 diffuse=[1.0, 0.84, 0.70], castshadow=True)
    s.specular = [0.1, 0.08, 0.06]


def _jezero_ground(spec, site, H, craters=()):
    """The heightfield (graded heightmap with the craters, 1:5) with the HiRISE orthoimage on it.
    Crater floors go below the dataset's lowest point: the heightfield spans zmin..zmax."""
    zmin, zmax = float(H.min()), float(H.max())
    n = H.shape[0]
    spec.add_hfield(name="jezero", size=[site.half, site.half, zmax - zmin, 0.5], nrow=n, ncol=n,
                    userdata=(np.flipud(H - zmin) / (zmax - zmin)).ravel().tolist())   # MuJoCo's row 0 is the south edge
    img = cv2.cvtColor(cv2.imread(str(J.ASSETS / "terrain_albedo.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = np.flipud(CR.paint(img, site, list(craters))).copy()                 # image row 0 is north
    tex = spec.add_texture(name="tex_ground", type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                           height=img.shape[0], nchannel=3)
    tex.data = img.tobytes()
    mat = spec.add_material(name="mat_ground", texrepeat=[1, 1], texuniform=False, specular=0.03, shininess=0.02)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tex_ground"
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="jezero", pos=[0, 0, zmin],
                            material="mat_ground", friction=[0.9, 0.005, 0.0005], condim=3, priority=0)


def _rocks(spec, site, keep_rects, pads, rng, rk=None):
    """The dataset's rocks (1:5): colliding meshes in the work zone and along the traverse (taller
    than 1 cm: smaller ones are below the wheels' notice), visual meshes elsewhere for the big ones.
    Graded pads are cleared. rk: the rocks to use (default the dataset's; craters.rocks moves some)."""
    rk = site.rocks if rk is None else rk
    x, y, h = rk["x"], rk["y"], rk["height"]
    near = _in_rects(x, y, keep_rects, 0.5)
    collide = near & (h > 0.010)
    visual = (~near) & (h > 0.03)
    for poly, r, _ in pads:
        pts = np.array(poly)
        d = np.min([J._seg_dist(x, y, pts[a], pts[min(a + 1, len(pts) - 1)]) for a in range(max(len(pts) - 1, 1))], axis=0)
        clear = d < r + rk["radius"] + 0.05
        collide &= ~clear
        visual &= ~clear
    meshes = [_read_obj(p) for p in J.rock_variants()]
    from scipy.spatial import ConvexHull
    hulls = [ConvexHull(V).vertices for V, _ in meshes]     # per variant: the hull's vertex indices
    rc = J.ROCK_CONTACT
    phys = dict(friction=[rc["traction"], 0.005, rc["rolling"]], solref=rc["solref"], solimp=rc["solimp"], condim=3,
                priority=3)
    # Drawing: one merged visual mesh per 8 m block and shade (a few dozen draw calls instead of
    # thousands: every rover's navcam renders this scene once per decision). Physics: each
    # colliding rock keeps its own (convex) mesh geom, invisible (group 3).
    hidden = dict(group=3, rgba=[0, 0, 0, 0])
    shades = [0.30, 0.36, 0.42]
    merged: dict = {}
    for i in np.nonzero(collide | visual)[0]:
        V, F = meshes[rk["variant"][i]]
        Vs = V * rk["scale"][i]
        if collide[i]:                                 # collision uses the convex hull: give only its vertices
            name = f"rock{i}"
            spec.add_mesh(name=name, uservert=Vs[hulls[rk["variant"][i]]].ravel().tolist())
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                    pos=[x[i], y[i], rk["z"][i]], quat=_quat_yaw(rk["yaw"][i]), **phys, **hidden)
        c, s = np.cos(rk["yaw"][i]), np.sin(rk["yaw"][i])
        Vw = Vs @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x[i], y[i], rk["z"][i]]
        key = (int((x[i] + 40) // 8), int((y[i] + 40) // 8), int(rng.integers(len(shades))))
        merged.setdefault(key, []).append((Vw, F))
    for (bx, by, k), parts in merged.items():
        off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        V = np.vstack([p[0] for p in parts])
        F = np.vstack([p[1] + o for p, o in zip(parts, off)])
        name = f"rocks_{bx}_{by}_{k}"
        ms = spec.add_mesh(name=name, uservert=V.ravel().tolist(), userface=F.ravel().tolist())
        ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = shades[k]
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                rgba=[g * 1.45, g * 0.95, g * 0.75, 1], contype=0, conaffinity=0, group=0)
    return int(collide.sum()), int(visual.sum())


def build(rovers=("r1", "r2", "r3"), gravity="mars", gui=False, realtime=1.0, seed=0, zone=0, flat=False,
          start_at=None, designs=None, rocks=True, rock_spots=None) -> World:
    """start_at: {rid: "A"} starts those rovers at the traverse's start A instead of their garage bay.
    designs: {rid: "scout" | "scientist" | "sampler"} (default: every rover is the sampler).
    rocks: scatter the loose sample rocks around the drill sites; rock_spots: [(x, y, yaw, size class)]
    puts them there instead (checks)."""
    from . import garage as GA
    from . import sample_rocks as SR
    from .survey import candidate_sites
    start_at = start_at or {}
    designs = {rid: ROVER.design((designs or {}).get(rid)) for rid in rovers}
    pr = probe()
    spec = mujoco.MjSpec()
    _base_options(spec, gravity)
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP      # resting rocks cost nothing
    rng = np.random.default_rng(seed)
    wb = spec.worldbody
    site = layout = gar = None
    craters: list = []
    drill_sites = np.zeros((0, 2))
    if flat:                                 # flat test ground (calibration): no site, no garage
        _sky_and_light(spec, [0.35, 0.45, -0.82])
        n = 101                              # a flat heightfield with the site's 10 cm spacing: the same
        spec.add_hfield(name="flat", size=[5.0, 5.0, 0.01, 0.5], nrow=n, ncol=n,   # contact generation as Jezero
                        userdata=[0.0] * (n * n))
        wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="flat", rgba=[0.6, 0.45, 0.35, 1],
                    friction=[0.9, 0.005, 0.0005], condim=3, priority=0)
        zrect = (-3.0, 3.0, -3.0, 3.0)
        starts = {rid: (0.0, -1.0 + 0.8 * i, np.pi / 2) for i, rid in enumerate(rovers)}
        ground = lambda x, y: 0.0           # noqa: E731
        tiles = {}
    else:
        site = J.load()
        layout = J.zones(max(zone + 1, 1))[zone]
        zrect = layout.zone
        if not hasattr(site, "cls0"):                 # the dataset's hidden maps, before any craters
            site.cls0, site.patch0 = site.cls, site.patch
        bay_rovers = [r for r in rovers if start_at.get(r) != "A"]
        gar = GA.place(site, zrect, designs, bay_rovers, lambda D: D.front,
                       reach=lambda D: probe(D).arm_base_xy[0] + PICK_LOCAL[0])
        craters = CR.plan(site, zrect, zone, CR.keepouts(site, gar.keepouts()))
        site.graded = CR.carve(site, J.grade(site, [gar.pad()]), craters)
        site.cls, site.patch = CR.ground_truth(site.cls0, site.patch0, site, craters)
        (pa, pb), pr_, _ = gar.pad()                   # the garage floor and apron: compacted, hard ground
        gx, gy = site.xy_grid()
        hard = J._seg_dist(gx[None, :], gy[:, None], pa, pb) < pr_
        site.cls = np.where(hard, J.CLASSES.index("bedrock"), site.cls)
        site.patch = np.where(hard, -1, site.patch)
        ground = site.height_at
        _sky_and_light(spec, J.sun_dir())
        _jezero_ground(spec, site, site.graded, craters)
        starts = dict(gar.park)
        site.rocks_placed = CR.rocks(site, craters, site.rocks, zone)  # the rocks as they lie (rim boulders too)
        drill_sites = candidate_sites(site, zrect, gar.keepouts())
        corridor = J.corridor_tiles(site)
        tiles = {"site_cam": zrect, **{f"tile_{k}": r for k, r in enumerate(corridor)}}
        clear = [([site.start[:2]], 0.45, 0.0)]                      # the traverse start A: rocks cleared
        _rocks(spec, site, [zrect] + corridor, [gar.pad()] + clear, rng, site.rocks_placed)
        GA.build(spec, gar, ground)
    SITE["x"], SITE["y"] = (zrect[0], zrect[1]), (zrect[2], zrect[3])

    # ---- the loose sample rocks, resting on the natural ground around the drill sites
    items = []
    grng = np.random.default_rng(int(abs(zrect[0]) * 13 + abs(zrect[2]) * 7) + 1)
    spots = rock_spots if rock_spots is not None else (SR.scatter(drill_sites, grng) if rocks else [])
    if len(spots):
        for k, (x, y, yaw, cls) in enumerate(spots):
            it, _ = SR.add_rock(spec, f"srock{k}", [x, y, float(ground(x, y)) + 0.03], yaw, cls, grng)
            items.append(it)

    # ---- fixed overhead survey cameras: the zone's and the traverse tiles' ("orbital" views)
    def tile_cam(name, r):
        cx, cy = (r[0] + r[1]) / 2, (r[2] + r[3]) / 2
        xs = np.linspace(r[0], r[1], 9)
        ys = np.linspace(r[2], r[3], 9)
        top = max(float(ground(a, b_)) for a in xs for b_ in ys)
        wide_y = (r[3] - r[2]) > (r[1] - r[0])         # image width (1600 px) along the long side
        kw = dict(xyaxes=[0, 1, 0, -1, 0, 0]) if wide_y else {}
        wb.add_camera(name=name, pos=[cx, cy, top + CAM_ABOVE], fovy=60.0, **kw)

    for name, r in (tiles.items() if tiles else [("site_cam", zrect)]):
        tile_cam(name, r)
    cx, cy = (zrect[0] + zrect[1]) / 2, (zrect[2] + zrect[3]) / 2
    cz = float(ground(cx, cy))
    bpos = [zrect[0] - 1.8, zrect[2] - 3.2, cz + 2.6]
    wb.add_camera(name="beauty", pos=bpos, xyaxes=_look(bpos, [cx, cy, cz]), fovy=55)
    if gar is not None:                                # the garage seen from its apron, a little to the side
        gc = gar.to_world([3.2, 1.4])
        gpos = [gc[0], gc[1], gar.z + 1.5]
        gl = gar.to_world([-0.6, 0.0])
        wb.add_camera(name="garage_view", pos=gpos, xyaxes=_look(gpos, [gl[0], gl[1], gar.z + 0.35]), fovy=60)
    if site is not None:
        a, b_ = site.start, site.goal
        tpos = [a[0] - 6.0, a[1] - 6.0, float(ground(*a[:2])) + 9.0]
        wb.add_camera(name="traverse_view", pos=tpos, xyaxes=_look(tpos, [(a[0] + b_[0]) / 2, (a[1] + b_[1]) / 2, 0.5]),
                      fovy=60)

    # ---- rovers, their brakes and their carry holds
    start = {}
    for rid in rovers:
        if start_at.get(rid) == "A" and site is not None:
            x, y, yaw = site.start
        else:
            x, y, yaw = starts[rid]
        start[rid] = (float(x), float(y), float(yaw))
        _attach_rover(spec, rid, start[rid], float(ground(x, y)) + probe(designs[rid]).chassis_z + 0.01, designs[rid])
        eq = spec.add_equality(name=f"brake_{rid}")
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = f"{rid}/{CHASSIS}"
        eq.name2 = "world"
        eq.active = False
        eq.solref = [0.004, 1.0]
        if ROVER.has_drill(designs[rid]):           # hole markers: parked out of sight until a hole is drilled
            for k in range(HOLE_MARKERS):
                hb = spec.worldbody.add_body(name=f"hole_{rid}_{k}", pos=[x, y, -5.0], mocap=True)
                hb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.034, 0.0012, 0], pos=[0, 0, 0.0004],
                            rgba=[0.74, 0.52, 0.36, 1], contype=0, conaffinity=0, group=1)      # tailings
                hb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0105, 0.0012, 0], pos=[0, 0, 0.0012],
                            rgba=[0.10, 0.07, 0.05, 1], contype=0, conaffinity=0, group=1)      # the hole
        for it in (items if designs[rid].arm else []):
            eq = spec.add_equality(name=f"hold_{rid}_{it['name']}")
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = f"{rid}/arm_gripper"
            eq.name2 = it["name"]
            eq.active = False
            eq.solref = [0.01, 1.0]

    m = spec.compile()
    # MuJoCo scales clip planes and fog by the model's extent (~57 m here): set them in metres, so
    # a camera 15 cm above the ground (the hazcam) still sees it, and fog starts at 15 m
    ext = float(m.stat.extent)
    m.vis.map.znear, m.vis.map.zfar = 0.02 / ext, 150.0 / ext
    m.vis.map.fogstart, m.vis.map.fogend = 15.0 / ext, 45.0 / ext
    d = mujoco.MjData(m)
    for rid in rovers:                         # arms folded: out of the cameras' view
        if designs[rid].arm:
            ROVER.set_arm_pose(m, d, rid, ROVER.STOW_Q)
    w = World(m=m, d=d, gravity=gravity, rovers=list(rovers), garage=gar, probe=pr, designs=designs,
              site=site, layout=layout, zone=tuple(zrect), tiles=tiles, start=start, realtime=realtime,
              craters=craters, drill_sites=drill_sites)
    for rid in rovers:                         # wheels win wheel-ground contacts: the hidden ground goes on them
        w.wheels[rid] = np.array([m.geom(f"{rid}/wheel_{k}_col").id for k in designs[rid].wheels])
        m.geom_priority[w.wheels[rid]] = 2
    for it in items:
        j = m.joint(f"{it['name']}_free")
        w.items.append(dict(it, body=m.body(it["name"]).id, qadr=j.qposadr[0], vadr=j.dofadr[0], state="ground"))
    if gui:
        from mujoco import viewer as mj_viewer
        w.viewer = mj_viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False, key_callback=w._key)
        if gar is not None:
            look = gar.to_world([0.8, 0.0])
            w.viewer.cam.lookat[:] = [look[0], look[1], gar.z + 0.3]
            w.viewer.cam.distance, w.viewer.cam.azimuth, w.viewer.cam.elevation = 6.0, np.degrees(gar.yaw) + 25, -28
        else:
            w.viewer.cam.lookat[:] = [cx, cy, cz]
            w.viewer.cam.distance, w.viewer.cam.azimuth, w.viewer.cam.elevation = 8.0, 230, -38
    w._advance(400)                           # settle on the ground, then park
    for rid in rovers:
        w.brake(rid, True)
    return w
