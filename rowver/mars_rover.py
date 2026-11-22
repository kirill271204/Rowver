"""The fleet's three 1:5 Mars rovers. Each is its own machine, built for its job and its payload.

  scout      "Pathfinder": goes first and finds the traps. Low wedge body, 4 big balloon wheels on
             independent spring-loaded swing arms (soaks up rocks at speed), 4.5 kg, 0.22 m/s.
             Senses wheel torque and sinkage (a camera on the front-left wheel); low navcam with
             DINOv2 codes, front and rear hazcams; roll cage, whip antenna, deck solar panel.
  scientist  "Spectra": reads the rocks from a distance. 6 wheels on a rocker-bogie (rocker + bogie per
             side, differential: Perseverance's suspension), ~8 kg, 0.15 m/s. Tall mast with a
             SuperCam-like turret (remote spectroscopy 0.5-4 m) and a narrow telescope camera (RMI);
             navcam with DINOv2 codes; RTG.
  sampler    "Hauler": drills and carries. Long low cargo deck on 6 wheels (a rigid middle axle, sprung
             front and rear swing arms), ~14 kg, 0.10 m/s. SO-101 arm on the front edge (folded back
             over the deck while driving), a drill tower with a sliding core drill, a sample carousel;
             navcam (colour + geometry), hazcam, a drill camera.

All three skid-steer: every wheel is driven, the rover turns by driving its sides at different
speeds and can not move sideways. Motor gain, torque cap and motor inertia scale with weight per
wheel and wheel speed, so every design meets the ground's rolling resistance the same way.
Collision: ellipsoid wheels (smooth in skid turns) and a body box; the rest is visual.

Cameras and arm never block each other (checked for each design): the navcam head
reaches forward to the nose line, so every ray points ahead of anything on the deck. The sampler's arm mounts LOW on
the body's front face (Perseverance-style: it reaches rocks on the ground) and folds across the
front while driving (STOW_Q): below the navcam's lowest ray and above the hazcam, which hangs under
the arm's base and looks forward and down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

CHASSIS = "chassis"
NAV_BASELINE = 0.084           # m: navcam stereo baseline (Perseverance 42 cm at 1:5)
HAZ_BASELINE = 0.050           # m: hazcam stereo baseline (Perseverance ~25 cm at 1:5)
WHEELS4 = ("fl", "fr", "rl", "rr")
WHEELS6 = ("fl", "fr", "ml", "mr", "rl", "rr")
WHEELS = WHEELS4                                # the default design's wheels are named per design: design.wheels
ROCKER_RANGE = np.radians(25)
# the arm's driving pose (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll): folded
# across the body's front face; valid for an arm mounted as `arm_mount` places it
STOW_Q = (-1.57, 1.70, 1.21, 0.0, 0.0)        # searched: 3.9 cm clear of body, wheels, ground; 0 camera pixels
STOW_FRONT = 0.075                              # m: how far the folded arm reaches ahead of its mount
ARM_MASS = 0.85                                # kg: the SO-101 with its mount (approx., for scaling)
# reference drive tuned on the first rover (8.85 kg, 6 cm wheels at 0.12 m/s): kv 3, 5 N m, 0.01 kg m^2
_REF = dict(mass=8.85, omega=2.0, kv=3.0, torque=5.0, inertia=0.01)

WHITE, GOLD, DARK, GREY = (0.90, 0.89, 0.86, 1), (0.80, 0.62, 0.22, 1), (0.16, 0.16, 0.17, 1), (0.50, 0.50, 0.52, 1)
LENS = (0.05, 0.05, 0.08, 1)
SO101_XML = Path(__file__).resolve().parents[1] / "so101" / "assets" / "so101" / "so101_new_calib.xml"


@dataclass(frozen=True)
class RoverDesign:
    name: str
    body: tuple                 # (x0, x1, y0, y1, z0, z1) of the body box, chassis frame (origin at ground level)
    mass: float                 # kg, the chassis alone
    wheel_r: float
    wheel_w: float
    wheel_x: float              # wheel centres at (+-wheel_x, +-wheel_y)
    wheel_y: float
    v_max: float                # m/s
    navcam_z: float             # m: navcam height
    navcam_pitch: float = np.radians(40.0)
    navcam_fovy: float = 70.0
    hazcam_pitch: float = np.radians(50.0)
    hazcam_fovy: float = 80.0
    arm: bool = False
    dino: bool = False          # runs the frozen DINOv2 codes on its navcam
    rear_hazcam: bool = False
    rmi: bool = False           # a narrow telescope camera on the mast (remote micro-imager)
    colour: tuple = WHITE       # body colour
    tint: tuple = GOLD          # accents (deck plate)
    sensors: tuple = ()         # unique readings it can make (dock.py / tasks.py)
    track_factor: float = 1.75  # skid steering's expansion factor (measured by check_rover)
    suspension: str = "swing4"  # swing4 (scout) | rocker_bogie (scientist) | cargo6 (sampler)
    mid_x: float = 0.0          # 6-wheel designs: x of the middle wheels
    arm_y: float = 0.0          # the arm's mount: y on the body's front face

    @property
    def wheels(self) -> tuple:
        return WHEELS4 if self.suspension == "swing4" else WHEELS6

    def wheel_xy(self, w: str):
        sx = {"f": self.wheel_x, "m": self.mid_x, "r": -self.wheel_x}[w[0]]
        return sx, (self.wheel_y if w[1] == "l" else -self.wheel_y)

    # ---------------------------------------------------------------- derived geometry
    @property
    def total_mass(self) -> float:
        return self.mass + len(self.wheels) * self.wheel_mass + 4 * 0.25 * self.mass / 7.0 + (ARM_MASS if self.arm else 0.0)

    @property
    def wheel_mass(self) -> float:
        return 0.18 * self.mass / 7.0

    @property
    def pivot(self):
        return (0.0, self.body[3] + 0.03, self.body[4] + 0.02)

    @property
    def nose(self) -> float:
        """The frontmost point of the rover at deck height and below: body or front wheels."""
        return max(self.body[1], self.wheel_x + self.wheel_r)

    @property
    def tail(self) -> float:
        return min(self.body[0], -(self.wheel_x + self.wheel_r))

    @property
    def navcam_pos(self):
        return (self.nose + 0.06, -(self.body[3] - 0.02), self.navcam_z)

    @property
    def hazcam_pos(self):
        if self.arm:                                  # under the arm's base, looking forward and down
            return (self.body[1] + 0.045, 0.0, self.body[4] - 0.028)
        return (self.nose + 0.01, 0.0, self.body[4] + 0.05)

    @property
    def front(self) -> float:
        """The frontmost point of the rover as it drives (the folded arm included)."""
        return max(self.nose, self.arm_mount[0] + STOW_FRONT) if self.arm else self.nose

    @property
    def arm_mount(self):
        return (self.body[1] + 0.035, self.arm_y, self.body[4] + 0.02)     # low on the front face

    @property
    def self_box(self):
        """The rover's own extent (x0, x1, y0, y1, z_above): camera points inside it that stand above
        the ground are the rover itself (body, wheels, folded arm), masked out."""
        yw = self.wheel_y + self.wheel_w / 2 + 0.045
        return (self.tail - 0.04, self.front + 0.07, -yw, yw, 0.025)

    @property
    def bekker(self) -> float:
        """Soft-ground resistance of this design's wheels relative to the reference design (the sampler,
        on which the ground's slip is calibrated). Bekker: sinkage z ~ (W / (b sqrt(D)))^(2/3) for a
        wheel load W, width b, diameter D; rolling resistance ~ sqrt(z / D). A light rover on big wide
        wheels sinks less and meets less resistance on the same sand."""
        def c(d):
            W_ = d.total_mass / len(d.wheels)
            D_ = 2 * d.wheel_r
            return (W_ / (d.wheel_w * np.sqrt(D_))) ** (1 / 3) / np.sqrt(D_)
        ref = DESIGNS.get(DEFAULT)
        return float(c(self) / c(ref)) if ref is not None else 1.0

    @property
    def drive(self) -> dict:
        """Servo gain, torque cap and motor inertia scaled from the reference rover by weight per
        wheel and nominal wheel speed (slip = resistance / (kv x wheel speed) stays the same)."""
        w = (self.total_mass / len(self.wheels)) / (_REF["mass"] / 4)
        om = _REF["omega"] / (0.8 * self.v_max / self.wheel_r)
        return dict(kv=_REF["kv"] * w * om, torque=_REF["torque"] * w, inertia=_REF["inertia"] * w * om)


DESIGNS = {
    "scout": RoverDesign(
        name="scout", body=(-0.17, 0.15, -0.10, 0.10, 0.13, 0.21), mass=3.4, wheel_r=0.075, wheel_w=0.07,
        wheel_x=0.15, wheel_y=0.20, v_max=0.22, navcam_z=0.40, dino=True, rear_hazcam=True, suspension="swing4", track_factor=1.70,
        colour=(0.93, 0.47, 0.13, 1), tint=(0.12, 0.14, 0.30, 1), sensors=("sinkage_mm", "rolling_res")),
    "scientist": RoverDesign(
        name="scientist", body=(-0.22, 0.22, -0.12, 0.12, 0.17, 0.33), mass=6.6, wheel_r=0.055, wheel_w=0.05,
        wheel_x=0.21, wheel_y=0.21, mid_x=-0.02, v_max=0.15, navcam_z=0.74, dino=True, rmi=True,
        suspension="rocker_bogie", track_factor=2.42, colour=WHITE, tint=GOLD, sensors=("photochem_score", "mineral_class")),
    "sampler": RoverDesign(
        name="sampler", body=(-0.32, 0.30, -0.16, 0.16, 0.15, 0.27), mass=11.5, wheel_r=0.065, wheel_w=0.06,
        wheel_x=0.25, wheel_y=0.24, mid_x=0.0, v_max=0.10, navcam_z=0.62, arm=True, arm_y=0.0, suspension="cargo6", track_factor=1.06,
        colour=(0.95, 0.95, 0.93, 1), tint=(0.20, 0.45, 0.75, 1), sensors=("drill_torque_nm", "sample_success")),
}
DRILL_FLUTES = (0.13, 0.22)     # m below the drill head: the string ends / the bit's tip (the flutes between)
DRILL_TRAVEL = 0.26             # m: the slide's stroke (ground contact from the deck, plus the core depth)
CAROUSEL_SLOTS = 8


def has_drill(dsg) -> bool:
    return design(dsg).name == "sampler"


DEFAULT = "sampler"             # the design with the arm (station work)


def design(name_or_design) -> RoverDesign:
    return name_or_design if isinstance(name_or_design, RoverDesign) else DESIGNS[name_or_design or DEFAULT]


def _box_inertia(m, sx, sy, sz):
    return [m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12]


def _need(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (the SO-101 model ships with so101/assets)")


def arm_spec() -> mujoco.MjSpec:
    """The SO-101 (with its actuators) as an attachable spec, base at the frame origin, with exactly
    the load-time changes so101/robot/so101_mujoco.py makes: the arm meshes are visual only and two
    box pads on the fingers are what grips (the lab's grasp rows were learned with these pads)."""
    from so101.robot.so101_mujoco import JAW_PAD, STATIC_PAD
    _need(SO101_XML)
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    for g in spec.geoms:                                 # arm meshes: visual only
        g.contype, g.conaffinity = 0, 0
    pad = dict(type=mujoco.mjtGeom.mjGEOM_BOX, friction=[1.6, 0.02, 0.002], condim=4,
               rgba=[0.15, 0.15, 0.15, 1], contype=1, conaffinity=1, solref=[0.004, 1])
    spec.body("gripper").add_geom(name="static_pad", pos=STATIC_PAD["pos"], size=STATIC_PAD["size"], **pad)
    spec.body("moving_jaw_so101_v1").add_geom(name="jaw_pad", pos=JAW_PAD["pos"], size=JAW_PAD["size"], **pad)
    return spec


def _quat(M) -> list:
    """Rotation matrix (columns = local axes in the parent frame) -> MuJoCo quaternion."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(M, float).ravel())
    return q.tolist()


def _frame_along(d):
    """A rotation whose local z is along d."""
    z = np.asarray(d, float) / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def rover_spec(dsg=None, tint=None) -> mujoco.MjSpec:
    """One rover of design `dsg` (bodies, joints, actuators, its payload), as an attachable spec.
    Physics: the body box, the wheel ellipsoids, the joints and the masses. Everything else is
    visual (rover_parts.py) and never collides or weighs anything."""
    from .rover_parts import Parts
    D = design(dsg)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    P = Parts(spec, D.colour)
    G = mujoco.mjtGeom

    def cam(body, name, pos, pitch, fovy, back=False):
        s_, c_ = np.sin(pitch), np.cos(pitch)                 # looks along +x (or -x), tilted down
        xy = [0, 1, 0, -s_, 0, c_] if back else [0, -1, 0, s_, 0, c_]
        body.add_camera(name=name, pos=list(pos), xyaxes=xy, fovy=fovy)

    ch = spec.worldbody.add_body(name=CHASSIS)
    ch.add_freejoint(name="root")
    x0, x1, y0, y1, z0, z1 = D.body
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    ch.add_geom(name="body_box", type=G.mjGEOM_BOX, pos=c.tolist(), size=[hx, hy, hz], rgba=[1, 1, 1, 0], group=3,
                density=0)
    ch.mass, ch.ipos, ch.explicitinertial = D.mass, [0.0, 0.0, (z0 + z1) / 2], True
    ch.inertia = _box_inertia(D.mass, 2 * hx, 2 * hy, 2 * hz)

    # ================================================================ the body (warm electronics box)
    P.chamfer(ch, c, [hx, hy, hz], 0.012, "paint")
    blanket = {"scout": "silver", "scientist": "gold", "sampler": "silver"}[D.name]
    for s in (-1, 1):                                         # side blankets with seams
        P.box(ch, [c[0], s * (hy + 0.0015), c[2] - 0.15 * hz], [hx - 0.02, 0.0015, hz * 0.62], blanket)
        for k in range(1, 4):
            P.box(ch, [x0 + k * 2 * hx / 4, s * (hy + 0.003), c[2] - 0.15 * hz], [0.0012, 0.0012, hz * 0.62], "dark")
    P.box(ch, [x1 + 0.0015, 0, c[2] + 0.35 * hz], [0.0015, hy * 0.55, 0.006], "accent")   # ID stripe on the nose
    deck = z1 + 0.002
    P.box(ch, [c[0] - 0.01, 0, deck], [hx - 0.03, hy - 0.025, 0.002], "black")          # radiator deck
    for k in range(5):                                        # radiator fins / cable trays across it
        P.box(ch, [x0 + 0.04 + k * (2 * hx - 0.08) / 4, 0, deck + 0.003], [0.002, hy - 0.03, 0.002], "dark")

    # ================================================================ mast + navcam head (front-right corner)
    nx, ny, nz = D.navcam_pos
    mx = x1 - 0.025                                           # mast column on the deck's front-right corner
    P.chamfer(ch, [mx, ny, deck + 0.022], [0.024, 0.024, 0.02], 0.006, "paint")         # deploy hinge housing
    P.cyl(ch, [mx, ny, deck + 0.04], [mx, ny, nz - 0.05], 0.011, "carbon")
    P.disc(ch, [mx, ny, nz - 0.045], 0.019, 0.01, "z", "alu")                            # azimuth actuator
    P.disc(ch, [mx, ny + 0.03, nz - 0.01], 0.014, 0.012, "y", "dark")                    # elevation actuator
    face = nx - 0.004                                          # everything on the head stays behind the lenses
    P.chamfer(ch, [(mx - 0.03 + face) / 2, ny, nz], [(face - mx + 0.03) / 2, 0.055, 0.024], 0.006, "paint")
    P.box(ch, [face - 0.001, ny, nz], [0.001, 0.05, 0.02], "black")                      # optics bench face
    P.camera_pair(ch, [face - 0.012, ny, nz], 0.05, 0.0065)                              # navcam stereo pair
    cam(ch, "navcam", D.navcam_pos, D.navcam_pitch, D.navcam_fovy)
    # the stereo partners (stereo.py): the right eye of each pair, along the camera's +x (the chassis' -y).
    # Baselines: Perseverance's navcam 42 cm and hazcam 25 cm, at 1:5
    cam(ch, "navcam_r", np.add(D.navcam_pos, [0, -NAV_BASELINE, 0]).tolist(), D.navcam_pitch, D.navcam_fovy)

    # ================================================================ hazcams on the nose
    hxp, _, hzp = D.hazcam_pos
    hf = hxp - 0.004
    if hf - 0.03 > x1:                                        # a bracket out to the nose (front wheels ahead)
        P.box(ch, [(x1 + hf - 0.03) / 2, 0, hzp], [(hf - 0.03 - x1) / 2, 0.01, 0.005], "titanium")
    P.chamfer(ch, [hf - 0.015, 0, hzp], [0.015, 0.05, 0.014], 0.004, "black")
    P.camera_pair(ch, [hf - 0.012, 0, hzp], 0.07, 0.006)
    cam(ch, "hazcam", D.hazcam_pos, D.hazcam_pitch, D.hazcam_fovy)
    cam(ch, "hazcam_r", np.add(D.hazcam_pos, [0, -HAZ_BASELINE, 0]).tolist(), D.hazcam_pitch, D.hazcam_fovy)

    # ================================================================ per design
    if D.name == "scout":
        _scout(P, ch, D, cam, deck)
    elif D.name == "scientist":
        _scientist(P, ch, D, cam, deck)
    else:
        _sampler(P, spec, ch, D, cam, deck)

    # ================================================================ suspension and wheels
    dr = D.drive
    load = D.total_mass * 3.71 / len(D.wheels)               # N per wheel on Mars

    def wheel(parent, w, at):
        wb = parent.add_body(name=f"wheel_{w}", pos=list(at))
        wb.add_joint(name=f"wheel_{w}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.002,
                     armature=dr["inertia"])
        q = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        R_, W2 = D.wheel_r, D.wheel_w / 2
        wb.add_geom(name=f"wheel_{w}_col", type=G.mjGEOM_ELLIPSOID, size=[R_, R_, W2], quat=q,
                    rgba=[0, 0, 0, 0], group=3, mass=D.wheel_mass, condim=6, friction=[1.0, 0.005, 0.0005])
        P.wheel(wb, R_, D.wheel_w)
        side = 1 if w[1] == "l" else -1                       # in-wheel drive motor, inboard
        P.disc(wb, [0, -side * (W2 + 0.012), 0], 0.022 * R_ / 0.06, 0.012, "y", "dark")

    def hinge(body, name, lim, stiff=0.0, damp=0.05):
        body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-lim, lim],
                       stiffness=stiff, damping=damp, armature=1e-4)
        body.add_geom(type=G.mjGEOM_SPHERE, size=[0.012, 0, 0], mass=0.25 * D.mass / 7.0, contype=0, conaffinity=0,
                      group=2, material="titanium")
        P.disc(body, [0, 0, 0], 0.02, 0.016, "y", "alu")       # pivot housing

    def axle(body, at, side):
        """A wheel's steering-less upright: from the tube end down/out to the hub."""
        P.tube(body, [at[0], 0, at[2] + 0.012], [at[0], at[1] - side * (D.wheel_w / 2 + 0.024), at[2] + 0.012], 0.008)
        P.chamfer(body, [at[0], at[1] - side * (D.wheel_w / 2 + 0.03), at[2] + 0.004], [0.016, 0.008, 0.02], 0.004,
                  "titanium")

    def swing_arm(w, pivot_x, sag):
        """A spring-loaded swing arm from the body at pivot_x to wheel w (independent suspension)."""
        wx, wy = D.wheel_xy(w)
        sy = 1 if wy > 0 else -1
        pv = np.array([pivot_x, sy * (D.body[3] + 0.02), D.wheel_r + 0.02])
        arm = ch.add_body(name=f"swing_{w}", pos=list(pv))
        L = abs(wx - pivot_x)
        hinge(arm, f"susp_{w}", np.radians(25), stiff=load * L / sag, damp=0.3 * load * L)
        at = np.array([wx, wy, D.wheel_r]) - pv
        P.tube(arm, [0, 0, 0], [at[0], 0, at[2] + 0.012], 0.011)
        P.tube(arm, [0, 0.012 * sy, 0.006], [at[0], 0.012 * sy, at[2] + 0.018], 0.006)      # twin-tube arm
        axle(arm, at, sy)
        a = np.array([0.55 * at[0], 0, 0.55 * at[2] + 0.012])  # coil-over: arm to the body's shock mount
        b = np.array([0.1 * at[0], 0, 0.085])
        P.cyl(arm, a, b, 0.008, "dark")
        Rm = _frame_along(b - a)
        P.coil(arm, a + 0.15 * (b - a), 0.014, 0.6 * np.linalg.norm(b - a), 6, 0.003, quat=_quat(Rm))
        wheel(arm, w, at)

    if D.suspension == "swing4":
        for w in D.wheels:
            wx, _ = D.wheel_xy(w)
            swing_arm(w, wx - np.sign(wx) * 0.11, sag=0.10)
    elif D.suspension == "rocker_bogie":
        for side, sy in (("l", 1), ("r", -1)):
            pv = np.array([0.03, sy * (D.body[3] + 0.03), D.body[4] + 0.03])
            P.disc(ch, [pv[0], sy * (D.body[3] + 0.012), pv[2]], 0.024, 0.012, "y", "dark")   # rocker bearing
            rk = ch.add_body(name=f"rocker_{side}", pos=list(pv))
            hinge(rk, f"rocker_{side}", ROCKER_RANGE)
            fx, fy = D.wheel_xy(f"f{side}")
            at = np.array([fx, fy, D.wheel_r]) - pv
            knee = np.array([0.55 * at[0], 0, 0.02])
            P.tube(rk, [0, 0, 0], knee, 0.012)
            P.tube(rk, knee, [at[0], 0, at[2] + 0.012], 0.011)
            axle(rk, at, sy)
            wheel(rk, f"f{side}", at)
            bx = (D.mid_x - D.wheel_x) / 2
            bp = np.array([bx, 0, D.wheel_r + 0.05]) - np.array([pv[0], 0, pv[2]])
            P.tube(rk, [0, 0, 0], list(bp), 0.012)
            bg = rk.add_body(name=f"bogie_{side}", pos=list(bp))
            hinge(bg, f"bogie_{side}", np.radians(30))
            for w in (f"m{side}", f"r{side}"):
                wx, wy = D.wheel_xy(w)
                at = np.array([wx - bx, wy - pv[1], -0.05])
                P.tube(bg, [0, 0, 0], [at[0], 0, at[2] + 0.012], 0.010)
                axle(bg, at, sy)
                wheel(bg, w, at)
        # the differential bar across the deck (it ties the rockers; drawn at rest)
        dx = 0.03
        P.tube(ch, [dx, -(hy - 0.01), z1 + 0.018], [dx, hy - 0.01, z1 + 0.018], 0.008, "titanium")
        P.disc(ch, [dx, 0, z1 + 0.012], 0.02, 0.008, "z", "alu")
        for s in (-1, 1):
            P.tube(ch, [dx, s * (hy - 0.01), z1 + 0.018], [dx, s * (hy + 0.025), z0 + 0.03], 0.005, "titanium")
    else:
        for side, sy in (("l", 1), ("r", -1)):
            mx_, my_ = D.wheel_xy(f"m{side}")
            hub = np.array([mx_, my_, D.wheel_r])
            P.chamfer(ch, [mx_, sy * (D.body[3] + 0.01), D.wheel_r + 0.03], [0.03, 0.012, 0.03], 0.006, "titanium")
            P.tube(ch, [mx_, sy * D.body[3], D.wheel_r + 0.012], [mx_, my_ - sy * (D.wheel_w / 2 + 0.024),
                                                                    D.wheel_r + 0.012], 0.011)
            wheel(ch, f"m{side}", hub)
            for w in (f"f{side}", f"r{side}"):
                wx, _ = D.wheel_xy(w)
                swing_arm(w, wx - np.sign(wx) * 0.13, sag=0.05)

    if D.suspension == "rocker_bogie":                        # the two rockers turn equal and opposite
        eq = spec.add_equality(name="differential")
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1, eq.name2 = "rocker_l", "rocker_r"
        eq.data[0:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
        eq.solref = [0.01, 1.0]

    wmax = D.v_max / D.wheel_r
    for w in D.wheels:
        a = spec.add_actuator(name=f"drive_{w}", target=f"wheel_{w}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=dr["kv"])
        a.ctrlrange = [-1.15 * wmax, 1.15 * wmax]
        a.ctrllimited = True
        a.forcerange = [-dr["torque"], dr["torque"]]
        a.forcelimited = True

    if D.arm:
        mount = ch.add_body(name="arm_mount", pos=list(D.arm_mount))
        P.chamfer(mount, [0.015, 0, -0.006], [0.05, 0.06, 0.006], 0.003, "alu")          # base plate
        back = x1 - D.arm_mount[0]                            # the bracket back to the body's face
        P.chamfer(mount, [back / 2, 0, -0.03], [abs(back) / 2 + 0.004, 0.05, 0.03], 0.004, "titanium")
        spec.attach(arm_spec(), frame=mount.add_frame(), prefix="arm_")
    return spec


def _rtg(P, body, a, b, r):
    """A radioisotope power unit: a finned cylinder from a to b on a bracket."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    P.cyl(body, a, b, r, "dark")
    u = (b - a) / np.linalg.norm(b - a)
    for e in (a, b):                                          # end caps
        P.cyl(body, e - 0.003 * u, e + 0.003 * u, r * 1.08, "alu")
    e1 = np.array([0, 1.0, 0])
    e2 = np.cross(u, e1)
    for k in range(8):
        ang = k * np.pi / 4
        rk = np.cos(ang) * e1 + np.sin(ang) * e2
        t = np.cross(u, rk)
        c = (a + b) / 2 + rk * (r + 0.009)
        P.box(body, c, [np.linalg.norm(b - a) / 2 * 0.85, 0.010, 0.0015], "black", quat=_quat(np.column_stack([u, rk, t])))


def _hex_panel(P, body, c, r, thick, quat=None, mat="paint"):
    ang = np.arange(6) * np.pi / 3
    v = [(r * np.cos(a), r * np.sin(a), z) for a in ang for z in (-thick, thick)]
    n = P._mesh(P.name("hex"), v)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=n, pos=list(c), material=mat,
                  **({"quat": quat} if quat is not None else {}), contype=0, conaffinity=0, group=2, density=0)


def _scout(P, ch, D, cam, deck):
    """A compact prospector (VIPER-class): body-mounted solar panels on both sides and the back,
    a lamp mast, a direct-to-Earth dish, a wheel camera that watches the sinkage."""
    x0, x1, y0, y1, z0, z1 = D.body
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    for s in (-1, 1):                                          # side solar arrays, proud of the body, cells outward
        Rs = np.column_stack([[1.0, 0, 0], [0, 0, -s], [0, s, 0]])
        P.solar_panel(ch, [cx - 0.01, s * (hy + 0.004), cz + 0.01], [hx - 0.025, hz * 0.85, 0.003], R=Rs)
    Rr = np.column_stack([[0, 1.0, 0], [0, 0, -1.0], [-1.0, 0, 0]])
    P.solar_panel(ch, [x0 - 0.004, 0, cz + 0.01], [hy - 0.02, hz * 0.85, 0.003], R=Rr)
    nx, ny, nz = D.navcam_pos
    for s in (-1, 1):                                          # LED lamps either side of the navcam head
        P.chamfer(ch, [nx - 0.03, ny + s * 0.075, nz - 0.004], [0.022, 0.016, 0.016], 0.004, "dark")
        P.disc(ch, [nx - 0.0085, ny + s * 0.075, nz - 0.004], 0.011, 0.0015, "x", "lamp")
    # direct-to-Earth dish on a post, rear-left of the deck, facing up and back
    P.cyl(ch, [x0 + 0.05, 0.05, deck], [x0 + 0.05, 0.05, deck + 0.07], 0.006, "titanium")
    P.disc(ch, [x0 + 0.05, 0.05, deck + 0.07], 0.012, 0.008, "y", "dark")
    P.dish(ch, [x0 + 0.045, 0.05, deck + 0.08], 0.045, quat=_quat(_frame_along([-0.6, 0.2, 0.75])))
    P.coil(ch, [x0 + 0.04, -0.06, deck], 0.008, 0.06, 5, 0.002, mat="alu")                # UHF helix
    P.cyl(ch, [x0 + 0.04, -0.06, deck], [x0 + 0.04, -0.06, deck + 0.064], 0.003, "dark")
    # wheel camera: on a bracket over the front-left wheel, looking at the wheel's contact patch
    wx = D.wheel_x - 0.03
    P.tube(ch, [wx, hy, z0 + 0.04], [wx, hy + 0.035, z0 + 0.05], 0.004, "titanium")
    P.chamfer(ch, [wx, hy + 0.04, z0 + 0.05], [0.012, 0.01, 0.01], 0.003, "black")
    # rear hazcam on a bracket behind the rear wheels
    rx = D.tail - 0.01
    P.box(ch, [(x0 + rx + 0.02) / 2, 0, z0 + 0.05], [max((x0 - rx - 0.02) / 2, 0.001), 0.01, 0.005], "titanium")
    P.chamfer(ch, [rx + 0.015, 0, z0 + 0.05], [0.013, 0.05, 0.013], 0.004, "black")
    for s in (-1, 1):
        P.disc(ch, [rx + 0.003, s * 0.035, z0 + 0.05], 0.006, 0.0015, "x", "glass")
    cam(ch, "hazcam_rear", (rx, 0.0, z0 + 0.05), D.hazcam_pitch, D.hazcam_fovy, back=True)


def _scientist(P, ch, D, cam, deck):
    """A remote-sensing rover (Perseverance-class): SuperCam-style laser/telescope head and a
    colour stereo pair on the mast, a gimballed high-gain antenna, a UHF helix, an RTG at the back."""
    x0, x1, y0, y1, z0, z1 = D.body
    nx, ny, nz = D.navcam_pos
    mx = x1 - 0.025
    face = nx - 0.004
    tz = nz + 0.062                                            # the SuperCam-style head on top of the camera head
    P.chamfer(ch, [face - 0.06, ny, tz], [0.06, 0.042, 0.036], 0.008, "paint")
    P.disc(ch, [face - 0.004, ny, tz], 0.028, 0.004, "x", "alu")
    P.disc(ch, [face - 0.001, ny, tz], 0.022, 0.0015, "x", "glass")                    # telescope aperture
    P.box(ch, [mx, ny, nz + 0.03], [0.012, 0.03, 0.008], "dark")                         # bracket between them
    P.camera_pair(ch, [face - 0.012, ny, nz + 0.034], 0.09, 0.008, mat="black")         # colour zoom pair
    cam(ch, "rmi", (nx + 0.002, ny, tz), np.radians(15), 8.0)
    # high-gain antenna: a hexagonal flat panel on a two-axis gimbal, deck left
    gx, gy = x0 + 0.13, 0.065
    P.cyl(ch, [gx, gy, deck], [gx, gy, deck + 0.05], 0.009, "titanium")
    P.disc(ch, [gx, gy, deck + 0.05], 0.016, 0.01, "z", "dark")
    _hex_panel(P, ch, [gx, gy + 0.005, deck + 0.075], 0.055, 0.006,
               quat=_quat(_frame_along([-0.35, 0.25, 0.9])))
    P.coil(ch, [x0 + 0.06, -0.07, deck], 0.01, 0.075, 6, 0.0022, mat="alu")              # UHF helix antenna
    P.cyl(ch, [x0 + 0.06, -0.07, deck], [x0 + 0.06, -0.07, deck + 0.078], 0.004, "dark")
    P.chamfer(ch, [x0 + 0.18, -0.03, deck + 0.01], [0.025, 0.025, 0.008], 0.003, "alu")  # calibration target
    # RTG: a finned cylinder at the back, tilted up, on a bracket
    a, b = [x0 - 0.02, 0, z1 - 0.03], [x0 - 0.13, 0, z1 + 0.05]
    P.box(ch, [x0 - 0.012, 0, z1 - 0.03], [0.012, 0.03, 0.03], "titanium")
    _rtg(P, ch, a, b, 0.032)


def _sampler(P, spec, ch, D, cam, deck):
    """A drilling rover (Rosalind Franklin-class) with an arm: deployed solar wings on both sides,
    a panoramic camera bar on the mast, a drill tower at the back, a sample carousel, the SO-101 at
    the front-left corner (it folds back along the left of the deck, clear of the mast on the right)."""
    x0, x1, y0, y1, z0, z1 = D.body
    hy = (y1 - y0) / 2
    nx, ny, nz = D.navcam_pos
    face = nx - 0.004
    # panoramic camera bar: two wide-angle cameras at the ends of an optical bench, a high-res one
    for s in (-1, 1):
        P.chamfer(ch, [face - 0.03, ny + s * 0.085, nz], [0.028, 0.026, 0.026], 0.005, "paint")
        P.disc(ch, [face - 0.002, ny + s * 0.085, nz], 0.012, 0.002, "x", "glass")
    P.chamfer(ch, [face - 0.025, ny, nz + 0.04], [0.025, 0.02, 0.016], 0.004, "dark")
    P.disc(ch, [face - 0.001, ny, nz + 0.04], 0.009, 0.0015, "x", "glass")
    # solar wings: two panels per side, deployed level with the deck over the rear wheels
    for s in (-1, 1):
        yi, yo = s * (hy + 0.012), s * (hy + 0.15)
        P.box(ch, [x0 + 0.17, s * (hy + 0.006), z1 - 0.008], [0.14, 0.006, 0.006], "titanium")     # hinge line
        for k, (xa, xb) in enumerate(((x0 + 0.02, x0 + 0.165), (x0 + 0.175, x0 + 0.32))):
            P.solar_panel(ch, [(xa + xb) / 2, (yi + yo) / 2, z1 - 0.004], [(xb - xa) / 2 - 0.003, abs(yo - yi) / 2, 0.003])
    # sample carousel, deck right-rear: a fixed drum and a turning lid with CAROUSEL_SLOTS core tubes
    cx_, cy_ = x0 + 0.12, -0.075
    P.disc(ch, [cx_, cy_, deck + 0.018], 0.055, 0.018, "z", "alu")
    car = ch.add_body(name="carousel", pos=[cx_, cy_, deck + 0.036])
    car.add_joint(name="carousel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.05, armature=1e-3)
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.002, 0], rgba=[1, 1, 1, 0], mass=0.05,
                 contype=0, conaffinity=0, group=3)
    P.disc(car, [0, 0, 0.002], 0.058, 0.003, "z", "paint")
    P.box(car, [0.045, 0, 0.006], [0.012, 0.003, 0.0015], "accent")                        # index mark
    for k in range(CAROUSEL_SLOTS):
        a_ = k * 2 * np.pi / CAROUSEL_SLOTS
        P.disc(car, [0.04 * np.cos(a_), 0.04 * np.sin(a_), 0.006], 0.007, 0.002, "z", "dark")
    a_ = spec.add_actuator(name="carousel", target="carousel", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a_.set_to_position(kp=2.0, kv=0.2)
    # UHF helix antenna mid-deck on the left (clear of the drill tower, where it read as a loose spring)
    P.disc(ch, [x0 + 0.26, 0.11, deck + 0.004], 0.014, 0.004, "z", "dark")
    P.coil(ch, [x0 + 0.26, 0.11, deck + 0.008], 0.009, 0.065, 5, 0.002, mat="alu")
    # drill tower at the back: two rails, a crossbeam, the sliding drill (a real joint) and its camera
    tx = x0 + 0.03 - 0.06
    for s_ in (-1, 1):
        P.box(ch, [tx, s_ * 0.042, (z0 + z1 + 0.16) / 2], [0.008, 0.006, (z1 + 0.16 - z0) / 2], "alu")
    P.chamfer(ch, [tx, 0, z1 + 0.17], [0.022, 0.06, 0.012], 0.004, "paint")
    P.box(ch, [(x0 + tx) / 2, 0, z1 - 0.02], [(x0 - tx) / 2, 0.05, 0.008], "titanium")
    dr_ = ch.add_body(name="drill", pos=[tx, 0, z1 + 0.10])
    dr_.add_joint(name="drill_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-DRILL_TRAVEL, 0.0],
                  damping=2.0, armature=0.01)
    dr_.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.022, 0.032, 0.03], rgba=[1, 1, 1, 0], mass=0.15,
                 contype=0, conaffinity=0, group=3)
    P.chamfer(dr_, [0, 0, 0], [0.022, 0.032, 0.03], 0.005, "paint")
    P.disc(dr_, [0, 0, 0.03], 0.018, 0.006, "z", "dark")                                  # drill motor
    P.box(dr_, [0.023, 0, 0], [0.001, 0.02, 0.012], "accent")
    P.disc(dr_, [0, 0, -0.033], 0.011, 0.003, "z", "titanium")                            # chuck
    # the bit turns in the chuck: string, flutes at the tip (the helix runs from the tip UP), crown
    bit = dr_.add_body(name="drill_bit", pos=[0, 0, 0])
    bit.add_joint(name="drill_spin", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=1e-4, armature=1e-4)
    bit.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[0, 0, -0.04, 0, 0, -0.2], size=[0.008, 0, 0],
                 rgba=[1, 1, 1, 0], mass=0.05, contype=0, conaffinity=0, group=3)
    P.cyl(bit, [0, 0, -0.03], [0, 0, -DRILL_FLUTES[0]], 0.0055, "alu")                    # drill string
    P.cyl(bit, [0, 0, -DRILL_FLUTES[0]], [0, 0, -DRILL_FLUTES[1]], 0.005, "titanium")    # bit core under the flutes
    P.coil(bit, [0, 0, -DRILL_FLUTES[1]], 0.0075, DRILL_FLUTES[1] - DRILL_FLUTES[0], 6, 0.0025, mat="titanium")
    P.disc(bit, [0, 0, -DRILL_FLUTES[1] - 0.002], 0.009, 0.002, "z", "dark")              # cutting crown
    bit.add_geom(name="drill_core", type=mujoco.mjtGeom.mjGEOM_CYLINDER,                  # shown only with a core
                 fromto=[0, 0, -DRILL_FLUTES[1] - 0.004, 0, 0, -DRILL_FLUTES[1] - 0.026], size=[0.0055, 0, 0],
                 rgba=[0.62, 0.46, 0.34, 0], contype=0, conaffinity=0, group=2, density=0)
    dr_.add_geom(name="drill_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, -DRILL_FLUTES[1] - 0.004],
                 size=[0.006, 0, 0], rgba=[1, 1, 1, 0], mass=0.005, contype=0, conaffinity=0, group=3,
                 friction=[0.3, 0.005, 0.0001])                   # touches the ground only while lowering
    a_ = spec.add_actuator(name="drill", target="drill_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a_.set_to_position(kp=200.0, kv=20.0)
    a_.ctrlrange = [-DRILL_TRAVEL, 0.0]
    a_.ctrllimited = True
    a_ = spec.add_actuator(name="drill_spin", target="drill_spin", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a_.set_to_velocity(kv=0.01)
    a_.ctrlrange = [-40.0, 40.0]
    a_.ctrllimited = True
    cam(ch, "drillcam", (x0 + 0.02 - 0.01, 0.0, z0 + 0.02), np.radians(80), 60.0, back=True)


class SkidKinematics:
    """Body twist (vx, wz) in the chassis frame -> four wheel speeds (skid steering).

    Wheel i at r_i rolls at (vx - wz * track_factor * y_i) / R: the rover turns as if its track were
    track_factor times wider (wheels scrub sideways in a turn; measured on flat bedrock). On other
    ground the turn achieved differs, and the rows measure it. vy is impossible and dropped.
    """
    holonomic = False

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, prefix: str, dsg=None):
        self.design = design(dsg)
        mujoco.mj_kinematics(m, d)
        ch = m.body(prefix + CHASSIS).id
        Rc, pc = d.xmat[ch].reshape(3, 3), d.xpos[ch]
        self.names = self.design.wheels
        self.r = np.array([(Rc.T @ (d.xpos[m.body(prefix + f"wheel_{w}").id] - pc))[:2] for w in self.names])
        self.R = float(m.geom_size[m.geom(prefix + "wheel_fl_col").id][0])
        self.drive = [m.actuator(prefix + f"drive_{w}").id for w in self.names]
        self.steer: list = []
        self.acts = list(self.drive)
        self.wmax = float(m.actuator_ctrlrange[self.drive[0], 1]) / 1.15
        self.v_max = self.wmax * self.R
        self.tf = self.design.track_factor
        self.w_max = self.v_max / (self.tf * float(np.max(np.abs(self.r[:, 1]))))
        self._delta = np.zeros(0)                          # no steering state (kept for snapshots)

    def targets(self, vx, vy, wz):
        speed = (vx - wz * self.tf * self.r[:, 1]) / self.R
        s = max(1.0, float(np.max(np.abs(speed))) / self.wmax)
        return speed / s, self._delta

    def command(self, d: mujoco.MjData, vx, vy, wz) -> np.ndarray:
        speed, _ = self.targets(vx, vy, wz)
        d.ctrl[self.drive] = speed
        return speed

    def wheel_speeds(self, vx, vy, wz):
        return self.targets(vx, vy, wz)[0]


def kinematics(m, d, prefix, dsg=None):
    return SkidKinematics(m, d, prefix, dsg)


def arm_joint_ids(m, rid):
    from so101.robot.so101_mujoco import ARM_JOINTS
    return [m.joint(f"{rid}/arm_{j}").id for j in ARM_JOINTS]


def set_arm_pose(m, d, rid, q):
    """Put rover rid's arm at joint pose q (qpos, and the servos' targets so it stays there)."""
    for jid, v in zip(arm_joint_ids(m, rid), q):
        d.qpos[m.jnt_qposadr[jid]] = v
    for i in range(m.nu):
        if m.actuator(i).name.startswith(f"{rid}/arm_"):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]


# the default design's geometry under the old names (scripts written for one rover)
_D = DESIGNS[DEFAULT]
SELF_BOX = _D.self_box
BODY = dict(x=_D.body[0:2], y=_D.body[2:4], z=_D.body[4:6])
R = _D.wheel_r
MASK_R = 0.48                                  # m: what the overhead site camera ignores around any rover
