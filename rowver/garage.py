"""The fleet's docking station: a 3-bay garage with a solar roof, the only structure on the site.

  layout   one bay per rover, side by side, open front. A rover parks NOSE IN (its arm and
           cameras face the back wall, where the charging contacts are) and leaves by reversing
           straight out onto the apron, then turning. Bay 0.95 m wide x 1.25 m deep, eaves at
           0.95 m (the scientist's mast head tops at ~0.85 m).
  roof     a mono-pitch slab falling toward the open front, covered in solar panels.
  bays     painted bay lines, a status light over each bay, charging contacts on the back wall,
           and a sample tray on the floor of the sampler's bay (its rim collides, so a rock stays in).
  physics  walls and dividers collide (and the navcam's planner knows them, navcam.infrastructure);
           the roof, panels and fittings are visual only.

The garage stands just outside the work zone on its levelest edge, opening into it; its footprint
and a 1.6 m apron are graded flat. The frame: origin at the middle of the open front on the floor,
+x out of the garage, bays along y. The geometry is the sim's; the rovers only know their bay's
park pose and apron point (the base's own map of its garage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .rover_parts import Parts

BAY_W = 0.95        # m between dividers
DEPTH = 1.25        # m from the open front to the back wall
EAVES = 0.95        # m: wall height at the front (the roof rises toward the back)
PITCH = np.radians(6.0)
WALL = 0.02         # half thickness of walls and dividers
APRON = 1.6         # m of graded apron in front of the bays
OUT = 1.0           # m: a bay's apron point beyond the open front (where a rover turns)
NOSE_GAP = 0.30     # m between a parked rover's nose and the back wall
TRAY = (0.075, 0.095, 0.015)   # sample tray: inner half length (x), half width (y), rim height


@dataclass
class Garage:
    xy: tuple                       # origin: the middle of the open front
    yaw: float                      # +x of the garage frame (out of the bays)
    z: float                        # floor height (the graded pad)
    n: int = 3
    park: dict = field(default_factory=dict)      # rid -> (x, y, yaw) nose in
    apron: dict = field(default_factory=dict)     # rid -> (x, y) on the bay's axis, OUT m beyond the front
    tray: tuple | None = None                     # (x, y) of the sample tray's centre (world)

    # ------------------------------------------------------------ frames
    def to_world(self, p):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        p = np.asarray(p, float)
        return np.array([self.xy[0] + c * p[0] - s * p[1], self.xy[1] + s * p[0] + c * p[1]])

    def to_local(self, xy):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(xy, float) - self.xy
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    @property
    def width(self) -> float:
        return self.n * BAY_W

    def bay_y(self, k) -> float:
        return (k - (self.n - 1) / 2) * BAY_W

    def inside(self, xy, margin=0.0) -> bool:
        lx, ly = self.to_local(xy)
        return bool(-DEPTH - margin <= lx <= margin and abs(ly) <= self.width / 2 + margin)

    # ------------------------------------------------------------ site preparation
    def pad(self):
        """The graded pad (jezero.grade's (polyline, radius, z)): footprint + apron, a sausage along
        the garage's axis."""
        r = self.width / 2 + 0.15
        a, b = self.to_world([-DEPTH - 0.15 + r, 0.0]), self.to_world([APRON - r, 0.0])
        return ([tuple(a), tuple(b)], r, self.z)

    def keepouts(self) -> list:
        """(x, y, r) circles that cover the footprint and the apron (no crater, no loose rock)."""
        out = []
        for lx in np.arange(-DEPTH, APRON + 0.01, 0.5):
            for ly in np.arange(-self.width / 2, self.width / 2 + 0.01, 0.5):
                out.append((*self.to_world([lx, ly]), 0.45))
        return out


def place(site, zone, designs, rids, nose, reach=None) -> Garage:
    """Deterministic: the garage on the levelest edge of the zone, its front 0.4 m inside the zone,
    opening inward. designs: rid -> RoverDesign (nose: design -> its frontmost x) for the park poses;
    reach: design -> x of the arm's work point ahead of the chassis (the tray goes there)."""
    x0, x1, y0, y1 = zone
    S = site.slope_deg(graded=False)
    best = None
    n = len(rids)
    W = n * BAY_W
    for edge, inward in (("w", 0.0), ("e", np.pi), ("s", np.pi / 2), ("n", -np.pi / 2)):
        along = (y0 + W / 2 + 0.3, y1 - W / 2 - 0.3) if edge in "we" else (x0 + W / 2 + 0.3, x1 - W / 2 - 0.3)
        for t in np.arange(along[0], along[1] + 1e-6, 0.25):
            if edge == "w":
                o = (x0 + 0.4, t)
            elif edge == "e":
                o = (x1 - 0.4, t)
            elif edge == "s":
                o = (t, y0 + 0.4)
            else:
                o = (t, y1 - 0.4)
            g = Garage(xy=o, yaw=inward, z=0.0, n=n)
            pts = np.array([g.to_world([lx, ly]) for lx in np.linspace(-DEPTH - 0.1, APRON, 12)
                            for ly in np.linspace(-W / 2 - 0.1, W / 2 + 0.1, 10)])
            if (np.abs(pts) > site.half - 1.0).any():
                continue
            h = site.height_at(pts[:, 0], pts[:, 1], graded=False) if _takes_graded(site) else site.height_at(pts[:, 0], pts[:, 1])
            i, j = site.ij(pts[:, 0], pts[:, 1])
            score = float(np.ptp(h)) + 0.01 * float(np.max(S[i, j])) + 0.3 * float(np.mean(site.cls0[i, j] == 2)
                                                                               if hasattr(site, "cls0") else 0.0)
            if best is None or score < best[0]:
                best = (score, g, float(np.median(h)))
    if best is None:
        raise ValueError(f"no room for a {n}-bay garage around zone {zone}")
    _, g, z = best
    g.z = z
    for k, rid in enumerate(rids):
        d = DEPTH - NOSE_GAP - nose(designs[rid])          # park: nose NOSE_GAP from the back wall
        c = g.to_world([-d, g.bay_y(k)])
        g.park[rid] = (float(c[0]), float(c[1]), float(g.yaw + np.pi))
        a = g.to_world([OUT, g.bay_y(k)])
        g.apron[rid] = (float(a[0]), float(a[1]))
        if designs[rid].arm:                               # the sample tray: under the arm's work point
            tx = -d - (reach(designs[rid]) if reach else 0.0)
            tx = min(max(tx, -DEPTH + WALL + TRAY[0] + 0.01), -d - nose(designs[rid]) - TRAY[0] + 0.03)
            g.tray = tuple(float(v) for v in g.to_world([tx, g.bay_y(k)]))
    return g


def _takes_graded(site) -> bool:
    import inspect
    try:
        return "graded" in inspect.signature(site.height_at).parameters
    except (TypeError, ValueError):
        return False


# ===================================================================== geometry
def build(spec: mujoco.MjSpec, g: Garage, ground):
    """The garage's geoms in the world (a body at the garage frame)."""
    P = Parts(spec, (0.85, 0.45, 0.15, 1), group=1)
    G = mujoco.mjtGeom
    b = spec.worldbody.add_body(name="garage", pos=[g.xy[0], g.xy[1], g.z], quat=[np.cos(g.yaw / 2), 0, 0, np.sin(g.yaw / 2)])
    W = g.width
    back_h = EAVES + DEPTH * np.tan(PITCH)

    def solid(name, c, half):
        """A colliding wall piece (invisible: the look is drawn on top of it)."""
        b.add_geom(name=name, type=G.mjGEOM_BOX, pos=list(c), size=list(half), rgba=[0, 0, 0, 0], group=3,
                   friction=[0.8, 0.005, 0.0001], condim=3)

    def wall_h(x):                                          # the roof's underside above x (local)
        return EAVES + (-x) * np.tan(PITCH)

    # ---- floor: a sintered-regolith slab with painted bay lines and a lip at the front
    P.box(b, [-DEPTH / 2, 0, 0.004], [DEPTH / 2 + WALL, W / 2 + WALL, 0.004], "regolith")
    P.box(b, [0.03, 0, 0.003], [0.03, W / 2 + WALL, 0.003], "concrete")
    for k in range(g.n):
        y = g.bay_y(k)
        for s in (-1, 1):
            P.box(b, [-DEPTH / 2 + 0.05, y + s * (BAY_W / 2 - 0.09), 0.0085], [DEPTH / 2 - 0.08, 0.008, 0.0006], "lines")
        for j in range(3):                                  # chevrons on the apron edge of each bay
            x = 0.12 + 0.09 * j
            for s in (-1, 1):
                q = _qz(s * np.radians(35))
                P.box(b, [x, y + s * 0.07, 0.0065], [0.012, 0.085, 0.0006], "lines", quat=q)

    # ---- walls: back wall, two side walls, dividers between the bays
    solid("garage_back", [-DEPTH - WALL, 0, back_h / 2], [WALL, W / 2 + 2 * WALL, back_h / 2])
    P.box(b, [-DEPTH - WALL, 0, back_h / 2], [WALL, W / 2 + 2 * WALL, back_h / 2], "paint")
    for s in (-1, 1):
        y = s * (W / 2 + WALL)
        solid(f"garage_side_{'l' if s > 0 else 'r'}", [-DEPTH / 2, y, (EAVES + back_h) / 4], [DEPTH / 2, WALL, (EAVES + back_h) / 4])
        _wall_panel(P, b, y, s, back_h)
    for k in range(1, g.n):
        y = (k - g.n / 2) * BAY_W
        solid(f"garage_div{k}", [-DEPTH / 2 - 0.02, y, (EAVES + back_h) / 4 - 0.01], [DEPTH / 2 - 0.02, WALL * 0.75,
                                                                                    (EAVES + back_h) / 4 - 0.01])
        _wall_panel(P, b, y, 0, back_h, thick=WALL * 0.75, inset=0.04)
    # columns at the open front (the roof beam rests on them)
    for y in [-(W / 2 + WALL)] + [(k - g.n / 2) * BAY_W for k in range(1, g.n)] + [W / 2 + WALL]:
        P.chamfer(b, [-0.02, y, EAVES / 2], [0.025, 0.03, EAVES / 2], 0.006, "alu")

    # ---- roof: a slab falling toward the front, a fascia beam, the solar array on top
    L = (DEPTH + 0.18) / np.cos(PITCH)
    q = _qy(PITCH)
    rc = [-DEPTH / 2 + 0.06, 0, (EAVES + back_h) / 2 + 0.015]
    P.box(b, rc, [L / 2, W / 2 + 0.10, 0.015], "paint", quat=q)
    P.box(b, [0.07, 0, EAVES + 0.02], [0.02, W / 2 + 0.10, 0.035], "paint2")            # fascia
    P.box(b, [0.091, 0, EAVES + 0.02], [0.0015, W / 2 + 0.09, 0.006], "accent")          # its stripe
    nrow, ncol = 3, 2 * g.n
    pw, pl = (W + 0.16) / ncol, (L - 0.10) / nrow
    up = np.array([np.sin(PITCH), 0, np.cos(PITCH)])
    for i in range(nrow):
        for j in range(ncol):
            u = -L / 2 + 0.05 + (i + 0.5) * pl                # along the slope
            v = -(W + 0.16) / 2 + (j + 0.5) * pw
            c = np.array(rc) + np.array([np.cos(PITCH) * u, v, -np.sin(PITCH) * u]) + up * 0.022
            Rm = np.array([[np.cos(PITCH), 0, np.sin(PITCH)], [0, 1, 0], [-np.sin(PITCH), 0, np.cos(PITCH)]])
            P.solar_panel(b, c, [pl / 2 - 0.008, pw / 2 - 0.008, 0.004], R=Rm)
    for j in range(ncol + 1):                               # the array's support rails
        v = -(W + 0.16) / 2 + j * pw
        P.box(b, np.array(rc) + up * 0.012 + [0, v, 0], [L / 2 - 0.04, 0.004, 0.006], "alu", quat=q)

    # ---- per bay: charging contacts on the back wall, a status light over the front, the tray
    for k in range(g.n):
        y = g.bay_y(k)
        P.chamfer(b, [-DEPTH + 0.004, y, 0.24], [0.012, 0.11, 0.07], 0.004, "black")
        for s in (-1, 1):
            P.box(b, [-DEPTH + 0.018, y + s * 0.045, 0.24], [0.004, 0.018, 0.02], "copper")
        P.chamfer(b, [-DEPTH + 0.006, y, 0.42], [0.008, 0.06, 0.035], 0.003, "dark")   # bay controller
        P.disc(b, [-DEPTH + 0.015, y + 0.035, 0.43], 0.006, 0.002, "x", "led")
        P.sphere(b, [0.075, y, EAVES - 0.025], 0.016, "led")                            # status light
        P.disc(b, [0.075, y, EAVES - 0.008], 0.022, 0.006, "z", "dark")
    if g.tray is not None:
        _tray(b, g, P)

    # ---- cable trunk from the array down the side wall into the ground
    s = 1
    P.cyl(b, [-DEPTH + 0.05, s * (W / 2 + 2 * WALL + 0.02), back_h], [-DEPTH + 0.05, s * (W / 2 + 2 * WALL + 0.02), 0.02],
          0.012, "dark")


def _wall_panel(P, b, y, s, back_h, thick=WALL, inset=0.0):
    """A wall's look: a white panel with a lower band of gold insulation and vertical ribs."""
    h_front, h_back = EAVES, back_h
    xs = np.linspace(-DEPTH, -inset, 6)
    for x0_, x1_ in zip(xs[:-1], xs[1:]):
        xm = (x0_ + x1_) / 2
        h = EAVES + (-xm) * np.tan(PITCH)
        P.box(b, [xm, y, h / 2], [(x1_ - x0_) / 2, thick, h / 2], "paint")
        P.box(b, [x1_, y, h / 2], [0.004, thick + 0.003, h / 2], "paint2")             # rib
    P.box(b, [-(DEPTH + inset) / 2, y, 0.09], [(DEPTH - inset) / 2, thick + 0.0015, 0.07], "gold")
    _ = (h_front, h_back, s)


def _tray(b, g, P):
    """The sample tray: a thin floor plate with a low rim (the rim collides, so rocks stay in)."""
    c = g.to_local(g.tray)
    hx, hy, rim = TRAY
    G = mujoco.mjtGeom
    P.box(b, [c[0], c[1], 0.002], [hx + 0.006, hy + 0.006, 0.002], "alu")
    for name, dc, half in (("x0", [-hx - 0.003, 0], [0.003, hy + 0.006]), ("x1", [hx + 0.003, 0], [0.003, hy + 0.006]),
                           ("y0", [0, -hy - 0.003], [hx, 0.003]), ("y1", [0, hy + 0.003], [hx, 0.003])):
        pos = [c[0] + dc[0], c[1] + dc[1], rim / 2]
        b.add_geom(name=f"tray_{name}", type=G.mjGEOM_BOX, pos=pos, size=[half[0], half[1], rim / 2], material="alu",
                   friction=[0.8, 0.005, 0.0001], condim=3, group=1)


def _qz(a):
    return [np.cos(a / 2), 0, 0, np.sin(a / 2)]


def _qy(a):
    return [np.cos(a / 2), 0, np.sin(a / 2), 0]
