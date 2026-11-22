"""Parts for the fleet's rovers (visual only): materials and procedural meshes.

Every part here is a visual geom (group 2, no collision, no mass): the physics of a rover (body box,
wheel ellipsoids, joints, masses) lives in mars_rover.py and is not touched by how it looks.

  materials   flight-hardware finishes: white thermal paint, gold and silver MLI blankets, black
              anodised aluminium and radiators, bare and machined aluminium, titanium tubing,
              solar cells (a cell-grid texture), camera glass, carbon-fibre mast tubes
  meshes      wheel      a machined aluminium wheel: thin tread with chevron grousers, a dished
                         face with curved spokes and a hub (one mesh per wheel size)
              chamfer    a box with bevelled edges (body shells, camera heads, housings)
              dish       a parabolic high-gain antenna reflector
              coil       a helical spring (coil-over shocks) / UHF helix antenna
"""
from __future__ import annotations

import mujoco
import numpy as np

G = mujoco.mjtGeom
VIS = dict(contype=0, conaffinity=0, group=2, density=0)

FINISH = {   # name: (rgba, specular, shininess, reflectance)
    "paint":    ((0.86, 0.85, 0.82, 1), 0.25, 0.25, 0.0),     # white thermal paint
    "paint2":   ((0.74, 0.73, 0.70, 1), 0.20, 0.20, 0.0),     # shaded panels
    "gold":     ((0.83, 0.62, 0.24, 1), 0.90, 0.70, 0.08),    # gold MLI blanket
    "silver":   ((0.78, 0.78, 0.80, 1), 0.90, 0.75, 0.06),    # silver MLI
    "black":    ((0.07, 0.07, 0.08, 1), 0.35, 0.40, 0.0),     # black anodised / radiator
    "dark":     ((0.18, 0.18, 0.19, 1), 0.30, 0.30, 0.0),
    "alu":      ((0.70, 0.71, 0.73, 1), 0.80, 0.60, 0.05),    # machined aluminium
    "titanium": ((0.47, 0.47, 0.49, 1), 0.70, 0.55, 0.03),    # suspension tubes
    "carbon":   ((0.13, 0.13, 0.14, 1), 0.55, 0.65, 0.02),    # mast tube
    "glass":    ((0.02, 0.03, 0.05, 1), 1.00, 0.95, 0.30),    # lenses
    "copper":   ((0.72, 0.42, 0.22, 1), 0.70, 0.55, 0.02),    # springs, cable trays
    "tread":    ((0.62, 0.63, 0.65, 1), 0.60, 0.40, 0.02),    # wheel skin (dusty aluminium)
    "lamp":     ((0.95, 0.93, 0.80, 1), 0.90, 0.90, 0.10),
    "regolith": ((0.55, 0.38, 0.28, 1), 0.05, 0.05, 0.0),     # sintered-regolith floor slab
    "concrete": ((0.50, 0.46, 0.43, 1), 0.08, 0.08, 0.0),
    "lines":    ((0.92, 0.78, 0.20, 1), 0.20, 0.20, 0.0),     # painted bay lines
}
EMISSIVE = {"led": ((0.25, 0.95, 0.40, 1), 0.8)}              # status lights (rgba, emission)


def materials(spec: mujoco.MjSpec, accent=(0.85, 0.45, 0.15, 1)):
    """Add the finishes (+ "accent": a rover's identification stripe, + "solar": cells) to spec."""
    for name, (rgba, spec_, shin, refl) in FINISH.items():
        spec.add_material(name=name, rgba=list(rgba), specular=spec_, shininess=shin, reflectance=refl)
    spec.add_material(name="accent", rgba=list(accent), specular=0.4, shininess=0.4)
    for name, (rgba, em) in EMISSIVE.items():
        spec.add_material(name=name, rgba=list(rgba), emission=em)
    tex = spec.add_texture(name="cells", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.06, 0.09, 0.22], rgb2=[0.10, 0.14, 0.32], width=64, height=64)
    m = spec.add_material(name="solar", texrepeat=[6, 6], texuniform=True, specular=0.9, shininess=0.8, reflectance=0.05)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name   # (a textures=[...] kwarg corrupts memory on attach)


# ===================================================================== mesh helpers
def _box_tris(c, h, R=np.eye(3)):
    """Vertices (8) and triangles (12) of an oriented box (centre c, half sizes h, rotation R)."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    v = (s * h) @ R.T + c
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3),
         (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    return v, np.array(f)


class _Mesh:
    def __init__(self):
        self.v, self.f = [], []

    def add(self, v, f):
        n = sum(len(x) for x in self.v)
        self.v.append(np.asarray(v, float))
        self.f.append(np.asarray(f, int) + n)

    def ring_strip(self, prof, n, phase=0.0):
        """Revolve a polyline profile [(r, y), ...] about the y axis; every segment its own strip
        (sharp creases between segments)."""
        a = phase + np.linspace(0, 2 * np.pi, n + 1)[:-1]
        for (r0, y0), (r1, y1) in zip(prof[:-1], prof[1:]):
            v = [(r0 * np.cos(t), y0, r0 * np.sin(t)) for t in a] + [(r1 * np.cos(t), y1, r1 * np.sin(t)) for t in a]
            f = []
            for i in range(n):
                j = (i + 1) % n
                f += [(i, j, n + j), (i, n + j, n + i)]
            self.add(v, f)

    def box(self, c, h, R=np.eye(3)):
        self.add(*_box_tris(np.asarray(c, float), np.asarray(h, float), R))

    def done(self):
        return np.vstack(self.v), np.vstack(self.f)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def wheel_mesh(R, W, grousers=24, spokes=6):
    """A machined aluminium wheel about the local y axis (the wheel's axle): tread, chevron
    grousers, dished outer face with curved spokes, hub. -> (verts, faces), skin and details
    separately: (skin (v, f), details (v, f))."""
    skin = _Mesh()
    w = W / 2
    prof = [(0.0, -w * 0.35), (0.30 * R, -w * 0.55), (0.92 * R, -w * 0.92), (0.985 * R, -w), (0.985 * R, w),
            (0.92 * R, w * 0.92), (0.30 * R, w * 0.55), (0.0, w * 0.35)]
    skin.ring_strip(prof, 48)
    det = _Mesh()
    g = 0.05 * R                                    # grouser height
    phi = np.radians(28)                            # chevron sweep
    ey = np.array([0, 1.0, 0])
    for k in range(grousers):                        # chevrons: two ridges meeting at the tread's centre line
        a = 2 * np.pi * k / grousers
        for side in (-1, 1):
            ang = a + side * (w / 2 * np.tan(phi)) / R
            er = np.array([np.cos(ang), 0, np.sin(ang)])
            et = np.array([-np.sin(ang), 0, np.cos(ang)])
            d = np.cos(phi) * ey * side + np.sin(phi) * et
            t = np.cross(er, d)
            c = (0.985 * R + g / 2) * er + side * w / 2 * ey
            det.box(c, [g / 2, w / 2 / np.cos(phi), 0.0035], np.column_stack([er, d, t]))
    for side in (-1, 1):                             # curved spokes on both faces
        y = side * (w * 0.62)
        for k in range(spokes):
            a0 = 2 * np.pi * k / spokes
            pts = [(0.22 * R + u * 0.68 * R, a0 + 0.5 * u * u) for u in np.linspace(0, 1, 5)]
            for (r0, t0), (r1, t1) in zip(pts[:-1], pts[1:]):
                p0 = np.array([r0 * np.cos(t0), y, r0 * np.sin(t0)])
                p1 = np.array([r1 * np.cos(t1), y, r1 * np.sin(t1)])
                L = np.linalg.norm(p1 - p0)
                ang = np.arctan2(p1[2] - p0[2], p1[0] - p0[0])
                det.box((p0 + p1) / 2, [L / 2 + 0.002, 0.004, 0.035 * R], _rot_y(-ang))
    det.ring_strip([(0.0, -w * 0.75), (0.2 * R, -w * 0.75), (0.2 * R, w * 0.75), (0.0, w * 0.75)], 20)
    return skin.done(), det.done()


def chamfer_box(hx, hy, hz, c):
    """24 vertices of a box with bevelled edges (bevel c): the convex hull is the shape."""
    v = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v += [(sx * hx, sy * (hy - c), sz * (hz - c)), (sx * (hx - c), sy * hy, sz * (hz - c)),
                      (sx * (hx - c), sy * (hy - c), sz * hz)]
    return np.array(v)


def dish_mesh(r, depth, n=32, rings=6):
    """A parabolic reflector opening along +z (front and back faces)."""
    m = _Mesh()
    prof = [(r * t, depth * t * t) for t in np.linspace(0, 1, rings + 1)]
    v, f = [], []
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    for rr, zz in prof:
        v += [(rr * np.cos(t), rr * np.sin(t), zz) for t in a]
    for i in range(rings):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            f += [(p, q, q + n), (p, q + n, p + n), (p, q + n, q), (p, p + n, q + n)]
    m.add(v, f)
    return m.done()


def coil_mesh(r, length, turns, wire, n=10):
    """A helix along +z (springs, the UHF helix antenna) as a chain of thin boxes."""
    m = _Mesh()
    steps = int(turns * 16)
    pts = [np.array([r * np.cos(2 * np.pi * turns * t), r * np.sin(2 * np.pi * turns * t), length * t])
           for t in np.linspace(0, 1, steps + 1)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d = p1 - p0
        L = np.linalg.norm(d)
        x = d / L
        z = np.array([0, 0, 1.0]) - x[2] * x
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        m.box((p0 + p1) / 2, [L / 2 + wire * 0.3, wire / 2, wire / 2], np.column_stack([x, y, z]))
    return m.done()


# ===================================================================== builder
class Parts:
    """Adds visual parts to the bodies of one rover spec (meshes are created once per spec)."""

    def __init__(self, spec: mujoco.MjSpec, accent, group=2):
        """group: 2 for a rover's parts, 1 for the site's props (the garage)."""
        self.spec = spec
        self.vis = {**VIS, "group": group}
        materials(spec, accent)
        self._n = 0

    def _mesh(self, name, v, f=None):
        m = self.spec.add_mesh(name=name)
        m.uservert = np.asarray(v, float).ravel().tolist()
        if f is not None:
            m.userface = np.asarray(f, int).ravel().tolist()
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        return name

    def name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    # ---------------------------------------------------------------- primitives with finishes
    def box(self, body, c, half, mat, quat=None, **kw):
        body.add_geom(type=G.mjGEOM_BOX, pos=list(c), size=list(half), material=mat,
                      **({"quat": list(quat)} if quat is not None else {}), **self.vis, **kw)

    def cyl(self, body, a, b, r, mat):
        body.add_geom(type=G.mjGEOM_CYLINDER, fromto=[*a, *b], size=[r, 0, 0], material=mat, **self.vis)

    def tube(self, body, a, b, r, mat="titanium"):
        body.add_geom(type=G.mjGEOM_CAPSULE, fromto=[*a, *b], size=[r, 0, 0], material=mat, **self.vis)

    def sphere(self, body, c, r, mat):
        body.add_geom(type=G.mjGEOM_SPHERE, pos=list(c), size=[r, 0, 0], material=mat, **self.vis)

    def disc(self, body, c, r, h, axis, mat):
        """A short cylinder (thickness 2h) with its axis along `axis` (x, y or z)."""
        q = {"x": [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], "y": [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0],
             "z": [1, 0, 0, 0]}[axis]
        body.add_geom(type=G.mjGEOM_CYLINDER, pos=list(c), size=[r, h, 0], quat=q, material=mat, **self.vis)

    def chamfer(self, body, c, half, bevel, mat, quat=None):
        n = self._mesh(self.name("chamfer"), chamfer_box(*half, bevel))
        body.add_geom(type=G.mjGEOM_MESH, meshname=n, pos=list(c), material=mat,
                      **({"quat": list(quat)} if quat is not None else {}), **self.vis)

    def mesh(self, body, v, f, mat, pos=(0, 0, 0), quat=None, base="part"):
        n = self._mesh(self.name(base), v, f)
        body.add_geom(type=G.mjGEOM_MESH, meshname=n, pos=list(pos), material=mat,
                      **({"quat": list(quat)} if quat is not None else {}), **self.vis)

    # ---------------------------------------------------------------- assemblies
    def wheel(self, body, R, W, hub_mat="alu"):
        (vs, fs), (vd, fd) = wheel_mesh(R, W)
        self.mesh(body, vs, fs, "tread", base="wheel_skin")
        self.mesh(body, vd, fd, hub_mat, base="wheel_detail")

    def camera_pair(self, body, c, base, lens_r, mat="dark"):
        """A stereo pair of lenses on the +x face at c, separated by `base`, with sunshades."""
        for s in (-1, 1):
            p = np.array(c) + [0, s * base / 2, 0]
            self.disc(body, p + [0.004, 0, 0], lens_r * 1.35, 0.006, "x", mat)          # sunshade barrel
            self.disc(body, p + [0.011, 0, 0], lens_r, 0.0015, "x", "glass")

    def solar_panel(self, body, c, half, R=None, frame=0.006):
        """A panel of cells in a thin aluminium frame. half = (a, b, thickness) in the panel's frame;
        R: the panel frame's axes as columns (cells face local +z); level (cells up) if None."""
        a, b, t = half
        R = np.eye(3) if R is None else np.asarray(R, float)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        c = np.asarray(c, float)
        self.box(body, c, [a, b, t], "alu", q.tolist())
        self.box(body, c + R @ [0, 0, t * 0.6], [a - frame, b - frame, t * 0.7], "solar", q.tolist())

    def dish(self, body, c, r, quat=None):
        v, f = dish_mesh(r, 0.3 * r)
        self.mesh(body, v, f, "paint", pos=c, quat=quat, base="dish")

    def coil(self, body, a, r, length, turns, wire, mat="copper", quat=None):
        v, f = coil_mesh(r, length, turns, wire)
        self.mesh(body, v, f, mat, pos=a, quat=quat, base="coil")
