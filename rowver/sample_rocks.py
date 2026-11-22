"""Loose rocks the sampler's arm picks up: irregular, faceted, dusty, like the float rocks on Jezero's
floor. No boxes, no bricks.

An ordinary weathered stone: a sphere made lumpy by a few low-frequency waves plus fine roughness,
longer than it is wide, with one or two soft breaks and a settled underside; every rock has its own
shape, proportions and size. MuJoCo collides with its convex hull; the visual mesh is the irregular
surface. Sizes stay within the gripper's range: about the lab's grasp size classes (so101 OBJ_TYPES 0
~3 cm and 2 ~3.5 cm long), which is what the arm's learned pick knows. Only the SIZE comes from them.

Where they lie (sim side): a few around each candidate drill site, on the natural ground, resting.
The rovers find them with the hazcam's depth (missions.find_rocks), never from this list.
"""
from __future__ import annotations

import mujoco
import numpy as np
from so101.robot.so101_mujoco import OBJ_TYPES

TONES = {    # Jezero lithologies: (rgba, name)
    "basalt":    (0.29, 0.26, 0.25, 1),
    "sandstone": (0.62, 0.45, 0.32, 1),
    "mudstone":  (0.52, 0.38, 0.29, 1),
    "carbonate": (0.66, 0.58, 0.48, 1),
}
SIZE_CLASSES = (0, 2)          # so101 OBJ_TYPES the rocks come in (bounding sizes)
DENSITY = 2700.0               # kg/m^3 (basalt ~2900, sandstone ~2300)


def _icosphere(level=2):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), np.array(F, int)


def rock_mesh(rng, half_len, height):
    """An ordinary weathered stone: rounded and lumpy, longer than wide, one or two soft breaks,
    a settled flat-ish underside. Length ~2 half_len (+-15 %), height ~height.
    -> (verts (N, 3) centred on the bounding box, faces)."""
    V, F = _icosphere(2)
    n = V.copy()
    r = np.ones(len(V))
    for _ in range(5):                                   # lumps: low-frequency waves on the surface
        k = rng.normal(size=3)
        k *= rng.uniform(1.2, 3.0) / np.linalg.norm(k)
        r += rng.uniform(0.03, 0.08) * np.cos(n @ k + rng.uniform(0, 2 * np.pi))
    for _ in range(4):                                   # chipped, uneven surface
        k = rng.normal(size=3)
        k *= rng.uniform(4.0, 7.0) / np.linalg.norm(k)
        r += rng.uniform(0.015, 0.03) * np.cos(n @ k + rng.uniform(0, 2 * np.pi))
    r += rng.uniform(-0.025, 0.025, len(r))              # per-vertex grit
    V = n * r[:, None]
    for _ in range(rng.integers(1, 3)):                  # one or two soft breaks (rounded, not sharp)
        d = rng.normal(size=3)
        d[2] *= 0.5
        d /= np.linalg.norm(d)
        s_ = V @ d
        off = rng.uniform(0.65, 0.85) * s_.max()
        cut = np.clip(s_ - off, 0, None)
        V -= (cut - 0.35 * cut ** 2 / max(cut.max(), 1e-9))[:, None] * d * 0.85
    base = np.quantile(V[:, 2], 0.15)                    # settled: the underside flattens
    V[:, 2] = np.where(V[:, 2] < base, base + 0.3 * (V[:, 2] - base), V[:, 2])
    V -= (V.max(0) + V.min(0)) / 2
    ext = V.max(0) - V.min(0)
    L = 2 * half_len * rng.uniform(0.9, 1.15)
    V *= [L / ext[0], L * rng.uniform(0.62, 0.85) / ext[1], height * rng.uniform(0.85, 1.1) / ext[2]]
    return V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3)    # unshared vertices: faceted shading


def add_rock(spec, name, pos, yaw, size_class, rng, tone=None):
    """A free rock body. -> (item dict, the half height) for World.items."""
    o = OBJ_TYPES[size_class]
    V, F = rock_mesh(rng, o["size"], o["height"])
    tone = tone or rng.choice(list(TONES))
    mat = f"{name}_mat"
    base = np.array(TONES[tone][:3]) * rng.uniform(0.9, 1.1)
    spec.add_material(name=mat, rgba=[*np.clip(base, 0, 1), 1], specular=0.05, shininess=0.05)
    m = spec.add_mesh(name=f"{name}_mesh")
    m.uservert = V.ravel().tolist()
    m.userface = F.ravel().tolist()
    b = spec.worldbody.add_body(name=name, pos=list(pos), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    b.add_freejoint(name=f"{name}_free")
    vol = _hull_volume(V)
    b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh", material=mat,
               mass=float(np.clip(DENSITY * vol, 0.02, 0.12)), friction=[0.85, 0.01, 0.001], condim=4)
    ext = V.max(0) - V.min(0)
    return dict(name=name, kind="rock", type=int(size_class), tone=tone, size_m=float(max(ext[0], ext[1]))), \
        float(-V[:, 2].min())


def _hull_volume(V):
    from scipy.spatial import ConvexHull
    return float(ConvexHull(V).volume)


def scatter(sites, rng, per_site=3, r=(0.30, 0.55)):
    """Rock positions around candidate drill sites: (x, y, yaw, size class) per rock."""
    out = []
    for p in sites:
        for k in range(per_site):
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(*r)
            out.append((p[0] + d * np.cos(a), p[1] + d * np.sin(a), rng.uniform(-np.pi, np.pi),
                        int(rng.choice(SIZE_CLASSES))))
    return out
