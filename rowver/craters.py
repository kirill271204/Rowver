"""Impact craters on the Jezero site (1:5): bowls with a raised rim, a sandy floor and rim boulders.

The same craters for every run of a zone (placed from a fixed seed per zone): what one rover
learns about them stays true for the next one.

Where
  in the work zone   2 small ones (0.6-1.0 m across) in the open ground between the station groups
  around the zone    3 medium ones (1.0-1.8 m) within 4 m of it
  on the traverse    4 (1.0-2.0 m) near the straight line A -> B: some cut into it, so a rover has to
                     go around; their rims stay at least 1 m apart, so there is always a way round
  scenery            10 large ones (1.6-5 m) elsewhere on the site, away from the zone and the traverse
  Never on a graded pad, a station, a start slot, the camp props, the item store or A / B.

Shape (r: radius, d: depth below the old ground, h: rim height; rr = distance / r)
  inside    z = (d + h) rr^2 - d             a paraboloid bowl, from -d at the centre to +h at the rim
  outside   z = h (rr^-3 - 2.5^-3) / (1 - 2.5^-3)   the rim and its ejecta, gone by 2.5 r
  fresh craters are deep (d = 0.18 x diameter: walls of ~35 deg, more than the rover can climb),
  old ones are shallow (d = 0.07 x diameter: ~14 deg walls, drivable but slow). Rim h = 0.04 x diameter.
Hidden ground truth (sim physics only, never seen by a rover): the floor (rr < 0.65) is soft sand
(wind-blown fill), the walls and rim gravel. The orthoimage is darkened on the floor and lightened
on the rim, and dataset rocks inside a crater are removed; boulders are scattered on the rim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import jezero as J

FRESH_DD, OLD_DD, RIM_DD = 0.18, 0.07, 0.04          # depth / diameter, rim height / diameter
EJECTA = 2.5                                          # the rim's ejecta is gone at EJECTA x r
SAND, GRAVEL = J.CLASSES.index("soft_sand"), J.CLASSES.index("gravel")


@dataclass(frozen=True)
class Crater:
    x: float
    y: float
    r: float
    depth: float
    rim: float
    where: str                                        # zone / near / traverse / scenery

    @property
    def fresh(self) -> bool:
        return self.depth > 0.12 * 2 * self.r


def profile(rr, c: Crater):
    """Height change at relative distance rr (array) from crater c's centre."""
    rr = np.asarray(rr, float)
    out = c.rim * (np.maximum(rr, 1.0) ** -3 - EJECTA ** -3) / (1 - EJECTA ** -3)
    out = np.where(rr < 1.0, (c.depth + c.rim) * rr * rr - c.depth, out)
    return np.where(rr < EJECTA, out, 0.0)


# ------------------------------------------------------------------ placement
def keepouts(site, extra=()) -> list:
    """(x, y, radius) circles no crater (with its ejecta) may touch: the traverse's ends + extra
    (the garage and its apron)."""
    return [(site.start[0], site.start[1], 1.5), (site.goal[0], site.goal[1], 1.5)] + list(extra)


def _free(x, y, r, keep, placed, gap):
    for kx, ky, kr in keep:
        if np.hypot(x - kx, y - ky) < kr + 1.3 * r + 0.3:
            return False
    for c in placed:
        if np.hypot(x - c.x, y - c.y) < r + c.r + gap:
            return False
    return True


def plan(site, zrect, zone_index: int, keep: list) -> list:
    """The zone's craters (deterministic for the zone)."""
    rng = np.random.default_rng(7000 + int(zone_index))
    x0, x1, y0, y1 = zrect
    lim = site.half - 2.0
    a, b = np.array(site.start[:2]), np.array(site.goal)
    ab = b - a
    nrm = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    placed: list = []

    def add(where, n, rlo, rhi, sample, gap, fresh_share):
        for k in range(n):
            fresh = k < int(round(n * fresh_share))
            for _ in range(3000):
                r = float(rng.uniform(rlo, rhi))
                x, y = sample(r)
                if not (-lim < x < lim and -lim < y < lim):
                    continue
                if _free(x, y, r, keep, placed, gap):
                    dd = FRESH_DD if fresh else OLD_DD
                    placed.append(Crater(float(x), float(y), r, dd * 2 * r, RIM_DD * 2 * r, where))
                    break

    def in_zone(r):
        return rng.uniform(x0 + r + 0.3, x1 - r - 0.3), rng.uniform(y0 + r + 0.3, y1 - r - 0.3)

    def near(r):
        while True:
            x, y = rng.uniform(x0 - 4, x1 + 4), rng.uniform(y0 - 4, y1 + 4)
            if not (x0 - r < x < x1 + r and y0 - r < y < y1 + r):
                return x, y

    def on_route(r):
        t = rng.uniform(0.12, 0.88)
        off = rng.choice([-1, 1]) * rng.uniform(0.2, 1.1) * (1 + r)
        p = a + t * ab + off * nrm
        return p[0], p[1]

    def scenery(r):
        while True:
            x, y = rng.uniform(-lim, lim, 2)
            p = np.array([x, y])
            t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
            route = np.linalg.norm(p - (a + t * ab))
            if route > r * 1.3 + 2.5 and not (x0 - 3 - r < x < x1 + 3 + r and y0 - 3 - r < y < y1 + 3 + r):
                return x, y

    add("zone", 2, 0.3, 0.5, in_zone, 0.6, 0.5)
    add("near", 3, 0.5, 0.9, near, 0.8, 0.67)
    add("traverse", 4, 0.5, 1.0, on_route, 1.0, 0.5)
    add("scenery", 10, 0.8, 2.5, scenery, 1.0, 0.6)
    return placed


# ------------------------------------------------------------------ applying them
def _window(site, c: Crater, reach):
    xs, ys = site.xy_grid()
    j = np.nonzero(np.abs(xs - c.x) <= reach)[0]
    i = np.nonzero(np.abs(ys - c.y) <= reach)[0]
    if not len(i) or not len(j):
        return None
    X, Y = np.meshgrid(xs[j], ys[i])
    return (slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)), np.hypot(X - c.x, Y - c.y) / c.r


def carve(site, H: np.ndarray, craters: list) -> np.ndarray:
    """H with the craters added to the ground (a copy)."""
    H = H.copy()
    for c in craters:
        w = _window(site, c, EJECTA * c.r)
        if w is not None:
            sl, rr = w
            H[sl] += profile(rr, c)
    return H


def ground_truth(cls: np.ndarray, patch: np.ndarray, site, craters: list):
    """(class map, patch map) with sandy floors and gravel walls and rims (copies; sim side only)."""
    cls, patch = cls.copy(), patch.copy()
    for c in craters:
        w = _window(site, c, 1.2 * c.r)
        if w is None:
            continue
        sl, rr = w
        inside = rr < 1.15
        cls[sl] = np.where(rr < 0.65, SAND, np.where(inside, GRAVEL, cls[sl]))
        patch[sl] = np.where(inside, -1, patch[sl])
    return cls, patch


def paint(img: np.ndarray, site, craters: list) -> np.ndarray:
    """The orthoimage (rows north first, as in terrain_albedo.png) with darker floors and lighter rims."""
    out = img.astype(np.float32)
    hgt, wid = img.shape[:2]
    span = 2 * site.half
    rng = np.random.default_rng(11)
    for c in craters:
        reach = 2.2 * c.r
        u0 = int(max(0, (c.x - reach - site.x_min) / span * wid))
        u1 = int(min(wid, (c.x + reach - site.x_min) / span * wid + 1))
        v0 = int(max(0, (site.y_max - (c.y + reach)) / span * hgt))
        v1 = int(min(hgt, (site.y_max - (c.y - reach)) / span * hgt + 1))
        if u1 <= u0 or v1 <= v0:
            continue
        u, v = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
        x = site.x_min + (u + 0.5) / wid * span
        y = site.y_max - (v + 0.5) / hgt * span
        rr = np.hypot(x - c.x, y - c.y) / c.r
        ang = np.arctan2(y - c.y, x - c.x)
        streaks = 1 + 0.03 * np.sin(ang * rng.integers(9, 15) + rng.uniform(0, 6.3))
        f = np.ones_like(rr)
        f = np.where(rr < 0.6, 0.80, f)
        f = np.where((rr >= 0.6) & (rr < 0.95), 0.80 + 0.20 * (rr - 0.6) / 0.35, f)
        f = np.where((rr >= 0.95) & (rr < 1.15), 1.10, f)
        f = np.where((rr >= 1.15) & (rr < 2.2), (1.0 + 0.07 * (2.2 - rr) / 1.05) * streaks, f)
        tint = np.where(rr[..., None] < 0.6, np.array([1.0, 0.93, 0.88], np.float32), 1.0)  # dust: redder floors
        out[v0:v1, u0:u1] *= (f[..., None] * tint)
    return np.clip(out, 0, 255).astype(np.uint8)


def rocks(site, craters: list, rk: dict, zone_index: int) -> dict:
    """The site's rocks with those inside a crater removed and boulders added on the rims."""
    rng = np.random.default_rng(9000 + int(zone_index))
    keep = np.ones(len(rk["x"]), bool)
    for c in craters:
        keep &= np.hypot(rk["x"] - c.x, rk["y"] - c.y) > 1.15 * c.r
    out = {k: v[keep] for k, v in rk.items()}
    donors = np.nonzero((rk["height"] > 0.015) & (rk["height"] < 0.06))[0]
    if not len(donors):
        return out
    add = {k: [] for k in rk}
    for c in craters:
        for _ in range(int(3 + 4 * c.r)):
            i = int(rng.choice(donors))
            a, d = rng.uniform(0, 2 * np.pi), rng.uniform(1.0, 1.5) * c.r
            x, y = c.x + d * np.cos(a), c.y + d * np.sin(a)
            embed = rk["z"][i] - float(site.height_at(rk["x"][i], rk["y"][i], graded=False))
            for k in rk:
                add[k].append(rk[k][i])
            add["x"][-1], add["y"][-1] = x, y
            add["z"][-1] = float(site.height_at(x, y)) + embed      # site.graded holds the craters by now
            add["yaw"][-1] = rng.uniform(-np.pi, np.pi)
    return {k: np.concatenate([out[k], np.asarray(add[k], dtype=out[k].dtype)]) for k in rk}
