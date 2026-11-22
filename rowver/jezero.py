"""The Jezero site ("jezero_s1", the user's HiRISE DTM of Jezero Crater) at 1:5, SIM SIDE ONLY.

The rovers are 1:5 Perseverances, so the site is scaled 1:5 too: 200 m -> 40 m, the 0.5 m
heightmap -> 0.1 m (exactly the terrain map's cell), 13.6 m of relief -> 2.7 m, rocks -> 1.5-40 cm.
Slopes are unchanged. Frame: x east, y north, z up; the site centre is the origin.

What is HIDDEN (world.json: "the brain must never read this block or the two .npy maps"): the
terrain classes (bedrock / gravel / soft sand), the 40 slip patches and every slip and sinkage
number. Only the simulator's physics (world.py sets each wheel's contact from the ground under
it), the calibration and check scripts, and the human-facing run summaries read them. A rover
gets what a camera sees and what its own drives do.

Also here: the outpost work zones (chosen automatically on mixed ground) and the station layout
in a zone (on flat spots of the right ground, with graded pads under them, as a real outpost
would level its work areas).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "jezero"
PHYSICS = ASSETS / "physics.json"
SCALE = 0.2
CLASSES = ("bedrock", "gravel", "soft_sand")
ZONE_W, ZONE_H = 7.2, 5.4                 # m: one outpost work zone (4:3, what one overhead camera sees)
TILE_W, TILE_H = ZONE_W, ZONE_H           # every survey tile (zone or traverse corridor) has this footprint

# hidden wheel-ground contact (MuJoCo). Per class a friction; the softness follows the sinkage
# (soft() below); the rolling friction (soft ground resisting the wheel) sets the slip and comes
# from an offline calibration, done at each class's SINK_LEVELS (m, 1:5).
CLASS_CONTACT = {"bedrock": {"traction": 1.00}, "gravel": {"traction": 0.90}, "soft_sand": {"traction": 0.70}}
# (sand deeper than ~18 mm at 1:5 swallows the 5.3 cm wheels on the heightfield: measured 130 mm of
# sinkage and slip 1.0 at a 28 mm setting, so sand sinkage is capped there; its slip range is covered)
SINK_LEVELS = {"bedrock": (0.0,), "gravel": (0.001, 0.004), "soft_sand": (0.010, 0.018)}


def soft(sinkage: float):
    """Contact softness for a sinkage (m): -> (solref, solimp). Measured: 0.08 s / dmin 0.5 sinks a
    1:5 rover ~11 mm, 0.15 s / dmin 0.3 ~22 mm; below 2 mm the contact is hard."""
    s_ = float(max(sinkage, 0.0))
    if s_ < 0.002:
        return [0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0]
    tc = 0.02 + 5.5 * s_
    return [tc, 1.0], [max(0.25, 0.9 - 25 * s_), max(0.6, 0.95 - 12 * s_), 0.03, 0.5, 2.0]


ROCK_CONTACT = {"traction": 1.0, "rolling": 0.0003, "solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]}


# ===================================================================== the site
@dataclass
class Site:
    meta: dict
    hm: np.ndarray                 # (401, 401) heights at 1:5 (m); row 0 = north
    cls: np.ndarray                # (401, 401) uint8 class id          HIDDEN
    patch: np.ndarray              # (401, 401) int16 patch id, -1      HIDDEN
    x_min: float
    y_max: float
    res: float
    rocks: dict                    # name -> (N,) arrays, scaled 1:5
    start: tuple                   # A (x, y, yaw)
    goal: tuple                    # B (x, y)
    patches: list = field(default_factory=list)   # HIDDEN
    graded: np.ndarray | None = None              # the heightmap with the outpost's pads levelled

    @property
    def n(self):
        return self.hm.shape[0]

    @property
    def half(self):
        return (self.n - 1) * self.res / 2

    def ij(self, x, y):
        j = np.clip(np.rint((np.asarray(x) - self.x_min) / self.res).astype(int), 0, self.n - 1)
        i = np.clip(np.rint((self.y_max - np.asarray(y)) / self.res).astype(int), 0, self.n - 1)
        return i, j

    def xy_grid(self):
        j = np.arange(self.n)
        return self.x_min + j * self.res, self.y_max - j * self.res      # x of columns, y of rows

    def height_at(self, x, y, graded=True):
        """Bilinear terrain height (the graded pads included by default)."""
        H = self.graded if (graded and self.graded is not None) else self.hm
        fx = np.clip((np.asarray(x, float) - self.x_min) / self.res, 0, self.n - 1.000001)
        fy = np.clip((self.y_max - np.asarray(y, float)) / self.res, 0, self.n - 1.000001)
        j0, i0 = fx.astype(int), fy.astype(int)
        tx, ty = fx - j0, fy - i0
        return ((1 - tx) * (1 - ty) * H[i0, j0] + tx * (1 - ty) * H[i0, j0 + 1] + (1 - tx) * ty * H[i0 + 1, j0]
                + tx * ty * H[i0 + 1, j0 + 1])

    def slope_deg(self, graded=False):
        H = self.graded if (graded and self.graded is not None) else self.hm
        gy, gx = np.gradient(H, self.res)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    # ------------------------------------------------------------ HIDDEN ground truth
    def cls_at(self, x, y) -> np.ndarray:
        i, j = self.ij(x, y)
        return self.cls[i, j]

    def cls_name(self, x, y) -> str:
        return CLASSES[int(self.cls_at(x, y))]

    def slip_sinkage(self, x, y):
        """-> (class id, slip, sinkage (m, 1:5)) arrays for points: the patch's values inside a patch,
        else the class range's middle (gravel outside every class: the dataset's default)."""
        i, j = self.ij(x, y)
        c = self.cls[i, j].astype(int)
        p = self.patch[i, j].astype(int)
        slip, sink = _CLASS_MID[0][c], _CLASS_MID[1][c]
        if (p >= 0).any():
            P = self._patch_arrays()
            slip = np.where(p >= 0, P[0][np.maximum(p, 0)], slip)
            sink = np.where(p >= 0, P[1][np.maximum(p, 0)], sink)
        return c, slip, sink

    def _patch_arrays(self):
        if not hasattr(self, "_pa"):
            self._pa = (np.array([q["slip"] for q in self.patches]),
                        np.array([q["sinkage_m"] * SCALE for q in self.patches]))
        return self._pa


_CLASS_MID: tuple = ()


@lru_cache(maxsize=1)
def load() -> Site:
    """The site at 1:5 (data only: json + npy with allow_pickle=False)."""
    global _CLASS_MID
    meta = json.loads((ASSETS / "world.json").read_text(encoding="utf-8"))
    hm = np.load(ASSETS / "heightmap.npy", allow_pickle=False).astype(float) * SCALE
    cls = np.load(ASSETS / "terrain_class.npy", allow_pickle=False)
    patch = np.load(ASSETS / "terrain_patches.npy", allow_pickle=False)
    b = meta["bounds"]
    res = meta["heightmap"]["resolution_m"] * SCALE
    R = meta["rocks"]
    rocks = {k: np.array([r[k] for r in R], float) * (SCALE if k in ("x", "y", "z", "radius", "height") else 1.0)
             for k in ("x", "y", "z", "radius", "height", "yaw")}
    rocks["variant"] = np.array([r["variant"] for r in R], int)
    rocks["scale"] = np.array([r["scale"] for r in R], float) * SCALE
    T = meta["terrain"]
    _CLASS_MID = (np.array([np.mean(T["classes"][c]["slip"]) for c in CLASSES]),
                  np.array([np.mean(T["classes"][c]["sinkage_m"]) * SCALE for c in CLASSES]))
    s, g = meta["start"], meta["goal"]
    return Site(meta=meta, hm=hm, cls=cls, patch=patch, x_min=b["x_min"] * SCALE, y_max=b["y_max"] * SCALE,
                res=res, rocks=rocks, start=(s["x"] * SCALE, s["y"] * SCALE, s["yaw"]), goal=(g["x"] * SCALE, g["y"] * SCALE),
                patches=T["patches"])


def class_slip_ranges() -> dict:
    T = load().meta["terrain"]["classes"]
    return {c: tuple(T[c]["slip"]) for c in CLASSES}


def rock_variants() -> list:
    return [ASSETS / "rocks" / Path(v["file"]).name for v in load().meta["rock_variants"]]


def sun_dir() -> list:
    """Direction the sunlight travels (world.json: azimuth from north, clockwise; elevation)."""
    s = load().meta["sun"]
    az, el = np.radians(s["azimuth_deg"]), np.radians(s["elevation_deg"])
    to_sun = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    return (-to_sun).tolist()


# ===================================================================== hidden contact (sim physics)
@lru_cache(maxsize=1)
def _physics():
    if not PHYSICS.exists():
        return None
    return json.loads(PHYSICS.read_text(encoding="utf-8"))


def _roll_for(table, slip):
    s, r = np.asarray(table["slip"]), np.asarray(table["rolling"])
    k = int(np.searchsorted(s, slip))                # first entry with slip >= target
    if k <= 0:
        return float(r[0])
    if k >= len(s):
        return float(r[-1])
    f = (slip - s[k - 1]) / max(s[k] - s[k - 1], 1e-9)   # interpolate in log(rolling)
    return float(np.exp(np.log(r[k - 1]) + f * (np.log(r[k]) - np.log(r[k - 1]))))


def contact_for(cls_id: int, slip: float, sinkage: float) -> dict:
    """MuJoCo contact parameters for a wheel on ground of this class, slip and sinkage (m, 1:5)."""
    name = CLASSES[int(cls_id)]
    lv = SINK_LEVELS[name]
    sk = float(np.clip(sinkage, lv[0], lv[-1]))
    solref, solimp = soft(sk)
    P = _physics()
    if P is None or "levels" not in P["classes"].get(name, {}):      # not calibrated yet
        roll = 0.0003
    else:
        T = P["classes"][name]["levels"]                  # one table per sinkage level
        if len(lv) == 1:
            roll = _roll_for(T[0], slip)
        else:
            k = int(np.clip(np.searchsorted(lv, sk), 1, len(lv) - 1))
            f = (sk - lv[k - 1]) / (lv[k] - lv[k - 1])
            ra, rb = _roll_for(T[k - 1], slip), _roll_for(T[k], slip)
            roll = float(np.exp((1 - f) * np.log(ra) + f * np.log(rb)))
    return {"traction": CLASS_CONTACT[name]["traction"], "rolling": roll, "solref": solref, "solimp": solimp}


# ===================================================================== work zones
@dataclass
class Layout:
    zone: tuple                    # (x0, x1, y0, y1); the garage goes on its levelest edge (garage.py)


def _corridor_dist(site, x, y):
    """Distance from points to the segment A -> B (the traverse)."""
    a, b = np.array(site.start[:2]), np.array(site.goal)
    d = b - a
    t = np.clip(((np.asarray(x) - a[0]) * d[0] + (np.asarray(y) - a[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(np.asarray(x) - (a[0] + t * d[0]), np.asarray(y) - (a[1] + t * d[1]))


def zone_candidates(site: Site):
    """Score every 7.2 x 5.4 m window (0.5 m steps): mixed ground (some soft sand and some bedrock,
    mostly gravel), moderate slopes, few big rocks, away from the A->B traverse."""
    S = site.slope_deg()
    xs, ys = site.xy_grid()
    rk = site.rocks
    big = rk["height"] > 0.035
    out = []
    step = 0.5
    for x0 in np.arange(-site.half + 1.0, site.half - ZONE_W - 1.0, step):
        for y0 in np.arange(-site.half + 1.0, site.half - ZONE_H - 1.0, step):
            j0, j1 = int((x0 - site.x_min) / site.res), int((x0 + ZONE_W - site.x_min) / site.res)
            i0, i1 = int((site.y_max - y0 - ZONE_H) / site.res), int((site.y_max - y0) / site.res)
            c = site.cls[i0:i1, j0:j1]
            fb, fg, fs = [(c == k).mean() for k in range(3)]
            p90 = np.percentile(S[i0:i1, j0:j1], 90)
            if fs < 0.06 or fb < 0.04 or fg < 0.35 or p90 > 13:
                continue
            cx, cy = x0 + ZONE_W / 2, y0 + ZONE_H / 2
            if _corridor_dist(site, cx, cy) < 5.0:
                continue
            nb = int((big & (rk["x"] > x0) & (rk["x"] < x0 + ZONE_W) & (rk["y"] > y0) & (rk["y"] < y0 + ZONE_H)).sum())
            score = -abs(fs - 0.2) * 3 - abs(fb - 0.12) * 3 - 0.05 * max(0, p90 - 8) - 0.01 * max(0, nb - 15)
            out.append((score, (float(x0), float(x0 + ZONE_W), float(y0), float(y0 + ZONE_H))))
    out.sort(key=lambda t: -t[0])
    return out


LAYOUT_VERSION = 4                  # the cache also holds the old station layouts; only "zone" is read
LAYOUTS = ASSETS / "layouts.json"
RAMP = 0.6          # m: a graded pad blends into the natural ground over this (grade)


@lru_cache(maxsize=8)
def zones(n: int = 4) -> tuple:
    """n non-overlapping work zones, best first. Deterministic; cached in assets/jezero/layouts.json."""
    if LAYOUTS.exists():
        c = json.loads(LAYOUTS.read_text(encoding="utf-8"))
        if c.get("version") == LAYOUT_VERSION and len(c["zones"]) >= n:
            return tuple(Layout(zone=tuple(d["zone"])) for d in c["zones"][:n])
    site = load()
    chosen = []
    for score, z in zone_candidates(site):
        if any(not (z[1] + 1 < c.zone[0] or c.zone[1] + 1 < z[0] or z[3] + 1 < c.zone[2] or c.zone[3] + 1 < z[2])
               for c in chosen):
            continue
        chosen.append(Layout(zone=z))
        if len(chosen) == n:
            break
    LAYOUTS.write_text(json.dumps({"version": LAYOUT_VERSION, "zones": [{"zone": L.zone} for L in chosen]}),
                       encoding="utf-8")
    return tuple(chosen)


def _seg_dist(x, y, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(d @ d)
    if L < 1e-12:
        return np.hypot(x - a[0], y - a[1])
    t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / L, 0, 1)
    return np.hypot(x - (a[0] + t * d[0]), y - (a[1] + t * d[1]))


def grade(site: Site, pads) -> np.ndarray:
    """The heightmap with level pads: inside a pad the ground is at its height, then it blends back
    to the natural ground over RAMP."""
    H = site.hm.copy()
    xs, ys = site.xy_grid()
    for poly, r, z in pads:
        pts = np.array(poly)
        k = int(np.ceil((r + RAMP) / site.res)) + 1
        i, j = site.ij(pts[:, 0], pts[:, 1])
        sl = (slice(max(i.min() - k, 0), i.max() + k + 1), slice(max(j.min() - k, 0), j.max() + k + 1))
        X, Y = np.meshgrid(xs[sl[1]], ys[sl[0]])
        d = np.min([_seg_dist(X, Y, pts[a], pts[min(a + 1, len(pts) - 1)]) for a in range(max(len(pts) - 1, 1))], axis=0)
        w = np.clip(1 - (d - r) / RAMP, 0, 1)
        w = w * w * (3 - 2 * w)                        # smooth step
        H[sl] = H[sl] * (1 - w) + z * w
    return H


def corridor_tiles(site: Site, width_tiles=1):
    """Survey tiles (x0, x1, y0, y1) along A -> B: overhead images of the traverse, like the orbital
    images a real rover plans on, with the same 10 cm cells as the work zones."""
    a, b = np.array(site.start[:2]), np.array(site.goal)
    ylo, yhi = min(a[1], b[1]) - 1.0, max(a[1], b[1]) + 1.0
    tiles = []
    y = ylo
    while y < yhi:                                   # tiles are TILE_H tall (along y) ... rotated: use TILE_W along y
        yc = y + TILE_W / 2
        t = np.clip((yc - a[1]) / (b[1] - a[1]), 0, 1)
        xc = a[0] + t * (b[0] - a[0])
        tiles.append((float(xc - TILE_H / 2), float(xc + TILE_H / 2), float(y), float(y + TILE_W)))
        y += TILE_W - 0.4                            # a little overlap
    return tiles
