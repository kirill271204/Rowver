"""Hidden geology of the site (SIM SIDE ONLY) and the instruments that measure it.

The brain never reads these fields. A rover learns them only through its own instruments, and
through the rows other rovers uploaded to the dock:
  photochem   0..1  the science value of a spot ("photochemistry potential": preserved organics
                    and alteration minerals). High on layered bedrock and on fresh crater rims
                    (freshly exposed rock), low on wind-blown sand; a smooth regional field on
                    top (buried delta layers), different in every zone.
  mineral     0 basalt, 1 carbonate, 2 clay, 3 aeolian sand (a category per spot)
  hardness    0..1  how hard the rock is to drill (drill torque); sand is soft but a core
                    collapses in it; the hardest rock stalls the bit.

Instruments (noise and range limits as on a real payload):
  spectrometer   the scientist's mast spectrometer (SuperCam-like LIBS / Raman / VISIR): reads
                 photochem and mineral of a target 0.5-4 m away; noise grows with range.
  drill          the sampler's core drill, at the spot under its drill: torque (N m) and whether
                 the core came out intact.
The scout's trafficability readings (sinkage, rolling resistance) come from its wheels while it
drives (physics), not from here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import jezero as J

MINERALS = ("basalt", "carbonate", "clay", "sand")
SPEC_RANGE = (0.5, 4.0)                      # m
SPEC_NOISE = (0.03, 0.02)                    # photochem noise sd = a + b x range (m)
TORQUE = (0.8, 3.0, 0.15)                    # N m: base + slope x hardness, noise sd


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Blob:
    x: float
    y: float
    s: float
    a: float


class Geology:
    """Deterministic fields for one world (site, its craters and its zone), plus noisy sensors."""

    def __init__(self, world, seed: int = 0):
        self.w = world
        self.site = world.site
        self.craters = list(world.craters)
        x0, x1, y0, y1 = world.zone
        g = np.random.default_rng(1000 + 7 * seed + int(abs(x0) * 13 + abs(y0) * 7))
        pad = 3.0                                   # the regional field reaches a bit outside the zone
        self.blobs = [Blob(float(g.uniform(x0 - pad, x1 + pad)), float(g.uniform(y0 - pad, y1 + pad)),
                           float(g.uniform(0.8, 2.0)), float(g.choice([-1, 1]) * g.uniform(0.20, 0.40)))
                      for _ in range(10)]
        self.hblobs = [Blob(float(g.uniform(x0 - pad, x1 + pad)), float(g.uniform(y0 - pad, y1 + pad)),
                            float(g.uniform(1.5, 3.0)), float(g.uniform(-0.2, 0.2))) for _ in range(5)]
        X, Y = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 30))
        self._f0 = float(np.mean(self._field(self.blobs, X, Y)))     # the regional field is centred on the zone
        self._h0 = float(np.mean(self._field(self.hblobs, X, Y)))
        self.rng = np.random.default_rng(5000 + seed)   # instrument noise

    # ---------------------------------------------------------------- hidden fields
    def _cls(self, x, y):
        if self.site is None:
            return np.zeros(np.shape(x), int)
        return self.site.cls_at(x, y).astype(int)

    def _field(self, blobs, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return sum(b.a * np.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.s * b.s)) for b in blobs)

    def _rim(self, x, y):
        """0..1: on a crater rim (freshly exposed rock), more for fresh craters."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape)
        for c in self.craters:
            rr = np.hypot(x - c.x, y - c.y) / c.r
            out = np.maximum(out, (1.0 if c.fresh else 0.5) * np.exp(-((rr - 1.05) / 0.15) ** 2))
        return out

    def photochem(self, x, y):
        c = self._cls(x, y)
        base = np.array([0.45, 0.25, 0.05])[c]
        v = base + (self._field(self.blobs, x, y) - self._f0) + 0.30 * self._rim(x, y)
        v = np.where(c == 2, np.minimum(v, 0.15), v)          # sand never preserves much
        return np.clip(v, 0.0, 1.0)

    def hardness(self, x, y):
        c = self._cls(x, y)
        base = np.array([0.70, 0.40, 0.08])[c]
        return np.clip(base + (self._field(self.hblobs, x, y) - self._h0) + 0.1 * self._rim(x, y), 0.0, 1.0)

    def mineral(self, x, y):
        c, p = self._cls(x, y), self.photochem(x, y)
        m = np.where(c == 2, 3, np.where(c == 0, np.where(p > 0.6, 1, 0), np.where(p > 0.35, 2, 0)))
        return m.astype(int)

    def p_sample(self, x, y):
        """Chance the core comes out intact: soft ground collapses, the hardest rock stalls the bit."""
        h = self.hardness(x, y)
        return np.clip(0.10 + 0.80 * _sig((h - 0.30) / 0.06) - 0.45 * _sig((h - 0.88) / 0.03), 0.02, 0.98)

    # ---------------------------------------------------------------- instruments
    def spectrometer(self, rover_xy, target_xy):
        """-> {"photochem_score", "mineral_class", "spec_range"} or None (target out of range)."""
        d = float(np.hypot(target_xy[0] - rover_xy[0], target_xy[1] - rover_xy[1]))
        if not SPEC_RANGE[0] <= d <= SPEC_RANGE[1]:
            return None
        x, y = target_xy
        p = float(self.photochem(x, y)) + self.rng.normal(0, SPEC_NOISE[0] + SPEC_NOISE[1] * d)
        m = int(self.mineral(x, y))
        if self.rng.random() > 0.97 - 0.04 * d:                 # misread grows with range
            m = int(self.rng.integers(0, len(MINERALS)))
        return {"photochem_score": float(np.clip(p, 0, 1)), "mineral_class": m, "spec_range": d}

    def drill(self, x, y):
        """-> {"drill_torque_nm", "sample_success"} at the spot under the drill."""
        h = float(self.hardness(x, y))
        tq = TORQUE[0] + TORQUE[1] * h + self.rng.normal(0, TORQUE[2])
        ok = bool(self.rng.random() < float(self.p_sample(x, y)))
        return {"drill_torque_nm": float(max(tq, 0.0)), "sample_success": float(ok)}

    # ---------------------------------------------------------------- summaries (sim side)
    def truth(self, x, y) -> dict:
        return {"photochem": float(self.photochem(x, y)), "mineral": MINERALS[int(self.mineral(x, y))],
                "hardness": float(self.hardness(x, y)), "p_sample": float(self.p_sample(x, y)),
                "ground": J.CLASSES[int(self._cls(x, y))]}
