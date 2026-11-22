"""A rover's own map of the site, built from its stereo navcam, kept for the whole run and shared at the dock.

The navcam's LocalMap is a 6 m window that follows the rover and forgets what falls out of it. The SiteMap
keeps everything the rover has looked at, at [map] cell (10 cm) over the whole site, as running sums:

  views           how many navcam frames measured the cell
  z, z2           mean height and its spread inside the cell (roughness)
  zmin, zmax      relief inside the cell
  val, sat, tex   brightness, saturation, texture of the ground in the colour frame
  e0..e7          the frozen DINOv2 codes (what the ground looks like)

features(x, y) -> the site-row columns M_FEATS (NaN where no rover looked): m_views, m_relief, m_step (largest
height jump to a neighbour cell), m_slope (deg, plane through the 3 x 3 cells), m_rough, m_val, m_sat, m_tex,
m_e0..m_e7. Dock: upload_delta() gives what the rover added since its last sync; load(dock_map) starts a
sortie from the dock's merged map (the briefing): that is how the scientist inherits the scout's eyes.
"""
from __future__ import annotations

import numpy as np

N_CODES = 8
M_FEATS = ["m_views", "m_relief", "m_step", "m_slope", "m_rough", "m_val", "m_sat", "m_tex"] + \
          [f"m_e{k}" for k in range(N_CODES)]
SUMS = ("n", "z", "z2", "val", "sat", "tex") + tuple(f"e{k}" for k in range(N_CODES))


class SiteMap:
    def __init__(self, x0, x1, y0, y1, cell=0.10):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.nx, self.ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        sh = (self.ny, self.nx)
        self.s = {k: np.zeros(sh) for k in SUMS}
        self.zmin, self.zmax = np.full(sh, np.inf), np.full(sh, -np.inf)
        self.base = None                                   # the sums at the last dock sync (for the delta)
        self.sync()

    # ---------------------------------------------------------------- building
    def add_local(self, lm, mask):
        """Fold the LocalMap cells in `mask` (measured this frame) into the site map."""
        iy, ix = np.nonzero(mask & np.isfinite(lm.z))
        if not len(iy):
            return 0
        x = lm.x0 + (ix + 0.5) * lm.cell
        y = lm.y0 + (iy + 0.5) * lm.cell
        cx = np.floor((x - self.x0) / self.cell).astype(int)
        cy = np.floor((y - self.y0) / self.cell).astype(int)
        ok = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        iy, ix, cx, cy = iy[ok], ix[ok], cx[ok], cy[ok]
        z = lm.z[iy, ix]
        vals = {"n": np.ones(len(z)), "z": z, "z2": z * z}
        for k in ("val", "sat", "tex"):
            a = getattr(lm, k)[iy, ix]
            vals[k] = np.nan_to_num(a)
        for k in range(N_CODES):
            vals[f"e{k}"] = np.nan_to_num(lm.E[iy, ix, k])
        for k, v in vals.items():
            np.add.at(self.s[k], (cy, cx), v)
        np.minimum.at(self.zmin, (cy, cx), z)
        zm = np.where(np.isfinite(lm.zmax[iy, ix]), lm.zmax[iy, ix], z)
        np.maximum.at(self.zmax, (cy, cx), zm)
        return len(z)

    # ---------------------------------------------------------------- dock
    def sync(self):
        self.base = {k: v.copy() for k, v in self.s.items()}

    def upload_delta(self) -> dict:
        d = {k: self.s[k] - self.base[k] for k in SUMS}
        d["zmin"], d["zmax"] = self.zmin.copy(), self.zmax.copy()
        self.sync()
        return d

    def add_delta(self, d: dict):
        for k in SUMS:
            self.s[k] += d[k]
        self.zmin = np.minimum(self.zmin, d["zmin"])
        self.zmax = np.maximum(self.zmax, d["zmax"])

    def load(self, other: "SiteMap"):
        for k in SUMS:
            self.s[k] = other.s[k].copy()
        self.zmin, self.zmax = other.zmin.copy(), other.zmax.copy()
        self.sync()

    @property
    def cells(self) -> int:
        return int((self.s["n"] > 0).sum())

    # ---------------------------------------------------------------- reading
    def features(self, x, y) -> dict:
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        cx = np.floor((x - self.x0) / self.cell).astype(int)
        cy = np.floor((y - self.y0) / self.cell).astype(int)
        inside = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        cx, cy = np.clip(cx, 0, self.nx - 1), np.clip(cy, 0, self.ny - 1)
        n = self.s["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            zm = self.s["z"] / n                                       # mean height per cell (NaN unseen)
            out = {"m_views": np.where(inside, n[cy, cx], 0.0)}
            seen = inside & (n[cy, cx] > 0)
            nn = np.maximum(n[cy, cx], 1)
            out["m_relief"] = np.where(seen, self.zmax[cy, cx] - self.zmin[cy, cx], np.nan)
            out["m_rough"] = np.where(seen, np.sqrt(np.maximum(self.s["z2"][cy, cx] / nn - (self.s["z"][cy, cx] / nn) ** 2, 0)),
                                      np.nan)
            for k in ("val", "sat", "tex"):
                out[f"m_{k}"] = np.where(seen, self.s[k][cy, cx] / nn, np.nan)
            for k in range(N_CODES):
                out[f"m_e{k}"] = np.where(seen, self.s[f"e{k}"][cy, cx] / nn, np.nan)
            # 3 x 3 neighbourhood: the largest step to a seen neighbour, and the slope of the plane through them
            steps, slopes = np.full(len(x), np.nan), np.full(len(x), np.nan)
            off = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
            for i in np.flatnonzero(seen):
                pts = []
                for a, b in off:
                    j, k = cy[i] + a, cx[i] + b
                    if 0 <= j < self.ny and 0 <= k < self.nx and n[j, k] > 0:
                        pts.append((b * self.cell, a * self.cell, zm[j, k]))
                P = np.array(pts)
                if len(P) >= 2:
                    steps[i] = float(np.max(np.abs(P[:, 2] - zm[cy[i], cx[i]])))
                if len(P) >= 3:
                    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
                    g = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
                    slopes[i] = float(np.degrees(np.arctan(np.hypot(g[0], g[1]))))
            out["m_step"], out["m_slope"] = steps, slopes
        return out
