r"""Fit the frozen PCA that turns DINOv2-S patch features (384-dim) into the 8 navcam codes.

Unlabelled: navcam frames from random poses across the whole Jezero site (no terrain class, no
slip, nothing hidden is read). Patches showing ground within the navcam's range are kept, and
the rover's own body is masked as in navcam.py. Run once; the result is frozen so every code
column means the same thing in every row ever written (refitting means starting a fresh table).

  .venv\Scripts\python.exe -m rowver.tools.fit_navcam_pca [--frames 240]
"""
from __future__ import annotations

import argparse

import numpy as np

from rowver import mars_rover as P
from rowver import world as W
from rowver.navcam import H_PX, W_PX
from rowver.rover import Rover
from rowver.vision import N_CODES, PATCH, PCA_FILE, Encoder
from so101.perception.camera import RGBDCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--range", type=float, default=3.5)
    a = ap.parse_args()
    w = W.build(("r1",), gravity="mars", zone=0)
    rv = Rover(w, "r1")
    cam = RGBDCamera(w.m, w.d, "r1/navcam", W_PX, H_PX)
    enc = Encoder(need_pca=False)
    site = w.site
    rng = np.random.default_rng(0)
    S = site.slope_deg(graded=True)
    xs, ys = site.xy_grid()
    ok_cells = np.argwhere(S < 12)
    ok_cells = ok_cells[(np.abs(xs[ok_cells[:, 1]]) < site.half - 3) & (np.abs(ys[ok_cells[:, 0]]) < site.half - 3)]
    feats = []
    gh, gw = H_PX // PATCH, W_PX // PATCH
    for k in range(a.frames):
        i, j = ok_cells[rng.integers(len(ok_cells))]
        rv.teleport(float(xs[j]), float(ys[i]), float(rng.uniform(-np.pi, np.pi)))
        rv.drive(0, 0, 0, 0.3)
        rgb, depth = cam.capture()
        Pw = cam.points(depth)
        cpos, _ = cam.pose
        x, y, _, R = rv.true_pose()
        loc = (Pw - w.d.xpos[rv.ch]) @ R
        x0, x1, y0, y1, zu = P.SELF_BOX
        own = (loc[..., 0] > x0) & (loc[..., 0] < x1) & (loc[..., 1] > y0) & (loc[..., 1] < y1) & (loc[..., 2] > zu)
        good = ~own & (np.linalg.norm(Pw - cpos, axis=-1) < a.range)
        frac = good.reshape(gh, PATCH, gw, PATCH).mean((1, 3))   # a patch counts if most of it is ground in range
        f = enc.patches([rgb])[0]
        feats.append(f[frac > 0.8])
        if (k + 1) % 40 == 0:
            print(f"  {k + 1} frames, {sum(len(x) for x in feats)} patches", flush=True)
    X = np.vstack(feats).astype(np.float64)
    mu = X.mean(0)
    U, s_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = s_ ** 2 / (s_ ** 2).sum()
    C = Vt[:N_CODES]
    PCA_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PCA_FILE, mean=mu.astype(np.float32), components=C.astype(np.float32),
             explained=ev[:N_CODES].astype(np.float32), n_patches=np.int64(len(X)))
    print(f"PCA on {len(X)} patches from {a.frames} frames: the 8 codes explain "
          f"{100 * ev[:N_CODES].sum():.0f}% of the variance ({', '.join(f'{100 * e:.0f}' for e in ev[:N_CODES])}%)")
    print(f"wrote {PCA_FILE}")
    cam.close()


if __name__ == "__main__":
    main()
