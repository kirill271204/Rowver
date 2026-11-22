"""Goals for the local driver (rover_body.py). There is no site map and no path planner on board.

  approach_point(park)     where the learned drive ends before a station; scripted docking does the rest
  orbital_route(site, a, b)
      the "ground team" plan for a long traverse, made ONCE before the drive from the orbital
      elevation model (HiRISE DTM: slopes only; no terrain class, no slip, nothing hidden):
      waypoints about every 3 m, like the rough route the Perseverance team draws on orbital
      images. Between them every driving decision is the rover's own (navcam + TabPFN).
"""
from __future__ import annotations

import heapq

import numpy as np

SQ2 = float(np.sqrt(2))
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def approach_point(park, cfg):
    x, y, yaw = park
    if yaw is None:                                  # a plain waypoint: drive onto it
        return (x, y)
    a = cfg("nav.approach")
    return (x - a * np.cos(yaw), y - a * np.sin(yaw))


def orbital_route(site, a, b, cell=1.0, spacing=3.0, max_slope=18.0, margin=1):
    """Waypoints from a to b on `cell` blocks of the orbital DTM (the ground as it is, craters and
    graded pads included): cost per metre 1 + (slope / 8 deg)^2, blocks whose steepest 10 cm cell is
    over max_slope avoided (a crater's wall). The last waypoint is b."""
    S = site.slope_deg(graded=True)
    k = max(1, int(round(cell / site.res)))
    n = S.shape[0] // k
    blk = S[:n * k, :n * k].reshape(n, k, n, k)
    mean, worst = blk.mean((1, 3)), blk.max((1, 3))
    from scipy.ndimage import binary_dilation
    steep = binary_dilation(worst > max_slope, iterations=margin)   # a block's margin: the rover's half-width
    cost = 1.0 + (mean / 8.0) ** 2 + np.where(steep, 50.0, 0.0)
    x_of = lambda j: site.x_min + (j + 0.5) * k * site.res            # noqa: E731  (column j -> x)
    y_of = lambda i: site.y_max - (i + 0.5) * k * site.res            # noqa: E731  (row i, north first -> y)

    def cell_of(p):
        j = int(np.clip((p[0] - site.x_min) / (k * site.res), 0, n - 1))
        i = int(np.clip((site.y_max - p[1]) / (k * site.res), 0, n - 1))
        return i, j

    s, g = cell_of(a), cell_of(b)
    dist, came, q = {s: 0.0}, {s: None}, [(0.0, s)]
    while q:
        c, u = heapq.heappop(q)
        if u == g:
            break
        if c > dist[u]:
            continue
        for di, dj, L in MOVES:
            v = (u[0] + di, u[1] + dj)
            if 0 <= v[0] < n and 0 <= v[1] < n:
                nc = c + L * cell * 0.5 * (cost[u] + cost[v])
                if nc < dist.get(v, np.inf):
                    dist[v], came[v] = nc, u
                    heapq.heappush(q, (nc, v))
    path, u = [], g
    while u is not None:
        path.append((x_of(u[1]), y_of(u[0])))
        u = came.get(u)
    path = np.array(path[::-1])
    path[0], path[-1] = a[:2], b[:2]
    seg = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    marks = np.arange(spacing, seg[-1], spacing)
    pts = [(float(np.interp(m, seg, path[:, 0])), float(np.interp(m, seg, path[:, 1]))) for m in marks]
    return pts + [tuple(map(float, b[:2]))]
