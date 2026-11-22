"""Find the ball and the cup in one RGB-D frame with classical OpenCV: colour + depth, no learned model.

  ball: orange, round, 2-12 cm above the table. Its centre is a least-squares sphere fit
        (known radius) to the 3D points of its pixels.
  cup:  red, the rim at cup height. Its centre is a circle fit to the 3D points of the rim.

Depth turns every pixel into a point on/above the table (RGBDCamera.points), so the
results come out in metres in the robot's frame. Height above the table rejects the
tabletop and the arm parked out of view; hue rejects the yellow arm and the pale table.
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV: H 0-179, S and V 0-255
BALL_HUE, CUP_HUE_HI, CUP_HUE_LO = (4, 18), 168, 4     # orange; red wraps around 0
MIN_SAT, MIN_VAL = 120, 35


def _hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _clean(mask, k=3, close=0):
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return m


def _fit_sphere(P, r, cam_pos, iters=20):
    c = P.mean(0)
    ray = c - cam_pos
    c = c + 0.5 * r * ray / np.linalg.norm(ray)        # the visible cap's mean sits ~r/2 in front of the centre
    for _ in range(iters):
        diff = P - c
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        J = -diff / dist[:, None]
        step = np.linalg.lstsq(J, -(dist - r), rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    res = np.linalg.norm(P - c, axis=1) - r
    return c, float(np.sqrt(np.mean(res ** 2)))


def _fit_circle(xy):
    """Algebraic (Kasa) circle fit, refitted once without the worst outliers. -> centre, radius, rms."""
    def fit(p):
        A = np.column_stack([p, np.ones(len(p))])
        sol = np.linalg.lstsq(A, -(p ** 2).sum(1), rcond=None)[0]
        c = -sol[:2] / 2
        return c, float(np.sqrt(max(c @ c - sol[2], 0.0)))
    c, R = fit(xy)
    res = np.abs(np.linalg.norm(xy - c, axis=1) - R)
    keep = res < max(3 * np.median(res), 0.002)
    if keep.sum() >= 8 and keep.sum() < len(xy):
        c, R = fit(xy[keep])
        res = np.abs(np.linalg.norm(xy[keep] - c, axis=1) - R)
    return c, R, float(np.sqrt(np.mean(res ** 2)))


def detect_ball(hsv, pts, cam, r):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    z = pts[..., 2]
    mask = _clean((H >= BALL_HUE[0]) & (H <= BALL_HUE[1]) & (S >= MIN_SAT) & (V >= MIN_VAL) & (z > 0.02) & (z < 0.12))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cam_pos = cam.pose[0]
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12:
            continue
        circ = 4 * np.pi * area / max(cv2.arcLength(cnt, True) ** 2, 1e-9)
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [cnt], -1, 1, -1)
        sel = (blob > 0) & (mask > 0)
        P = pts[sel]
        expect = np.pi * (r * cam.px_per_m(P.mean(0))) ** 2
        size_fit = min(area / expect, expect / area)
        score = min(circ, 1.0) * size_fit
        if best is None or score > best[0]:
            best = (score, P, cnt)
    if best is None or best[0] < 0.25:
        return None
    score, P, cnt = best
    c, rms = _fit_sphere(P, r, cam_pos)
    return {"x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "conf": round(float(score), 3),
            "pixels": int(len(P)), "fit_rms_mm": round(1000 * rms, 2)}


def detect_cup(hsv, pts, cam, rim_z, rim_r):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    z = pts[..., 2]
    red = ((H >= CUP_HUE_HI) | (H <= CUP_HUE_LO)) & (S >= MIN_SAT) & (V >= MIN_VAL) & (z > 0.003) & (z < rim_z + 0.03)
    mask = _clean(red, 3, close=5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [cnt], -1, 1, -1)
        sel = (blob > 0) & red
        rim = sel & (z > rim_z - 0.012)
        if rim.sum() < 15:
            continue
        c, R, rms = _fit_circle(pts[rim][:, :2])
        score = float(np.exp(-abs(R - rim_r) / 0.006) * np.exp(-rms / 0.004))
        if best is None or score > best[0]:
            best = (score, c, R, rms, int(rim.sum()))
    if best is None or best[0] < 0.2:
        return None
    score, c, R, rms, n = best
    return {"x": float(c[0]), "y": float(c[1]), "rim_r": round(float(R), 4), "conf": round(score, 3),
            "pixels": n, "fit_rms_mm": round(1000 * rms, 2)}


def detect(rgb, depth, cam, ball_r, rim_z, rim_r) -> dict:
    """-> {"ball": dict | None, "cup": dict | None, "frame": annotated BGR image}."""
    hsv = _hsv(rgb)
    pts = cam.points(depth)
    ball = detect_ball(hsv, pts, cam, ball_r)
    cup = detect_cup(hsv, pts, cam, rim_z, rim_r)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s = max(1, round(frame.shape[1] / 640))
    for name, det, z, rad, col in (("ball", ball, None, ball_r, (0, 200, 0)), ("cup", cup, rim_z, rim_r, (255, 120, 0))):
        if det is None:
            cv2.putText(frame, f"no {name}", (10, 30 * s if name == "ball" else 60 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                        (0, 0, 255), 2 * s)
            continue
        p = [det["x"], det["y"], det["z"] if z is None else z]
        u, v = cam.project(p)
        rp = max(3, int(rad * cam.px_per_m(p)))
        cv2.circle(frame, (int(round(u)), int(round(v))), rp + 3 * s, col, 2 * s)
        cv2.drawMarker(frame, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 10 * s, 2 * s)
        cv2.putText(frame, f"{name} {100 * det['x']:.1f},{100 * det['y']:.1f} cm  conf {det['conf']:.2f}",
                    (int(u) + rp + 6 * s, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, max(1, s))
    return {"ball": ball, "cup": cup, "frame": frame}
