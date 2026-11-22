"""The TabPFN decision panel, presentation edition (for the Isaac Sim view and the recorded video).

Same data as panel_card.py (DecisionPanel.view(), read-only), drawn as an engineering instrument panel:
one dark sidebar material, sharp edges, hairline rules instead of cards, hierarchy from type weight and
size (Inter, tight tracking on the large name), colour only where it means something (the rover's own
colour, the chosen arc, good / warning / bad). panel_card.py and the MuJoCo viewer are unchanged.

  render(v, W, H) -> (H, W, 3) uint8
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 2                                     # supersampling
U = 480.0                                  # layout units across the panel
INTER = os.environ.get("ROWVER_PANEL_FONT", os.path.expanduser("~/rowver_fonts/Inter.ttf"))

BG = (13, 14, 16)
RULE = (36, 38, 43)
TRACK = (30, 32, 37)
TXT = (242, 243, 245)
TXT2 = (162, 166, 174)
TXT3 = (106, 110, 118)
GOOD = (48, 209, 88)
WARN = (255, 159, 10)
BAD = (255, 69, 58)
GOLD = (255, 214, 10)
BLUE = (10, 132, 255)
CYAN = (100, 210, 255)
STATUS = {"DRIVING": ("Driving", TXT), "COMMITTED": ("Committed", GOOD), "LEARNING": ("Learning", (191, 90, 242)),
          "STUCK": ("Stuck", BAD), "SHIELD STOP": ("Shield stop", WARN), "IDLE": ("Idle", TXT3),
          "PICKING": ("Picking", CYAN), "CHOOSING SITES": ("Choosing sites", (191, 90, 242))}
WEIGHT = {"regular": 400, "medium": 500, "semibold": 600, "bold": 700}
M = 28.0                                   # side margin


@lru_cache(maxsize=256)
def font(weight: str, px: float):
    size = max(int(round(px * SS)), 6)
    try:
        f = ImageFont.truetype(INTER, size)
        try:
            axes = f.get_variation_axes()
            vals = []
            for a in axes:
                nm = a.get("name", b"")
                nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                vals.append(WEIGHT.get(weight, 400) if "eight" in nm else (0 if "lant" in nm else a.get("default", 0)))
            f.set_variation_by_axes(vals)
        except Exception:
            pass
        return f
    except OSError:
        for name in ("DejaVuSans-Bold.ttf" if weight in ("semibold", "bold") else "DejaVuSans.ttf",):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default(size)


def _mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


class Pen:
    def __init__(self, W, H):
        self.k = W * SS / U
        self.Hu = H * U / W
        self.img = Image.new("RGB", (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def p(self, v):
        return v * self.k

    def text(self, x, y, s, weight="regular", size=14, fill=TXT, anchor="la", track=0.0):
        f = font(weight, size * self.k / SS)
        if not track:
            self.d.text((self.p(x), self.p(y)), s, font=f, fill=fill, anchor=anchor)
            return
        # manual tracking (PIL has none): -0.02 em on large display text
        w = self.width(s, weight, size, track)
        cx = self.p(x) - (w * self.k if anchor[0] == "r" else w * self.k / 2 if anchor[0] == "m" else 0)
        for ch in s:
            self.d.text((cx, self.p(y)), ch, font=f, fill=fill, anchor="l" + anchor[1])
            cx += f.getlength(ch) + track * size * self.k

    def width(self, s, weight="regular", size=14, track=0.0):
        f = font(weight, size * self.k / SS)
        return (f.getlength(s) + track * size * self.k * max(len(s) - 1, 0)) / self.k

    def wrap(self, s, weight, size, w):
        out, line = [], ""
        for word in s.split():
            t = (line + " " + word).strip()
            if self.width(t, weight, size) <= w or not line:
                line = t
            else:
                out.append(line)
                line = word
        if line:
            out.append(line)
        return out

    def para(self, x, y, s, weight="regular", size=14, fill=TXT2, w=U - 2 * M, lh=1.42):
        for ln in self.wrap(s, weight, size, w):
            self.text(x, y, ln, weight, size, fill)
            y += size * lh
        return y

    def rect(self, x0, y0, x1, y1, fill):
        self.d.rectangle([self.p(x0), self.p(y0), self.p(x1), self.p(y1)], fill=fill)

    def hline(self, y, x0=M, x1=U - M, fill=RULE):
        self.rect(x0, y, x1, y + 1.0, fill)

    def line(self, pts, fill, width, dash=None):
        P = [(self.p(a), self.p(b)) for a, b in pts]
        if dash is None:
            self.d.line(P, fill=fill, width=max(int(round(width * self.k)), 1), joint="curve")
            return
        on, off = dash
        acc, draw = 0.0, True
        for (a, b), (c, d) in zip(P, P[1:]):
            seg = math.hypot(c - a, d - b)
            if seg < 1e-6:
                continue
            t = 0.0
            while t < seg:
                lim = (on if draw else off) * self.k - acc
                step = min(lim, seg - t)
                if draw:
                    self.d.line([(a + (c - a) * t / seg, b + (d - b) * t / seg),
                                 (a + (c - a) * (t + step) / seg, b + (d - b) * (t + step) / seg)],
                                fill=fill, width=max(int(round(width * self.k)), 1))
                t += step
                acc += step
                if acc >= (on if draw else off) * self.k - 1e-6:
                    acc, draw = 0.0, not draw

    def dot(self, x, y, r, fill, outline=None, width=1.0):
        self.d.ellipse([self.p(x - r), self.p(y - r), self.p(x + r), self.p(y + r)], fill=fill,
                       outline=outline, width=max(int(round(width * self.k)), 1))

    def out(self, W, H):
        return np.asarray(self.img.resize((W, H), Image.LANCZOS), dtype=np.uint8)


def _pct(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.0%}"


def _section(pen, y, title, right=None, right_fill=TXT3):
    pen.text(M, y, title, "semibold", 13, TXT2)
    if right:
        pen.text(U - M, y, right, "regular", 12, right_fill, anchor="ra")
    return y + 24


# ------------------------------------------------------------------ sections
def header(pen, v, y):
    col = tuple(v.get("color", TXT2))
    st, scol = STATUS.get(v.get("status", "IDLE"), (str(v.get("status", "")).capitalize(), TXT2))
    pen.rect(0, 0, 4, 116, col)                                   # the rover's own colour, a single accent edge
    pen.text(M, y, v.get("role", ""), "medium", 13, TXT2)
    pen.text(M, y + 20, v.get("name", ""), "semibold", 34, TXT, track=-0.02)
    pen.dot(U - M - pen.width(st, "medium", 13) - 12, y + 7, 4, scol)
    pen.text(U - M, y, st, "medium", 13, scol, anchor="ra")
    mem = v.get("memory", "")
    mem = mem.replace("TabPFN brain  ·  ", "TabPFN  ·  ")
    pen.text(M, y + 66, mem, "regular", 12.5, TXT3)
    y += 92
    pen.hline(y)
    return y + 18


def mission(pen, v, y):
    m = v.get("mission")
    if not m:
        return y
    lab = m["label"][:1].upper() + m["label"][1:]
    pen.text(M, y, lab, "medium", 15, TXT)
    pen.text(U - M, y + 1, f"{m['to_go']:.1f} m to go", "regular", 13, TXT2, anchor="ra")
    d0 = max(m["d0"], 1e-3)
    f = float(np.clip(1 - m["to_go"] / d0, 0, 1))
    pen.rect(M, y + 26, U - M, y + 29, TRACK)
    pen.rect(M, y + 26, M + (U - 2 * M) * f, y + 29, tuple(v.get("color", TXT)))
    el = int(m.get("elapsed", 0))
    pen.text(M, y + 38, f"{d0 - m['to_go']:.1f} of {d0:.1f} m  ·  {el // 60}:{el % 60:02d} on this drive", "regular", 12,
             TXT3)
    y += 62
    pen.hline(y)
    return y + 18


def decision(pen, v, y):
    lines = list(v.get("headline", []) or [])
    if not lines:
        return y
    y = _section(pen, y, "Decision")
    y = pen.para(M, y, lines[0], "semibold", 21, TXT, lh=1.22) + 6
    for s in lines[1:2 if v.get("extras") else 3]:     # with science lines below, keep the decision short
        y = pen.para(M, y, s, "regular", 14, TXT2) + 4
    res = v.get("result")
    if res:
        s, good = res
        c = GOOD if good > 0 else BAD if good < 0 else WARN
        y += 4
        top = y
        y = pen.para(M + 14, y, s, "medium", 14, c)
        pen.rect(M, top + 2, M + 3, y - 6, c)
    y += 10
    pen.hline(y)
    return y + 18


def arc_diagram(pen, v, y):
    fan = v.get("fan")
    if not fan:
        return y
    cen = v.get("census", {})
    think = v.get("think", "")
    right = think.split("  ·  ")[0].replace("TabPFN scored ", "") if think else None
    y = _section(pen, y, "Arcs it weighed", right)
    h = 150.0
    cx, cy = U / 2, y + h - 12
    reach = max(float(fan.get("reach", 0.5)), 0.3)
    g = fan.get("goal")
    if g is not None:
        reach = max(reach, min(float(np.hypot(*g)), 2.5 * reach))
    s = (h - 30) / reach
    s = min(s, (U / 2 - M - 10) / reach)

    def P(pts):                                                    # rover frame (x fwd, y left) -> panel
        pts = np.asarray(pts, float)
        return [(cx - b * s, cy - a * s) for a, b in pts]

    for r in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):                     # range rings, faint
        if r > reach * 1.05:
            break
        rr = r * s
        pts = [(cx + rr * math.cos(t), cy - rr * math.sin(t)) for t in np.linspace(0.1, math.pi - 0.1, 48)]
        pen.d.line([(pen.p(a), pen.p(b)) for a, b in pts], fill=RULE, width=max(int(pen.k), 1))
        pen.text(cx + rr * math.cos(0.12) + 2, cy - 4, f"{r:g} m", "regular", 10, TXT3, anchor="ls")
    for pts, cat, col in fan.get("arcs", []):
        if cat == 0:
            pen.line(P(pts), (70, 72, 78), 1.0)
        elif cat == 1:
            pen.line(P(pts), BAD + (200,), 1.2, dash=(4, 4))
        else:
            c = _mix((58, 62, 72), (150, 165, 190), col if np.isfinite(col) else 0.0)
            pen.line(P(pts), c, 1.4)
    if fan.get("geo") is not None:
        pen.line(P(fan["geo"]), CYAN, 1.6, dash=(5, 4))
    ch = P(fan["chosen"])
    pen.line(ch, GOLD, 3.2)
    if len(ch) >= 2:
        (x0, y0), (x1, y1) = ch[-2], ch[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        tip = [(x1 + 7 * math.cos(a), y1 + 7 * math.sin(a)), (x1 + 6 * math.cos(a + 2.5), y1 + 6 * math.sin(a + 2.5)),
               (x1 + 6 * math.cos(a - 2.5), y1 + 6 * math.sin(a - 2.5))]
        pen.d.polygon([(pen.p(a_), pen.p(b_)) for a_, b_ in tip], fill=GOLD)
    if g is not None:
        gx, gy = P([g])[0]
        gx, gy = float(np.clip(gx, M, U - M)), float(np.clip(gy, y + 4, cy))
        pen.dot(gx, gy, 5, None, outline=TXT, width=1.4)
        pen.text(gx + 9, gy, f"goal {float(np.hypot(*g)):.1f} m", "medium", 11, TXT, anchor="lm")
    pen.rect(cx - 6, cy - 2, cx + 6, cy + 10, (70, 74, 82))                 # the rover
    y = cy + 22
    x = M
    for lab, c, dash in (("chosen", GOLD, None), ("geometry only", CYAN, (4, 3)), ("ruled out by TabPFN", BAD, (3, 3)),
                         ("blocked by map", (70, 72, 78), None)):
        pen.line([(x, y + 6), (x + 16, y + 6)], c, 2.0, dash=dash)
        pen.text(x + 21, y, lab, "regular", 11, TXT3)
        x += 21 + pen.width(lab, "regular", 11) + 14
    y += 22
    if cen:
        pen.text(M, y, f"{cen.get('open', 0)} open  ·  {cen.get('tab', 0)} ruled out by TabPFN  ·  "
                       f"{cen.get('map', 0)} blocked by the map", "regular", 11, TXT3)
        y += 18
    pen.hline(y + 4)
    return y + 22


def _bar(pen, y, lo, mid, hi, actual, limit=None, scale=1.0, fill_to=None, fill_col=BLUE):
    x0, x1 = M, U - M
    X = lambda v: x0 + (x1 - x0) * float(np.clip(v / scale, 0, 1))      # noqa: E731
    pen.rect(x0, y, x1, y + 4, TRACK)
    if fill_to is not None and np.isfinite(fill_to):
        pen.rect(x0, y, X(fill_to), y + 4, fill_col)
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        pen.rect(X(lo), y - 2, X(hi), y + 6, BLUE + (150,))
    if mid is not None and np.isfinite(mid):
        pen.rect(X(mid) - 1, y - 4, X(mid) + 1, y + 8, TXT)
    if limit is not None and np.isfinite(limit):
        pen.rect(X(limit) - 1, y - 4, X(limit) + 1, y + 8, BAD)
    if actual is not None and np.isfinite(actual):
        pen.dot(X(actual), y + 2, 4.5, GOLD, outline=BG, width=1.5)


def gauges(pen, v, y):
    p = v.get("pred")
    if not p:
        return y
    a = v.get("actual") or {}
    y = _section(pen, y, "Predicted vs measured")
    ap = a.get("prog")
    pen.text(M, y, "Progress along the arc", "regular", 14, TXT)
    pen.text(U - M, y, f"{_pct(p['prog50'])}  →  {_pct(ap)}" if ap is not None else f"{_pct(p['prog50'])}  →  …",
             "semibold", 14, TXT, anchor="ra")
    _bar(pen, y + 24, p["prog10"], p["prog50"], p["prog90"], ap)
    y += 46
    failed = a.get("failed")
    out = "" if failed is None else ("  ·  stuck" if failed else "  ·  didn't")
    pf = p.get("pfail", 0.0) or 0.0
    pen.text(M, y, "Chance of getting stuck", "regular", 14, TXT)
    pen.text(U - M, y, f"{_pct(pf)}{out}", "semibold", 14, BAD if failed else (GOOD if failed is not None else TXT),
             anchor="ra")
    _bar(pen, y + 24, None, None, None, None, fill_to=pf, fill_col=WARN if pf > 0.3 else GOOD)
    y += 46
    t90, lim = p.get("tilt90"), p.get("tilt_limit")
    if t90 is not None and lim:
        at = a.get("tilt")
        pen.text(M, y, "Body tilt, worst case", "regular", 14, TXT)
        pen.text(U - M, y, f"≤ {math.degrees(t90):.0f}° of {math.degrees(lim):.0f}°"
                 + (f"  ·  {math.degrees(at):.0f}°" if at is not None and np.isfinite(at) else ""),
                 "semibold", 14, TXT, anchor="ra")
        _bar(pen, y + 24, 0.0, None, t90, at, limit=lim, scale=lim * 1.15)
        y += 46
    pen.text(M, y - 6, "blue: TabPFN's 80% range   white: best guess   yellow: measured", "regular", 10.5, TXT3)
    y += 14
    pen.hline(y)
    return y + 18


def track_record(pen, v, y):
    h = [x for x in v.get("hist", []) if x[1] is not None and np.isfinite(x[1])]
    if len(h) < 3:
        return y
    inside = sum(1 for q10, _, q90, r, _, _ in h if r is not None and np.isfinite(q10) and np.isfinite(q90)
                 and q10 - 1e-9 <= r <= q90 + 1e-9)
    frac = inside / len(h)
    y = _section(pen, y, "Track record", f"measured inside TabPFN's range  {inside} of {len(h)}",
                 GOOD if frac >= 0.7 else WARN)
    H, x0, x1 = 50.0, M, U - M
    n = len(h)
    X = lambda i: x0 + (x1 - x0) * i / max(n - 1, 1)                     # noqa: E731
    Y = lambda v_: y + H - H * float(np.clip(v_ if np.isfinite(v_) else 0, 0, 1.1)) / 1.1   # noqa: E731
    band = [(X(i), Y(q90)) for i, (q10, _, q90, *_r) in enumerate(h)] + \
           [(X(i), Y(q10)) for i, (q10, *_r) in reversed(list(enumerate(h)))]
    pen.d.polygon([(pen.p(a), pen.p(b)) for a, b in band], fill=BLUE + (70,))
    pen.line([(X(i), Y(q50)) for i, (_, q50, *_r) in enumerate(h)], (210, 214, 222), 1.2)
    for i, (_, _, _, r, _, f) in enumerate(h):
        if r is not None and np.isfinite(r):
            pen.dot(X(i), Y(r), 2.2, BAD if f else GOLD)
    y += H + 14
    pen.hline(y)
    return y + 18


def extras(pen, v, y, floor=None):
    """The other TabPFN decisions and the science chain, compact: as many lines as fit above the fleet,
    the science lines (rocks read, rock chosen) first."""
    ex = list(v.get("extras") or [])
    if not ex:
        return y
    sci = [e for e in ex if e[1].startswith(("Sampler", "Scientist"))]
    ex = sci + [e for e in ex if e not in sci]
    floor = pen.Hu if floor is None else floor
    if y + 24 + 17 > floor:
        return y
    y = _section(pen, y, "Science and other TabPFN decisions")
    for c, s in ex:
        lines = pen.wrap(s, "regular", 12, U - 2 * M - 14)[:2]
        if y + 16 * len(lines) > floor:
            break
        if len(pen.wrap(s, "regular", 12, U - 2 * M - 14)) > 2:
            lines[-1] = lines[-1].rstrip(" ,;·") + " …"
        pen.dot(M + 3, y + 7, 3, tuple(c))
        for ln in lines:
            pen.text(M + 14, y, ln, "regular", 12, TXT2)
            y += 16
        y += 5
    return y + 4


def fleet_height(v):
    return 30 + 36 * len(v.get("fleet", []))


def fleet(pen, v, y_bottom):
    rows = v.get("fleet", [])
    y = y_bottom - fleet_height(v)
    pen.hline(y - 12)
    y = _section(pen, y, "Fleet")
    for r in rows:
        col = tuple(r["color"])
        st, scol = STATUS.get(r["status"], (str(r["status"]).capitalize(), TXT2))
        if r["active"]:
            pen.rect(0, y - 4, 4, y + 28, col)
        pen.dot(M + 4, y + 9, 4, col)
        pen.text(M + 16, y, r["name"], "semibold" if r["active"] else "medium", 14, TXT if r["active"] else TXT2)
        pen.text(M + 16, y + 19, f"{r['decisions']} decisions  ·  {r['stucks']} stuck  ·  overrode geometry "
                                 f"{r['overrides']}", "regular", 11, TXT3)
        pen.text(U - M, y + 1, st, "medium", 12.5, scol, anchor="ra")
        y += 36


def render(v: dict, W: int, H: int) -> np.ndarray:
    """-> (H, W, 3) uint8: the presentation panel for view v (DecisionPanel.view())."""
    pen = Pen(W, H)
    y = 26.0
    y = header(pen, v, y)
    y = mission(pen, v, y)
    y = decision(pen, v, y)
    floor = pen.Hu - fleet_height(v) - 22
    for sec, need in ((gauges, 175), (arc_diagram, 235), (track_record, 95)):
        if sec is track_record and v.get("extras"):
            continue                                      # with science lines, they take the room instead
        if y + need <= floor:
            y = sec(pen, v, y)
    y = extras(pen, v, y, floor - 16)
    fleet(pen, v, pen.Hu - 14)
    return pen.out(W, H)
