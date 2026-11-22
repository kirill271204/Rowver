"""The decision panel's sidebar, drawn as an image (PIL) and laid over the MuJoCo viewer.

Pure: a `view` dict in (rowver/panel.py builds it), an RGB array out. Sections, top to bottom:
  header        the rover, its status, its memory
  decision      what TabPFN chose, in plain words, and what happened
  arc diagram   the same arcs as in 3D, seen from above; the legend counts open / ruled out / blocked
  gauges        TabPFN's prediction vs what happened: progress, stuck risk, tilt
  track record  the last moves: TabPFN's 10-90% band vs reality, and how often reality landed inside
  fleet         one card per rover (pinned to the bottom), then the hotkeys
Layout is in units of a 460-wide column; everything scales with the window (high-DPI included) and is
drawn at 2x, then downsampled, so lines and text are smooth.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COL_W = 460.0                       # layout units across the sidebar
SS = 2                              # supersampling
FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FACES = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf", "light": "segoeuil.ttf"}

BG = (13, 16, 23)
CARD = (22, 26, 37)
CARD_HI = (29, 34, 48)
RULE = (44, 51, 69)
TXT = (234, 237, 244)
TXT2 = (156, 165, 184)
TXT3 = (102, 111, 131)
GOLD = (255, 205, 96)
CYAN = (120, 220, 255)
VETO = (255, 84, 112)
MAPV = (88, 96, 112)
GOOD = (106, 226, 172)
WARN = (255, 184, 77)
GRAD = [(88, 92, 222), (190, 82, 186), (255, 158, 66)]         # TabPFN score: low -> high
STATUS = {"DRIVING": GOLD, "COMMITTED": (46, 204, 182), "LEARNING": (178, 140, 255), "STUCK": VETO,
          "SHIELD STOP": (255, 128, 82), "IDLE": TXT3, "PICKING": (120, 200, 255), "CHOOSING SITES": (178, 140, 255)}


@lru_cache(maxsize=128)
def font(face: str, px: int):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, FACES[face]), px)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", px)
        except OSError:
            return ImageFont.load_default(px)


def mix(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def score_rgb(t):
    t = float(np.clip(t, 0.0, 1.0)) * (len(GRAD) - 1)
    k = min(int(t), len(GRAD) - 2)
    return mix(GRAD[k], GRAD[k + 1], t - k)


def risk_rgb(p):
    return GOOD if p < 0.15 else (WARN if p < 0.4 else VETO)


class Pen:
    """Drawing in layout units (the 460-wide column), at SS x the output resolution."""

    def __init__(self, W, H):
        self.k = W / COL_W * SS
        self.img = Image.new("RGB", (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.img)
        self.Hu = H / (W / COL_W)

    def p(self, v):
        return int(round(v * self.k))

    def xy(self, x, y):
        return (self.p(x), self.p(y))

    def f(self, face, size):
        return font(face, max(6, self.p(size)))

    def text(self, x, y, s, face="regular", size=14, fill=TXT, anchor="la"):
        self.d.text(self.xy(x, y), s, font=self.f(face, size), fill=fill, anchor=anchor)

    def width(self, s, face="regular", size=14):
        return self.d.textlength(s, font=self.f(face, size)) / self.k

    def wrap(self, s, face, size, w):
        out, line = [], ""
        for word in s.split():
            t = (line + " " + word).strip()
            if self.width(t, face, size) <= w or not line:
                line = t
            else:
                out.append(line)
                line = word
        if line:
            out.append(line)
        return out

    def rrect(self, x, y, w, h, r, fill=None, outline=None, width=1.0):
        self.d.rounded_rectangle([self.xy(x, y), self.xy(x + w, y + h)], radius=self.p(r), fill=fill,
                                 outline=outline, width=max(1, self.p(width)) if outline else 0)

    def line(self, pts, fill, width, joint="curve"):
        if len(pts) >= 2:
            self.d.line([self.xy(*q) for q in pts], fill=fill, width=max(1, self.p(width)), joint=joint)

    def dot(self, x, y, r, fill, outline=None):
        self.d.ellipse([self.xy(x - r, y - r), self.xy(x + r, y + r)], fill=fill, outline=outline,
                       width=max(1, self.p(1.2)) if outline else 0)

    def poly(self, pts, fill):
        self.d.polygon([self.xy(*q) for q in pts], fill=fill)

    def eyebrow(self, x, y, s, fill=TXT3):
        self.text(x, y, " ".join(s.upper()), "semibold", 10.5, fill)

    def pill(self, x_right, y, s, col):
        w = self.width(s, "semibold", 11) + 18
        self.rrect(x_right - w, y, w, 22, 11, fill=mix(CARD, col, 0.22))
        self.text(x_right - w / 2, y + 11, s, "semibold", 11, col, anchor="mm")
        return w

    def out(self, W, H):
        return np.asarray(self.img.resize((W, H), Image.LANCZOS), dtype=np.uint8)


def _resample(pts, step):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return pts[:1]
    t = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def _dashed(pen, pts, fill, width, dash=7.0):
    q = _resample(pts, dash / 2)
    for k in range(0, len(q) - 1, 2):
        pen.line([q[k], q[k + 1]], fill, width)


def _arrowhead(pen, pts, fill, size):
    if len(pts) < 2:
        return
    a, b = np.asarray(pts[-2], float), np.asarray(pts[-1], float)
    u = b - a
    n = np.hypot(*u)
    if n < 1e-6:
        return
    u /= n
    v = np.array([-u[1], u[0]])
    tip = b + u * size * 0.6
    pen.poly([tuple(tip), tuple(b - u * size * 0.5 + v * size * 0.55), tuple(b - u * size * 0.5 - v * size * 0.55)], fill)


# ---------------------------------------------------------------- sections
def header(pen, v, y):
    x0 = 22
    pen.dot(x0 + 6, y + 15, 7, v["color"])
    pen.text(x0 + 22, y + 2, v["name"], "bold", 24, TXT)
    w = pen.width(v["name"], "bold", 24)
    pen.text(x0 + 30 + w, y + 11, v["role"], "regular", 13, TXT2)
    pen.pill(COL_W - 20, y + 4, v["status"], STATUS.get(v["status"], TXT2))
    y += 38
    pen.text(x0, y, v["memory"], "regular", 13, TXT2)
    y += 26
    pen.line([(x0, y), (COL_W - 20, y)], RULE, 1)
    return y + 14


def mission(pen, v, y):
    """Where the rover is going: the drive's final destination, distance covered, to go, time."""
    m = v.get("mission")
    if not m:
        return y
    x0, w = 16, COL_W - 32
    pen.rrect(x0, y, w, 58, 12, fill=CARD)
    pen.text(x0 + 14, y + 8, "Heading to " + m["label"], "semibold", 14, TXT)
    mins, secs = divmod(int(m["elapsed"]), 60)
    pen.text(x0 + w - 14, y + 10, f"{mins}:{secs:02d} on this drive", "regular", 11.5, TXT3, anchor="ra")
    by = y + 34
    done = float(np.clip(1 - m["to_go"] / max(m["d0"], 1e-6), 0, 1))
    pen.rrect(x0 + 14, by, w - 28, 7, 3.5, fill=mix(CARD, RULE, 1.0))
    pen.rrect(x0 + 14, by, max((w - 28) * done, 7), 7, 3.5, fill=v["color"])
    pen.text(x0 + 14, by + 10, f"{m['d0'] - m['to_go']:.1f} of {m['d0']:.1f} m covered", "regular", 10.5, TXT3)
    pen.text(x0 + w - 14, by + 10, f"{m['to_go']:.1f} m to go", "semibold", 10.5, TXT2, anchor="ra")
    return y + 70


def decision(pen, v, y):
    x0, w = 22, COL_W - 44
    pen.eyebrow(x0, y, "What it decided")
    y += 22
    lines = v.get("headline") or ["Waiting for its first decision."]
    for k, s in enumerate(lines):
        face, size, col = ("semibold", 18, TXT) if k == 0 else ("regular", 14, TXT2)
        for ln in pen.wrap(s, face, size, w):
            pen.text(x0, y, ln, face, size, col)
            y += size * 1.38
        y += 4
    if v.get("result"):
        txt, good = v["result"]
        col = GOOD if good == 1 else (VETO if good == -1 else WARN)
        pen.rrect(x0, y + 2, 3, 18 * len(pen.wrap(txt, "semibold", 14, w - 14)) + 2, 1.5, fill=col)
        for ln in pen.wrap(txt, "semibold", 14, w - 14):
            pen.text(x0 + 12, y, ln, "semibold", 14, col)
            y += 19.5
        y += 4
    return y + 10


def arc_diagram(pen, v, y):
    x0, w, h = 16, COL_W - 32, 282
    pen.eyebrow(x0 + 6, y, "The arcs it weighed, seen from above")
    y += 22
    pen.rrect(x0, y, w, h, 14, fill=CARD)
    fan = v.get("fan")
    cx, cy = x0 + w / 2, y + 184
    if not fan:
        pen.text(cx, y + h / 2, "No scored arcs yet: still exploring", "regular", 14, TXT3, anchor="mm")
        return y + h + 16
    reach = max(fan["reach"], 0.3)
    sc = 160.0 / reach

    def P(pts):
        pts = np.asarray(pts, float)
        return [(cx - q[1] * sc, cy - q[0] * sc) for q in pts]
    step = 0.25 if reach <= 1.2 else 0.5
    for r in np.arange(step, reach + 1e-6, step):                     # range rings ahead of the rover
        a = np.linspace(-np.pi / 2, np.pi / 2, 72)
        pen.line([(cx - r * sc * np.sin(t), cy - r * sc * np.cos(t)) for t in a], mix(CARD, RULE, 0.85), 1)
        pen.text(cx - r * sc, cy + 5, f"{r:g} m", "regular", 9.5, TXT3, anchor="ma")
    pen.line([(cx - reach * sc, cy), (cx + reach * sc, cy)], mix(CARD, RULE, 0.85), 1)
    for pts, cat, col in fan["arcs"]:
        if cat == 0:
            pen.line(P(pts), mix(CARD, MAPV, 0.8), 1.4)
    for pts, cat, col in fan["arcs"]:
        if cat == 1:
            _dashed(pen, P(pts), VETO, 2.2)
    for pts, cat, col in sorted([a for a in fan["arcs"] if a[1] == 2 and a[2] == a[2]], key=lambda a: a[2]):
        pen.line(P(pts), mix(CARD, score_rgb(col), 0.55 + 0.45 * col), 1.6 + 2.2 * col)
    if fan.get("geo") is not None:
        _dashed(pen, P(fan["geo"]), CYAN, 3.2, dash=9)
    ch = P(fan["chosen"])
    pen.line(ch, mix(CARD, GOLD, 0.35), 10)
    pen.line(ch, GOLD, 4.6)
    _arrowhead(pen, ch, GOLD, 13)
    pen.rrect(cx - 11, cy - 15, 22, 30, 6, fill=CARD_HI, outline=v["color"], width=2)     # the rover, from above
    pen.poly([(cx, cy - 22), (cx - 5, cy - 15), (cx + 5, cy - 15)], v["color"])
    g = fan.get("goal")
    if g is not None:
        dist = float(np.hypot(*g))
        ang = float(np.clip(np.arctan2(g[1], g[0]), -np.pi / 2, np.pi / 2))
        rr = min(dist, reach * 0.93)
        gx, gy = cx - np.sin(ang) * rr * sc, cy - np.cos(ang) * rr * sc
        gy = max(gy, y + 34)
        pen.line([(gx, gy), (gx, gy - 20)], TXT, 1.6)
        pen.poly([(gx, gy - 20), (gx + 12, gy - 16), (gx, gy - 12)], v["color"])
        right = gx < cx + 110
        label = f"goal {dist:.1f} m" + ("" if dist <= reach else " away")
        lw = pen.width(label, "semibold", 11)
        lx_ = gx + 15 if right else gx - 6 - lw
        pen.rrect(lx_ - 5, gy - 24, lw + 10, 19, 6, fill=mix(BG, CARD, 0.5))
        pen.text(lx_, gy - 22, label, "semibold", 11, TXT)
    c = v.get("census") or {}
    ly = y + 202                                                     # legend with the counts
    items = [("chosen", GOLD, "solid", ""), ("geometry-only", CYAN, "dash", ""),
             ("ruled out by TabPFN", VETO, "dash", c.get("tab")),
             ("blocked by the map", mix(CARD, MAPV, 0.95), "solid", c.get("map"))]
    lx = x0 + 14
    for k, (name, col, kind, cnt) in enumerate(items):
        if k == 2:
            lx, ly = x0 + 14, ly + 21
        if kind == "dash":
            _dashed(pen, [(lx, ly + 8), (lx + 18, ly + 8)], col, 3, dash=6)
        else:
            pen.line([(lx, ly + 8), (lx + 18, ly + 8)], col, 3.5)
        label = name + (f" {cnt}" if cnt not in ("", None) else "")
        pen.text(lx + 24, ly, label, "regular", 11.5, TXT2)
        lx += 24 + pen.width(label, "regular", 11.5) + 18
    ly += 21
    for k in range(40):
        pen.line([(x0 + 14 + k * 0.45, ly + 8), (x0 + 14 + (k + 1) * 0.45, ly + 8)], score_rgb(k / 39), 4)
    pen.text(x0 + 38, ly, f"open {c.get('open', '')}: TabPFN score low to high", "regular", 11.5, TXT2)
    if v.get("think"):
        pen.text(x0 + 14, y + h - 19, v["think"], "regular", 10.5, TXT3)
    return y + h + 16


def gauges(pen, v, y):
    x0, w = 22, COL_W - 44
    pen.eyebrow(x0, y, "TabPFN predicted vs what happened")
    y += 24
    pr, ac = v.get("pred"), v.get("actual")
    if not pr:
        pen.text(x0, y, "No prediction: this move was a random try to learn from.", "regular", 13.5, TXT3)
        return y + 30

    def track(yy, lo, hi):
        pen.rrect(x0, yy, w, 8, 4, fill=CARD_HI)
        return lambda t: x0 + w * (float(np.clip((t - lo) / (hi - lo), 0, 1)))
    # progress
    got = f" · got {ac['prog']:.0%}" if ac else ""
    pen.text(x0, y, "Progress along the arc", "regular", 13, TXT2)
    pen.text(x0 + w, y, f"expects {pr['prog50']:.0%}{got}", "semibold", 13, TXT, anchor="ra")
    y += 21
    X = track(y, 0.0, 1.2)
    pen.rrect(X(pr["prog10"]), y, max(X(pr["prog90"]) - X(pr["prog10"]), 3), 8, 4, fill=mix(CARD_HI, GRAD[0], 0.85))
    pen.rrect(X(pr["prog50"]) - 1.5, y - 4, 3, 16, 1.5, fill=TXT)
    if ac:
        ax = X(ac["prog"])
        pen.poly([(ax, y - 6), (ax + 6, y + 4), (ax, y + 14), (ax - 6, y + 4)], GOLD)
    pen.text(X(1.0), y + 11, "100%", "regular", 9, TXT3, anchor="ma")
    y += 26
    # stuck risk
    pf = pr["pfail"]
    pen.text(x0, y, "Chance of getting stuck", "regular", 13, TXT2)
    tag = " · got STUCK" if ac and ac.get("failed") else (" · it didn't" if ac else "")
    pen.text(x0 + w, y, f"{pf:.0%}{tag}", "semibold", 13, risk_rgb(pf), anchor="ra")
    y += 22
    X = track(y, 0.0, 1.0)
    pen.rrect(x0, y, max(X(pf) - x0, 4), 8, 4, fill=risk_rgb(pf))
    y += 21
    # tilt
    lim = pr.get("tilt_limit")
    t90 = pr.get("tilt90")
    if t90 is not None and lim:
        pen.text(x0, y, "Body tilt (worst case)", "regular", 13, TXT2)
        got = f" · got {np.degrees(ac['tilt']):.0f}°" if ac and ac.get("tilt") is not None else ""
        pen.text(x0 + w, y, f"up to {np.degrees(t90):.0f}° of {np.degrees(lim):.0f}° limit{got}", "semibold", 13,
                 TXT if t90 < 0.8 * lim else WARN, anchor="ra")
        y += 21
        X = track(y, 0.0, lim * 1.25)
        pen.rrect(x0, y, max(X(t90) - x0, 4), 8, 4, fill=mix(CARD_HI, (150, 170, 220), 0.8))
        pen.rrect(X(lim) - 1.5, y - 4, 3, 16, 1.5, fill=VETO)
        if ac and ac.get("tilt") is not None:
            ax = X(ac["tilt"])
            pen.poly([(ax, y - 6), (ax + 6, y + 4), (ax, y + 14), (ax - 6, y + 4)], GOLD)
        y += 21
    pen.text(x0, y, "blue bar = TabPFN's likely range   white = its best guess   gold = what happened",
             "regular", 10.5, TXT3)
    return y + 24


def track_record(pen, v, y):
    x0, w, h = 16, COL_W - 32, 82
    hist = v.get("hist") or []
    hist = [r for r in hist if r[0] is not None and r[3] is not None and np.isfinite(r[0]) and np.isfinite(r[3])]
    if len(hist) < 3:
        return y
    pen.rrect(x0, y, w, h, 14, fill=CARD)
    inside = sum(1 for r in hist if r[0] - 1e-9 <= r[3] <= r[2] + 1e-9)
    pen.eyebrow(x0 + 14, y + 12, "Track record")
    pen.text(x0 + w - 14, y + 9, f"reality inside TabPFN's range {inside} of {len(hist)} times", "semibold", 11.5,
             GOOD if inside >= 0.7 * len(hist) else WARN, anchor="ra")
    gx, gy, gw, gh = x0 + 14, y + 32, w - 70, h - 42
    n = len(hist)
    vals = np.array([[r[0], r[2], r[3]] for r in hist], float)
    lo = float(np.clip(np.nanmin(vals), -0.2, 0.9)) - 0.03
    hi = float(np.clip(np.nanmax(vals), lo + 0.2, 1.4)) + 0.03
    X = lambda i: gx + gw * i / max(n - 1, 1)                                   # noqa: E731
    Y = lambda p: gy + gh * (1 - (float(np.clip(p, lo, hi)) - lo) / (hi - lo))  # noqa: E731
    if lo < 1.0 < hi:
        _dashed(pen, [(gx, Y(1.0)), (gx + gw, Y(1.0))], mix(CARD, RULE, 1.0), 1, dash=5)
        pen.text(gx + gw + 4, Y(1.0), "100%", "regular", 9, TXT3, anchor="lm")
    pen.text(gx + gw + 4, gy + gh - 12, "gold =", "regular", 9, TXT3, anchor="lm")
    pen.text(gx + gw + 4, gy + gh, "reality", "regular", 9, GOLD, anchor="lm")
    band = [(X(i), Y(r[2])) for i, r in enumerate(hist)] + [(X(i), Y(r[0])) for i, r in reversed(list(enumerate(hist)))]
    pen.poly(band, mix(CARD, GRAD[0], 0.45))
    pen.line([(X(i), Y(r[1])) for i, r in enumerate(hist)], mix(CARD, TXT, 0.6), 1.4)
    pen.line([(X(i), Y(r[3])) for i, r in enumerate(hist)], GOLD, 2.2)
    for i, r in enumerate(hist):
        pen.dot(X(i), Y(r[3]), 3.4 if r[5] else 2.2, VETO if r[5] else GOLD)
    return y + h + 14


def extras(pen, v, y):
    x0, w = 22, COL_W - 44
    ex = v.get("extras") or []
    if not ex:
        return y
    pen.eyebrow(x0, y, "Other TabPFN decisions")
    y += 22
    for col, txt in ex:
        pen.dot(x0 + 5, y + 9, 4.5, col)
        for ln in pen.wrap(txt, "regular", 13, w - 18):
            pen.text(x0 + 16, y, ln, "regular", 13, TXT2)
            y += 18.5
        y += 4
    return y + 8


FLEET_ROW = 30


def fleet_height(v):
    return len(v.get("fleet") or []) * (FLEET_ROW + 6) + 22 + 24


def fleet(pen, v, y_bottom):
    rows = v.get("fleet") or []
    x0, w = 16, COL_W - 32
    y = y_bottom - fleet_height(v)
    pen.eyebrow(x0 + 6, y, "The fleet")
    y += 20
    for r in rows:
        pen.rrect(x0, y, w, FLEET_ROW, 10, fill=CARD_HI if r["active"] else CARD,
                  outline=r["color"] if r["active"] else None, width=1.5)
        pen.dot(x0 + 14, y + FLEET_ROW / 2, 4.5, r["color"])
        pen.text(x0 + 26, y + 6, r["name"], "semibold", 13, TXT)
        nx = x0 + 26 + pen.width(r["name"], "semibold", 13) + 10
        pw = pen.width(r["status"], "semibold", 10.5) + 22                  # the status pill's width
        line = f"{r['decisions']} decisions · {r['stucks']} stuck · overrode geometry {r['overrides']}"
        size = 10.5
        while size > 8 and pen.width(line, "regular", size) > (x0 + w - 14 - pw) - nx:
            size -= 0.5
        pen.text(nx, y + 9 + (10.5 - size) / 2, line, "regular", size, TXT3)
        pen.pill(x0 + w - 8, y + 4, r["status"], STATUS.get(r["status"], TXT2))
        y += FLEET_ROW + 6
    pen.text(x0 + 6, y + 2, v.get("keys", ""), "regular", 11, TXT3)
    return y


def render_sidebar(v: dict, W: int, H: int) -> np.ndarray:
    """-> (H, W, 3) uint8: the sidebar for view v."""
    pen = Pen(W, H)
    y = 16.0
    y = header(pen, v, y)
    y = mission(pen, v, y)
    y = decision(pen, v, y)
    floor = pen.Hu - fleet_height(v) - 16                                 # the fleet is pinned to the bottom
    for sec, need in ((arc_diagram, 320), (gauges, 172), (track_record, 92), (extras, 60)):
        if y + need <= floor:
            y = sec(pen, v, y)
    fleet(pen, v, pen.Hu - 8)
    return pen.out(W, H)
