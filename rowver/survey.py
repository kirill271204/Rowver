"""site_survey: where is it worth drilling, and is it safe to get there? One table, many sensors.

Rows are spots on the ground, not drives. Every rover adds the readings its own instruments make:
  scout       sinkage_mm, slip           (trafficability probe: a short straight drive, wheel camera)
  scientist   photochem_score, mineral   (mast spectrometer, 0.5-4 m away)
  sampler     drill_torque_nm, sample_fail (core drill)
Every other target of a row is NaN. Features are what any planner knows about a spot from orbit:
its position and the orbital data there (HiRISE orthoimage colour, DTM slope); no rover identity
(the rows describe the ground, not the body). Each target's model fits only on the rows that
measured it (brain/world_model.py), so a rover can predict photochem without a spectrometer of its
own, from rows the scientist uploaded to the dock.

Score of a candidate drill site (the sampler's own TabPFN):
  photochem lower quantile  -  w_risk x (sinkage q90 / sink_ref + slip q90 + P(core fails))
Terms whose model does not exist yet are skipped; with no photochem model at all there is nothing
to choose by, and the caller takes the nearest candidate (the --no-brief control).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from brain.tabpfn_model import qcol
from brain.world_model import WorldModel

from . import craters as CR
from . import jezero as J

SKILL, FAMILY = "site_survey", "site"
from .sitemap import M_FEATS

ORB_FEATS = ["sx", "sy", "o_slope", "o_hue", "o_sat", "o_val"]
FEATURES = ORB_FEATS + M_FEATS           # orbital data + what the rovers' own stereo maps measured there
TARGETS = ["photochem_score", "sinkage_mm", "slip", "drill_torque_nm", "rolling_res", "vib"]
FAIL = "sample_fail"
# a row about one loose rock (the scientist's remote readings): which rock, and what each instrument measured
ROCK_COLS = ["rock_id", "rock_organics", "rock_sio2", "rock_hydration", "rock_class", "rock_size_cm",
             "rock_val", "rock_sat", "rock_tex", "drill_val", "drill_tex",
             "rmi_px", "rmi_val", "rmi_sat", "rmi_tex", "rmi_size_cm",            # RMI telescope image of the rock
             "zoom_px", "zoom_range", "zoom_size_cm",                            # colour zoom pair, stereo
             "pano_px", "pano_range", "pano_size_cm", "slip_vo"]                 # panoramic bar; visual odometry
ROCK_FEATURES = FEATURES + ["rock_class", "rock_sio2", "rock_hydration", "rock_size_cm", "rock_val", "rock_sat",
                            "rock_tex", "rmi_val", "rmi_sat", "rmi_tex", "rmi_size_cm", "zoom_size_cm"]


class Orbital:
    """Orbital data at a spot: the orthoimage's colour (with the craters painted, as the rover sees
    the ground) and the DTM slope. Public data: no hidden class or patch map is read."""

    def __init__(self, world):
        site = world.site
        self.site = site
        img = cv2.cvtColor(cv2.imread(str(J.ASSETS / "terrain_albedo.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = CR.paint(img, site, list(world.craters))                 # row 0 is north
        img = cv2.GaussianBlur(img, (0, 0), 2.0)                         # orbital resolution, not navcam
        self.hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        self.slope = site.slope_deg(graded=True)
        self.h, self.wd = img.shape[:2]

    def at(self, x, y) -> dict:
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        span = 2 * self.site.half
        u = np.clip(((x - self.site.x_min) / span * self.wd).astype(int), 0, self.wd - 1)
        v = np.clip(((self.site.y_max - y) / span * self.h).astype(int), 0, self.h - 1)
        H = self.hsv[v, u]
        i, j = self.site.ij(x, y)
        return {"sx": x, "sy": y, "o_slope": self.slope[i, j].astype(float),
                "o_hue": H[:, 0] * 2.0, "o_sat": H[:, 1] / 255.0, "o_val": H[:, 2] / 255.0}


def site_features(orb: Orbital, sitemap, x, y) -> pd.DataFrame:
    """Features of spots: orbital data + the rover's site map there (NaN where no rover looked)."""
    f = dict(orb.at(x, y))
    if sitemap is not None:
        f.update(sitemap.features(x, y))
    else:
        f.update({k: np.full(len(np.atleast_1d(x)), np.nan) for k in M_FEATS})
    return pd.DataFrame(f)


def site_row(rid, episode, step, x, y, orb: Orbital, sitemap=None, **readings) -> dict:
    """One spot -> a site_survey row (unmeasured targets NaN)."""
    f = {k: float(v[0]) for k, v in site_features(orb, sitemap, x, y).items()}
    row = {"robot_id": rid, "skill": SKILL, "family": FAMILY, "episode": episode, "step": step,
           "failed": 0, "source": "sim", "ctx_x": float(x), "ctx_y": float(y), **f}
    for t in TARGETS + [FAIL, "mineral_class", "spec_range", "probe"] + ROCK_COLS:
        row[t] = float(readings[t]) if t in readings and readings[t] is not None else np.nan
    return row


class SiteSurvey:
    """A rover's site model: TabPFN on the site rows of its own table (its pack + its own rows)."""

    def __init__(self, rid, mem, cfg):
        self.rid, self.mem, self.cfg = rid, mem, cfg
        self.wm = WorldModel(FEATURES, TARGETS, fail=FAIL, family=FAMILY)
        self.n_rows = 0

    def rows(self) -> pd.DataFrame:
        df = self.mem.df
        if df.empty or "skill" not in df:
            return df
        return df[df.skill == SKILL]

    def refit(self):
        rows = self.rows()
        cap = int(self.cfg("memory.recall_cap"))
        if len(rows) > cap:
            rows = rows.tail(cap)
        self.n_rows = len(rows)
        self.wm.fit(rows)
        return self

    def heads(self) -> list:
        return sorted(self.wm.reg) + ([FAIL] if self.wm.clf is not None and not self.wm.is_rate(FAIL) else [])

    def risk(self, X: pd.DataFrame):
        """Trafficability risk from the scout's rows (and anyone's slip): q90 sinkage / sink_ref + q90 slip.
        -> (n,) or None when neither model exists yet."""
        if not any(t in self.wm.reg for t in ("sinkage_mm", "slip")):
            return None
        q, _ = self.wm.predict(X)
        c, hi = self.cfg, qcol(0.9)
        r = np.zeros(len(X))
        if "sinkage_mm" in q:
            r += np.clip(q["sinkage_mm"][:, hi], 0, None) / c.opt("survey.sink_ref_mm", 10.0)
        if "slip" in q:
            r += np.clip(q["slip"][:, hi], 0, 1)
        return r      # rolling_res / vib are recorded targets TabPFN learns, not yet calibrated into the risk

    def score(self, X: pd.DataFrame):
        """-> (score (n,) or None when there is no photochem model, details per candidate)."""
        if "photochem_score" not in self.wm.reg:
            return None, {}
        q, p = self.wm.predict(X)
        c = self.cfg
        lo = qcol(c.opt("survey.photochem_q", 0.2))
        hi = qcol(0.9)
        s = q["photochem_score"][:, lo].copy()
        risk = np.zeros(len(X))
        if "sinkage_mm" in q:
            risk += np.clip(q["sinkage_mm"][:, hi], 0, None) / c.opt("survey.sink_ref_mm", 10.0)
        if "slip" in q:
            risk += np.clip(q["slip"][:, hi], 0, 1)
        if self.wm.clf is not None:
            risk += p
        s -= c.opt("survey.w_risk", 0.5) * risk
        det = {"photochem_lo": q["photochem_score"][:, lo], "risk": risk,
               "photochem_mid": q["photochem_score"][:, qcol(0.5)]}
        return s, det


class RockModel:
    """The sampler's rock model: TabPFN on the scientist's rock rows (position, orbital colour, rock type,
    SiO2, hydration, size -> Raman organics). It fuses every rock the scientist read, so one noisy reading
    is pulled towards what similar rocks nearby measured; the lower quantile ranks the rocks."""

    def __init__(self, mem, cfg):
        self.mem, self.cfg = mem, cfg
        self.wm = WorldModel(ROCK_FEATURES, ["rock_organics"], fail="rock_fail", family=FAMILY)
        self.rows = pd.DataFrame()

    def refit(self):
        df = self.mem.df
        if df.empty or "rock_id" not in df:
            self.rows = df.iloc[0:0]
            return self
        r = df[(df.skill == SKILL) & df.rock_id.notna() & df.rock_organics.notna()]
        self.rows = r
        self.wm.fit(r)
        return self

    def read_rocks(self) -> pd.DataFrame:
        """One line per rock the scientist read (its readings averaged when it read one twice)."""
        if self.rows.empty:
            return self.rows
        cols = ["ctx_x", "ctx_y"] + ROCK_FEATURES + ["rock_organics"]
        return self.rows.groupby("rock_id")[cols].mean().reset_index()

    def rank(self, q=0.2):
        """-> (rocks DataFrame with organics_mid / organics_lo, sorted best first) or None (no model)."""
        R = self.read_rocks()
        if R.empty or "rock_organics" not in self.wm.reg:
            return None if R.empty else R.assign(organics_mid=R.rock_organics, organics_lo=R.rock_organics) \
                .sort_values("organics_lo", ascending=False)
        qq, _ = self.wm.predict(R)
        R = R.assign(organics_mid=qq["rock_organics"][:, qcol(0.5)], organics_lo=qq["rock_organics"][:, qcol(q)])
        return R.sort_values("organics_lo", ascending=False)


def open_ground(site, p, max_slope=15.0, rock_h=0.04, rock_r=0.30, relief=0.03, relief_r=0.45) -> bool:
    """A spot the ground team would pick from orbital data: gentle slope (the DTM), even all around
    (the ground within relief_r departs from its best plane by less than `relief`: not on a crater's
    rim or wall, not in its bowl), no rock taller than the wheels' clearance close by (rocks, the rim boulders included, are
    visible terrain, not hidden data)."""
    if float(site.slope_deg(graded=True)[site.ij(*p)]) > max_slope:
        return False
    if in_bowl(site, p):
        return False
    a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    rr = np.array([0.0, relief_r / 2, relief_r])
    X, Y = p[0] + np.outer(rr, np.cos(a)).ravel(), p[1] + np.outer(rr, np.sin(a)).ravel()
    Z = site.height_at(X, Y)
    A = np.column_stack([X - p[0], Y - p[1], np.ones_like(X)])
    res = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]          # off the best plane: a rim, a wall, a bowl
    if float(np.ptp(res)) > relief:                              # (an even slope is fine: max_slope covers it)
        return False
    rk = getattr(site, "rocks_placed", None) or site.rocks
    near = np.hypot(rk["x"] - p[0], rk["y"] - p[1]) < rock_r + rk["radius"]
    return not bool((near & (rk["height"] > rock_h)).any())


def in_bowl(site, p, ring=(0.5, 1.2), rise=0.04, frac=0.6) -> bool:
    """From the orbital DTM: the spot lies in a depression (a crater floor) when most of the ground
    0.5-1.2 m around it stands more than `rise` above it. A flat floor passes the slope test, but a
    rover sent there has to cross the bowl's wall twice."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rr = np.linspace(*ring, 4)
    X = p[0] + np.outer(rr, np.cos(a)).ravel()
    Y = p[1] + np.outer(rr, np.sin(a)).ravel()
    h0 = float(site.height_at(np.array([p[0]]), np.array([p[1]]))[0])
    return bool(np.mean(site.height_at(X, Y) > h0 + rise) > frac)


def candidate_sites(site, zone, keep=(), n=30, min_gap=0.8) -> np.ndarray:
    """Candidate drill sites the ground team marks from orbit: open spots (DTM slope, no big rock)
    spread over the zone, clear of the garage and its apron (keep: (x, y, r) circles). Sand is NOT
    excluded: from orbit it is just a colour. Deterministic for the zone."""
    x0, x1, y0, y1 = zone
    g = np.random.default_rng(int(abs(x0) * 31 + abs(y0) * 17))
    K = np.array(keep, float).reshape(-1, 3)
    out = []
    for _ in range(4000):
        p = np.array([g.uniform(x0 + 0.5, x1 - 0.5), g.uniform(y0 + 0.5, y1 - 0.5)])
        if len(K) and (np.hypot(*(K[:, :2] - p).T) - K[:, 2]).min() < 0.7:
            continue
        if not open_ground(site, p):
            continue
        if out and np.hypot(*(np.array(out) - p).T).min() < min_gap:
            continue
        out.append(p)
        if len(out) >= n:
            break
    return np.array(out)


def drill_candidates(fleet) -> np.ndarray:
    """The zone's candidate drill sites (chosen when the world is built: the loose rocks lie around them)."""
    return fleet.w.drill_sites
