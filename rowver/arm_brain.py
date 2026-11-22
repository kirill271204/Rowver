"""The lab's arm brain (so101/brain.py), unchanged, plus the two things a fleet on Mars needs:
  * a `site` feature (earth 0 / mars 1 / moon 2) in every pick and throw row: the robot knows
    WHERE it is, not what gravity is. TabPFN decides from the rows how much the site matters.
  * robot_id on every row, so the table says which rover (or the lab) each row came from.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from so101.brain import Brain
from so101.cerebellum import Cerebellum
from so101.config import CFG
from so101.planner import Planner
from so101.skills import SkillSpec, pick_spec, throw_spec


@dataclass
class SiteSpec(SkillSpec):
    use_site: bool = True

    def featurize(self, state, acts):
        df = super().featurize(state, acts)
        if self.use_site:
            df["site"] = float(state.get("site", 0))
        return df


def site_spec(base: SkillSpec, use_site: bool) -> SiteSpec:
    kw = {f.name: getattr(base, f.name) for f in fields(base)}
    if use_site:
        kw["features"] = list(base.features) + ["site"]
    return SiteSpec(**kw, use_site=use_site)


from collections import Counter

ARM = {"refits": 0, "kept": 0, "why": Counter()}   # arm refit decisions (run summary)
_SHARED: dict = {}                                   # csv path -> the one so101 Memory for that table


class TableView:
    """A rover's handle on a shared so101 Memory: appends are tagged with its robot_id."""

    def __init__(self, shared, rid):
        object.__setattr__(self, "_m", shared)
        object.__setattr__(self, "_rid", rid)

    def append(self, row: dict):
        self._m.append({"robot_id": self._rid, **row})

    def __len__(self):
        return len(self._m)

    def __getattr__(self, k):
        return getattr(self._m, k)

    def __setattr__(self, k, v):
        setattr(self._m, k, v)


class ArmBrain(Brain):
    def __init__(self, robot, skill, run_dir: Path, use_site=True, frozen=False, seed=0):
        CFG.run_dir = Path(run_dir)                  # so101 memory files go to the fleet's runs/
        super().__init__(robot, skills=(skill,), fresh=False, seed=seed)
        sp = site_spec({"pick": pick_spec(), "throw": throw_spec()}[skill], use_site)
        ok = getattr(robot, "action_ok", None)
        land_ok = getattr(robot, "landing_ok", robot.reachable)
        self.spec[skill] = sp
        self.cb[skill] = Cerebellum(sp)
        self.planner[skill] = Planner(self.cb[skill], land_ok, seed=seed,
                                      action_ok=(lambda st, a, s=skill: ok(s, st, a)) if ok else None)
        self._dirty[skill] = True
        self.skill, self.frozen = skill, frozen
        self.n0 = len(self.mem[skill])
        mem = self.mem[skill]
        if len(mem.df) and "site" not in mem.df:     # rows recorded before the site column existed: Earth lab
            mem.df["site"] = 0
        if len(mem.df) and "robot_id" not in mem.df:
            mem.df["robot_id"] = "lab"
        # ONE table per skill for every rover (each so101 Brain would otherwise load its own copy of
        # the CSV and save over the other rovers' rows); every row is tagged with its robot_id
        shared = _SHARED.setdefault(str(mem.path), mem)
        self.mem[skill] = TableView(shared, robot.rid)
        self._fit_len = -1

    def _refit(self, skill):
        """Refit the arm's TabPFN context only when the new pick rows (this rover's AND every other
        rover's) are news: an outcome the classifiers did not expect (log-loss above
        memory.news.tau_fail_miss), a placement error over twice its predicted uncertainty, or a
        row without a prediction; or after 3 new rows. Fits go through brain/fit_cache."""
        df = self.mem[skill].df
        n = self.n0 if self.frozen else len(df)
        if not (self._dirty[skill] or n != self._fit_len):
            return
        cb = self.cb[skill]
        new = df.iloc[self._fit_len:n] if self._fit_len >= 0 else df
        why = self._news(skill, new) if cb.ready and self._fit_len >= 0 else "no fit yet"
        if why is None and len(new) < 3:
            ARM["kept"] += 1
            return
        cb.fit(df.iloc[:self.n0] if self.frozen else df)
        ARM["refits"] += 1
        ARM["why"][why or "3 new rows"] += 1
        self._dirty[skill] = False
        self._fit_len = n

    def _news(self, skill, new):
        import math
        tau = 1.2
        for r in new.to_dict("records"):
            preds = [c for c in self.spec[skill].classifiers if f"pred_p_{c}" in r and r[f"pred_p_{c}"] == r[f"pred_p_{c}"]]
            if not preds:
                return "a row without a prediction"
            for c in preds:
                p = min(max(float(r[f"pred_p_{c}"]), 1e-4), 1 - 1e-4)
                y = int(r.get(c, 0) or 0)
                if (-math.log(p) if y else -math.log(1 - p)) > tau:
                    return f"unexpected {c} = {y} (P was {p:.2f})"
            e, u = r.get("pred_err_cm"), r.get("pred_unc_cm")
            if e == e and u == u and e is not None and u is not None and float(e) > 2 * float(u):
                return f"placement {float(e):.1f} cm off (expected {float(u):.1f})"
        return None

    def rows_by_source(self) -> dict:
        df = self.mem[self.skill].df
        if not len(df) or "robot_id" not in df:
            return {}
        return df.robot_id.value_counts().to_dict()

    def site_rows(self, site_code) -> int:
        df = self.mem[self.skill].df
        return int((df.site == site_code).sum()) if len(df) and "site" in df else 0
