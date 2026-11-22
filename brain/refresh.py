"""When to refresh TabPFN's context, per model (each regressor, the classifier).

Every row in the table is evaluated once per model (brain/novelty.is_news) against the fit that
made its prediction; news is kept as pending. A model is refitted:
  critical     an unexpected failure (classifier), or the model has no fit yet: before the next decision
  normal       a calibrated surprise / false alarm: once max_news_before_refit are pending
  exploration  new ground, random tries: counted with the normal ones
  staleness    max_rows_since_fit new training rows, or max_fit_age_s of driving since the fit
refit = "every5" in [memory.news] reproduces the old rule (all models every 5 rows or after a
failure) as the control arm.

State per (robot, skill, model): last_fit_seq (the newest event_seq in the fitted rows),
evaluated_seq, held (classifier rows whose delayed fail label is not final yet), pending, rows and
seconds since the fit. The watermark moves only after a fit succeeded (Brain calls fitted()).
It persists in refresh_state.json (atomic writes), so a restart resumes where it stopped, and
the brain restores each model from the rows up to its watermark: the same rows, so the stored
fit is reused without a request.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .novelty import is_news, settings

REPORT = {"news": Counter(), "reasons": Counter(), "refits": Counter(), "rates": 0}
_FILES: dict = {}
_LOCK = threading.Lock()


class _StateFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        except (OSError, ValueError):
            self.data = {}

    def save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)


def _state_file(path):
    with _LOCK:
        if str(path) not in _FILES:
            _FILES[str(path)] = _StateFile(path)
        return _FILES[str(path)]


def _new():
    return {"last_fit_seq": None, "evaluated_seq": -1, "held": [], "pending": [], "rows_since": 0,
            "age_s": 0.0, "fails_since": 0, "new_since": 0, "fit_key": None, "refits": 0}


class Refresh:
    def __init__(self, robot_id, skill, models: dict, cfg, state_path=None, lookahead=1):
        """models: {name: 'reg' | 'clf'} (a regressor's name is its target; the classifier's is
        the fail label)."""
        self.robot_id, self.skill, self.models = robot_id, skill, dict(models)
        self.s = settings(cfg)
        self.lookahead = int(lookahead)
        self.file = _state_file(state_path) if state_path is not None else None
        self.st = {}
        for n in self.models:
            k = self._k(n)
            saved = self.file.data.get(k) if self.file is not None else None
            self.st[n] = {**_new(), **saved} if saved else _new()

    def _k(self, name):
        return f"{self.robot_id}|{self.skill}|{name}"

    def save(self):
        if self.file is not None:
            with _LOCK:
                for n, st in self.st.items():
                    self.file.data[self._k(n)] = st
                self.file.save()

    # ------------------------------------------------------------ evaluation
    @staticmethod
    def _executed(df):
        """Rows the robot actually drove (counterfactual rollouts never relabel or mature anything)."""
        return df if "source" not in df else df[df.source.fillna("sim") != "counterfactual"]

    @staticmethod
    def _is_cf(row):
        return str(row.get("source", "sim")) == "counterfactual"

    def _matured(self, df, row):
        if self._is_cf(row):
            return True                                       # one rolled-out segment: its label is final
        ex = self._executed(df)
        later = ex[(ex.robot_id == row["robot_id"]) & (ex.event_seq > row["event_seq"])]
        return len(later) >= self.lookahead

    def _judge(self, name, row, wm, df):
        st = self.st[name]
        kind = self.models[name]
        pr, why, _ = is_news(row, name, kind, wm.columns, wm.stats.get(name), self.s)
        if kind == "clf" and int(row.get("failed", 0)) and st["last_fit_seq"] is not None and not self._is_cf(row):
            ex = self._executed(df)
            prev = ex[(ex.robot_id == row["robot_id"]) & (ex.episode == row["episode"])
                      & (ex.event_seq < row["event_seq"])].tail(self.lookahead)
            if len(prev) and (prev.event_seq <= st["last_fit_seq"]).any() and pr is None:
                pr, why = "normal", "a fitted row's fail label changed"
        if pr is not None:
            st["pending"].append([int(row["event_seq"]), pr, why])
            REPORT["news"][(name, pr)] += 1
            cat = "surprise" if " outside " in why else why.split(" (")[0]
            REPORT["reasons"][f"{name}: {cat}"] += 1
        return pr, why

    def evaluate(self, wm, table):
        """Judge every row of `table` (this skill family, every robot) not judged yet."""
        if table is None or not len(table) or "event_seq" not in table:
            return
        df = table.sort_values("event_seq")
        recs = df.to_dict("records")
        by_seq = {int(r["event_seq"]): r for r in recs}
        for name, kind in self.models.items():
            st = self.st[name]
            still = []
            for q in st["held"]:                              # classifier rows waiting for their label
                r = by_seq.get(int(q))
                if r is None:
                    continue
                if self._matured(df, r):
                    self._judge(name, r, wm, df)
                else:
                    still.append(int(q))
            st["held"] = still
            for r in recs:
                q = int(r["event_seq"])
                if q <= st["evaluated_seq"]:
                    continue
                st["evaluated_seq"] = q
                if st["last_fit_seq"] is not None and q <= st["last_fit_seq"]:
                    continue                                  # already in the fitted context
                failed = int(r.get("failed", 0))
                st["new_since"] += 1
                st["fails_since"] += failed
                if kind == "clf" or not failed:
                    st["rows_since"] += 1
                    st["age_s"] += float(r.get("rep_duration", 0.0) or 0.0)
                if kind == "clf" and not failed and not self._matured(df, r):
                    st["held"].append(q)
                    continue
                self._judge(name, r, wm, df)

    # ------------------------------------------------------------ decision
    def due(self, wm, train) -> dict:
        """{model: reason} to refit before the next decision (train: the recalled rows)."""
        out = {}
        for name in self.models:
            st = self.st[name]
            ready = wm.has(name)
            can = wm.can_fit(name, train)
            if not can:
                continue
            if ready and wm.is_rate(name):                    # a rate, not a model: free to update
                if st["rows_since"] > 0:
                    out[name] = "both outcomes seen" if wm.both_classes(train) else "rate"
                continue
            if self.s["refit"] == "every5":
                if not ready or st["new_since"] >= 5 or st["fails_since"] > 0:
                    out[name] = "every 5 rows / failure" if ready else "no fit yet"
                continue
            if not ready:
                out[name] = "no fit yet"
            elif any(p[1] == "critical" for p in st["pending"]):
                out[name] = "critical: " + next(p[2] for p in st["pending"] if p[1] == "critical")
            elif len(st["pending"]) >= self.s["max_news_before_refit"]:
                out[name] = f"{len(st['pending'])} news rows"
            elif st["rows_since"] >= self.s["max_rows_since_fit"]:
                out[name] = f"stale: {st['rows_since']} rows since the fit"
            elif st["rows_since"] > 0 and st["age_s"] >= self.s["max_fit_age_s"]:
                out[name] = f"stale: {st['age_s']:.0f} s since the fit"
        return out

    def fitted(self, name, train, key):
        """The fit of `name` on `train` succeeded: move its watermark, clear its news."""
        st = self.st[name]
        seq = train["event_seq"].max() if len(train) and "event_seq" in train else None
        st["last_fit_seq"] = None if seq is None or (isinstance(seq, float) and np.isnan(seq)) else int(seq)
        st["pending"], st["rows_since"], st["age_s"], st["fails_since"], st["new_since"] = [], 0, 0.0, 0, 0
        st["fit_key"] = key
        if key is not None:
            st["refits"] += 1
            REPORT["refits"][name] += 1
        else:
            REPORT["rates"] += 1
        self.save()

    def resumed_at(self, name):
        return self.st[name]["last_fit_seq"]

    def adopt(self, name, seq, key):
        """Take over a fit another rover made up to `seq`: rows after it are judged against it."""
        st = self.st[name]
        st.update(_new())
        st["last_fit_seq"] = st["evaluated_seq"] = int(seq)
        st["fit_key"] = key
        self.save()
