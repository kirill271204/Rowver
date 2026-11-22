"""Cerebellum: TabPFN v3.5-fast as an in-context world model. Nothing is trained: a fit only
loads the recalled experience rows as TabPFN's context (on the Prior Labs cloud: the server's
KV cache, fit_with_cache).

  regressors   one per target, on rows where the try itself did not fail -> 25 quantiles
  classifier   P(fail) on all rows (fail = failed now or within the next tries)
Columns: the skill's context features + its action + the body descriptors, so rows from other
bodies carry what kind of body they came from and TabPFN can weigh them itself. Bookkeeping
columns (pred_*, event_*, rep_*, ctx_*) are never training features.

Missing values: a feature may be NaN (a sensor the row's robot did not have); TabPFN takes it as
missing. A target may be NaN too (not measured on that row): each model trains only on the rows
that have its own target (or a valid fail label). A model with too few such rows is not fitted and
predicts nothing: callers skip it.

Each model is fitted on its own (fit_model), through brain/fit_cache: the same rows never cost a
second fit, and Brain refits only the models that have news (brain/refresh.py).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import fit_cache
from .novelty import context_stats
from .tabpfn_model import QUANTILES

MIN_ROWS = 3
BOOKKEEPING = ("pred_", "event_", "rep_", "ctx_", "exec_")


class WorldModel:
    def __init__(self, columns: list[str], targets: list[str], fail: str = "fail", family: str = ""):
        bad = [c for c in columns if c.startswith(BOOKKEEPING)]
        if bad:
            raise ValueError(f"bookkeeping columns can not be features: {bad}")
        self.columns, self.targets, self.fail, self.family = list(columns), list(targets), fail, family
        self.kinds = {**{t: "reg" for t in self.targets}, self.fail: "clf"}
        self.reg: dict[str, object] = {}
        self.clf = None                  # a TabPFN classifier, or a float rate while only one class is seen
        self.stats: dict[str, dict] = {}  # model -> its fitted context (novelty test)
        self.keys: dict[str, str | None] = {}
        self.n_rows = self.n_reg = 0
        self.fit_seconds = 0.0

    # ------------------------------------------------------------ which rows each model sees
    def _train(self, name, rows):
        if not len(rows):
            return rows
        if self.kinds[name] == "reg":
            failed = rows["failed"].to_numpy(float) if "failed" in rows else np.zeros(len(rows))
            y = rows[name].to_numpy(float) if name in rows else np.full(len(rows), np.nan)
            return rows[(np.nan_to_num(failed) == 0) & ~np.isnan(y)]
        y = rows[self.fail].to_numpy(float) if self.fail in rows else np.full(len(rows), np.nan)
        return rows[~np.isnan(y)]

    def can_fit(self, name, rows) -> bool:
        tr = self._train(name, rows) if len(rows) else rows
        return len(tr) >= MIN_ROWS

    def has(self, name) -> bool:
        if self.kinds[name] == "reg":
            return name in self.reg
        return self.clf is not None

    def is_rate(self, name) -> bool:
        return self.kinds[name] == "clf" and isinstance(self.clf, float)

    def both_classes(self, rows) -> bool:
        tr = self._train(self.fail, rows)
        y = tr[self.fail].to_numpy(int) if len(tr) else np.zeros(0, int)
        return bool(0 < y.sum() < len(y))

    def lineage(self, name) -> str:
        return fit_cache.lineage_id(self.kinds[name], name, self.family, self.columns)

    def fit_model(self, name, rows: pd.DataFrame):
        """Fit one model on `rows` -> its fit key (None: the classifier is a rate, no request)."""
        t = time.time()
        tr = self._train(name, rows)
        X = tr[self.columns].to_numpy(float)
        seqs = tr["event_seq"].to_numpy() if "event_seq" in tr else None
        key = None
        if self.kinds[name] == "reg":
            m = fit_cache.fit_cached("reg", X, tr[name].to_numpy(float), columns=self.columns, target=name,
                                     family=self.family, seqs=seqs, lineage=(tuple(self.columns), name))
            self.reg[name], key = m, getattr(m, "key", None)
            self.n_reg = len(tr)
        else:
            y = tr[self.fail].to_numpy(int)
            if 0 < y.sum() < len(y):
                m = fit_cache.fit_cached("clf", X, y, columns=self.columns, target=self.fail, family=self.family,
                                         seqs=seqs, lineage=(tuple(self.columns), self.fail))
                self.clf, key = m, getattr(m, "key", None)
            else:
                self.clf = float((y.sum() + 1) / (len(y) + 2))     # Laplace: no evidence -> 0.5
        self.stats[name] = context_stats(X)
        self.keys[name] = key
        self.n_rows = max(self.n_rows, len(rows))
        self.fit_seconds = time.time() - t
        return key

    def fit(self, rows: pd.DataFrame):
        """Every model at once (callers outside the refresh scheduler)."""
        self.reg, self.clf, self.stats, self.keys = {}, None, {}, {}
        self.n_rows = len(rows)
        if rows.empty:
            return self
        for name in self.kinds:
            if self.can_fit(name, rows):
                self.fit_model(name, rows)
        return self

    @property
    def ready(self) -> bool:
        return bool(self.reg) and all(t in self.reg for t in self.targets)

    @property
    def any_ready(self) -> bool:
        """At least one target model (a table whose targets are measured by different robots)."""
        return bool(self.reg)

    def predict(self, X: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """-> ({target: (n, Q) quantiles}, P(fail) (n,)). One request per model, one after another."""
        A = X[self.columns].to_numpy(float)
        kw = dict(output_type="quantiles", quantiles=list(QUANTILES))
        clf = self.clf if self.clf is not None else 0.5
        if self.reg and all(hasattr(m, "predict_async") for m in self.reg.values()):
            # a shared brain server (rowver/hive): every request goes out first, then wait
            pend = {t: m.predict_async("predict", A, **kw) for t, m in self.reg.items()}
            pc = None if isinstance(clf, float) else clf.predict_async("predict_proba", A)
            q = {t: np.sort(np.column_stack(f.result()), axis=1) for t, f in pend.items()}
            p = np.full(len(A), clf) if pc is None else pc.result()[:, 1]
            return q, p
        q = {t: np.sort(np.column_stack(m.predict(A, **kw)), axis=1) for t, m in self.reg.items()}
        p = np.full(len(A), clf) if isinstance(clf, float) else clf.predict_proba(A)[:, 1]
        return q, p
