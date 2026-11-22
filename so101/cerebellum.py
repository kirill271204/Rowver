"""Cerebellum: TabPFN as an in-context world model, one per skill.

Memory rows are TabPFN's context. Nothing is trained; every refit just loads the latest
experience table. For a skill spec it provides:

* TabPFNRegressors for each outcome target, returning full predictive quantiles,
* a chained TabPFNRegressor  b | (features, a)  so the landing point is sampled from a
  *joint* distribution, not two independent marginals,
* TabPFNClassifiers (grasp success, topple reflex),
* inverse-CDF sampling from those quantiles for Monte-Carlo planning.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from brain.fit_cache import fit_cached
from brain.tabpfn_model import QUANTILES, backend_name, make_model, qcol, sample_from_quantiles  # noqa: F401
from .skills import SkillSpec

MIN_ROWS = 3


# ---------------------------------------------------------------- the module
class Cerebellum:
    def __init__(self, spec: SkillSpec):
        self.spec = spec
        self.models: dict[str, object] = {}
        self.chain = None
        self.clf: dict[str, object] = {}
        self.n_rows = 0
        self.n_reg = 0
        self._reg = None
        self.scale = 1.0          # conformal spread multiplier (set by the brain)
        self.fit_seconds = 0.0
        self.predict_calls = 0
        self.predicted_rows = 0

    @property
    def F(self):
        return self.spec.feature_cols

    # -------------------------------------------------------- fit = load context
    def fit(self, mem: pd.DataFrame):
        t = time.time()
        sp = self.spec
        self.n_rows = len(mem)
        self.models, self.chain, self.clf = {}, None, {}
        if self.n_rows == 0:
            return self
        reg = mem.query(sp.reg_rows) if len(mem) else mem
        self.n_reg = len(reg)
        self._reg = reg
        if len(reg) >= MIN_ROWS:
            X = reg[self.F].to_numpy(float)
            a, b = sp.chain
            for t_ in (a, b):                 # other targets are fitted lazily
                self.models[t_] = self._fit("reg", X, reg[t_].to_numpy(float), self.F, t_)
            self.chain = self._fit("reg", np.column_stack([X, reg[a].to_numpy(float)]), reg[b].to_numpy(float),
                                   self.F + [a], f"{b}|{a}")
        for c in sp.classifiers:
            y = mem[c].to_numpy(int)
            if 0 < y.sum() < len(y) and len(y) >= MIN_ROWS:
                self.clf[c] = self._fit("clf", mem[self.F].to_numpy(float), y, self.F, c)
            else:   # one class seen so far: shrink the prior towards the observed rate
                p0 = sp.prior_p.get(c, 0.5)
                self.clf[c] = float((y.sum() + 2 * p0) / (len(y) + 2))
        self.fit_seconds = time.time() - t
        return self

    def _fit(self, kind, X, y, columns, target):
        """Through brain/fit_cache: the same rows (in any order) never cost a second fit, and on the
        cloud a fit made in an earlier run is reloaded without a request."""
        return fit_cached(kind, X, y, columns=list(columns), target=target,
                          family="arm:" + str(getattr(self.spec, "name", "skill")))

    @property
    def ready(self) -> bool:
        return bool(self.models)

    # -------------------------------------------------------- predictive distributions
    def _prior(self, X: pd.DataFrame, t: str) -> np.ndarray:
        """Cold start (no experience yet): wide Gaussian prior from the spec."""
        z = np.array([np.sqrt(2) * _erfinv(2 * q - 1) for q in QUANTILES])
        mu, sd = self.spec.prior.get(t, (0.0, 0.05))
        return np.full((len(X), 1), mu) + sd * z[None, :]

    def quantiles(self, X: pd.DataFrame, targets=None) -> dict[str, np.ndarray]:
        """Return {target: (n, Q) quantile matrix}."""
        targets = targets or list(self.spec.chain)
        if not self.ready:
            return {t: self._prior(X, t) for t in targets}
        Xa = X[self.F].to_numpy(float)
        out = {}
        for t in targets:
            if t not in self.models:
                self.models[t] = self._fit("reg", self._reg[self.F].to_numpy(float), self._reg[t].to_numpy(float),
                                           self.F, t)
            q = np.column_stack(self.models[t].predict(Xa, output_type="quantiles", quantiles=list(QUANTILES)))
            out[t] = self._calibrate(q) if t in self.spec.chain else q
            self.predict_calls += 1
            self.predicted_rows += len(Xa)
        return out

    def chain_quantiles(self, X: pd.DataFrame, a_vals: np.ndarray) -> np.ndarray:
        Xa = np.column_stack([X[self.F].to_numpy(float), a_vals])
        q = self.chain.predict(Xa, output_type="quantiles", quantiles=list(QUANTILES))
        self.predict_calls += 1
        self.predicted_rows += len(Xa)
        return self._calibrate(np.column_stack(q))

    def _calibrate(self, qm: np.ndarray) -> np.ndarray:
        """Stretch quantiles around the median by the conformal scale."""
        if self.scale == 1.0:
            return qm
        med = qm[:, [qm.shape[1] // 2]]
        return med + self.scale * (qm - med)

    def proba(self, X: pd.DataFrame, name: str) -> np.ndarray:
        c = self.clf.get(name, self.spec.prior_p.get(name, 0.5))
        if isinstance(c, float):
            return np.full(len(X), c)
        self.predict_calls += 1
        self.predicted_rows += len(X)
        return c.predict_proba(X[self.F].to_numpy(float))[:, 1]

    def joint_samples(self, X: pd.DataFrame, n_samples: int, rng, qd: dict):
        """Samples (n, S) of the chain pair (a, b) from the joint distribution."""
        a, b = self.spec.chain
        sa = sample_from_quantiles(qd[a], n_samples, rng)
        if self.chain is None:
            return sa, sample_from_quantiles(qd[b], n_samples, rng)
        Xrep = X.loc[X.index.repeat(n_samples)].reset_index(drop=True)
        qc = self.chain_quantiles(Xrep, sa.reshape(-1))
        sb = sample_from_quantiles(qc, 1, rng).reshape(len(X), n_samples)
        return sa, sb


def _erfinv(y):
    # Winitzki approximation, good enough for a cold-start prior
    a = 0.147
    ln = np.log(1 - y * y)
    t = 2 / (np.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)
