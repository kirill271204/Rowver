"""Is a new row news for a model? Only news (or staleness) makes the brain refit TabPFN's context.

Every row already carries the prediction made BEFORE the try (pred_<target>, pred_<target>_q10,
pred_<target>_q90, pred_p_fail). After the try we know what happened. Per model:

Regressor (target y):
  surprise   S = max(q10 - y, y - q90, 0) / (q90 - q10 + eps). A calibrated model still puts about
             20% of outcomes outside q10..q90, so a miss alone is not news: it is news when
             S > tau[target] and the miss is larger than abs_margin[target] (target units).
             tau comes from history (calibrated offline: the 90th percentile of S).
  new ground the band q90 - q10 is wider than novel_band[target] (TabPFN is unsure) AND the
             row's features are far from every row of the fitted context (Gower distance to the
             nearest one > novelty_dist): uncertain and unfamiliar.
Classifier (fail):
  log-loss   L = -[y log p + (1 - y) log(1 - p)]. News if L > tau_fail_miss when the try failed
             (default 1.2: failed while P(fail) < 0.30), or L > tau_fail_false when it did not
             (default 2.3: fine while P(fail) > 0.90). Missing a stuck costs more than a false alarm.
Exploration: a deliberate random try is news of the "exploration" priority; a row made while no
fit was ready ("no_model") is news only while the model is still not ready (the scheduler).

Priorities: critical (refit before the next decision), normal, exploration (batched).
"""
from __future__ import annotations

import math
import os

import numpy as np

EPS = 1e-6
DEFAULTS = {
    "refit": "news", "max_news_before_refit": 3, "max_rows_since_fit": 25, "max_fit_age_s": 600.0,
    "tau": {}, "abs_margin": {}, "novel_band": {}, "tau_default": 0.5,
    "novelty_dist": 0.25, "tau_fail_miss": 1.2, "tau_fail_false": 2.3,
}


def settings(cfg) -> dict:
    """[memory.news] merged over the defaults (other projects' configs need not have it)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    try:
        user = cfg("memory.news")
    except KeyError:
        user = {}
    for k, v in user.items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) else v
    if os.getenv("CORTEX_REFIT"):                     # A/B runs: CORTEX_REFIT=every5 | news
        out["refit"] = os.getenv("CORTEX_REFIT")
    return out


# ------------------------------------------------------------------ context statistics
def context_stats(X: np.ndarray) -> dict:
    """What the novelty test needs of a fitted context: its rows, a robust scale per column, and
    which columns are categorical (a few integer codes, e.g. site, payload)."""
    X = np.asarray(X, float)
    q75, q25 = np.nanpercentile(X, 75, axis=0), np.nanpercentile(X, 25, axis=0)
    scale = q75 - q25
    rng = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
    scale = np.where(scale > EPS, scale, np.where(rng > EPS, rng, 1.0))
    cat = np.array([len(np.unique(c[~np.isnan(c)])) <= 4 and np.allclose(c[~np.isnan(c)], np.round(c[~np.isnan(c)]))
                    for c in X.T]) if len(X) else np.zeros(X.shape[1], bool)
    return {"X": X, "scale": scale, "cat": cat}


def nearest_distance(x: np.ndarray, st: dict) -> float:
    """Gower-style distance from one row to the nearest row of the context, in [0, 1]:
    numeric columns |dx| / robust scale, capped at 1; categorical columns 0 or 1."""
    X = st["X"]
    if not len(X):
        return 1.0
    d = np.abs(X - x) / st["scale"]
    d = np.where(st["cat"], (np.abs(X - x) > EPS).astype(float), np.minimum(d, 1.0))
    d = np.where(np.isnan(X - x), np.nan, d)            # categorical columns too: missing is unknown, not equal
    known = ~np.isnan(d)                                # only the columns both rows have (sensors)
    n = known.sum(1)
    if not n.any():
        return 1.0
    mean = np.where(n > 0, np.where(known, d, 0.0).sum(1) / np.maximum(n, 1), 1.0)
    return float(mean.min())


# ------------------------------------------------------------------ the scores
def regression_surprise(y, q10, q90):
    width = max(q90 - q10, 0.0)
    miss = max(q10 - y, y - q90, 0.0)
    return miss / (width + EPS), miss, width


def fail_logloss(failed: int, p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return -math.log(p) if failed else -math.log(1 - p)


def _num(row, k):
    v = row.get(k)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def is_news(row: dict, model: str, kind: str, columns, stats: dict | None, s: dict):
    """-> (priority or None, reason, scores). model: a target name (kind 'reg') or the fail
    label (kind 'clf'). Reads only the row's pred_* columns, outcome and features."""
    how = str(row.get("how", ""))
    if how == "random":
        return "exploration", "random try", {}
    if how in ("no_model", "reference", "baseline"):
        return None, "", {}
    if kind == "clf":
        p = _num(row, "pred_p_fail")
        if p is None:
            return None, "", {}
        failed = int(_num(row, "failed") or 0)
        L = fail_logloss(failed, p)
        if failed and L > s["tau_fail_miss"]:
            return "critical", f"unexpected failure (P(fail) was {p:.2f})", {"logloss": L}
        if not failed and L > s["tau_fail_false"]:
            return "normal", f"false alarm (P(fail) was {p:.2f})", {"logloss": L}
        return None, "", {"logloss": L}
    if int(_num(row, "failed") or 0):                 # regressors are fitted on rows that did not fail
        return None, "", {}
    y, q10, q90 = _num(row, model), _num(row, f"pred_{model}_q10"), _num(row, f"pred_{model}_q90")
    if y is None or q10 is None or q90 is None:
        return None, "", {}
    S, miss, width = regression_surprise(y, q10, q90)
    tau = s["tau"].get(model, s["tau_default"])
    if S > tau and miss > s["abs_margin"].get(model, 0.0):
        return "normal", f"{model} {y:.3f} outside {q10:.3f}..{q90:.3f} (S {S:.2f})", {"S": S}
    band = s["novel_band"].get(model)
    if band is not None and width > band and stats is not None:
        x = np.array([_num(row, c) if _num(row, c) is not None else np.nan for c in columns], float)
        D = nearest_distance(x, stats)
        if D > s["novelty_dist"]:
            return "exploration", f"new ground ({model} band {width:.2f}, distance {D:.2f})", {"S": S, "D": D}
    return None, "", {"S": S}
