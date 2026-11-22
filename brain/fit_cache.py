"""Reuse of TabPFN fits: the same training rows never cost a second fit.

A fit only loads rows as TabPFN's context (the weights never change); on the Prior Labs cloud it
builds the server's KV cache (fit_with_cache). So a fit is fully defined by the rows and the
estimator settings, and can be reused:
  in process   rover 2 recalling the rows rover 1 just fitted gets rover 1's fit
  on disk      (cloud) runs/fits.json keeps each fit's save_model() record, the server's id
               for it: a later run loads it with no request at all

Canonical data. The rows are sorted by a stable per-row BLAKE2b hash of (features, target),
duplicates kept with their multiplicity, and the model is FITTED on that order, so the same
rows recalled in another order give the same key and the same fit.

Key: BLAKE2b-256 over the estimator kind, target, skill family (never the robot), feature names
and dtypes in schema order (column order is not canonicalized: it changes the model's input),
the estimator settings (tabpfn_model.model_params: backend, model version, n_estimators,
fit_mode, random_state, client version) and the canonical dataset hash.

Single-flight: one lock per key (check, lock, check again, fit). Records are written atomically
(temp file + os.replace). A fit the server no longer has (LostFit) is dropped and refitted once;
every other error goes up (see tabpfn_model.CloudModel). A cross-process lock is not needed
while one process owns the store (M2 will need a file lock).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import tabpfn_model as T

STATS = {"fits": 0, "hits_memory": 0, "hits_disk": 0, "refits_lost": 0}
_MEM: OrderedDict = OrderedDict()            # key -> CachedModel
_MEM_MAX = 64
_LOCKS: dict = {}
_GUARD = threading.Lock()
_STORE: "FitStore | None" = None


def _lock(key):
    with _GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


# ------------------------------------------------------------------ canonical data and keys
def canonical(X: np.ndarray, y: np.ndarray):
    """-> (row order, dataset hash). Rows sorted by their own hash; duplicates stay."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64).reshape(len(X), -1)
    dig = np.array([hashlib.blake2b(X[i].tobytes() + y[i].tobytes(), digest_size=16).digest()
                    for i in range(len(X))], dtype="S16")
    order = np.argsort(dig, kind="stable")
    h = hashlib.blake2b(digest_size=32)
    h.update(np.array([X.shape[0], X.shape[1]], np.int64).tobytes())
    h.update(dig[order].tobytes())
    return order, h.hexdigest()


def fit_key(kind, target, family, columns, dtypes, dataset_hash) -> str:
    head = {"kind": kind, "target": target, "family": family, "columns": list(columns),
            "dtypes": list(dtypes), "params": T.model_params(kind)}
    h = hashlib.blake2b(digest_size=32)
    h.update(json.dumps(head, sort_keys=True).encode())
    h.update(dataset_hash.encode())
    return h.hexdigest()


def lineage_id(kind, target, family, columns) -> str:
    return hashlib.blake2b(json.dumps([kind, target, family, list(columns)]).encode(), digest_size=12).hexdigest()


# ------------------------------------------------------------------ the on-disk store
class FitStore:
    """{key: {"record": save_model() record, "kind", "lineage", "max_seq", "n"}} in one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        try:
            self.rec = json.loads(self.path.read_text()) if self.path.exists() else {}
        except (OSError, ValueError):
            self.rec = {}

    def _write(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.rec))
        os.replace(tmp, self.path)

    def get(self, key):
        return self.rec.get(key)

    def put(self, key, entry):
        with self.lock:
            self.rec[key] = entry
            self._write()

    def drop(self, key):
        with self.lock:
            if self.rec.pop(key, None) is not None:
                self._write()


def set_store(path):
    """Where fit records persist (cloud only); None: in-process reuse only."""
    global _STORE
    _STORE = FitStore(path) if path is not None else None


_WATERMARKS: dict = {}            # lineage -> [(max_seq, key)] of fits made or loaded in this process


def latest_seq(lineage: str):
    """The newest event_seq any known fit of this lineage was made up to (None: none known)."""
    seqs = [s for s, _ in _WATERMARKS.get(lineage, []) if s is not None]
    if _STORE is not None:
        seqs += [e["max_seq"] for e in _STORE.rec.values() if e.get("lineage") == lineage and e.get("max_seq") is not None]
    return max(seqs) if seqs else None


# ------------------------------------------------------------------ the cached model
class CachedModel:
    """A fitted model with its key; refits once if the server has lost the fit."""

    def __init__(self, kind, model, key, X, y, entry):
        self.kind, self.model, self.key = kind, model, key
        self._X, self._y, self._entry = X, y, entry

    def _again(self):
        STATS["refits_lost"] += 1
        if _STORE is not None:
            _STORE.drop(self.key)
        self.model = T.make_model(self.kind, local=True).fit(self._X, self._y)
        STATS["fits"] += 1
        if _STORE is not None and T.is_cloud(self.model):
            _STORE.put(self.key, {**self._entry, "record": self.model.record()})

    def _call(self, name, *a, **kw):
        try:
            return getattr(self.model, name)(*a, **kw)
        except T.LostFit:
            with _lock(self.key):
                self._again()
            return getattr(self.model, name)(*a, **kw)

    def predict(self, X, **kw):
        return self._call("predict", X, **kw)

    def predict_proba(self, X):
        return self._call("predict_proba", X)


def fit_cached(kind: str, X, y, *, columns, target, family="", seqs=None, lineage=None):
    """A fitted TabPFN model ('reg' | 'clf') on rows X, y: from this process, from the store, or a
    new fit. X is fitted in canonical row order. seqs: the rows' event_seq (watermark)."""
    if T._CLIENT is not None:                     # a shared brain server: it has its own context cache
        return T.make_model(kind, lineage=lineage).fit(X, y)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    order, dh = canonical(X, y)
    Xc, yc = X[order], y[order]
    dtypes = ["float64"] * X.shape[1]
    key = fit_key(kind, target, family, columns, dtypes, dh)
    lin = lineage_id(kind, target, family, columns)
    max_seq = int(np.max(seqs)) if seqs is not None and len(seqs) else None
    entry = {"kind": kind, "lineage": lin, "max_seq": max_seq, "n": int(len(X))}
    with _lock(key):
        if key in _MEM:
            _MEM.move_to_end(key)
            STATS["hits_memory"] += 1
            return _MEM[key]
        m = None
        if _STORE is not None and T.backend() == "cloud":
            rec = _STORE.get(key)
            if rec is not None:
                m = T.CloudModel.from_record(kind, rec["record"])
                STATS["hits_disk"] += 1
        if m is None:
            m = T.make_model(kind, local=True).fit(Xc, yc)
            STATS["fits"] += 1
            if _STORE is not None and T.is_cloud(m):
                _STORE.put(key, {**entry, "record": m.record()})
        cm = CachedModel(kind, m, key, Xc, yc, entry)
        _MEM[key] = cm
        while len(_MEM) > _MEM_MAX:
            _MEM.popitem(last=False)
        _WATERMARKS.setdefault(lin, []).append((max_seq, key))
        return cm


def clear_memory():
    """Forget the in-process fits (tests: simulate a new process)."""
    _MEM.clear()
    _WATERMARKS.clear()
    for k in STATS:
        STATS[k] = 0
