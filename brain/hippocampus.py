"""Hippocampus: one shared experience table for every robot and skill.

Every try is one row: who (robot_id, body descriptors), what (skill, family), the situation
(context features), the action, the measured outcome, the brain's prediction made *before*
the try, and bookkeeping (episode, step, mode, source). Rows of other robots and other skills
stay in the same table; recall() decides which ones the cerebellum sees for a decision:
  1. the same robot and skill (all of it, newest first)
  2. the same skill family on other bodies, nearest body descriptors first
  capped at recall_cap rows (TabPFN's context and speed).
With [memory] curate set and more candidate rows than recall_cap, the context is curated instead
of cut (deterministic, so the same table gives the same rows and the stored fit is reused):
  1. rare events (up to curate.rare of the budget): every failure / shield stop, the 5% highest
     tilt_max and the 5% lowest progress_n, newest first
  2. recent rows (curate.recent): this robot's newest first, then the other robots' newest
  3. the rest by diversity: greedy farthest-point over robust-scaled columns; rows closer than
     curate.dup (mean scaled distance) to a chosen row are near-duplicates and never added
  counterfactual rows fill at most curate.max_counterfactual of the budget. The curation is
  redone every curate.every new rows; in between, the newest rows take the last places.
Failure labels are delayed: a row is "fail" if the robot fails during it or in the next
fail_lookahead tries of the same episode (a bad step often shows only one step later). Only
executed rows are relabelled: a counterfactual row (source "counterfactual") is one segment
rolled out from a snapshot, so its fail is its own failed.
Every row gets an immutable event_seq (monotonic, also across runs) and an event_id (uuid4):
the refresh scheduler (brain/refresh.py) judges each event exactly once by its seq.
"""
from __future__ import annotations

import atexit
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


class Hippocampus:
    def __init__(self, run_dir: Path, fail_lookahead: int = 1, fresh_for: str | None = None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path, self.events = self.dir / "memory.csv", self.dir / "events.jsonl"
        self.df = pd.read_csv(self.path) if self.path.exists() and self.path.stat().st_size else pd.DataFrame()
        if fresh_for is not None and len(self.df):        # forget one robot's rows, keep the others'
            self.df = self.df[self.df.robot_id != fresh_for].reset_index(drop=True)
            self._save()
        self.lookahead = int(fail_lookahead)
        if len(self.df) and "event_seq" not in self.df:     # a table from before event ids
            self.df["event_seq"] = np.arange(len(self.df), dtype=np.int64)
            self.df["event_id"] = [str(uuid.uuid4()) for _ in range(len(self.df))]
            self._save()
        self._seq = int(self.df["event_seq"].max()) + 1 if len(self.df) else 0
        self._dirty = 0
        atexit.register(self.flush)

    def __len__(self):
        return len(self.df)

    SAVE_EVERY = 25                                       # new rows between writes of the table

    def _save(self):
        self.df.to_csv(self.path, index=False)
        self._dirty = 0

    def flush(self):
        """Write the table if rows were added since the last write (end of an episode, at exit)."""
        if getattr(self, "_dirty", 0):
            self._save()

    def remember(self, row: dict):
        self.remember_many([row])

    def remember_many(self, rows: list):
        """Append rows (each gets its event_seq / event_id), relabel, save once."""
        if not rows:
            return
        new = []
        for r in rows:
            new.append({**r, "event_seq": self._seq, "event_id": str(uuid.uuid4())})
            self._seq += 1
        self.df = pd.concat([self.df, pd.DataFrame(new)], ignore_index=True)
        for rid, ep in {(r["robot_id"], r["episode"]) for r in new}:
            self._relabel(rid, ep)
        self._dirty = getattr(self, "_dirty", 0) + len(new)
        if self._dirty >= self.SAVE_EVERY:                  # the table lives in memory; disk is the backup
            self._save()

    def _relabel(self, robot_id, episode):
        """fail = this try or one of the next `lookahead` tries of the episode failed (executed rows)."""
        m = (self.df.robot_id == robot_id) & (self.df.episode == episode)
        if "source" in self.df:
            m &= self.df.source.fillna("sim") != "counterfactual"
        idx = self.df.index[m]
        failed = self.df.loc[idx, "failed"].to_numpy(bool)
        fail = np.array([failed[i:i + 1 + self.lookahead].any() for i in range(len(idx))], int)
        self.df.loc[idx, "fail"] = fail

    def log(self, ev: dict):
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **ev}, default=_plain) + "\n")

    def next_episode(self, robot_id) -> int:
        if self.df.empty or "episode" not in self.df:
            return 0
        own = self.df[self.df.robot_id == robot_id]
        return int(own.episode.max()) + 1 if len(own) else 0

    def recall(self, robot_id: str, skill: str, family: str, descriptors: dict, cap: int,
               curate: dict | None = None, columns: list | None = None) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        own = self.df[(self.df.robot_id == robot_id) & (self.df.skill == skill)]
        other = self.df[(self.df.robot_id != robot_id) & (self.df.family == family)]
        if curate and len(own) + len(other) > cap and "event_seq" in self.df:
            return self._curated(robot_id, own, other, cap, curate, columns or [])
        if len(own) >= cap:
            return own.tail(cap)
        if len(other):
            keys = [k for k in descriptors if k in other]
            ref = np.array([float(descriptors[k]) for k in keys])
            X = other[keys].to_numpy(float)
            scale = np.maximum(np.abs(ref), 1e-9)
            dist = np.abs((X - ref) / scale).sum(1)
            other = other.iloc[np.argsort(dist, kind="stable")].head(cap - len(own))
        return pd.concat([own, other])

    # ------------------------------------------------------------ the curated context
    def _curated(self, robot_id, own, other, cap, cur, columns):
        pool = pd.concat([own, other])
        every = int(cur.get("every", 25))
        key = (robot_id, tuple(columns), cap)
        memo = getattr(self, "_memo", {}).get(key)
        top = int(pool.event_seq.max())
        if memo is not None and 0 <= top - memo["top"] < every:
            keep = pool.event_seq.isin(memo["seqs"]) | (pool.event_seq > memo["top"])
            return pool[keep].sort_values("event_seq")
        seqs = curate(pool, robot_id, cap - every, cur, columns)
        if not hasattr(self, "_memo"):
            self._memo = {}
        self._memo[key] = {"top": top, "seqs": seqs}
        return pool[pool.event_seq.isin(seqs)].sort_values("event_seq")


def curate(pool: pd.DataFrame, robot_id, budget: int, cur: dict, columns: list) -> np.ndarray:
    """event_seq of the `budget` rows chosen from `pool` (see the module docstring)."""
    budget = max(int(budget), 1)
    df = pool.sort_values("event_seq", ascending=False)            # newest first everywhere
    seq = df.event_seq.to_numpy(np.int64)
    n = len(df)
    if n <= budget:
        return seq
    is_cf = (df.source.fillna("sim") == "counterfactual").to_numpy() if "source" in df else np.zeros(n, bool)
    cf_cap = int(cur.get("max_counterfactual", 0.5) * budget)
    chosen = np.zeros(n, bool)

    def take(mask, limit):
        k = 0
        for i in np.flatnonzero(mask & ~chosen):
            if chosen.sum() >= budget or k >= limit:
                break
            if is_cf[i] and (chosen & is_cf).sum() >= cf_cap:
                continue
            chosen[i] = True
            k += 1

    events = df.failed.fillna(0).to_numpy(bool) if "failed" in df else np.zeros(n, bool)
    if "rep_shield" in df:
        sh = df.rep_shield
        events |= (sh.notna() & (sh.astype(str) != "") & (sh.astype(str) != "nan")).to_numpy()
    near = np.zeros(n, bool)
    if "tilt_max" in df:
        t = df.tilt_max.to_numpy(float)
        near |= t >= np.nanpercentile(t, 95)
    if "progress_n" in df:
        p = df.progress_n.to_numpy(float)
        near |= p <= np.nanpercentile(p, 5)
    n_rare = int(cur.get("rare", 0.25) * budget)
    take(events, n_rare)                                       # failures and shield stops first ...
    take(near, n_rare - int(chosen.sum()))                     # ... then near-misses
    own = (df.robot_id == robot_id).to_numpy()
    n_recent = int(cur.get("recent", 0.4) * budget)
    before = int(chosen.sum())
    take(own, n_recent)
    take(~own, n_recent - (int(chosen.sum()) - before))
    cols = [c for c in columns if c in df]
    if chosen.sum() < budget and cols:
        X = df[cols].to_numpy(float)
        med = np.nanmedian(X, 0)
        X = np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0.0))
        q75, q25 = np.percentile(X, 75, 0), np.percentile(X, 25, 0)
        sc = np.where(q75 - q25 > 1e-12, q75 - q25, np.maximum(np.ptp(X, 0), 1e-12))
        X = (X - np.median(X, 0)) / sc
        mind = np.full(n, np.inf)
        idx = np.flatnonzero(chosen)
        for chunk in np.array_split(idx, max(1, len(idx) // 128 + 1)):
            if len(chunk):
                mind = np.minimum(mind, np.abs(X[:, None] - X[chunk][None]).mean(2).min(1))
        dup = float(cur.get("dup", 0.02))
        while chosen.sum() < budget:
            block = chosen | (is_cf & ((chosen & is_cf).sum() >= cf_cap))
            cand = np.where(block, -np.inf, mind)
            i = int(np.argmax(cand))                          # ties: the newest (first) row
            if cand[i] == -np.inf or cand[i] < dup:
                break
            chosen[i] = True
            mind = np.minimum(mind, np.abs(X - X[i]).mean(1))
    return seq[chosen]


def _plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
