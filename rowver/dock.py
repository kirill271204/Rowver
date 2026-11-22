"""The docking station: the fleet's Mission Briefing Server.

One master table at the dock holds every row any rover ever uploaded (all skills, all bodies; a
sensor a rover lacks is NaN). Rovers do not share rows while they are out: each carries its own
table (RoverMemory) and its own TabPFN.

  undock   brief(rid, design, mission) -> a Mission Context Pack: at most [dock] pack_rows rows
           of the master table, picked for THIS sortie (retrieval-augmented in-context learning,
           as in TabPFN-kNN / LoCalPFN: a local context beats a global sample):
             1. corridor     rows recorded near the planned route and the target area
                             ([dock] corridor_m) first; the nearest ones beyond it fill up if few
             2. relevance    site_survey rows that measured a target this mission needs
             3. body         drive rows from bodies like this one first (continuous descriptors:
                             mass, wheel radius, wheel base, track)
             4. rare events  every failure / stuck / failed core in the corridor is kept
             5. diversity    when the corridor holds more than the budget: in rank order (rare,
                             near, similar body), skipping rows that nearly duplicate a chosen one
           The rover's table becomes: the pack + its own rows not yet uploaded.
  dock     upload(rid): the rover's new rows go into the master table, only while it is parked
           in its bay ([dock] bay_tol).
  --no-brief (the control): no pack; the rover keeps only its own rows, sortie after sortie.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from brain.hippocampus import Hippocampus


class RoverMemory(Hippocampus):
    """A rover's onboard table: the briefing pack of this sortie + its own rows."""

    def __init__(self, run_dir: Path, rid: str, fail_lookahead=1, fit_dir: Path | None = None):
        super().__init__(run_dir, fail_lookahead)
        self.rid = rid
        self.fit_dir = fit_dir                       # fits are shared through one cache (same rows -> same fit)
        self.mark = self._seq                        # rows from here on are new (not uploaded yet)
        self.pack_info: dict = {}

    def unsent(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        return self.df[(self.df.event_seq >= self.mark) & (self.df.robot_id == self.rid)]

    def mark_sent(self):
        self.mark = self._seq

    def load_pack(self, pack: pd.DataFrame | None, keep_own: bool):
        """The table for the next sortie: the pack (new seqs) + own rows (all of them with keep_own,
        else only the ones not uploaded yet: the pack brings back what the dock thinks is useful)."""
        own = self.df[self.df.robot_id == self.rid] if (keep_own and len(self.df)) else self.unsent()
        parts = [own]
        if pack is not None and len(pack):
            p = pack.copy()
            p["event_seq"] = np.arange(self._seq, self._seq + len(p), dtype=np.int64)
            p["event_id"] = [str(uuid.uuid4()) for _ in range(len(p))]
            self._seq += len(p)
            parts.insert(0, p)
        self.df = pd.concat(parts, ignore_index=True) if any(len(x) for x in parts) else pd.DataFrame()
        self.mark = self._seq
        self._memo = {}
        self._save()


def _seg_dist(P, route):
    """Distance from points P (n, 2) to a polyline / point set `route` (m, 2)."""
    R = np.asarray(route, float).reshape(-1, 2)
    if len(R) == 1:
        return np.hypot(P[:, 0] - R[0, 0], P[:, 1] - R[0, 1])
    d = np.full(len(P), np.inf)
    for a, b in zip(R[:-1], R[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(*(P - (a + t[:, None] * ab)).T))
    return d


class DockServer:
    def __init__(self, run_dir: Path, cfg, world):
        self.dir = Path(run_dir) / "dock"
        self.master = Hippocampus(self.dir, cfg("memory.fail_lookahead"))
        self.cfg, self.w = cfg, world
        self.log: list = []

    # ---------------------------------------------------------------- docking
    def bay(self, rid):
        return self.w.start[rid]

    def docked(self, rid, rover) -> bool:
        x, y, _, _ = rover.true_pose()
        bx, by, _ = self.bay(rid)
        return bool(np.hypot(x - bx, y - by) < self.cfg.opt("dock.bay_tol", 0.3))

    def upload(self, rid, rover, mem: RoverMemory) -> int:
        """The rover's new rows -> the master table (only while it is parked in its bay)."""
        if not self.docked(rid, rover):
            self.log.append({"rid": rid, "event": "upload refused: not in its bay"})
            return 0
        rows = mem.unsent()
        if len(rows):
            new = rows.copy()
            new["event_seq"] = np.arange(self.master._seq, self.master._seq + len(new), dtype=np.int64)
            new["event_id"] = [str(uuid.uuid4()) for _ in range(len(new))]
            self.master._seq += len(new)
            self.master.df = pd.concat([self.master.df, new], ignore_index=True)
            self.master._save()
        mem.flush()
        mem.mark_sent()
        self.log.append({"rid": rid, "event": "upload", "rows": len(rows)})
        return len(rows)

    # ---------------------------------------------------------------- the briefing
    def brief(self, rid, desc: dict, mission: dict) -> pd.DataFrame:
        """mission: {"route": [(x, y), ...], "skills": {skill: share of the pack},
        "targets": {skill: [targets this mission needs]}, "columns": {skill: model columns}}."""
        M = self.master.df
        cap = int(self.cfg.opt("dock.pack_rows", 1500))
        corridor = float(self.cfg.opt("dock.corridor_m", 1.5))
        info, parts = {"cap": cap}, []
        if M.empty or "skill" not in M:
            self.log.append({"rid": rid, "event": "brief", "rows": 0, "cap": cap})
            return M
        shares = mission["skills"]
        tot = sum(shares.values())
        for skill, share in shares.items():
            budget = int(cap * share / tot)
            pool = M[M.skill == skill]
            need = mission.get("targets", {}).get(skill)
            if need is not None and len(pool):                       # 2. relevance
                have = [t for t in need if t in pool]
                pool = pool[pool[have].notna().any(axis=1)] if have else pool.iloc[:0]
            if not len(pool):
                info[skill] = {"rows": 0}
                continue
            P = pool[["ctx_x", "ctx_y"]].to_numpy(float) if "ctx_x" in pool else np.zeros((len(pool), 2))
            dist = np.nan_to_num(_seg_dist(P, mission["route"]), nan=1e9)
            near = dist <= corridor                                   # 1. corridor
            keys = [k for k in desc if k in pool]                     # 3. body similarity
            body = np.zeros(len(pool))
            if keys:
                ref = np.array([float(desc[k]) for k in keys])
                X = pool[keys].to_numpy(float)
                body = np.nansum(np.abs((X - ref) / np.maximum(np.abs(ref), 1e-9)), 1)
            rank = dist / max(corridor, 1e-9) + body                  # near the route, similar body first
            rare = _rare(pool) & near                                 # 4. rare events in the corridor
            order = np.argsort(rank + np.where(rare, -1e6, 0.0), kind="stable")
            n_near = int(near.sum())
            if n_near <= budget:                                      # the whole corridor, then the nearest beyond
                shortlist = pool.iloc[np.concatenate([order[near[order]], order[~near[order]]])[:budget]]
            else:                                                     # 5. diversity: in rank order, no near-duplicates
                ranked = pool.iloc[order[near[order]]]
                cols = [c for c in mission.get("columns", {}).get(skill, [])    # not the body: ranked above
                        if c in ranked and c not in desc]
                shortlist = ranked.iloc[_diverse(ranked, budget, cols, float(self.cfg.opt("dock.dup", 0.05)))]
            parts.append(shortlist)
            rows_near = int(near[np.isin(pool.event_seq.to_numpy(), shortlist.event_seq.to_numpy())].sum())
            info[skill] = {"rows": len(shortlist), "in_corridor": rows_near, "rare": int(rare.sum()),
                           "by_robot": shortlist.robot_id.value_counts().to_dict()}
        pack = pd.concat(parts, ignore_index=True) if parts else M.iloc[:0]
        info["rows"] = len(pack)
        self.log.append({"rid": rid, "event": "brief", **info})
        return pack


def _rare(df: pd.DataFrame) -> np.ndarray:
    """Failures (stuck arcs, failed cores) and trafficability hazards (deep sinkage, high slip)."""
    r = np.zeros(len(df), bool)
    for c in ("failed", "sample_fail"):
        if c in df:
            r |= df[c].fillna(0).to_numpy(float) > 0.5
    if "sinkage_mm" in df:
        r |= df["sinkage_mm"].fillna(0).to_numpy(float) > 8.0
    if "slip" in df:
        r |= df["slip"].fillna(0).to_numpy(float) > 0.4
    return r


def _diverse(ranked: pd.DataFrame, budget: int, cols: list, dup: float) -> np.ndarray:
    """Positions of `budget` rows of `ranked` (best first): in order, skipping a row whose mean
    robust-scaled distance to a chosen row is below `dup`; skipped rows fill up only if needed."""
    n = len(ranked)
    if n <= budget or not cols:
        return np.arange(min(n, budget))
    X = ranked[cols].to_numpy(float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0.0))
    q75, q25 = np.percentile(X, 75, 0), np.percentile(X, 25, 0)
    X = (X - np.median(X, 0)) / np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
    chosen, skipped = [], []
    for i in range(n):
        if len(chosen) >= budget:
            break
        if chosen and np.abs(X[chosen] - X[i]).mean(1).min() < dup:
            skipped.append(i)
        else:
            chosen.append(i)
    chosen += skipped[:budget - len(chosen)]
    return np.array(sorted(chosen))
