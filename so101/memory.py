"""Hippocampus: one experience table per skill, one row per action. It is TabPFN's context."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG


class Memory:
    def __init__(self, path: Path, events: Path = CFG.events_jsonl, fresh=False):
        self.path, self.events = Path(path), Path(events)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            for f in (self.path, self.events):
                f.unlink(missing_ok=True)
        self.df = pd.read_csv(self.path) if self.path.exists() and self.path.stat().st_size else pd.DataFrame()

    def __len__(self):
        return len(self.df)

    def append(self, row: dict):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)

    def log_event(self, ev: dict):
        ev = {"t": time.time(), **ev}
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=_json_default) + "\n")

    def stats(self) -> dict:
        df = self.df
        if df.empty:
            return {"rows": 0, "note": "memory is empty: cold start"}
        # predicted rows (practice + game) test the model; only game rows count as hits
        ev = df[df["mode"] != "seed"] if "mode" in df else df
        game = ev[ev["mode"] != "practice"] if "mode" in ev else ev
        out = {
            "rows": int(len(df)),
            "practice": int((df["mode"] == "practice").sum()) if "mode" in df else 0,
            "attempts": int(len(game)),
            "hits": int(game.hit.sum()) if len(game) else 0,
            "hit_rate": _r(game.hit.mean()) if len(game) else None,
            "hit_rate_last10": _r(game.hit.tail(10).mean()) if len(game) else None,
            "topples": int(df.toppled.sum()),
        }
        e = ev.dropna(subset=["pred_err_cm"]) if "pred_err_cm" in ev else ev.iloc[0:0]
        if len(e):
            out |= {
                "pred_err_cm_first5": _r(e.pred_err_cm.head(5).mean()),
                "pred_err_cm_last5": _r(e.pred_err_cm.tail(5).mean()),
                "interval80_coverage": _r(e.in_80.mean()),
                "surprises": int(e.surprise.sum()),
                "mean_pred_uncertainty_cm_last5": _r(e.pred_unc_cm.tail(5).mean()),
            }
        per = df.groupby("type_name").agg(rows=("hit", "size"), hit_rate=("hit", "mean"),
                                          topple_rate=("toppled", "mean")).round(3)
        out["per_object"] = per.reset_index().to_dict("records")
        return out


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
