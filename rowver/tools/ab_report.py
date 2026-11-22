"""Compare arms of the same mission over several runs (seeds): mean +- sd per arm.

  .venv\\Scripts\\python.exe -m rowver.tools.ab_report NAME=RUN_DIR[::LOG] [NAME=RUN_DIR[::LOG] ...]

The same NAME may be given several times (one per seed). RUN_DIR is a copy of runs/ after the
run, LOG the run's console log (API counts, job results). Every quality metric uses only what the
rover actually drove in the MISSION (source "sim", phase "mission"): practice rows and
counterfactual rollouts are context, never scored. Predictions are the ones recorded BEFORE each
arc (pred_* of "play" rows), never a model refitted on the outcome.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _mission(d):
    if "source" in d:
        d = d[d.source.fillna("sim") != "counterfactual"]
    if "phase" in d:
        d = d[d.phase.fillna("mission") == "mission"]
    return d


def _mae(p, t):
    ok = p[(p.failed == 0) & p[f"pred_{t}"].notna()] if f"pred_{t}" in p else p.iloc[:0]
    return float(np.mean(np.abs(ok[f"pred_{t}"] - ok[t]))) if len(ok) else float("nan")


def metrics(run_dir: Path, log: Path | None) -> dict:
    full = pd.read_csv(run_dir / "memory.csv")
    full = full[full.skill == "drive_arc"]
    d = _mission(full).sort_values("event_seq") if "event_seq" in full else _mission(full)
    p = d[d.how == "play"]
    out = {"mission drive rows": len(d)}
    if "phase" in full:
        ex = full[full.source.fillna("sim") != "counterfactual"] if "source" in full else full
        out["practice rows"] = int((ex.phase == "practice").sum())
        out["counterfactual rows"] = int((full.source == "counterfactual").sum()) if "source" in full else 0
    out["stucks/tips"] = int(d.failed.sum())
    if "rep_shield" in d:
        sh = d.rep_shield.fillna("").astype(str)
        out["shield stops"] = int(((sh != "") & (sh != "nan")).sum())
    for t in ("progress_n", "drift", "tilt_max"):
        out[f"{t} MAE"] = _mae(p, t)
    if "pred_progress_n" in p:
        e = p[(p.failed == 0) & p.pred_progress_n.notna()]
        out["progress MAE first 10"] = float(np.mean(np.abs(e.pred_progress_n - e.progress_n).head(10))) if len(e) else np.nan
        out["progress MAE last 10"] = float(np.mean(np.abs(e.pred_progress_n - e.progress_n).tail(10))) if len(e) else np.nan
        r2 = e[e.robot_id == "r2"].head(20)
        out["r2 first 20 MAE (cross-robot)"] = float(np.mean(np.abs(r2.pred_progress_n - r2.progress_n))) if len(r2) else np.nan
    if "pred_p_fail" in p and len(p):
        y, pf = p["fail"].to_numpy(int), p["pred_p_fail"].to_numpy(float)
        out["fail Brier"] = float(np.nanmean((pf - y) ** 2))
        out["fail recall @0.3"] = float(np.mean(pf[y == 1] > 0.3)) if (y == 1).any() else float("nan")
    t = p["decide_s"].to_numpy(float) if "decide_s" in p else np.zeros(0)
    if len(t):
        out["decide p50 s"], out["decide p95 s"] = float(np.percentile(t, 50)), float(np.percentile(t, 95))
    if log is not None and log.exists():
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Prior Labs API: (\d+) fit and (\d+) predict requests", txt)
        if m:
            out["fit requests"], out["predict requests"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"drive refits by model: (\{[^}]*\})", txt)
        if m:
            out["drive model refits"] = sum(int(v) for v in re.findall(r": (\d+)", m.group(1)))
        for what in ("samples in the depot", "bricks in the wall", "beacons on pads"):
            m = re.search(rf"{what}: (\d+)/(\d+)", txt)
            if m:
                out[what] = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"\| ([0-9.]+) min", txt)
        if m:
            out["wall minutes"] = float(m.group(1))
        out["run finished"] = (1, 1) if "=== summary ===" in txt else (0, 1)
    return out


def aggregate(runs: list[dict]) -> dict:
    keys = list(dict.fromkeys(k for r in runs for k in r))
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals and isinstance(vals[0], tuple):
            out[k] = f"{sum(v[0] for v in vals)}/{sum(v[1] for v in vals)}"
            continue
        v = np.array([x for x in vals if x is not None], float)
        v = v[np.isfinite(v)]
        if not len(v):
            out[k] = "n/a"
        elif len(v) == 1:
            out[k] = f"{v[0]:.3f}" if not float(v[0]).is_integer() else f"{int(v[0])}"
        else:
            out[k] = f"{v.mean():.3f} +- {v.std(ddof=1):.3f}"
    return out


def report(groups: dict) -> str:
    agg = {n: aggregate(rs) for n, rs in groups.items()}
    keys = list(dict.fromkeys(k for a in agg.values() for k in a))
    w = max(len(k) for k in keys) + 2
    cw = max(18, *(len(v) + 2 for a in agg.values() for v in a.values()))
    lines = ["".ljust(w) + "".join(f"{n} (n={len(groups[n])})".rjust(cw) for n in agg)]
    for k in keys:
        lines.append(k.ljust(w) + "".join(str(a.get(k, "")).rjust(cw) for a in agg.values()))
    lines.append("\nmean +- sd over runs (seeds). Quality: mission arcs the rover drove, predictions made before each arc.")
    return "\n".join(lines)


def main():
    groups = defaultdict(list)
    for arg in sys.argv[1:]:
        name, rest = arg.split("=", 1)
        parts = rest.split("::")
        groups[name].append(metrics(Path(parts[0]), Path(parts[1]) if len(parts) > 1 else None))
    print(report(groups))


if __name__ == "__main__":
    main()
