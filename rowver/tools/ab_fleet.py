r"""A/B: the sampler briefed by the dock vs the sampler with no briefing, over seeds.

Per seed: the scout and the scientist go out once (their rows reach the master table at the dock);
then the sampler runs twice on copies of that same master table: briefed (a Mission Context Pack
when it undocks) and --no-brief (its own rows only). Same zone, same geology, same candidates.

Reported per arm (mean +- sd over seeds): true photochem of the drilled sites (sim truth, never
shown to the rovers), intact cores, stucks, sortie time, sites drilled on sand, pack size.

  .venv\Scripts\python.exe -m rowver.tools.ab_fleet --seeds 0 1 2 [--zone 0] [--sites 2]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from rowver.run_fleet import run


def arm_stats(events):
    dr = [e for e in events if e.get("event") == "drill"]
    so = [e for e in events if e.get("job") == "sample" and e.get("event") == "sortie"]
    return {"photochem": float(np.mean([e["photochem"] for e in dr])) if dr else np.nan,
            "cores": float(np.mean([e["success"] for e in dr])) if dr else np.nan,
            "on_sand": float(np.mean([e["ground"] == "soft_sand" for e in dr])) if dr else np.nan,
            "reached": float(np.mean([e["reached"] for e in dr])) if dr else np.nan,
            "stucks": float(sum(e["stucks"] for e in so)),
            "seconds": float(sum(e["seconds"] for e in so)),
            "pack": float(np.mean([e["pack"] for e in so])) if so else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--zone", type=int, default=0)
    ap.add_argument("--scout", type=int, default=6)
    ap.add_argument("--vantage", type=int, default=2)
    ap.add_argument("--sites", type=int, default=2)
    a = ap.parse_args()
    res = {"brief": [], "no-brief": []}
    for seed in a.seeds:
        prep = f"ab_prep_z{a.zone}_s{seed}"
        print(f"\n##### seed {seed}: scout + scientist #####", flush=True)
        _, prep_dir = run(zone=a.zone, seed=seed, scout=a.scout, vantage=a.vantage, run_name=prep, sampler=False)
        for arm, brief in (("brief", True), ("no-brief", False)):
            print(f"\n##### seed {seed}: sampler, {arm} #####", flush=True)
            ev, _ = run(zone=a.zone, seed=seed, brief=brief, sites=a.sites, run_name=f"ab_{arm}_z{a.zone}_s{seed}",
                        master_from=prep_dir, prep=False)
            st = arm_stats(ev)
            res[arm].append(st)
            print(f"seed {seed} {arm}: " + ", ".join(f"{k} {v:.2f}" for k, v in st.items()), flush=True)
    print("\n=== A/B: sampler briefed by the dock vs no briefing (mean +- sd over seeds "
          f"{a.seeds}, zone {a.zone}) ===")
    keys = list(res["brief"][0])
    print(f"{'':12s}" + "".join(f"{k:>18s}" for k in keys))
    for arm, rows in res.items():
        M = np.array([[r[k] for k in keys] for r in rows], float)
        print(f"{arm:12s}" + "".join(f"{np.nanmean(M[:, i]):10.2f} +-{np.nanstd(M[:, i]):5.2f}" for i in range(len(keys))))
    out = f"ab_fleet_z{a.zone}.json"
    from brain.provenance import ROOT, Config
    from rowver.fleet import CONFIG
    p = ROOT / Config(CONFIG)("memory.dir") / out
    p.write_text(json.dumps({"seeds": a.seeds, "zone": a.zone, **res}, indent=1), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
