"""The heterogeneous fleet demo: scout, scientist and sampler share experience only through the dock.

  .venv\\Scripts\\python.exe -m rowver.run_fleet --fresh [--gui [--no-panel]] [--no-brief]

  r1 scout "Pathfinder"   probes trafficability at candidate drill sites (sinkage, slip)
  r2 scientist "Spectra"  reads the sites with its mast spectrometer (photochem, mineral)
  r3 sampler "Hauler"     briefed by the dock, picks drill sites with its own TabPFN, drives, drills,
                          picks a loose rock with its arm and brings it home to the tray in its bay
Each rover has its own onboard table and TabPFN; rows reach the master table at the dock only
when a rover is parked in its bay, and a rover takes a mission-specific pack when it undocks.

A/B (tools/ab_fleet.py): --prep-only runs the scout and the scientist and keeps the master table;
--master-from DIR --sampler-only runs only the sampler on a copy of it, briefed or --no-brief.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

DESIGNS = {"r1": "scout", "r2": "scientist", "r3": "sampler"}


def run(zone=0, seed=0, brief=True, fresh=True, gui=False, realtime=1.0, scout=6, vantage=2, sites=2,
        sampler_sorties=1, run_name="fleet", master_from=None, prep=True, sampler=True, verbose=True,
        panel=True):
    """-> (events, the run dir)."""
    from brain.provenance import ROOT, Config
    from .fleet import CONFIG, Fleet
    from .missions import FleetOps, sample_sortie, scout_sortie, survey_sortie
    cfg = Config(CONFIG)
    run_dir = ROOT / cfg("memory.dir") / run_name
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    if master_from is not None:                    # the A/B: both arms start from the same master table
        src = Path(master_from)
        (run_dir / "dock").mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "dock" / "memory.csv", run_dir / "dock" / "memory.csv")
        if (src / "fits.json").exists():           # the same rows give the same fits: no new requests
            shutil.copy(src / "fits.json", run_dir / "fits.json")
    t0 = time.time()
    fleet = Fleet(tuple(DESIGNS), zone=zone, gui=gui, realtime=realtime, seed=seed, fresh=False, designs=DESIGNS,
                  run_name=run_name, verbose=verbose, seed_from_lab=True, panel=panel)   # the arm starts from the lab's pick rows
    ops = FleetOps(fleet, seed=seed, brief=brief)
    fleet.say(f"fleet on zone {zone}: " + ", ".join(f"{r} {d} ({fleet.w.designs[r].total_mass:.1f} kg)"
                                                     for r, d in DESIGNS.items())
              + f" | {len(ops.cands)} candidate drill sites | master table {len(fleet.dock.master.df)} rows | "
              f"{'briefed' if brief else 'NO briefing (control)'}")
    events = []
    try:
        if prep:
            fleet.say("\n=== r1 scout: trafficability probes ===")
            events += list(scout_sortie(ops, "r1", n=scout))
            fleet.say("\n=== r2 scientist: spectra ===")
            events += list(survey_sortie(ops, "r2", n_vantage=vantage))
        if sampler:
            for k in range(sampler_sorties):
                fleet.say(f"\n=== r3 sampler: sortie {k + 1}/{sampler_sorties} ===")
                events += list(sample_sortie(ops, "r3", n_sites=sites))
    except KeyboardInterrupt:
        fleet.say("\nstopped")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps({**ev, "zone": zone, "seed": seed}, default=float) + "\n")
    summary(fleet, ops, events, time.time() - t0)
    if gui and fleet.w.viewer is not None:           # keep the viewer open until its window is closed
        print("\nrun finished: close the viewer window to exit", flush=True)
        while fleet.w.viewer.is_running():
            time.sleep(0.2)
    fleet.close()
    return events, run_dir


def summary(fleet, ops, events, seconds):
    say = print
    say("\n=== summary ===")
    for ev in events:
        if ev.get("event") == "sortie":
            say(f"{ev['rid']} {ev['job']}: {'docked' if ev['ok'] else 'NOT docked'}, uploaded {ev['uploaded']} rows, "
                f"{ev['seconds']:.0f} s, {ev['stucks']} stucks")
    dr = [e for e in events if e.get("event") == "drill"]
    if dr:
        say(f"sampler drilled {len(dr)} sites: true photochem {np.mean([e['photochem'] for e in dr]):.2f} "
            f"(best candidate {max(ops.geo.photochem(*c) for c in ops.cands):.2f}, mean candidate "
            f"{np.mean([ops.geo.photochem(*c) for c in ops.cands]):.2f}), cores intact "
            f"{sum(e['success'] for e in dr):.0f}/{len(dr)}, grounds {[e['ground'] for e in dr]}")
    rk = [e for e in events if e.get("event") == "rock"]
    if rk:
        say(f"sampler rocks: {sum(e['ok'] for e in rk)}/{len(rk)} in the tray of its bay, grasped "
            f"{sum(e.get('grasped', False) for e in rk)}/{len(rk)}, " + ", ".join(e.get("tone", e.get("why", "")) for e in rk))
    say(f"dock: master table {len(fleet.dock.master.df)} rows | "
        + ", ".join(f"{e['rid']} {e['event']} {e.get('rows', '')}" for e in fleet.dock.log))
    say(f"{seconds / 60:.1f} min | run dir {ops.f.dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-brief", action="store_true", help="the control: no Mission Context Pack")
    ap.add_argument("--fresh", action="store_true", help="wipe this run's tables first")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-panel", action="store_true", help="--gui without the TabPFN decision panel")
    ap.add_argument("--realtime", type=float, default=1.0)
    ap.add_argument("--scout", type=int, default=6, help="scout probe sites")
    ap.add_argument("--vantage", type=int, default=2, help="scientist vantage points")
    ap.add_argument("--sites", type=int, default=2, help="drill sites per sampler sortie")
    ap.add_argument("--sampler-sorties", type=int, default=1)
    ap.add_argument("--run-name", default="fleet")
    ap.add_argument("--master-from", default=None, help="copy this run dir's master table first")
    ap.add_argument("--prep-only", action="store_true", help="scout + scientist only")
    ap.add_argument("--sampler-only", action="store_true", help="sampler only")
    a = ap.parse_args()
    run(zone=a.zone, seed=a.seed, brief=not a.no_brief, fresh=a.fresh, gui=a.gui, realtime=a.realtime, scout=a.scout,
        vantage=a.vantage, sites=a.sites, sampler_sorties=a.sampler_sorties, run_name=a.run_name,
        master_from=a.master_from, prep=not a.sampler_only, sampler=not a.prep_only, panel=not a.no_panel)


if __name__ == "__main__":
    main()
