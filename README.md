# Rowver: a Mars rover fleet with TabPFN as its brain

Rowver is a fleet of three rovers on Jezero crater (MuJoCo, 1:5 scale): the scout Pathfinder, the scientist Spectra and the sampler Hauler. They learn with **Cortex**, the TabPFN brain in `brain/`. There's no neural-network training, no LLM and no gradient descent. Every arc a rover drives becomes a row in an experience table, and TabPFN (v3.5-fast, running locally) uses those rows in context to predict what the next action will do. The full story is in `docs/idea.md`.

## Run it
```
.venv\Scripts\python.exe -m brain.check_tabpfn          # once: download the TabPFN v3.5-fast weights, smoke check
.venv\Scripts\python.exe -m rowver.run_fleet --gui --fresh --scout 1 --vantage 1 --sites 1 --run-name demo
```
- Leave out `--gui` to run headless.
- `--no-brief` runs the A/B control: the sampler gets no briefing from the dock.
- With `--gui`, the viewer shows the TabPFN decision panel. Keys: `9` hides or shows it, `7` switches rover, `8` turns camera follow on or off.

## Layout
| Path | What it is |
|---|---|
| `rowver/` | The fleet: the world (`world.py`, `jezero.py`, `terrain.py`), the rovers (`mars_rover.py`, `rover*.py`, `navcam.py`), the garage and dock, the missions, and the TabPFN decision panel (`panel.py`, `panel_card.py`). Entry point: `run_fleet.py` |
| `rowver/rowver.toml` | The fleet's settings: drive, safety, memory, refits. Every value is tagged with its source |
| `rowver/assets/jezero/` | The Jezero site the simulation loads (heightmap, albedo, rocks, physics) |
| `rowver/tools/` | `import_jezero.py` and `fit_navcam_pca.py` rebuild the assets; `ab_fleet.py` and `ab_report.py` run and score the briefing A/B |
| `brain/` | Cortex, the robot-agnostic brain: hippocampus (memory), cerebellum (TabPFN), prefrontal (the planner), and the TabPFN loader. See `brain/README.md` |
| `brain.toml` | The brain's settings |
| `so101/` | The SO-101 arm on Hauler: its code, its MuJoCo model, and the lab's pick rows that seed it (`runs/memory_pick.csv`) |
| `docs/` | The write-up (`idea.md`) and the deck (`Rowver.pptx`, `Rowver.pdf`) |
| `runs/` | Created by each run: the experience tables and logs (git-ignored) |
| `.env` | TabPFN device and estimator count. Keep it out of any commit (`.env.example` shows the keys) |

`rowver` and `so101` import `brain`. `brain` imports no robot.

Licence: the TabPFN v3.5 weights are non-commercial (CC BY-NC).
