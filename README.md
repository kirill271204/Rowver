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

## Isaac Sim view (Linux GPU box, e.g. Brev L40S with the Isaac Launchable)

MuJoCo keeps all the logic (physics, TabPFN, decisions). Isaac Sim 6.0.1 only mirrors and renders it: live over
a local socket, or offline from a recording. Nothing in `rowver/tools/isaac_*.py` simulates anything.

Setup on the box (once):
```
cd ~/rowver && python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt usd-core timm         # usd-core: the recorder; timm: the navcam's DINOv2
.venv/bin/python -m brain.check_tabpfn                          # TabPFN v3.5-fast on the GPU
docker cp rowver/tools/. vscode:/tmp/rowver_tools/              # the Isaac scripts, into the Isaac container
# + /tmp/rowver_tools/mars_sky.hdr, backdrop/ (HiRISE + CTX rings), navcams.json, dust_wheels.json
```

Live (watch at `https://<the instance's port-80 secure link>/viewer`):
```
./run_mission.sh live --scout 1 --vantage 2 --sites 1 --sampler-sorties 2 --seed 0 --live --record
./restart_mirror.sh                      # Isaac mirror: chase view + top + overview, TabPFN panel on the left
```
- `run_fleet.py --live` (`rowver/live.py`) writes the static scene once (`scene.usdc`, `terrain.usdc`), copies it into
  the container and streams the poses that moved (UDP, 30 Hz of sim time) and the panel image (TCP). It paces the
  sim to real time (`--live-realtime`); the brain counts sim seconds, so decisions do not change.
- `restart_mirror.sh` flags: `--views top,navcam,overview`, `--haze 0.3`, `--film-iso 70`, `--motes`, `--no-fx`,
  `--chrome` (keep the editor UI). The mission keeps running while the mirror restarts.

Fallback video from a recording (`--record`, `rowver/record.py`: animated USD + poses + panel images):
```
docker cp runs/<run>/record vscode:/tmp/rowver_rec
docker exec -u ubuntu:ubuntu -w /isaac-sim vscode /isaac-sim/python.sh /tmp/rowver_tools/isaac_render.py \
    --dir /tmp/rowver_rec --out /tmp/rowver_render --step 2          # --count 10 for a look first
docker cp vscode:/tmp/rowver_render ~/render
ffmpeg -framerate 15 -pattern_type glob -i "$HOME/render/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 18 demo.mp4
.venv/bin/python -m rowver.tools.compose_panel --frames ~/render --record runs/<run>/record --out ~/render_panel
ffmpeg -framerate 15 -pattern_type glob -i "$HOME/render_panel/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 18 demo_panel.mp4
```

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
