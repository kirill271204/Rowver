<div align="center">

# Rowver

**Rovers that learn from each other.**

Three Mars rovers on real Jezero crater terrain, one docking station, zero training.<br>
TabPFN reads a table of their shared experience to make every decision.

[Website](https://safe-growth-661667.framer.app) · [Write-up](docs/idea.md) · [Slides](docs/Rowver.pptx) · [Brain](brain/README.md)

<img src="docs/images/hero.jpg" alt="Pathfinder driving the Jezero delta front, rendered in NVIDIA Isaac Sim" width="100%">

<sub>Pathfinder on the Jezero delta front. Physics in MuJoCo, rendered in NVIDIA Isaac Sim.</sub>

</div>

---

## The idea

Mars is up to 22 minutes away by radio, so a rover cannot wait for Earth every time the ground changes under its wheels. It has to decide alone.

Rowver does not train anything. **Every metre a rover drives becomes one row** in an experience table: what it saw, what it did, its body, what happened. At every decision, **TabPFN v3.5-fast**, a tabular foundation model, reads those rows in context and predicts what each candidate action will do, with uncertainty. The rover picks the best action, tries it, and the outcome becomes the next row.

- **No gradient steps.** The weights never change; only the table does. One new row changes the next prediction.
- **No LLM in the loop.** Every action and every number comes from the table and TabPFN.
- **Different bodies, one table.** Body descriptors (mass, wheel radius, track) are columns, and a sensor a rover lacks is `NaN`. One table spans three very different machines.

Built at HackLab Robotics × BRISA, CIC Berlin (21–22 November 2026), for the Prior Labs track *"TabPFN for spatial awareness: how far can a table take a robot?"*

## The fleet

<img src="docs/images/station.jpg" alt="The docking station on the Jezero terrain" width="100%">

| | Pathfinder (scout) | Spectra (scientist) | Hauler (sampler) |
|---|---|---|---|
| Job | Drives to candidate sites first and probes the ground: slip, sinkage | Reads every rock in range from 0.5 to 4 m with its mast instruments | Ranks rocks, drills a core, picks the best rock with its arm and brings it home |
| Mass | 4.24 kg | 8.56 kg | 15.8 kg |
| Suspension | 4 balloon wheels on sprung swing arms | 6-wheel rocker-bogie | 6 wheels, sprung front and rear |
| Unique sensors | wheel torque, sinkage camera | RMI telescope, LIBS, Raman, VISIR | core drill, SO-101 arm, drill camera |
| Writes | `slip`, `sinkage_mm` | `photochem_score`, `mineral_class` | `drill_torque_nm`, `sample_success` |

All three have stereo navcams and hazcams. **Each rover carries its own TabPFN and its own table.** A model uses only the columns its own sensors produce, so another rover's rows help through the columns they share.

## How it decides

<table>
<tr>
<td width="32%"><img src="docs/images/decision_panel.jpg" alt="The live TabPFN decision panel"></td>
<td>

About once a second, while driving:

1. **See.** The rover's own stereo cameras build a 6 m local map. No orbital maps, no hidden ground truth.
2. **Predict.** One batched TabPFN pass scores 128 candidate arcs: progress, drift and tilt as quantiles, and P(stuck).
3. **Choose.** Score the pessimistic quantiles and veto unsafe arcs. A safety shield in the body watches every 20 ms, independent of TabPFN.
4. **Try.** Drive 1 s of the chosen arc.
5. **Remember.** Add one row: what it saw, what it predicted, what really happened.

The live panel (left) writes each decision out: what TabPFN expected, what a geometry-only planner would have picked, and what happened.

</td>
</tr>
</table>

## The dock: one master table, a pack per sortie

Rovers do not stream rows to each other while they are out.

- **Upload.** When a rover parks in its bay, its new rows go into the dock's **master table**.
- **Brief.** Before each undock, the dock hands the rover a **Mission Context Pack**: at most 1,500 rows of the master table, picked for this rover and this mission (`rowver/dock.py`):

| Rule | What goes in |
|---|---|
| 1. Corridor | Rows recorded near the planned route and the target area first |
| 2. Relevance | Survey rows that measured a target this mission needs |
| 3. Body | Drive rows from bodies like this one: similar mass, wheels and track |
| 4. Rare events | Every stuck arc, failed core or deep sinkage in the corridor is kept |
| 5. Diversity | When the corridor holds more than the budget, near-duplicates are skipped |

So when Pathfinder sinks in soft sand, that probe row becomes a rare event. It rides in every pack whose route passes nearby, and Spectra and Hauler steer around sand they have never seen. The cheapest rover makes the mistake once, and the whole fleet remembers it.

<table>
<tr>
<td width="50%"><img src="docs/images/overhead_tracks.jpg" alt="Overhead camera: tracks leading out of the garage bay"></td>
<td width="50%"><img src="docs/images/station_ripples.jpg" alt="The station between the wind ripples"></td>
</tr>
<tr>
<td><sub>Overhead camera: tracks leading out of the garage bay.</sub></td>
<td><sub>The station between the ripples. Every rover returns here to charge.</sub></td>
</tr>
</table>

## Measured

First full sortie (scout, then scientist, then sampler), in simulation:

| Result | Value |
|---|---|
| Stucks across all three rovers | **0** |
| Rows in the master table after one sortie | **330** |
| True photochemistry of the drilled site (intact core in carbonate bedrock) | **0.66** |
| Rock picked, carried home and set in the tray | **1 / 1** |
| Docking error, from four headings | **5–16 mm** |
| Open-ground arcs blocked by mistake | **0 / 592** |

The headline experiment is an A/B on the same world and seed: the sampler runs once **briefed** by the dock and once with only its own rows (`--no-brief`). `rowver/tools/ab_fleet.py` runs it and `rowver/tools/ab_report.py` scores it (mean ± sd over seeds, mission rows only).

## Built on real Mars

<img src="docs/images/navcam.jpg" alt="A rover's NavCam view across the Jezero wind ripples" width="100%">

<sub>What a rover sees: the NavCam view across the wind ripples of the delta front.</sub>

| | |
|---|---|
| Terrain | HiRISE elevation model of the Jezero delta front at 1 m, orbital image at 0.25 m, sub-metre relief from photoclinometry, at 1:5 scale |
| Site | 18.53° N 77.39° E, 6.2 km from the Perseverance landing site |
| Rocks | 128 boulders found in the HiRISE image, about 12,900 more from the Golombek and Rapp size model |
| Physics | MuJoCo, with terramechanics for slip and sinkage that the brain never sees |
| Brain | TabPFN v3.5-fast, local on the GPU (an RTX 3060 answers a 1k-row predict in about 0.25 s) |
| Render | NVIDIA Isaac Sim 6.0 on an L40S: chase, overhead and NavCam views |

## Run it

```
.venv\Scripts\python.exe -m brain.check_tabpfn          # once: download the TabPFN v3.5-fast weights, smoke check
.venv\Scripts\python.exe -m rowver.run_fleet --gui --fresh --scout 1 --vantage 1 --sites 1 --run-name demo
```

- Leave out `--gui` to run headless.
- `--no-brief` runs the A/B control: the sampler gets no briefing from the dock.
- With `--gui`, the viewer shows the TabPFN decision panel. Keys: `9` hides or shows it, `7` switches rover, `8` turns camera follow on or off.
- Settings: `rowver/rowver.toml` (the fleet) and `brain.toml` (the brain). Copy `.env.example` to `.env` for the TabPFN device and backend.

<details>
<summary><b>Isaac Sim view</b> (Linux GPU box, e.g. Brev L40S with the Isaac Launchable)</summary>

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

</details>

## Repository layout

| Path | What it is |
|---|---|
| `rowver/` | The fleet: the world (`world.py`, `jezero.py`, `terrain.py`), the rovers (`mars_rover.py`, `rover*.py`, `navcam.py`, `stereo.py`), the garage and dock (`dock.py`), the missions, and the TabPFN decision panel (`panel.py`, `panel_card.py`). Entry point: `run_fleet.py` |
| `rowver/rowver.toml` | The fleet's settings: drive, safety, memory, refits. Every value is tagged with its source |
| `rowver/assets/jezero/` | The Jezero site the simulation loads (heightmap, albedo, rocks, physics) |
| `rowver/tools/` | `import_jezero.py` and `fit_navcam_pca.py` rebuild the assets; `ab_fleet.py` and `ab_report.py` run and score the briefing A/B; `isaac_*.py` mirror and render in Isaac Sim |
| `brain/` | Cortex, the robot-agnostic brain: hippocampus (memory), cerebellum (TabPFN), prefrontal (the planner), and the TabPFN loader. See [`brain/README.md`](brain/README.md) |
| `brain.toml` | The brain's settings |
| `so101/` | The SO-101 arm on Hauler: its code, its MuJoCo model, and the lab's pick rows that seed it |
| `docs/` | The write-up ([`idea.md`](docs/idea.md)), the slides (`Rowver.pptx`, `Rowver.pdf`) and the images in this README |
| `runs/` | Created by each run: the experience tables and logs (git-ignored) |
| `.env` | TabPFN device, backend and estimator count. Keep it out of any commit (`.env.example` shows the keys) |

`rowver` and `so101` import `brain`. `brain` imports no robot.

## Why a fleet

One rover is a single point of failure and a physics trade-off: a heavy arm rover sinks in the same soft sand a light scout crosses safely. That is why NASA (CADRE) and DLR and ESA (ARCHES) are moving to teams of different robots. The catch is that standard learned models break when robots carry different sensors. TabPFN handles missing columns natively and needs no retraining, so the docking table turns separate, specialised rovers into one shared spatial brain.

<p align="center"><img src="docs/images/rover_at_station.jpg" alt="A rover outside its garage bay at the station" width="60%"></p>

<p align="center"><i>They don't call home. They add a row.</i></p>

## Licence

The TabPFN v3.5 weights are non-commercial (CC BY-NC).
