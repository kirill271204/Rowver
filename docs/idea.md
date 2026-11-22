# Rowver: Mars rovers whose only brain is a table and TabPFN

## The idea in one paragraph

A robot that learns usually needs a neural network trained by gradient descent on thousands or millions of tries. **Rowver does not train anything.** Every try a robot makes (a 1 s drive arc, a grasp, a step) becomes one row in an experience table: what it saw, what it did, its body, what happened. At every decision, **TabPFN v3.5-fast**, a tabular foundation model, reads those rows *in context* and predicts what each candidate action will do, with uncertainty (quantiles, P(fail)). The robot picks the best action, tries it, and the outcome becomes the next row. Learning is just adding rows. The weights never change, so a robot can learn from its first few dozen tries. Its predictions come with calibrated uncertainty, and rows from another robot with a different body help too: each row carries body descriptors, so TabPFN weighs them by how similar the body is.

The flagship demo is a **heterogeneous Mars rover fleet on real Jezero Crater terrain**. Three rovers with different bodies, sensors and jobs share what they learn only through their docking station. The headline experiment asks whether a heavy sampling rover does better when the dock *briefs* it with the other rovers' experience before it leaves.

Built for the Prior Labs challenge *"TabPFN for spatial awareness: how far can a table take a robot?"*

---

## 1. Why a table and TabPFN

| What robots usually need | What Rowver does instead |
|---|---|
| A trained policy (RL, imitation) with thousands of episodes | No training. TabPFN reads the robot's experience rows in context at decision time |
| A separate uncertainty model (ensembles, GPs) | TabPFN gives quantiles (q10..q90) and P(fail) for free; the planner scores pessimistic quantiles |
| Retraining to adapt | A new row changes the next prediction. A context is reloaded only when a row is *news* (a surprise, a missed failure, new ground) |
| Per-robot models | One table schema for every robot. Body descriptors (mass, wheel radius, track…) are features, so experience transfers across bodies |
| Opaque decisions | Every decision is a row with its prediction stored beside the outcome. You can audit what the robot expected and what happened |

Rules Rowver holds itself to:
- **No LLM in the control loop.** An LLM may only turn a sentence into a job list. It never chooses an action or a number.
- **No gradient descent, no fine-tuning.** TabPFN's weights are frozen; only its context changes.
- **Provenance on every number.** Each tunable is tagged with its source (`dataset:`, `hardware:`, `experience:`, `user:`). Nothing is hand-tuned in code.
- **Hidden ground truth stays hidden.** The simulator knows the terrain classes, slip patches and geology; the robots never read them. They learn from their own sensors and outcomes only.

---

## 2. The brain: Cortex (`brain/`, robot-agnostic)

The brain is named after brain regions. No robot code lives here; a robot package implements a *body* and imports the brain.

| Region | File | Job |
|---|---|---|
| Nerves | `nerves.py` | The body contract: `describe`, `reset`, `context`, `act` (+ optional `commit`, `counterfactual`). `LocalNerve` in one process, `UDPNerve` for a body in another process or on real hardware |
| Hippocampus | `hippocampus.py` | The experience table. `recall()` gives the robot's own rows first, then the same skill family on the nearest other bodies, capped and curated. Delayed failure labels. Every row has an immutable `event_seq` and `event_id` |
| Cerebellum | `world_model.py` | TabPFN heads: outcome quantiles per target and a P(fail) classifier, on context features + action + body descriptors. Each head trains only on rows that have its target (NaN-safe) |
| Prefrontal | `prefrontal.py` | Searches the action space (128 candidates per decision, batched), scores them with the objective, vetoes high-P(fail) actions |
| Loop | `brain.py` | sense → recall → predict → decide → act → remember. Modes: `baseline`, `practice`, `play` |
| Refit on news | `novelty.py`, `refresh.py` | A context is reloaded only when a row is news: an outcome outside its predicted band, a missed failure (critical: refit before the next action), new ground far from every fitted row, or staleness |
| Fit reuse | `fit_cache.py` | A fit is keyed by a hash of its settings and rows, so the same rows are never fitted twice: across rovers, across runs, after a restart |
| Model | `tabpfn_model.py` | TabPFN v3.5-fast, **local on the GPU** by default (an RTX 3060 6 GB answers a 1k-row predict in about 0.25 s), or the Prior Labs cloud with `fit_with_cache` |

**The body runs its own fast loop** (sensors → controller → actuators at 500 Hz). The brain talks to it about once per skill execution (about 1 Hz for driving) and never sits inside the fast loop. A **safety shield** in the body, independent of TabPFN, watches every 20 ms.

---

## 3. The robots

### 3.1 SO-101 arm (`so101/`): the lab
A LeRobot SO-101 arm in MuJoCo learns **push, pick-and-place, and throwing a ball into a cup**. Features: the object's pose in the arm frame and its type. Actions: grasp yaw, height, offsets, grip force, lift speed, aim. Targets: P(grasp), P(topple), placement error. The lab's pick rows seed the Mars sampler's arm (20 rows, tagged `robot_id=lab, site=0`). That's the first cross-domain transfer, Earth lab to Mars.

### 3.2 Unitree G1 humanoid (`humanoid_robot/`)
The G1 learns to walk on the same brain: a recorded human walk (LAFAN1, retargeted to the G1) is the reference, and TabPFN learns per-step corrections. The baseline is the recording with zero corrections, so the brain's contribution is measurable.

### 3.3 The Mars fleet (`rowver/`): the flagship
Described in the rest of this document.

---

## 4. The Mars fleet

### 4.1 The site: real Jezero Crater at 1:5
- **Terrain:** a HiRISE heightmap and orthoimage of Jezero, 40 m × 40 m at 10 cm cells, 2.7 m of relief, scaled 1:5 to match the rovers.
- **Rocks:** the dataset's rock meshes (one convex collider each); boulders added on crater rims.
- **Craters:** fresh (walls 25–35°) and old (19–23°) craters carved into the DTM.
- **Hidden ground:** bedrock, gravel and soft sand with calibrated slip (sand 0.35–0.70 as in the dataset). The ground class only reaches the wheel contacts; no rover feature reads it.
- **Hidden geology** (`geology.py`): a `photochem` field (richest on layered bedrock and crater rims, poorest on sand), mineral class, hardness and core success probability, varying per zone and seed. The robots measure it only through their instruments.
- **Gravity and air:** Mars by default; the site code is a feature, so the robots are never told the gravity.

### 4.2 The garage: docking station and briefing server
- A **garage with three bays** under a **solar-panel roof**, on a graded hard-ground pad with an apron in front. All other station structures were removed.
- A rover **backs out** of its bay onto the apron to leave, and **drives in nose-first** to dock (scripted line-up and creep-in, 5–16 mm accuracy measured from four headings).
- A **sample tray** in the sampler's bay, where the arm sets down the rock it carried home.
- The dock is also the **Mission Briefing Server** (`dock.py`):
  - **Upload** happens only while a rover is parked in its bay: its new rows go into the **master table**.
  - **Brief** happens before a rover leaves: it gets a **Mission Context Pack**, at most 1500 rows chosen for this mission:
    1. rows recorded along the planned route or in the target area (corridor, 1.5 m);
    2. rows relevant to the targets this mission needs;
    3. rows from similar bodies;
    4. rare hazards (stucks, shield stops) kept;
    5. the rest by diversity (no near-duplicates).
  - The sampler gets a two-stage briefing: first the site rows (where to drill), then the drive rows along its route.
- Between sorties, a rover's onboard table is the pack + its own rows. With `--no-brief` (the control arm), it has only its own rows.

### 4.3 Three different machines

| | Scout "Pathfinder" | Scientist "Spectra" | Sampler "Hauler" |
|---|---|---|---|
| Job | Trafficability probes: drive to sites, measure slip and sinkage | Remote spectra of candidate sites from vantage points (0.5–4 m) | Choose a drill site, drill a core, pick a loose rock with the arm, carry it home to the tray |
| Mass | 4.24 kg | 8.56 kg | 15.8 kg |
| Suspension | 4 balloon wheels on independent sprung swing arms | 6-wheel rocker-bogie with a differential | 6 wheels: rigid middle axle, sprung front and rear swing arms |
| Top speed | 0.22 m/s | 0.15 m/s | 0.10 m/s |
| Cameras | RGB-D navcam + DINOv2 codes, front and rear hazcams | RGB-D navcam + DINOv2 codes, hazcam, RMI telescope camera | RGB-D navcam (colour + geometry), hazcam, drillcam |
| Unique sensors | wheel torque, sinkage camera | spectrometer | core drill, SO-101 arm |
| Unique table columns | `sinkage_mm`, `rolling_res` | `photochem_score`, `mineral_class` | `drill_torque_nm`, `sample_success` |
| Look | orange, roll cage, body-mounted solar, LED lamps, dish | gold, SuperCam-style head, gimballed antenna, RTG | white, solar wings, drill tower, sample carousel, folded arm |

Each design has its own measured skid-steering track factor and Bekker soft-ground scaling (the light scout sinks less on the same sand). **Each rover runs its own TabPFN**. A model only uses the columns its own sensors produce, so another rover's rows help through the shared columns.

### 4.4 How a rover drives (AutoNav-style, no map on board)
1. **Look:** the navcam (RGB-D on the mast) and the hazcam build a **6 m local map at 5 cm** around the rover: height, colour, texture and 8 frozen DINOv2-S codes.
2. **Candidates:** 128 arcs (speed × turn rate). For each, features of the ground it would cross: slope, cross-slope, step, drop, roughness, grade per wheel track, how much was seen, colour, texture, codes; plus IMU, payload, site and body descriptors.
3. **Predict:** one batched TabPFN pass gives each arc's progress (1 − slip), drift and tilt as quantiles, and P(stuck).
4. **Score:** predicted progress × speed × closeness to the goal, minus risk. **Pessimistic quantiles** (q10 progress, q90 tilt) are scored, not means.
5. **Vetoes (geometry, not learned):** a step over 5.5 cm, a drop over 6 cm, ground steeper than **18°** along any wheel track (a crater wall can't be climbed back out of), mostly unseen ground, a structure in the way, a predicted q90 tilt past the shield's limit.
6. **Drive 1 s**, measure what happened, store the row.
7. **Commit when sure:** if the brain was confident (P(fail) < 0.1, q10 progress ≥ 0.4, a narrow band, the last segment matched its prediction) and the arc is still safe on the map, the rover drives the same arc for up to 2 more segments **without a new look or a new plan**. That's always inside the 0.9 m it judged when it chose. Each segment is still its own row. On a 2.8 m sampler leg this cut decisions from 18.5 to 8.7 per metre and the leg from 113 s to 71 s, with no failed segments.

**Routing:** legs longer than 1.5 m follow waypoints from an **orbital route** over the graded DTM: the slopes a ground team gets from HiRISE, with crater walls dilated by a margin. Every arc between waypoints is still the rover's own choice.

**Safety shield** (`safety.py`, no TabPFN):
- Its limits come from the rover's own geometry: the static tip angle, with a stop at 0.6 × it, and a stopping distance of v²/2a + 0.15 m.
- Every 20 ms it checks tilt: a stop happens when tilt is over the limit **and still rising**, so a rover resting tilted can drive off but can't tip further.
- Every 0.2 s it takes a fresh hazcam frame and checks the ground within the stopping distance (a step over 7 cm or a drop).
- A stop brakes the rover, and the row is labelled failed, so TabPFN learns which arcs end in a stop.
- The escape after a stop checks both 60° headings on the map *before* turning, takes the gentler safe one, and watches tilt while turning and backing up.

**Site selection:** drill sites and vantage points must be open ground:
- a gentle slope;
- not in a crater's bowl;
- even all around (less than 3 cm off the best-fit plane within 0.45 m, so not on a rim or a wall);
- no rock taller than the wheels' clearance nearby, rim boulders included.

### 4.5 The arm and the drill (sampler)
- **Arm:** an SO-101 mounted low on the front (base 16.9 cm above the ground), folded across the front while driving: 0 pixels in any camera, 3.9 cm clearance. Its reach band on the ground is 7.5–27.5 cm, and it picks at 17 cm.
- **Grasping:** the lab's TabPFN pick brain, seeded with 20 lab rows. A failed grasp is a row: the sampler re-stands on the rock where it now lies and tries again, up to 3 times. In the last full run, try 1 toppled the rock; with that one row, the predicted P(grasp) rose from 0.75 to 0.81 and the landing uncertainty fell from 4.2 to 1.4 cm, and try 2 grasped it.
- **Carry:** the rock is held folded over the deck while the rover drives home (a gripper weld only during the drive; the grasp, lift and place are physical), docks, unfolds and sets it in the tray.
- **Drill** (a real sequence, not a plunge):
  1. Lower until the bit's tip touches the ground: the drill's own contact, within 0.3 mm of the true surface.
  2. Spin up the bit; the flutes are at its tip.
  3. Feed 4 cm at a rate set by the measured torque: slower in hard rock.
  4. Stop and break the core off.
  5. Retract; the core shows in the bit if it held together.
  6. Turn the carousel one slot (45°) to store the core, and leave the hole marked on the ground.

### 4.6 What is learned, and from what

| Skill | Features (measured, no labels) | Actions | Targets |
|---|---|---|---|
| `drive_arc` (1 s) | the arc's ground from the rover's own cameras, ground under the rover, IMU, payload, site, body descriptors | speed (only slower than nominal), turn rate | progress, drift, tilt, stuck |
| `site_survey` | position, orbital slope and orbital colour (public HiRISE-like data only, no body features) | — (a candidate grid is scored) | photochem score, sinkage, slip, drill torque, core success |
| `pick` | object pose in the arm frame, object type, site | grasp yaw, height, offsets, force, lift speed, aim | P(grasp), P(topple), placement error |

The sampler's site model scores candidate drill sites by **pessimistic photochem (q20) minus a risk term** (predicted sinkage and slip, P(core fail)). It can only do that if it was briefed with the scientist's spectra and the scout's probes. Without a briefing, it has no photochem head and drills the nearest candidate.

### 4.7 Seeing TabPFN decide: the decision panel (`rowver/panel.py`, `panel_card.py`)

With `--gui`, the viewer shows what TabPFN weighed for every decision. It is on by default; `--no-panel` turns it off. The panel only observes: a check fingerprinted the physics, the RNGs, the map, the tables and the fan around every panel callback, and none of them changes.

- **In 3D:** one marker per rover, at its drive's final destination (not the route's waypoints); the marker disappears on arrival. The camera glides to the rover that is deciding. The arcs are drawn only in the sidebar.
- **The sidebar (left):** the decision in plain words, not just numbers.
  - **Mission:** where the rover is heading (for example "drill site 1/1"), distance covered against the total, metres to go and time on this drive.
  - **What it decided:** for example "TabPFN chose a gentle left curve. It expects to cover 83% of the planned distance, with a 6% chance of getting stuck. It overruled the geometry-only choice: TabPFN rated that arc 31% likely to get stuck." Then the result in green, amber or red.
  - **The arcs it weighed, seen from above:** about 24 representative arcs (one per band of curvature, trimmed after a 120° turn). Open arcs are coloured indigo → orange by TabPFN's score; arcs TabPFN ruled out are coral dashes; map-blocked arcs are grey; the geometry-only pick is dashed cyan; the chosen arc is gold with an arrowhead. The diagram also shows range rings, the goal and a legend that counts the open, ruled-out and blocked arcs.
  - **Predicted vs what happened:** gauges for progress (TabPFN's likely range, its best guess, and the outcome), the chance of getting stuck, and body tilt against the limit.
  - **Track record:** TabPFN's range against reality over the last 40 segments, and how often reality fell inside it.
  - **Other decisions:** the drill site the site model ranked, and the arm grasp.
  - **The fleet:** one card per rover, with its status, decisions, commits, predicted vs real stuck rate and overrides.
- Note: MuJoCo 4.0.1's viewer draws user images only while a user text is set, so the panel also sets a small caption.
- **Hotkeys** (every letter and 0–5 belong to the viewer):
  - `9` shows or hides the panel;
  - `7` switches the rover it follows (auto follows the rover that decided last);
  - `8` turns camera follow on or off.

---

## 4b. The science chain: scout → scientist → sampler, and every sensor in the tables

The three rovers now work like one Mars science campaign, and each one hands the next what it learned.

1. **The scout learns where it is safe to drive.** At each probe it records slip (odometry against motion),
   sinkage (wheel camera), rolling resistance (mean drive-motor torque ÷ (m·g·r), from the wheel motors) and
   vibration (spread of the IMU acceleration). Every rover also adds a passive trafficability row every third
   straight drive segment (slip, rolling resistance, vibration), so all driving becomes experience.
2. **The scientist uses that experience to drive safely and find the important rocks.** It undocks with the
   scout's rows, fits its own TabPFN site model on them and scores each vantage point by coverage minus the
   predicted risk (q90 sinkage and slip); risky vantage points are skipped. At each vantage it takes ground
   spectra, then looks around (four navcam frames) and reads every loose rock in range with its RMI camera
   (size), LIBS (SiO₂), Raman (organics) and VISIR (hydration, mineral class). One row per rock (`rock_id`).
3. **The sampler uses both.** Its briefing carries the site rows and the rock rows. Its TabPFN rock model
   (position, orbital colour, rock type, SiO₂, hydration, size → organics) ranks the read rocks by the lower
   quantile of organics, minus the trafficability risk at the rock; it drives there and picks the best rock
   it has not taken yet. A second sortie takes a different rock. With no readings it falls back to the rock
   nearest its drill hole.

Sim side, the rocks are real geology now: a rock's type is the ground's mineral where it lies (basalt,
carbonate, clay → mudstone, sand → sandstone), its colour follows its type, and its organics come from the
spot's photochemistry plus its type. The layout changes with `--seed`.

**Stereo vision.** Each rover's navcam and front hazcam are now stereo pairs (baselines 8.4 cm and 5 cm:
Perseverance's 42 cm and 25 cm at 1:5). `rowver/stereo.py` renders both eyes, matches them with OpenCV's
semi-global block matcher (sub-pixel, uniqueness and left-right checks, speckle filter) and triangulates
depth = f·B/disparity; unmatched pixels stay unseen. Measured against the renderer's depth on field frames:
median height error 5 mm under 1 m, 8 mm at 1–2 m, 19 mm at 2–3.5 m, with 65–70 % of pixels matched.
`rowver.toml [navcam] depth = "stereo" | "render"` (env `ROWVER_DEPTH`) switches it, so the two can be
compared on the same seed. On seed 0 both gave 0 stucks and 2 shield stops, so stereo is the default.

**Visual odometry.** Between two navcam frames the rover matches ORB features, takes their 3D positions
from the stereo depth of the first frame and solves for the camera's motion (solvePnPRansac). Slip =
1 − visual distance ÷ the distance the wheels rolled (wheel angle × radius): how real rovers measure slip.
The passive trafficability rows use it (`slip_vo`).

**A map that outlives the window.** The navcam's 6 m local map forgets what falls behind. Each rover now also
keeps a whole-run site map at 10 cm (`rowver/sitemap.py`): how often each cell was seen, its height spread,
relief, colour, texture and DINOv2 codes. Every site row and every candidate the rovers score carries 16 map
columns from it (`m_views, m_relief, m_step, m_slope, m_rough, m_val, m_sat, m_tex, m_e0..m_e7`; NaN where
nobody looked), so FEATURES grows from 6 to 22. At the dock a rover uploads what it added; a briefed rover
undocks with the dock's merged map, so the scientist starts with the scout's eyes (3,425 cells in the test
run) and the sampler with both (3,997, then 5,520 on its second sortie).

**Pointing the mast.** Instruments that only look straight ahead cannot see a rock off to the side, so
`rowver/mast.py` aims them like a pan-tilt head, from their real position on the rover:
- the scientist's RMI telescope (8°): the rock's colour and texture, and its size from its pixel extent;
- the scientist's colour zoom pair (9 cm baseline, 25°): a parallel stereo pair, block-matched at the rock's
  pixels -> its range and its size from the 3D extent of its points;
- the sampler's panoramic camera bar (17 cm, 70°): the chosen rock's range and size as it arrives;
- the sampler's drill camera: brightness and texture of the ground under the bit, before it drills.
Every number comes from the images; the rock rows carry `rmi_*`, `zoom_*`, `pano_*` and `drill_*` columns.

## 5. The headline experiment: does the briefing help?

**A/B:** the same world and seed. Scout and scientist run once, and their master table is saved. Then the sampler runs twice on copies of that table:
- **brief:** the dock briefs it (site pack + route pack);
- **no-brief:** only its own rows.

**Measured** (mean ± sd over seeds, mission rows only):
- stucks and mean slip on the way;
- time to target;
- **true photochem of the drilled site** (ground truth, used only for scoring);
- core success;
- rock in the tray;
- pack size.

**Hypothesis:** the briefed sampler drills richer sites, because the scientist's spectra train its site model. It also gets stuck less, because the scout's sand failures along its route are in its pack. That's experience transfer across *different bodies and sensors*, through a dock, with no training.

---

## 6. Where it stands (2026-11-22)

**Working, measured:**
- **Full fleet sortie** (scout → scientist → sampler), all three back in their bays, 0 stucks, master table 330 rows.
  - The sampler's briefed site model chose its site and drilled an intact core in carbonate bedrock (true photochem 0.66).
  - It then picked a rock (try 2), carried it home and set it in the tray: 1/1.
- **Docking:** all three designs dock within 5–16 mm from four headings, and back out to within 3–6 mm of the apron point.
- **Crater safety:**
  - fresh crater walls are blocked from all 8 sides;
  - 0 of 592 arcs on open ground are wrongly blocked;
  - routes clear crater rims;
  - no flips: the worst escape tilt is 29° against a 43° tip-over.
- **Sampler check:** cameras clear, brake drift 0.01 mm, ground pick, carry and tray all OK.
- **Drill sequence:** tested on bedrock and sand: contact, 4 cm depth, carousel 45°/90°, hole marker on the ground.
- **Commit:** decisions per metre halved on a sampler leg, with no failed segments.
- **Memory:** a fleet build fits in about 7.7 GB, using shared renderers and hull-only rock colliders.

**Next:**
1. A full fleet run with the viewer, showing the new drill sequence and the commits. The last attempt was stopped by Claude Code for low system memory during the scout's trip home.
2. **The A/B over several seeds** (`tools/ab_fleet.py`, headless).
3. Feed known rock positions into the orbital route: the sampler lost time stopping three times at one 7 cm rock it couldn't see on the DTM.

**Later**:
- stereo depth instead of the simulator's depth;
- visual odometry with station fixes instead of a noisy simulator pose;
- a goodness map and Field D*;
- calibrating drift and the tilt envelope on PDS data.

After that, a real-hardware path: the `UDPNerve` body contract already lets a body run in another process.

---

## 7. Honest limits
- **Terrain physics:** friction, rolling resistance and contact softness, not granular sand. Sinkage is capped (about 18 mm at 1:5).
- **Sand is hard to see:** the orthoimage gives about 2 cm per pixel with no grain or ripples, so sand looks only a little different from gravel, even to DINOv2 (AUC 0.75–0.83). Rovers learn sand partly by its look and partly by where they or another rover already drove.
- **Perception is simplified:** depth comes from the simulator's RGB-D cameras, and pose is the simulator's with added noise (stereo and visual odometry are planned).
- **Arm help:** object positions for the arm come from the simulator, with noise. A weld holds the rock only while the rover drives with it.
- **Speed vs real time:** decisions take wall time. TabPFN's planning pass is about 1.2 s per decision on an RTX 3060, so a mission runs slower than real time (commits help).
- **Novelty:** each piece exists somewhere in research (rover slip prediction, GP-based control, AutoNav). What's new is getting all of it from **one off-the-shelf tabular foundation model, with zero training**, and sharing it across different robots through a dock.

---

## 8. Stack and how to run

- **Stack:**
  - MuJoCo (Python bindings), TabPFN v3.5-fast (local GPU or Prior Labs cloud), DINOv2 ViT-S/14 (frozen, 8 PCA codes), Python on Windows (`.venv`).
  - Licences: the TabPFN v3.5 weights and LAFAN1 are non-commercial (CC BY-NC).
- **Secrets:** the TabPFN key lives only in `.env` as `TABPFN_TOKEN`. Never commit it.

```bat
:: the fleet: scout, scientist, sampler (add --gui for the viewer)
.venv\Scripts\python.exe -m rowver.run_fleet --fresh --scout 2 --vantage 1 --sites 1

:: the control arm: the sampler without the dock's briefing
.venv\Scripts\python.exe -m rowver.run_fleet --fresh --no-brief

:: checks
.venv\Scripts\python.exe -m rowver.tools.check_rover --design sampler
.venv\Scripts\python.exe -m rowver.tools.check_navcam

:: the other robots
.venv\Scripts\python.exe so101\run_throw.py --gui
.venv\Scripts\python.exe -m humanoid_robot.run_brain --practice 1000
```

Key files of the fleet:
- `mars_rover.py`: the three designs, the arm mount, the drill and carousel.
- `world.py`: the site, garage, rocks, hole markers.
- `garage.py`: the docking station.
- `rover.py` / `rover_body.py`: driving, docking, escape, commit.
- `navcam.py`: the local map and arc features.
- `safety.py`: the shield.
- `navigation.py`: orbital routes.
- `survey.py`: site selection and the site model.
- `geology.py`: the hidden fields and instruments.
- `dock.py`: upload and briefing.
- `missions.py`: the sorties, drill and rock pick.
- `fleet.py`: wiring.
- `run_fleet.py`: the CLI.
- `rowver.toml`: every tunable, tagged.
