# brain: the robot-agnostic Cortex brain

No robot code lives here. A robot package (`so101/`, `humanoid_robot/`, ...) implements a **body** and imports this package. Nothing in `brain/` imports a robot.

## Regions

| Region | File | Job |
|---|---|---|
| Thalamus (nerves) | `nerves.py` | The body contract: `describe`, `reset`, `context`, `act`. `LocalNerve` (one process) and `UDPNerve` + `serve_udp` (JSON datagrams with sequence numbers, ack, retry) |
| Hippocampus | `hippocampus.py` | One experience table for all robots and skills (`runs_brain/memory.csv`). `recall()`: same robot+skill first, then the same skill family on the nearest other bodies, capped. Delayed failure labels |
| Cerebellum | `world_model.py` | TabPFN: 25 outcome quantiles per target and P(fail), on the recalled rows (context features + action + body descriptors) |
| Prefrontal | `prefrontal.py` | Cross-entropy search over actions, scored by the objective in `brain.toml`. The amygdala vetoes high-P(fail) actions |
| Loop | `brain.py` | sense → recall → predict → decide → act → remember. Modes: `baseline` (no corrections), `practice`, `play` |
| Provenance | `provenance.py` | Loads `brain.toml`. Every number is a `Tagged(value, source)`: `dataset:`, `hardware:`, `experience:`, `user:` |
| Model | `tabpfn_model.py` | TabPFN **v3.5-fast** only, local and offline. `python -m brain.check_tabpfn` downloads it once |

The body runs its own fast loop: sensors → spinal-cord controller → actuators, at 500 Hz. The brain talks to the body once per skill execution, about 2 Hz, and never sits inside that fast loop.

## Adding a robot
1. Make a package next to `brain/`, e.g. `my_robot/`.
2. Write a body class with:
   - `robot_id`;
   - `descriptors`: tagged, body-independent numbers such as leg length and mass, read from the robot's model;
   - `skills`: `{name: SkillSchema}`, where `SkillSchema` sets the action bounds, context features, targets, the goal mapping and the targets to minimise;
   - `reset()`, `context(skill, goal)` and `act(skill, action)`.

   `act` returns the targets, `failed` and `done`. Extra keys are stored for reporting only.
3. Run it with `Brain(LocalNerve(body)).episode(skill, goal, mode)`. To run the body in another process, use `serve_udp(body)` on its side and `UDPNerve()` on the brain's side.

Rows from other robots in the same skill family are used automatically. They carry the body descriptors, so TabPFN can weigh them by how similar the body is.

## Rules
- Every number a skill uses is tagged with its source. Values come from recordings, the robot's model file, the robot's own experience, or `brain.toml`, which belongs to you. None are hand-tuned in the code.
- The baseline is the recorded reference with zero corrections. Nothing tuned by hand is used as a baseline.
