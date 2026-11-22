"""Offline Isaac Sim render of a recorded fleet run (the fallback video): renders, never simulates.

  docker exec -u ubuntu:ubuntu -w /isaac-sim vscode /isaac-sim/python.sh /tmp/rowver_tools/isaac_render.py \
      --dir /tmp/rowver_rec --out /tmp/rowver_render [--step 4] [--count 10]

Reads <dir>/manifest.json, usd/frames/fleet.usdc (animated geoms), terrain.usdc and track.npz (written by
rowver/record.py), builds the same stage as the live mirror (isaac_scene.py) and writes
<out>/frame_NNNNNN.png for every --step-th recorded frame (frame N = recorded frame N), chase camera on the
rover the TabPFN panel followed (or --camera overview). Encode on the host with ffmpeg (see the README).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/rowver_rec")
ap.add_argument("--out", default="/tmp/rowver_render")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=0, help="frames to render (0: all)")
ap.add_argument("--step", type=int, default=1, help="render every Nth recorded frame")
ap.add_argument("--camera", default="chase", choices=["chase", "overview"])
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--sky-hdr", default="/tmp/rowver_tools/mars_sky.hdr")
ap.add_argument("--film-iso", type=float, default=70.0)
ap.add_argument("--fog", type=float, default=0.0)
ap.add_argument("--haze", type=float, default=0.3)
ap.add_argument("--no-fx", action="store_true")
ap.add_argument("--backdrop", default="/tmp/rowver_tools/backdrop/scene_backdrop.usdc")
ap.add_argument("--sun", type=float, default=2500.0)
ap.add_argument("--sky", type=float, default=600.0)
ap.add_argument("--renderer", default="RaytracedLighting")
ap.add_argument("--subframes", type=int, default=2)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp(launch_config={"width": args.width, "height": args.height, "headless": True,
                                   "renderer": args.renderer, "display_options": 3094})
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac_fx as FX  # noqa: E402
import isaac_scene as S  # noqa: E402


def log(*a):
    print("[rowver render]", *a, flush=True)


base, out = Path(args.dir), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
man = json.loads((base / "manifest.json").read_text())
fps, n_frames = float(man["fps"]), int(man["frames"])
tr = np.load(base / "track.npz")
rovers = [str(r) for r in tr["rovers"]]
poses = np.stack([tr[f"{r}_pose"] for r in rovers], 1)          # (frames, rovers, 7): xyz + quat (w x y z)
active = tr["active"].astype(int)
log(f"{n_frames} recorded frames at {fps:g} fps ({n_frames / fps:.0f} s), rovers {rovers}")

# The recorded USD is animated (a time sample per frame for every moving geom). Letting Kit ingest 20k frames
# x 675 prims stalls it, so: Kit renders a STATIC copy, and each frame's transforms are read here with plain
# pxr from the recording and set on that copy (what the live mirror does with its UDP packets).
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

scene_file = base / man["files"]["scene"]
anim = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(scene_file)))    # the recording, outside Kit
static_file = scene_file.with_name("static_scene.usdc")
if not static_file.exists():
    lay = Sdf.Layer.OpenAsAnonymous(str(scene_file))
    st0 = Usd.Stage.Open(lay)
    for p in st0.Traverse():
        for a in p.GetAttributes():
            if a.GetNumTimeSamples():
                v = a.Get(0)
                a.Clear()
                if v is not None:
                    a.Set(v)
    lay.Export(str(static_file))
anim_attrs = []
for p in anim.GetPseudoRoot().GetChildren()[0].GetChildren():
    if p.GetName().startswith("Mesh_Xform_"):
        anim_attrs.append((p.GetName(), p.GetAttribute("xformOp:transform"), UsdGeom.Imageable(p).GetVisibilityAttr()))
log(f"recording: {len(anim_attrs)} animated geoms; Kit renders {static_file.name}")

ctx = omni.usd.get_context()
ctx.new_stage()
app.update()
stage = ctx.get_stage()
cams = S.build(stage, base, str(static_file.relative_to(base)), man["files"]["terrain"], args.sky_hdr, args.sun,
               args.sky, fps=fps)
if not args.no_fx:
    FX.backdrop(stage, args.backdrop)
for _ in range(600):
    app.update()
    _, loaded, total = ctx.get_stage_loading_status()
    if total == 0 or loaded >= total:
        break
S.render_settings(carb.settings.get_settings(), args.film_iso, args.fog, args.renderer)
dust = None
if not args.no_fx:
    FX.haze(carb.settings.get_settings(), density=args.haze)
    log(f"dust film on {FX.dusty(stage)} materials")
    wj = Path("/tmp/rowver_tools/dust_wheels.json")
    dust = FX.WheelDust(stage, json.loads(wj.read_text()) if wj.exists() else {}, [man["rovers"][r] for r in rovers])
kit_attrs = []
for name, ta, va in anim_attrs:
    kp = stage.GetPrimAtPath(f"/World/Mars/Fleet/{name}")
    if kp.IsValid():
        kt, kv = kp.GetAttribute("xformOp:transform"), UsdGeom.Imageable(kp).GetVisibilityAttr()
        kt.Set(kt.Get())
        kv.Set(kv.Get() or "inherited")
        kit_attrs.append((ta, va, kt, kv))
log(f"bound {len(kit_attrs)} geoms")


def pose_frame(k):
    with Sdf.ChangeBlock():
        for ta, va, kt, kv in kit_attrs:
            kt.Set(ta.Get(k))
            kv.Set(va.Get(k) or "inherited")


cam_prim, cam_op = cams[args.camera]
rp = rep.create.render_product(str(cam_prim.GetPath()), (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
tl = omni.timeline.get_timeline_interface()
tl.stop()


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


chase = S.Chase()
frames = list(range(args.start, n_frames, max(args.step, 1)))
if args.count:
    frames = frames[:args.count]
for _ in range(60):                                              # warm-up (shaders, textures)
    app.update()
cur = max(int(active[frames[0]]) if len(frames) else 0, 0)
t0 = time.time()
for i, k in enumerate(frames):
    if active[k] >= 0:
        cur = int(active[k])
    pose_frame(k)
    if args.camera == "chase":
        p = poses[k, cur]
        cam_op.Set(chase.step(p[0], p[1], p[2], yaw_of(p[3:]), args.step / fps))
    else:
        cam_op.Set(S.overview(poses[k, :, :3]))
    if dust is not None:                                         # dust and tracks from the recorded poses
        dust.step([(p[0], p[1], p[2], yaw_of(p[3:])) for p in poses[k]], args.step / fps)
    rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=True)
    img = np.asarray(rgb.get_data())[..., :3].astype(np.uint8)
    Image.fromarray(img).save(out / f"frame_{k:06d}.png")
    if i % 50 == 0:
        el = time.time() - t0
        log(f"{i + 1}/{len(frames)} frames, {el:.0f} s ({(i + 1) / max(el, 1e-3):.1f} fps)")
log(f"done: {len(frames)} frames in {time.time() - t0:.0f} s -> {out}")
(out / "render.json").write_text(json.dumps({"fps": fps, "step": args.step, "frames": frames[:1] + frames[-1:],
                                             "count": len(frames), "camera": args.camera}))
app.close()
