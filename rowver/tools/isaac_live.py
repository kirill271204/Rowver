"""Live Isaac Sim mirror of a Rowver fleet run: renders, never simulates.

  docker exec -u ubuntu:ubuntu -w /isaac-sim vscode /isaac-sim/python.sh /tmp/rowver_tools/isaac_live.py \
      --dir /tmp/rowver_live
  then open <the instance's secure link>/viewer in a browser (WebRTC stream of the Kit UI).

Waits for <dir>/ready.json (written by rowver/live.py when the mission starts), builds the stage
(isaac_scene.py: x5 root, terrain, the MuJoCo geoms, sun, Mars sky, haze, cameras), then applies the UDP
pose packets from the MuJoCo process as they arrive. No physics runs here. The chase camera follows the
rover the TabPFN panel follows; the panel image (TCP) is shown in a window docked left of the viewport.
--test N: render N frames, save the viewport to <dir>/first_frame.png, quit.
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/rowver_live")
ap.add_argument("--port", type=int, default=47410)
ap.add_argument("--panel-port", type=int, default=47411)
ap.add_argument("--sky-hdr", default="/tmp/rowver_tools/mars_sky.hdr")
ap.add_argument("--film-iso", type=float, default=70.0)
ap.add_argument("--fog", type=float, default=0.0, help="old short-range fog (keep 0; --haze is the Mars haze)")
ap.add_argument("--haze", type=float, default=0.3, help="dust haze density at ~9 km (0: off)")
ap.add_argument("--motes", action="store_true", help="also fine dust drifting in the air near the rover")
ap.add_argument("--wheel-dust", action="store_true", help="dust and grains thrown up by the wheels (off by default)")
ap.add_argument("--no-fx", action="store_true", help="no backdrop, dust film, drifting dust or wheel dust")
ap.add_argument("--backdrop", default="/tmp/rowver_tools/backdrop/scene_backdrop.usdc")
ap.add_argument("--sun", type=float, default=2500.0)
ap.add_argument("--sky", type=float, default=600.0)
ap.add_argument("--renderer", default="RaytracedLighting")
ap.add_argument("--camera", default="chase", choices=["chase", "overview"])
ap.add_argument("--test", type=int, default=0, help="render this many frames, save first_frame.png, quit")
ap.add_argument("--no-stream", action="store_true")
ap.add_argument("--views", default="top,overview", help="extra viewport windows, from top,navcam,overview ('' : none)")
ap.add_argument("--chrome", action="store_true", help="keep the editor UI (menus, toolbars, tabs, axis guides)")
ap.add_argument("--shot", default="", help="save a screenshot of the whole streamed window here once, keep running")
ap.add_argument("--shot-after", type=int, default=300, help="... after this many pose updates")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp(launch_config={"width": 1920, "height": 1080, "window_width": 1920, "window_height": 1080,
                                   "headless": True, "hide_ui": False, "renderer": args.renderer,
                                   "display_options": 3094})   # 3094: no grid
from isaacsim.core.experimental.utils.app import enable_extension  # noqa: E402

if not args.no_stream:
    enable_extension("omni.kit.livestream.app")

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from pxr import Sdf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac_fx as FX  # noqa: E402
import isaac_scene as S  # noqa: E402

HEAD = struct.Struct("<4sIdbBHH")
ROVER = struct.Struct("<4f")
ENTRY = struct.Struct("<H12f")


def log(*a):
    print("[rowver live]", *a, flush=True)


base = Path(args.dir)
log(f"waiting for {base / 'ready.json'} (start the mission with --live)")
while not (base / "ready.json").exists():
    app.update()
    time.sleep(0.05)
time.sleep(0.5)
ready = json.loads((base / "ready.json").read_text())
names = json.loads((base / ready["names"]).read_text())
log(f"scene: {ready['geoms']} geoms, rovers {ready['rovers']}")

ctx = omni.usd.get_context()
ctx.new_stage()
app.update()
stage = ctx.get_stage()
cams = S.build(stage, base, ready["scene"], ready["terrain"], args.sky_hdr, args.sun, args.sky)
if not args.no_fx:
    log(f"backdrop: {FX.backdrop(stage, args.backdrop)}")
for _ in range(600):                                   # layers, textures, shaders
    app.update()
    _, loaded, total = ctx.get_stage_loading_status()
    if total == 0 or loaded >= total:
        break
S.render_settings(carb.settings.get_settings(), args.film_iso, args.fog, args.renderer)
motes = dust = None
if not args.no_fx:
    FX.haze(carb.settings.get_settings(), density=args.haze)
    log(f"dust film on {FX.dusty(stage)} materials; wheel dust {'on' if args.wheel_dust else 'off'}")
    motes = FX.Motes(stage) if args.motes else None
    wheels = json.loads(Path("/tmp/rowver_tools/dust_wheels.json").read_text()) \
        if Path("/tmp/rowver_tools/dust_wheels.json").exists() else {}
    dust = FX.WheelDust(stage, wheels, [ready["rovers"][r] for r in ready["rovers"]]) if args.wheel_dust else None

attrs = []
missing = 0
for n in names:
    p = stage.GetPrimAtPath(f"/World/Mars/Fleet/Mesh_Xform_{n}")
    a = p.GetAttribute("xformOp:transform") if p.IsValid() else None
    if a is None or not a.IsValid():
        missing += 1
        attrs.append(None)
        continue
    a.Set(a.Get())                                     # make the over now: later writes are value-only
    attrs.append(a)
log(f"bound {len(names) - missing}/{len(names)} geom transforms")

vp = get_active_viewport()
vp.camera_path = str(cams[args.camera][0].GetPath())

# ------------------------------------------------------------ the TabPFN panel window (left of the viewport)
panel_img = {"rgba": None, "new": False}


def panel_server():
    from io import BytesIO

    from PIL import Image
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.panel_port))
    srv.listen(1)
    while True:
        c, _ = srv.accept()
        f = c.makefile("rb")
        try:
            while True:
                hdr = f.read(4)
                if len(hdr) < 4:
                    break
                n = struct.unpack("<I", hdr)[0]
                data = f.read(n)
                im = Image.open(BytesIO(data)).convert("RGBA")
                panel_img["rgba"], panel_img["new"] = np.asarray(im), True
                panel_img["n"] = panel_img.get("n", 0) + 1
        except Exception as e:  # a dropped connection: wait for the next one
            log(f"panel link: {e}")
        c.close()


threading.Thread(target=panel_server, daemon=True).start()
provider = ui.ByteImageProvider()
win = ui.Window("TabPFN decision panel", width=480, height=1080, padding_x=0, padding_y=0)
PANEL_BG = 0xFF17100D                                   # the panel card's own background, rgb(13, 16, 23) as ABGR
win.frame.set_style({"Window": {"background_color": PANEL_BG}})
with win.frame:
    with ui.ZStack():
        ui.Rectangle(style={"background_color": PANEL_BG})
        ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
for _ in range(5):
    app.update()
vpw = ui.Workspace.get_window("Viewport")
if vpw is not None:
    win.dock_in(vpw, ui.DockPosition.LEFT, 0.22)
# ------------------------------------------------------------ extra views (stacked right of the main viewport)
from omni.kit.viewport.utility import create_viewport_window  # noqa: E402

mounts = json.loads(Path("/tmp/rowver_tools/navcams.json").read_text()) if Path("/tmp/rowver_tools/navcams.json").exists() else {}
TITLES = {"top": "Top (deciding rover)", "navcam": "NavCam (deciding rover)", "overview": "Overview", "chase": "Chase"}
views = [v for v in args.views.split(",") if v in cams]
extra = []
for v in views:
    extra.append(create_viewport_window(TITLES[v], width=640, height=360, camera_path=str(cams[v][0].GetPath())))
for _ in range(5):
    app.update()
if vpw is not None and extra:
    extra[0].dock_in(vpw, ui.DockPosition.RIGHT, 0.34)
    for prev, w in zip(extra, extra[1:]):
        w.dock_in(prev, ui.DockPosition.BOTTOM, 0.5 if len(extra) == 2 else (0.67 if w is extra[1] else 0.5))
log(f"views: main = {args.camera}, extra = {views}")


def clean_ui():
    """A presentation layout: every view renders at its window's size (no letterbox) and the editor chrome
    is hidden (main menu, toolbar, viewport menubars, tab bars, axis guides, HUD)."""
    st = carb.settings.get_settings()
    for w in [vpw] + extra:
        if w is None:
            continue
        try:
            w.viewport_api.fill_frame = True
        except Exception as e:
            log(f"fill_frame: {e}")
    if args.chrome:
        return
    for k in ("render", "camera", "display", "settings", "lighting", "delegate", "waypoint", "core"):
        st.set(f"/exts/omni.kit.viewport.menubar.{k}/visible", False)
    for w in [vpw] + extra:                       # the toolbar over each view is the window's "Menubar" layer
        try:
            layer = w._find_viewport_layer("Menubar", "menubar") if w is not None else None
            if layer is not None:
                layer.visible = False
        except Exception as e:
            log(f"menubar layer: {e}")
    for w in [vpw, win] + extra:
        if w is not None:
            w.dock_tab_bar_visible = False
    tb = ui.Workspace.get_window("Main ToolBar")
    if tb is not None:
        tb.visible = False
    try:
        import omni.kit.mainwindow
        mw = omni.kit.mainwindow.get_main_window()
        mw.get_main_menu_bar().visible = False
        try:
            mw.get_status_bar_frame().visible = False
        except Exception:
            pass
    except Exception as e:
        log(f"main menu: {e}")

    def walk(prefix, node):                       # axis guide, grid and HUD off in every viewport
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}", v)
        elif prefix.endswith(("/guide/axis/visible", "/guide/grid/visible")) or ("/hud/" in prefix and prefix.endswith("/visible")):
            st.set(prefix, False)
    for root in ("/persistent/app/viewport", "/app/viewport"):
        walk(root, st.get(root) or {})
    for k in ("guide/axis", "guide/grid", "hud/renderFPS", "hud/cameraSpeed", "hud/toastMessage", "hud/renderResolution"):
        st.set(f"/app/viewport/defaults/{k}/visible", False)


clean_ui()

for wname in ("Stage", "Property", "Content", "Console", "Layer", "Render Settings", "Isaac Sim Assets [Beta]",
              "Semantics Schema Editor", "Physics", "Assets"):
    w = ui.Workspace.get_window(wname)
    if w is not None:
        w.visible = False


def show_panel():
    a = panel_img["rgba"]
    panel_img["new"] = False
    h, w = a.shape[:2]
    try:
        provider.set_data_array(np.ascontiguousarray(a, np.uint8).reshape(-1), [w, h])   # flat RGBA bytes
    except Exception as e:
        if not panel_img.get("warned"):
            log(f"panel: set_data_array failed ({e}), using set_bytes_data")
            panel_img["warned"] = True
        provider.set_bytes_data(a.ravel().tolist(), [w, h])
    panel_img["shown"] = panel_img.get("shown", 0) + 1


# ------------------------------------------------------------ pose packets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
sock.bind(("127.0.0.1", args.port))
sock.setblocking(False)

chase = S.Chase()
rovers = list(ready["rovers"])
last_frame, active, rover_pose = -1, 0, None
t_last, n_pk, n_frames, t_log = time.time(), 0, 0, time.time()
log("live: applying pose packets; open <secure link>/viewer")
while app.is_running():
    latest = {}
    while True:
        try:
            msg = sock.recv(65536)
        except BlockingIOError:
            break
        magic, frame, t, ai, nr, ne, flags = HEAD.unpack_from(msg, 0)
        if magic != b"RWV1":
            continue
        n_pk += 1
        if frame < last_frame - 30:                     # a new run started: accept it
            last_frame = -1
        if frame < last_frame:
            continue
        last_frame = frame
        off = HEAD.size
        rover_pose = [ROVER.unpack_from(msg, off + k * ROVER.size) for k in range(nr)]
        off += nr * ROVER.size
        active = ai if ai >= 0 else active
        for k in range(ne):
            e = ENTRY.unpack_from(msg, off + k * ENTRY.size)
            latest[e[0]] = e[1:]
    if latest:
        with Sdf.ChangeBlock():
            for j, v in latest.items():
                a = attrs[j] if j < len(attrs) else None
                if a is not None:
                    a.Set(S.matrix(v[:3], v[3:]))
        n_frames += 1
    now = time.time()
    if rover_pose is not None:
        x, y, z, yaw = rover_pose[active]
        cams["chase"][1].Set(chase.step(x, y, z, yaw, now - t_last))
        if motes is not None:
            motes.step(chase.tgt, min(now - t_last, 0.1))
        if dust is not None:
            dust.step(rover_pose, min(now - t_last, 0.1))
        cams["overview"][1].Set(S.overview(np.array([p[:3] for p in rover_pose])))
        cams["top"][1].Set(S.top(x, y, z))
        m = mounts.get(ready["rovers"].get(rovers[active], ""), None)
        if m is not None:
            cams["navcam"][1].Set(S.navcam(x, y, z, yaw, m))
    t_last = now
    if panel_img["new"]:
        show_panel()
    app.update()
    if n_frames in (30, 120):
        clean_ui()
    if args.shot and n_frames >= args.shot_after:
        shot, args.shot = args.shot, ""
        try:
            enable_extension("omni.kit.renderer.capture")
            import omni.kit.renderer_capture
            omni.kit.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(shot)
            log(f"window screenshot -> {shot}")
        except Exception as e:
            log(f"screenshot failed: {e}")
    if now - t_log > 10:
        log(f"packets {n_pk}, pose updates {n_frames}, last sim frame {last_frame}, following {rovers[active]}, "
            f"panel images in {panel_img.get('n', 0)} / shown {panel_img.get('shown', 0)}")
        t_log = now
    if args.test and n_frames >= args.test:
        for _ in range(30):
            app.update()
        from omni.kit.viewport.utility import capture_viewport_to_file
        for cname in ("chase", "overview"):
            vp.camera_path = str(cams[cname][0].GetPath())
            for _ in range(40):
                app.update()
            capture_viewport_to_file(vp, str(base / f"first_frame_{cname}.png"))
            for _ in range(20):
                app.update()
            log(f"saved {base / f'first_frame_{cname}.png'}")
        break
app.close()
