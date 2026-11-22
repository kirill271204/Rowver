"""Live link to Isaac Sim: a read-only mirror of a fleet run (run_fleet.py --live).

MuJoCo keeps every bit of the logic; this only reads it. At start it writes the static scene once:

  <run dir>/live/usd/frames/scene.usdc   every visible geom at its start pose (MuJoCo's USDExporter geometry)
  <run dir>/live/terrain.usdc            the compiled ground hfield as a textured mesh (craters included)
  <run dir>/live/names.json              geom index -> prim name (the order the pose packets use)
  <run dir>/live/ready.json              written last: Isaac starts loading when it sees it

and copies the folder into the Isaac container (docker cp, --live-container). Then, through World.on_step,
at --live-fps of sim time it sends UDP pose packets to Isaac (127.0.0.1: the container shares the host
network): the geoms that moved since the last packet, plus a full keyframe every second, and each
rover's chassis pose and the rover the panel follows. The TabPFN panel image goes over a TCP side
channel whenever it changes. Sending never blocks the sim: UDP is fire and forget, the panel is sent from
its own thread (latest image only).

--live-realtime R paces the sim to R x wall time when it runs ahead (as the GUI viewer does). That changes
only wall time: the brain's staleness counts sim seconds, decisions and physics are the same.
"""
from __future__ import annotations

import json
import queue
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from .record import _exporter_class, export_terrain, shared_panel

MAGIC = b"RWV1"
HEAD = struct.Struct("<4sIdbBHH")        # magic, frame, sim time, active rover, n rovers, n entries, flags
ROVER = struct.Struct("<4f")             # chassis x, y, z, yaw (sim scale)
ENTRY = struct.Struct("<H12f")           # geom index, position (3), rotation matrix (9, row major)
MAX_ENTRIES = 1100                       # per datagram (~55 kB, under the 64 kB UDP limit)
FLAG_KEY = 1


def scene_snapshot(m, d, out: Path):
    """Every visible geom at its current pose as a USD without time samples -> (exporter, names)."""
    ground = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD]
    Exp = _exporter_class()
    exp = Exp(m, ground, output_directory="usd", output_directory_root=str(out), verbose=False)
    exp.update_scene(d)
    for ref in exp.geom_refs.values():                # frame-0 samples -> plain (default) values
        for attr in (ref.transform_op.GetAttr(), ref.usd_xform.GetVisibilityAttr()):
            v = attr.Get(0)
            attr.Clear()
            if v is not None:
                attr.Set(v)
    path = Path(exp.frames_directory) / "scene.usdc"
    exp.stage.Export(str(path))
    return exp, list(exp.geom_refs.keys()), path


class LiveLink:
    def __init__(self, fleet, out_dir: Path, fps=30.0, realtime=1.0, host="127.0.0.1", port=47410,
                 panel_port=47411, container="vscode", container_dir="/tmp/rowver_live", panel=True):
        self.f, self.w = fleet, fleet.w
        self.m, self.d = self.w.m, self.w.d
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dt, self.realtime = 1.0 / float(fps), float(realtime)
        self.addr, self.panel_addr = (host, int(port)), (host, int(panel_port))
        self.frame, self.t0 = 0, float(self.d.time)
        self.rovers = list(fleet.rv)
        self.ch = {r: self.m.body(f"{r}/chassis").id for r in self.rovers}

        terrain = export_terrain(self.m, self.out)
        self.exp, self.names, scene = scene_snapshot(self.m, self.d, self.out)
        self.index = {n: k for k, n in enumerate(self.names)}
        self.key = {}                                     # (objtype, objid) -> geom index
        self.last = np.full((len(self.names), 12), np.nan, np.float32)
        (self.out / "names.json").write_text(json.dumps(self.names))
        (self.out / "ready.json").write_text(json.dumps({
            "scene": str(scene.relative_to(self.out)), "terrain": "terrain.usdc", "names": "names.json",
            "geoms": len(self.names), "rovers": {r: self.w.designs[r].name for r in self.rovers},
            "port": port, "panel_port": panel_port, "fps": fps, "scale_to_real": 5.0, "terrain_info": terrain,
            "written": time.time()}, indent=2))
        if container:
            subprocess.run(["docker", "exec", container, "rm", "-rf", container_dir], check=False)
            subprocess.run(["docker", "cp", str(self.out), f"{container}:{container_dir}"], check=True)
            print(f"live: scene ({len(self.names)} geoms) copied to {container}:{container_dir}", flush=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        self.panel = shared_panel(fleet) if panel else None
        self._pq: queue.Queue = queue.Queue(maxsize=1)
        self._panel_ver, self._panel_t = None, -1e9
        threading.Thread(target=self._panel_sender, daemon=True).start()
        self._w0, self._s0 = time.time(), float(self.d.time)
        self.sent = 0
        self.w.on_step.append(self._on_step)

    # ------------------------------------------------------------ per physics chunk
    def _on_step(self, w):
        t = float(self.d.time)
        if t + 1e-9 < self.t0 + self.frame * self.dt:
            return
        while t + 1e-9 >= self.t0 + (self.frame + 1) * self.dt:   # behind: skip to now (live, not a video)
            self.frame += 1
        self._send(t)
        self._panel(t)
        self.frame += 1
        if self.realtime > 0:                             # pace when ahead; never catch up in bursts
            target = self._w0 + (t - self._s0) / self.realtime
            now = time.time()
            if target > now:
                time.sleep(min(target - now, 0.5))
            elif now - target > 0.25:
                self._w0 += now - target

    def _send(self, t):
        self.exp._update_scene(self.d)
        sc = self.exp.scene
        n = len(self.names)
        cur = self.last.copy()
        for i in range(sc.ngeom):
            g = sc.geoms[i]
            k = (int(g.objtype), int(g.objid))
            j = self.key.get(k)
            if j is None:
                j = self.index.get(self.exp._get_geom_name(g), -1)
                self.key[k] = j
            if j < 0:
                continue
            cur[j, :3] = g.pos
            cur[j, 3:] = np.asarray(g.mat, np.float32).ravel()
        keyframe = self.frame % int(round(1 / self.dt)) == 0
        moved = np.arange(n) if keyframe else np.nonzero(~np.all(np.isclose(cur, self.last, atol=1e-6), 1))[0]
        self.last = cur
        active = self.panel.follow() if self.panel is not None else None
        ai = self.rovers.index(active) if active in self.rovers else -1
        rov = b"".join(ROVER.pack(*self.d.xpos[self.ch[r]], self._yaw(self.ch[r])) for r in self.rovers)
        chunks = [moved[a:a + MAX_ENTRIES] for a in range(0, max(len(moved), 1), MAX_ENTRIES)]
        for c in chunks:
            body = b"".join(ENTRY.pack(int(j), *cur[j]) for j in c)
            msg = HEAD.pack(MAGIC, self.frame, t, ai, len(self.rovers), len(c), FLAG_KEY if keyframe else 0) + rov + body
            try:
                self.sock.sendto(msg, self.addr)
                self.sent += 1
            except OSError:
                pass

    def _yaw(self, b):
        R = self.d.xmat[b].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------ the TabPFN panel (TCP, own thread)
    def _panel(self, t):
        p = self.panel
        if p is None or p._ver == self._panel_ver or t - self._panel_t < 0.25:
            return
        from .record import panel_image
        with p.lock:
            v, ver = p.view(), p._ver
        img = panel_image(v)
        self._panel_ver, self._panel_t = ver, t
        try:
            self._pq.get_nowait()
        except queue.Empty:
            pass
        self._pq.put_nowait(img)

    def _panel_sender(self):
        import io

        from PIL import Image
        s = None
        while True:
            img = self._pq.get()
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG", compress_level=1)
            data = buf.getvalue()
            for _ in range(2):
                try:
                    if s is None:
                        s = socket.create_connection(self.panel_addr, timeout=0.5)
                        s.settimeout(2.0)
                    s.sendall(struct.pack("<I", len(data)) + data)
                    break
                except OSError:
                    if s is not None:
                        s.close()
                    s = None

    def close(self):
        if self._on_step in self.w.on_step:
            self.w.on_step.remove(self._on_step)
        self.sock.close()
        return {"frames": self.frame, "packets": self.sent}


def main():
    """Preview: build the same world run_fleet builds (no brain, rovers parked) and stream it, to warm up Isaac
    and check the look before a mission.   python -m rowver.live --seconds 120"""
    import argparse
    from types import SimpleNamespace

    from brain.provenance import ROOT

    from . import world as W
    from .run_fleet import DESIGNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--zone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--container", default="vscode")
    a = ap.parse_args()
    w = W.build(tuple(DESIGNS), seed=a.seed, zone=a.zone, designs=DESIGNS)
    fake = SimpleNamespace(w=w, rv={r: None for r in DESIGNS}, panel=None)
    link = LiveLink(fake, ROOT / "runs" / "live_preview", container=a.container or None, panel=False)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        w._advance(17)                                  # parked rovers, braked: the scene just sits there
    print(link.close(), flush=True)


if __name__ == "__main__":
    main()
