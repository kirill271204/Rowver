"""Recorder for Isaac Sim renders: a read-only observer of a fleet run (run_fleet.py --record).

It never changes the simulation, a rover or a decision. Through World.on_step it samples the MuJoCo
state at a fixed rate of SIM time (default 30 fps) and writes, in <run dir>/record/:

  usd/frames/fleet.usdc   every visible geom (rovers, arm, garage, rocks, sample rocks, hole markers) as an
                          animated USD (MuJoCo's USDExporter geometry and materials). A geom gets a time sample
                          only when it moves, so the static scene costs one sample. The ground hfield is skipped
  usd/assets/             the textures the exporter writes for those geoms
  terrain.usdc            the ground once, as a mesh built from the COMPILED hfield (craters and graded pads
                          included) with the ground's albedo texture (terrain_albedo.png)
  track.npz               per frame: sim time, each rover's chassis pose, the rover the panel follows
  panel/panel_NNNNNN.png  the TabPFN decision panel (panel_card.render_sidebar), saved when it changes, at
                          the frame it changed; panels.jsonl lists them with their sim time
  manifest.json           fps, frame count, rover ids and designs, scale, files

Counterfactual rollouts (world.quiet) never call on_step, so they are not recorded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

PANEL_W, PANEL_H = 480, 1080      # the panel at the video's height (1080 p)
PANEL_MIN_DT = 0.25               # s of sim time between two panel images (the viewer redraws every 0.2 s)


def _exporter_class():
    from mujoco.usd import exporter as E
    from mujoco.usd import utils as U
    class FleetExporter(E.USDExporter):
        """USDExporter that skips the hfield and lights / cameras (Isaac adds its own), and writes a
        geom's transform only when it changes (held values between samples, no drift)."""

        def __init__(self, model, skip_geoms, **kw):
            self.skip = set(int(g) for g in skip_geoms)
            self.last = {}               # name -> (matrix, frame written)
            self.shown = {}              # name -> visible at the last frame
            super().__init__(model, **kw)

        def _load_lights(self):
            pass

        def _update_lights(self):
            pass

        def _load_cameras(self):
            pass

        def _update_cameras(self, data, scene_option=None):
            pass

        def _update_geoms(self):
            f = self.updates
            seen = set()
            for i in range(self.scene.ngeom):
                g = self.scene.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and int(g.objid) in self.skip:
                    continue
                if g.type == mujoco.mjtGeom.mjGEOM_HFIELD:
                    continue
                name = self._get_geom_name(g)
                if name not in self.geom_names:
                    self._load_geom(g)
                    ref = self.geom_refs[name]
                    if f > 0:                                    # appears later: hidden until now
                        ref.update_visibility(False, 0)
                    ref.update_visibility(True, f)
                    self.shown[name] = True
                ref = self.geom_refs[name]
                seen.add(name)
                M = U.create_transform_matrix(rotation_matrix=g.mat, translation_vector=g.pos).T
                prev = self.last.get(name)
                if prev is None or not np.allclose(prev[0], M, atol=1e-6):
                    if prev is not None and prev[1] < f - 1:     # held since prev: pin it before moving
                        ref.transform_op.Set(_m4(prev[0]), f - 1)
                    ref.transform_op.Set(_m4(M), f)
                    self.last[name] = (M, f)
                if not self.shown.get(name, True):
                    ref.update_visibility(True, f)
                    self.shown[name] = True
            for name, vis in self.shown.items():
                if vis and name not in seen:
                    self.geom_refs[name].update_visibility(False, f)
                    self.shown[name] = False

    def _m4(M):
        from pxr import Gf
        return Gf.Matrix4d(M.tolist())

    return FleetExporter


def export_terrain(m: mujoco.MjModel, out: Path, geom_name="ground") -> dict:
    """The ground hfield of the COMPILED model -> terrain.usdc (a triangle mesh, x east, y north, z up,
    metres at sim scale) + terrain_albedo.png. Returns what the manifest needs."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    g = m.geom(geom_name).id
    h = int(m.geom_dataid[g])
    nrow, ncol = int(m.hfield_nrow[h]), int(m.hfield_ncol[h])
    rx, ry, elev, _base = (float(v) for v in m.hfield_size[h])
    adr = int(m.hfield_adr[h])
    H = np.asarray(m.hfield_data[adr:adr + nrow * ncol], float).reshape(nrow, ncol)   # row 0 = south (y = -ry)
    pos = m.geom_pos[g].astype(float)
    xs = np.linspace(-rx, rx, ncol) + pos[0]
    ys = np.linspace(-ry, ry, nrow) + pos[1]
    X, Y = np.meshgrid(xs, ys)
    Z = H * elev + pos[2]
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(nrow * ncol).reshape(nrow, ncol)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]    # CCW seen from above
    faces = np.stack([a, b, c, a, c, d], -1).reshape(-1, 3)
    u, v = np.meshgrid(np.linspace(0, 1, ncol), np.linspace(0, 1, nrow))
    st = np.stack([u, v], -1).reshape(-1, 2)                               # v = 0 at the south edge

    # the ground's texture: MuJoCo's row 0 is the south edge; an image file's top row is v = 1 (north)
    mat = int(m.geom_matid[g])
    tid = int(m.mat_texid[mat][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    th, tw, tc = int(m.tex_height[tid]), int(m.tex_width[tid]), int(m.tex_nchannel[tid])
    tadr = int(m.tex_adr[tid])
    img = np.asarray(m.tex_data[tadr:tadr + th * tw * tc], np.uint8).reshape(th, tw, tc)[:, :, :3]
    Image.fromarray(np.flipud(img)).save(out / "terrain_albedo.png")

    stage = Usd.Stage.CreateNew(str(out / "terrain.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Terrain")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Terrain/ground")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
    # normals from the height grid (smooth shading)
    gy, gx = np.gradient(Z, ys, xs)
    N = np.stack([-gx, -gy, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    mtl = UsdShade.Material.Define(stage, "/Terrain/mat_ground")
    sh = UsdShade.Shader.Define(stage, "/Terrain/mat_ground/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    rd = UsdShade.Shader.Define(stage, "/Terrain/mat_ground/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, "/Terrain/mat_ground/albedo")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./terrain_albedo.png")
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mtl)
    stage.GetRootLayer().Save()
    return {"geom": geom_name, "grid": [nrow, ncol], "half_size_m": [rx, ry], "z_range_m": [float(Z.min()), float(Z.max())],
            "albedo_px": [tw, th]}


def panel_image(v):
    """The presentation panel (panel_pro.py); the original card (panel_card.py) if it fails."""
    try:
        from .panel_pro import render
        return render(v, PANEL_W, PANEL_H)
    except Exception as e:  # never let a drawing problem touch the run
        print(f"panel_pro failed ({e!r}); using panel_card", flush=True)
        from .panel_card import render_sidebar
        return render_sidebar(v, PANEL_W, PANEL_H)


def shared_panel(fleet):
    """The fleet's decision panel; headless, the same read-only observer --gui uses (one per fleet, shared by
    the recorder and the live link: it hooks each brain's on_decide)."""
    if fleet.panel is None:
        from .panel import DecisionPanel
        fleet.panel = DecisionPanel(fleet)
    return fleet.panel


class Recorder:
    def __init__(self, fleet, out_dir: Path, fps: float = 30.0, panel: bool = True):
        self.f, self.w = fleet, fleet.w
        self.m, self.d = self.w.m, self.w.d
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps, self.dt = float(fps), 1.0 / float(fps)
        self.t0 = float(self.d.time)
        self.frame = 0
        self.t_wall = time.time()

        ground = [i for i in range(self.m.ngeom) if self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD]
        self.terrain = export_terrain(self.m, self.out, self.m.geom(ground[0]).name)
        Exp = _exporter_class()
        self.exp = Exp(self.m, ground, output_directory="usd", output_directory_root=str(self.out), verbose=False)

        self.rovers = list(fleet.rv)
        self.ch = {r: self.m.body(f"{r}/chassis").id for r in self.rovers}
        self.track = {"t": [], "active": [], **{f"{r}_pose": [] for r in self.rovers}}

        self.panel = None
        if panel:
            self.panel = shared_panel(fleet)
            (self.out / "panel").mkdir(exist_ok=True)
            self.panels = []
            self._panel_ver, self._panel_t = None, -1e9
        self.w.on_step.append(self._on_step)

    # ------------------------------------------------------------ per physics chunk
    def _on_step(self, w):
        t = float(self.d.time)
        while t + 1e-9 >= self.t0 + self.frame * self.dt:
            self._capture(t)

    def _capture(self, t):
        self.exp.update_scene(self.d)                     # one USD frame (time code = self.frame)
        active = None
        if self.panel is not None:
            active = self.panel.follow()
        self.track["t"].append(t)
        self.track["active"].append(self.rovers.index(active) if active in self.rovers else -1)
        for r in self.rovers:
            b = self.ch[r]
            self.track[f"{r}_pose"].append(np.r_[self.d.xpos[b], self.d.xquat[b]])
        if self.panel is not None:
            self._capture_panel(t)
        self.frame += 1

    def _capture_panel(self, t):
        p = self.panel
        if p._ver == self._panel_ver or t - self._panel_t < PANEL_MIN_DT:
            return
        with p.lock:
            v = p.view()
            ver = p._ver
        img = panel_image(v)
        name = f"panel_{self.frame:06d}.png"
        Image.fromarray(img).save(self.out / "panel" / name)
        self.panels.append({"frame": self.frame, "t": t, "file": f"panel/{name}"})
        self._panel_ver, self._panel_t = ver, t

    # ------------------------------------------------------------ end of run
    def close(self):
        if self.on_step_registered():
            self.w.on_step.remove(self._on_step)
        st = self.exp.stage                               # (the exporter makes its stage on the first frame)
        st.SetTimeCodesPerSecond(self.fps)
        st.SetFramesPerSecond(self.fps)
        st.SetStartTimeCode(0)
        st.SetEndTimeCode(max(self.frame - 1, 0))
        usd_path = Path(self.exp.frames_directory) / "fleet.usdc"
        st.Export(str(usd_path))
        np.savez_compressed(self.out / "track.npz", t=np.array(self.track["t"]),
                            active=np.array(self.track["active"], np.int16),
                            rovers=np.array(self.rovers),
                            **{k: np.array(v, np.float32) for k, v in self.track.items() if k.endswith("_pose")})
        if self.panel is not None:
            with (self.out / "panels.jsonl").open("w", encoding="utf-8") as fh:
                for p in self.panels:
                    fh.write(json.dumps(p) + "\n")
        man = {
            "fps": self.fps, "frames": self.frame, "sim_seconds": self.frame / self.fps,
            "rovers": {r: self.w.designs[r].name for r in self.rovers},
            "chassis_body": "<rid>/chassis",
            "scale": {"sim_to_real": 5.0, "note": "the site and rovers are 1:5 (jezero.SCALE = 0.2); render at x5"},
            "frame": "x east, y north, z up, metres (sim scale), origin = site centre",
            "files": {"scene": str(usd_path.relative_to(self.out)), "terrain": "terrain.usdc", "track": "track.npz",
                      "panels": "panels.jsonl" if self.panel is not None else None},
            "terrain": self.terrain,
            "geoms": len(self.exp.geom_names),
            "panel_images": len(self.panels) if self.panel is not None else 0,
            "panel_px": [PANEL_W, PANEL_H],
            "wall_seconds": time.time() - self.t_wall,
        }
        (self.out / "manifest.json").write_text(json.dumps(man, indent=2))
        return man

    def on_step_registered(self):
        return self._on_step in self.w.on_step
