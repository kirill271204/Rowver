r"""Copy the Jezero site ("jezero_s1") into rowver/assets/jezero/, as DATA only.

  .venv\Scripts\python.exe -I rowver\tools\import_jezero.py C:\Users\VINAYAK\Downloads\world\world

The download is untrusted: it is read with json, numpy (allow_pickle=False) and OpenCV only, and
the rock meshes are re-written from their parsed vertices and faces (nothing else is kept).
No code from it is run. The 1:5 scaling is done at load time (rowver/jezero.py).
Kept: world.json, heightmap / terrain_class / terrain_patches .npy, the albedo (2048 px) and the
rock OBJs. Not kept: surface_fields.npz, the backdrops, the normal/roughness/displacement maps
and mujoco_terrain.xml/.bin (the site is rebuilt at 1:5).
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "assets" / "jezero"


def clean_obj(src: Path, dst: Path):
    V, F = [], []
    for line in src.read_text(encoding="ascii", errors="strict").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            idx = [int(t.split("/")[0]) for t in p[1:]]
            for k in range(1, len(idx) - 1):                     # triangulate
                F.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(V)
    assert len(V) >= 4 and np.isfinite(V).all() and np.abs(V).max() < 10
    assert all(1 <= i <= len(V) for f in F for i in f)
    dst.write_text("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in V) +
                   "".join(f"f {a} {b} {c}\n" for a, b, c in F), encoding="ascii")
    return len(V), len(F)


def main(src):
    src = Path(src)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rocks").mkdir(exist_ok=True)
    meta = json.loads((src / "world.json").read_text(encoding="utf-8"))
    assert meta.get("contract_version") == 1 and meta.get("name") == "jezero_s1"
    meta.pop("backdrop", None)                                    # not used
    (OUT / "world.json").write_text(json.dumps(meta), encoding="utf-8")
    for name in ("heightmap", "terrain_class", "terrain_patches"):
        a = np.load(src / f"{name}.npy", allow_pickle=False)
        assert a.shape == (401, 401) and a.dtype.kind in "fiu", (name, a.shape, a.dtype)
        np.save(OUT / f"{name}.npy", a, allow_pickle=False)
    img = cv2.imread(str(src / "terrain_albedo.png"), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] == img.shape[1]
    cv2.imwrite(str(OUT / "terrain_albedo.png"), cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_AREA))
    nv = 0
    for v in meta["rock_variants"]:
        rel = Path(v["file"])
        assert rel.parent.name == "rocks" and rel.suffix == ".obj" and ".." not in rel.parts
        n, _ = clean_obj(src / rel, OUT / "rocks" / rel.name)
        nv += n
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"jezero_s1 -> {OUT}: {len(meta['rocks'])} rocks, {len(meta['rock_variants'])} meshes ({nv} vertices), "
          f"{len(meta['terrain']['patches'])} hidden patches, {total / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -I rowver/tools/import_jezero.py <the downloaded world folder>")
    main(sys.argv[1])
