"""Put the TabPFN decision panel beside the Isaac frames (the fallback video, demo_panel.mp4).

  .venv/bin/python -m rowver.tools.compose_panel --frames ~/render --record runs/<run>/record --out ~/render_panel

For each rendered frame_NNNNNN.png (N = recorded frame), takes the last panel image saved at or before frame N
(record.py: panels.jsonl) and lays it over the left edge of the frame, scaled to the frame's height.
Then encode with ffmpeg (README).
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fr, rec, out = Path(a.frames), Path(a.record), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = [json.loads(ln) for ln in (rec / "panels.jsonl").read_text().splitlines() if ln.strip()]
    keys = [p["frame"] for p in panels]
    files = sorted(fr.glob("frame_*.png"))
    cache = {}
    for f in files:
        k = int(f.stem.split("_")[1])
        img = Image.open(f).convert("RGB")
        i = bisect.bisect_right(keys, k) - 1
        if i >= 0:
            pf = rec / panels[i]["file"]
            if pf not in cache:
                cache.clear()
                p = Image.open(pf).convert("RGB")
                cache[pf] = p.resize((round(p.width * img.height / p.height), img.height), Image.LANCZOS)
            img.paste(cache[pf], (0, 0))
        img.save(out / f.name)
    print(f"composited {len(files)} frames with {len(panels)} panel images -> {out}")


if __name__ == "__main__":
    main()
