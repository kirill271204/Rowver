"""A small frozen vision model for the navcam: DINOv2 ViT-S/14 patch features -> 8 PCA codes.

Why DINOv2-S: 22M parameters (85 MB of GPU memory next to TabPFN's ~3.8 GB), its patch features
are strong for dense tasks (segmentation, depth) out of the box, and its licence is Apache-2.0.
It is never trained or fine-tuned here: TabPFN stays the only learner. The 384-dim patch vector
is far too wide for a table with a few hundred rows, so it is reduced to 8 codes by a PCA fitted
ONCE on unlabelled navcam frames (tools/fit_navcam_pca.py) and then frozen, so a code column
means the same thing in every row ever written.

  enc = encoder()                 one per process (the GPU model is shared by every rover)
  enc.codes(rgb)                  (H, W, 3) uint8 -> (H/14, W/14, 8) float32
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = "vit_small_patch14_dinov2.lvd142m"
PCA_FILE = HERE / "assets" / "jezero" / "navcam_pca.npz"
PATCH = 14
N_CODES = 8


class Encoder:
    def __init__(self, pca_file=PCA_FILE, need_pca=True):
        import timm
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        m = timm.create_model(MODEL, pretrained=True, dynamic_img_size=True, num_classes=0)
        self.m = m.to(self.dev, self.dtype).eval()
        cfg = m.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=self.dev, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.dev, dtype=self.dtype).view(1, 3, 1, 1)
        self.pca = None
        if Path(pca_file).exists():
            z = np.load(pca_file, allow_pickle=False)
            self.pca = (z["mean"].astype(np.float32), z["components"].astype(np.float32))
        elif need_pca:
            raise FileNotFoundError(f"{pca_file} missing: run `python -m rowver.tools.fit_navcam_pca` once")

    def patches(self, rgbs) -> np.ndarray:
        """list of (H, W, 3) uint8 (H, W multiples of 14) -> (n, H/14, W/14, 384) float32."""
        torch = self.torch
        x = torch.from_numpy(np.stack(rgbs)).to(self.dev).permute(0, 3, 1, 2).to(self.dtype) / 255.0
        x = (x - self.mean) / self.std
        n, _, H, W = x.shape
        with torch.inference_mode():
            f = self.m.forward_features(x)[:, self.m.num_prefix_tokens:]
        return f.float().cpu().numpy().reshape(n, H // PATCH, W // PATCH, -1)

    def codes(self, rgb) -> np.ndarray:
        return self.codes_many([rgb])[0]

    def codes_many(self, rgbs) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("no PCA fitted yet (tools/fit_navcam_pca.py)")
        mu, C = self.pca
        return (self.patches(rgbs) - mu) @ C.T


@functools.lru_cache(maxsize=1)
def encoder() -> Encoder:
    return Encoder()
