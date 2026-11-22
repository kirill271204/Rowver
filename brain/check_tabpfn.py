"""Smoke check of the brain's model, TabPFN v3.5-fast (local, offline).

  .venv\\Scripts\\python.exe -m brain.check_tabpfn

Downloads the weights once if missing (huggingface.co/Prior-Labs/tabpfn_3_5, not gated: no login
or token; the LICENSE in that repo applies, non-commercial). Then, on synthetic data (no robot
code involved), it checks that the regressor returns quantiles and the classifier returns
probabilities, and times one planning pass (fit + 2048 candidates) and peak GPU memory.
"""
from __future__ import annotations

import time

import numpy as np

from .tabpfn_model import QUANTILES, backend_name, download_weights, make_model


def main():
    import torch
    p = download_weights()
    print(f"weights: {p} ({p.stat().st_size / 1e6:.0f} MB)")
    print("model:", backend_name())
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 10))
    y = X[:, 0] - 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, 200)
    yc = (X[:, 2] + X[:, 3] > 0).astype(int)
    cand = rng.uniform(-1, 1, (2048, 10))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t = time.time()
    m = make_model("reg")
    m.fit(X, y)
    t_fit = time.time() - t
    t = time.time()
    q = np.column_stack(m.predict(cand, output_type="quantiles", quantiles=list(QUANTILES)))
    t_pred = time.time() - t
    assert q.shape == (2048, len(QUANTILES)) and np.isfinite(q).all(), q.shape
    print(f"regressor  ok: fit {t_fit:.2f}s, 2048 candidates x {len(QUANTILES)} quantiles in {t_pred:.2f}s")

    c = make_model("clf")
    t = time.time()
    c.fit(X, yc)
    pr = c.predict_proba(cand)[:, 1]
    print(f"classifier ok: P range {pr.min():.2f}-{pr.max():.2f}, fit+predict {time.time() - t:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
