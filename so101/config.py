"""Settings, loaded from .env at the repo root."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent            # so101/: its runs/ and assets/ live here
load_dotenv(ROOT.parent / ".env")


def _f(name, default):
    return float(os.getenv(name, default))


@dataclass
class Config:
    # --- features (TabPFN itself is brain.tabpfn_model: v3.5-fast, settings in .env)
    use_position_features: bool = os.getenv("USE_POSITION_FEATURES", "0") == "1"  # turn on for a real table

    # --- planner
    n_candidates: int = int(os.getenv("PLAN_CANDIDATES", "2048"))
    cem_iters: int = int(os.getenv("PLAN_CEM_ITERS", "2"))
    mc_samples: int = int(os.getenv("PLAN_MC_SAMPLES", "128"))
    top_k: int = int(os.getenv("PLAN_TOP_K", "6"))

    # --- files
    run_dir: Path = field(default_factory=lambda: ROOT / os.getenv("RUN_DIR", "runs"))

    def memory_csv(self, skill: str) -> Path:
        return self.run_dir / f"memory_{skill}.csv"

    @property
    def events_jsonl(self) -> Path:
        return self.run_dir / "events.jsonl"


CFG = Config()
