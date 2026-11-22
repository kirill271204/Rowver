"""Martian terrain types: how each one LOOKS (procedural texture + visual relief) and how it
DRIVES (hidden contact physics of the terrain geom: friction, rolling resistance, softness).

The rovers never read the physics columns below. They see only the overhead camera image
and learn from their own and each other's drives which looks slip.
All physics numbers are scene design choices (user:rowver.toml would be the place to change
them; they are defaults of the simulated world, not of the robot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Terrain:
    name: str
    rgb: tuple            # base colour of the texture
    grain: float          # strength of the fine colour noise (texture)
    ripples: float        # amplitude of periodic ripples in the texture (0 = none)
    pebbles: int          # visual pebbles per m^2 (relief for the depth camera, no physics)
    # ---- hidden physics of the terrain geom (it has contact priority, so these win over the roller's):
    traction: float       # sliding friction
    rolling: float        # rolling friction (m): sinkage / bulldozing resistance of soft ground
    solref: tuple         # contact softness: a larger time constant sinks deeper
    solimp: tuple


TERRAINS: dict[str, Terrain] = {
    "bedrock": Terrain("bedrock", (0.36, 0.17, 0.12), 0.10, 0.0, 0, 1.00, 0.0003, (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)),
    "dust":    Terrain("dust",    (0.66, 0.36, 0.24), 0.05, 0.0, 0, 0.75, 0.0008, (0.03, 1.0), (0.8, 0.9, 0.002, 0.5, 2.0)),
    "gravel":  Terrain("gravel",  (0.45, 0.40, 0.37), 0.35, 0.0, 60, 0.85, 0.0005, (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)),
    "sand":    Terrain("sand",    (0.86, 0.62, 0.40), 0.04, 0.6, 0, 0.30, 0.0030, (0.05, 1.0), (0.6, 0.8, 0.004, 0.5, 2.0)),
}


def geom_physics(t: Terrain) -> dict:
    """MuJoCo geom attributes for a terrain surface (priority 1: its parameters decide the contact)."""
    return dict(friction=[t.traction, 0.005, t.rolling], condim=6, solref=list(t.solref), solimp=list(t.solimp),
                priority=1)


def texture(t: Terrain, size=256, seed=0) -> np.ndarray:
    """(size, size, 3) uint8 procedural texture for terrain t."""
    rng = np.random.default_rng(seed + sum(map(ord, t.name)))     # stable across runs (no str hash)
    img = np.ones((size, size, 3)) * np.array(t.rgb)[None, None, :]
    noise = rng.normal(0, 1, (size, size))
    # smooth a little so the grain is not pure pixel noise
    k = np.ones(3) / 3
    noise = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, noise)
    noise = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, noise)
    img *= (1 + t.grain * noise)[..., None]
    if t.ripples:
        yy, xx = np.mgrid[0:size, 0:size] / size
        wave = np.sin(2 * np.pi * (9 * yy + 0.8 * np.sin(2 * np.pi * xx)))
        img *= (1 + 0.12 * t.ripples * wave)[..., None]
    if t.name == "bedrock":                      # dark cracks
        for _ in range(14):
            x, y = rng.integers(0, size, 2)
            ang = rng.uniform(0, np.pi)
            for s in range(rng.integers(20, 70)):
                xi, yi = int(x + s * np.cos(ang)) % size, int(y + s * np.sin(ang)) % size
                img[yi, xi] *= 0.55
                ang += rng.normal(0, 0.15)
    return np.clip(img * 255, 0, 255).astype(np.uint8)
