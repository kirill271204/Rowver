"""Robot interface: every body (simulated or real) the controller can drive.

State dicts use the same keys everywhere, so the TabPFN features never change:
  x, y, theta, type, type_name, shape, size, height, upright, on_table,
  target_x, target_y, target_radius, dist_to_target
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


class Robot(ABC):
    name = "robot"
    skills: tuple[str, ...] = ("push",)
    target = np.zeros(2)
    target_radius = 0.03

    @abstractmethod
    def get_state(self) -> dict: ...

    @abstractmethod
    def push(self, angle: float, speed: float, offset: float, push_dist: float) -> dict:
        """Returns {"toppled", "out_of_bounds", "out_of_reach"}."""

    def pick_place(self, **params) -> dict:
        raise NotImplementedError(f"{self.name} cannot pick and place")

    @abstractmethod
    def reachable(self, x, y) -> np.ndarray:
        """Vectorised: True where the gripper can work at table height."""

    # --- sim-only conveniences (a real robot needs a human or a reset routine)
    def spawn(self, obj_type=None, pose=None):
        raise NotImplementedError

    def spawn_near_target(self, obj_type=None):
        raise NotImplementedError

    def set_target(self, xy):
        self.target = np.asarray(xy, float)

    def close(self):
        pass
