"""Provenance: every number a skill uses says where it came from.

Sources:  dataset:<file>   hardware:<mjcf>   experience:<rows>   user:brain.toml
A skill is built only from tagged values (`Tagged`), and `describe()` lists them, so it
can be checked that nothing was set by hand in the code.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root: brain.toml, runs_brain/
CONFIG_FILE = ROOT / "brain.toml"
SOURCES = ("dataset:", "hardware:", "experience:", "user:")


@dataclass(frozen=True)
class Tagged:
    value: object
    source: str

    def __post_init__(self):
        if not self.source.startswith(SOURCES):
            raise ValueError(f"untagged value {self.value!r}: source must start with one of {SOURCES}")


class Config:
    """brain.toml, with every value tagged user:brain.toml."""

    def __init__(self, path: Path = CONFIG_FILE):
        self.path = Path(path)
        self.data = tomllib.loads(self.path.read_text(encoding="utf-8"))

    def get(self, dotted: str):
        node = self.data
        for k in dotted.split("."):
            if k not in node:
                raise KeyError(f"{dotted} is not set in {self.path.name}")
            node = node[k]
        return node

    def opt(self, dotted: str, default=None, env: str | None = None):
        """An optional setting: the environment variable `env` if set (A/B runs), else the file,
        else `default`. The type follows the default (bool/int/float/str)."""
        import os
        raw = os.environ.get(env) if env else None
        if raw is None or raw == "":
            try:
                return self.get(dotted)
            except KeyError:
                return default
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, (int, float)):
            return type(default)(raw)
        return raw

    def tag(self, dotted: str) -> Tagged:
        return Tagged(self.get(dotted), f"user:{self.path.name}#{dotted}")

    def __call__(self, dotted: str):
        return self.get(dotted)


def table(tagged: dict[str, Tagged]) -> list[dict]:
    """Rows for printing / describe(): name, value, source."""
    return [{"name": k, "value": v.value, "source": v.source} for k, v in tagged.items()]
