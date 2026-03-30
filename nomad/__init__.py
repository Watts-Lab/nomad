"""NOMAD package.

Submodules are loaded lazily to avoid importing optional heavy dependencies
unless they are explicitly needed.
"""

from __future__ import annotations

import importlib

__all__ = [
    "constants",
    "filters",
    "city_gen",
    "traj_gen",
    "stop_detection",
    "io",
    "map_utils",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"nomad.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'nomad' has no attribute {name!r}")
