"""
pyomo_config.py
───────────────
Reads the pyomo: section from solver.yaml and returns it as a plain dict.
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig


def load_pyomo_config(config: FullConfig) -> dict[str, Any]:
    """
    Return the pyomo: section from solver.yaml, or an empty dict if absent.

    Raises ValueError if pyomo.enabled key is missing.
    """
    pyomo_cfg = config.solver.get("pyomo", {})
    return pyomo_cfg


def pyomo_enabled(config: FullConfig) -> bool:
    """Return True if pyomo.enabled is set to true in solver.yaml."""
    return bool(load_pyomo_config(config).get("enabled", False))
