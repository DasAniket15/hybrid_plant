"""
config_loader.py
────────────────
Loads all YAML configuration files and bundles them into an immutable
``FullConfig`` dataclass.  Path resolution is driven by
``hybrid_plant._paths.find_project_root`` so the loader works regardless
of the working directory the process was started from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hybrid_plant._paths import find_project_root

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Config container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FullConfig:
    """Immutable bundle of all project configuration namespaces."""

    project:    dict[str, Any]
    regulatory: dict[str, Any]
    tariffs:    dict[str, Any]
    bess:       dict[str, Any]
    finance:    dict[str, Any]
    solver:     dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate(config: FullConfig) -> None:
    """
    Run lightweight sanity checks on a freshly loaded config.

    Raises
    ------
    ValueError
        If any constraint is violated.
    """
    if config.project["simulation"]["resolution"] != "hourly":
        raise ValueError("Only hourly simulation resolution is supported.")

    if config.bess["bess"]["container"]["size_mwh"] <= 0:
        raise ValueError("BESS container size must be positive.")

    if config.finance["capex"]["solar"]["ac_dc_ratio"] <= 0:
        raise ValueError("Solar AC/DC ratio must be positive.")

    logger.info("Config validation passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> FullConfig:
    """
    Discover the project root, load all YAML configs, validate, and return
    a ``FullConfig`` instance.

    Returns
    -------
    FullConfig
        Immutable configuration bundle.
    """
    config_dir = find_project_root() / "configs"

    config = FullConfig(
        project    = _load_yaml(config_dir / "project.yaml"),
        regulatory = _load_yaml(config_dir / "regulatory.yaml"),
        tariffs    = _load_yaml(config_dir / "tariffs.yaml"),
        bess       = _load_yaml(config_dir / "bess.yaml"),
        finance    = _load_yaml(config_dir / "finance.yaml"),
        solver     = _load_yaml(config_dir / "solver.yaml"),
    )

    _validate(config)
    return config