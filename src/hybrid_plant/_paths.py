"""
_paths.py
─────────
Central path resolver for the hybrid_plant package.

Uses ``pyproject.toml`` as the project-root sentinel so path resolution is
independent of the working directory and the depth at which modules live
inside the package tree.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def find_project_root() -> Path:
    """
    Walk upward from this file until ``pyproject.toml`` is found.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Raises
    ------
    RuntimeError
        If no ``pyproject.toml`` is found in the directory tree.
    """
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Project root not found. "
        "Ensure pyproject.toml exists at the repository root."
    )
