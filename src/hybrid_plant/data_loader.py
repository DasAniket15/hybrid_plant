"""
data_loader.py
──────────────
Loads all time-series CSVs (8760-hour profiles) and degradation curves
required by the simulation engine.

All file paths are resolved relative to the project root so the loader works
regardless of the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import HOURS_PER_YEAR, KWH_TO_MWH


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Degradation curve helper
# ─────────────────────────────────────────────────────────────────────────────

def operating_value(curve: dict[int, float], age_or_year: int) -> float:
    """
    Return the operating efficiency/SOH value for a given project year using
    the end-of-year curve convention.

    curve[N] = residual value at end of year N (after N years of degradation).
    The operating value DURING year N is the start-of-year value:
      Year 1 → 1.0 (fresh)
      Year N (N≥2) → curve[N-1]  (= end of prior year)
      Beyond curve end → clamped at curve[max(curve)]
    """
    if age_or_year <= 1:
        return 1.0
    lookup = age_or_year - 1
    if lookup in curve:
        return curve[lookup]
    return curve[max(curve)]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(relative_path: str) -> Path:
    """Resolve a project-relative path string to an absolute ``Path``."""
    return find_project_root() / relative_path


def _load_csv_column(path: Path) -> np.ndarray:
    """
    Read the first column of a header-less CSV as a float64 array.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.

    Returns
    -------
    np.ndarray
        1-D float64 array of all values in the first column.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].values.astype(np.float64)


def _validate_8760(array: np.ndarray, name: str) -> None:
    """Raise if *array* does not contain exactly 8760 values."""
    if len(array) != HOURS_PER_YEAR:
        raise ValueError(
            f"{name} must contain exactly {HOURS_PER_YEAR} values; "
            f"got {len(array)}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_timeseries_data(config: FullConfig) -> dict[str, Any]:
    """
    Load all time-series profiles and degradation curves defined in
    ``project.yaml`` and ``bess.yaml``.

    Parameters
    ----------
    config : FullConfig
        Fully loaded and validated configuration bundle.

    Returns
    -------
    dict
        Keys
        ────
        solar_cuf               : np.ndarray  shape (8760,), fraction [0–1]
        wind_cuf                : np.ndarray  shape (8760,), fraction [0–1]
        load_profile            : np.ndarray  shape (8760,), MWh per hour
        solar_degradation_curve : pd.DataFrame  columns [year, efficiency]
        wind_degradation_curve  : pd.DataFrame  columns [year, efficiency]
        bess_soh_curve          : pd.DataFrame  columns [year, soh]
    """
    gen_cfg  = config.project["generation"]
    load_cfg = config.project["load"]

    # ── Solar CUF ────────────────────────────────────────────────────────────
    solar_cuf = _load_csv_column(_resolve(gen_cfg["solar"]["cuf_profile_file"])) / 100.0
    _validate_8760(solar_cuf, "Solar CUF")

    # ── Wind CUF ─────────────────────────────────────────────────────────────
    wind_cuf = _load_csv_column(_resolve(gen_cfg["wind"]["cuf_profile_file"])) / 100.0
    _validate_8760(wind_cuf, "Wind CUF")

    # ── Load profile ─────────────────────────────────────────────────────────
    load_profile = _load_csv_column(_resolve(load_cfg["source_file"]))
    if load_cfg["unit"].lower() == "kwh":
        load_profile = load_profile * KWH_TO_MWH   # kWh → MWh
    _validate_8760(load_profile, "Load profile")

    # ── Degradation curves ───────────────────────────────────────────────────
    solar_deg = pd.read_csv(_resolve(gen_cfg["solar"]["degradation"]["file"]))
    wind_deg  = pd.read_csv(_resolve(gen_cfg["wind"]["degradation"]["file"]))
    bess_deg  = pd.read_csv(_resolve(config.bess["bess"]["degradation"]["file"]))

    return {
        "solar_cuf":                solar_cuf,
        "wind_cuf":                 wind_cuf,
        "load_profile":             load_profile,
        "solar_degradation_curve":  solar_deg,
        "wind_degradation_curve":   wind_deg,
        "bess_soh_curve":           bess_deg,
    }