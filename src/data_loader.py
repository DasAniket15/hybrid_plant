import os
import numpy as np
import pandas as pd
from typing import Dict, Any


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(relative_path: str) -> str:
    """
    Resolves paths defined in YAML relative to project root.
    """
    root = _project_root()
    return os.path.join(root, relative_path)


def _load_csv_column(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].values.astype(np.float64)


def _validate_8760(array: np.ndarray, name: str):
    if len(array) != 8760:
        raise ValueError(f"{name} must contain exactly 8760 values.")


# -------------------------------------------------
# Main Loader
# -------------------------------------------------

def load_timeseries_data(config: Dict[str, Any]) -> Dict[str, np.ndarray]:

    # ---------------------------
    # Solar CUF
    # ---------------------------
    solar_path = _resolve_path(
        config.project["generation"]["solar"]["cuf_profile_file"]
    )
    solar_cuf = _load_csv_column(solar_path) / 100.0
    _validate_8760(solar_cuf, "Solar CUF")

    # ---------------------------
    # Wind CUF
    # ---------------------------
    wind_path = _resolve_path(
        config.project["generation"]["wind"]["cuf_profile_file"]
    )
    wind_cuf = _load_csv_column(wind_path) / 100.0
    _validate_8760(wind_cuf, "Wind CUF")

    # ---------------------------
    # Load Profile
    # ---------------------------
    load_path = _resolve_path(
        config.project["load"]["source_file"]
    )
    load_profile = _load_csv_column(load_path)
    
    if config.project["load"]["unit"].lower() == "kwh":
        load_profile = load_profile / 1000.0 # convert th MWh

    _validate_8760(load_profile, "Load Profile")

    # ---------------------------
    # Degradation Curves
    # ---------------------------
    solar_deg = pd.read_csv(
        _resolve_path(config.project["generation"]["solar"]["degradation"]["file"])
    )

    wind_deg = pd.read_csv(
        _resolve_path(config.project["generation"]["wind"]["degradation"]["file"])
    )

    bess_deg = pd.read_csv(
        _resolve_path(config.bess["bess"]["degradation"]["file"])
    )

    return {
        "solar_cuf": solar_cuf,
        "wind_cuf": wind_cuf,
        "load_profile": load_profile,
        "solar_degradation_curve": solar_deg,
        "wind_degradation_curve": wind_deg,
        "bess_soh_curve": bess_deg,
    }