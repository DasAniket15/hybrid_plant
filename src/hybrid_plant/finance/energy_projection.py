"""
energy_projection.py
────────────────────
Projects annual energy delivery across the full project lifetime by applying
technology-specific degradation curves to Year-1 hourly simulation results.

Outputs
───────
  delivered_pre_mwh   busbar energy (pre-loss)  → LCOE denominator
  delivered_meter_mwh meter energy  (post-loss) → savings & landed tariff
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import FullConfig


class EnergyProjection:
    """
    Scales Year-1 dispatch actuals by annual degradation factors to produce
    a 25-year energy projection.

    Parameters
    ----------
    config            : FullConfig
    data              : dict  — time-series data dict (used for type consistency)
    year1_results     : dict  — output of Year1Engine.evaluate()
    solar_capacity_mw : float — AC MW (informational only, not recalculated)
    wind_capacity_mw  : float — MW
    loss_factor       : float — from GridInterface, passed through directly
    """

    def __init__(
        self,
        config:            FullConfig,
        data:              dict[str, Any],
        year1_results:     dict[str, Any],
        solar_capacity_mw: float,
        wind_capacity_mw:  float,
        loss_factor:       float,
    ) -> None:
        self._year1       = year1_results
        self._loss_factor = loss_factor
        self._project_life = config.project["project"]["project_life_years"]

        root = find_project_root()

        self._solar_eff = self._load_curve(
            root / config.project["generation"]["solar"]["degradation"]["file"],
            column="efficiency",
        )
        self._wind_eff = self._load_curve(
            root / config.project["generation"]["wind"]["degradation"]["file"],
            column="efficiency",
        )
        self._soh = self._load_curve(
            root / config.bess["bess"]["degradation"]["file"],
            column="soh",
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_curve(path: Path, column: str) -> dict[int, float]:
        """
        Load a degradation CSV into a {year: value} dict.

        Parameters
        ----------
        path   : absolute path to CSV
        column : target column name (case-insensitive)

        Returns
        -------
        dict[int, float]
        """
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        if "year" not in df.columns:
            raise ValueError(f"'year' column not found in {path}")
        col = column.lower()
        if col not in df.columns:
            raise ValueError(f"'{column}' column not found in {path}")

        return dict(zip(df["year"].astype(int), df[col]))

    # ─────────────────────────────────────────────────────────────────────────

    def project(self) -> dict[str, np.ndarray]:
        """
        Apply year-by-year degradation factors and return energy arrays.

        Returns
        -------
        dict
            solar_direct_mwh    : np.ndarray  shape (project_life,)
            wind_direct_mwh     : np.ndarray  shape (project_life,)
            battery_mwh         : np.ndarray  shape (project_life,)
            delivered_pre_mwh   : np.ndarray  busbar (pre-loss)
            delivered_meter_mwh : np.ndarray  at client meter (post-loss)
        """
        solar_1   = float(np.sum(self._year1["solar_direct_pre"]))
        wind_1    = float(np.sum(self._year1["wind_direct_pre"]))
        battery_1 = float(np.sum(self._year1["discharge_pre"]))

        solar_arr   = np.zeros(self._project_life)
        wind_arr    = np.zeros(self._project_life)
        battery_arr = np.zeros(self._project_life)
        pre_arr     = np.zeros(self._project_life)
        meter_arr   = np.zeros(self._project_life)

        for i, year in enumerate(range(1, self._project_life + 1)):
            solar_eff = self._solar_eff.get(year, 1.0)
            wind_eff  = self._wind_eff.get(year, 1.0)
            soh       = self._soh.get(year, 1.0)

            s = solar_1   * solar_eff
            w = wind_1    * wind_eff
            b = battery_1 * soh

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = s + w + b
            meter_arr[i]   = (s + w + b) * self._loss_factor

        return {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,    # busbar — LCOE denominator
            "delivered_meter_mwh":  meter_arr,  # at meter — savings calc
        }
