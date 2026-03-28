"""
energy_projection.py
────────────────────
Projects annual energy delivery across the full project lifetime by
re-running the complete 8760-hour plant dispatch for each calendar year
with that year's degraded capacities.

Why re-simulate per year (vs. scaling Year-1 totals)
─────────────────────────────────────────────────────
The simple approach of multiplying Year-1 scalar totals by degradation
factors is inaccurate because the three energy streams interact non-linearly:

  • Degraded solar capacity  → less surplus available to charge the BESS
    → different charge/discharge pattern → different shortfall coverage
  • Degraded BESS SOH        → lower energy capacity AND lower power caps
    (C-rate × degraded capacity) → the ToD reservation planner behaves
    differently with less headroom

Capturing these effects requires a full simulation, not scalar scaling.

Degradation model per year t
────────────────────────────
  Solar AC capacity   = base_solar_mw  × solar_eff[t]   (efficiency curve)
  Wind capacity       = base_wind_mw   × wind_eff[t]    (efficiency curve)
  BESS energy cap     = containers × container_size × soh[t]
                        (passed to PlantEngine as bess_soh_factor)
  Power caps, BESS    = C-rate × degraded energy cap    (inside PlantEngine)

Outputs (same keys as before — backward compatible)
───────
  solar_direct_mwh    annual solar direct delivery (busbar, pre-loss)
  wind_direct_mwh     annual wind direct delivery  (busbar, pre-loss)
  battery_mwh         annual BESS discharge        (busbar, pre-loss)
  delivered_pre_mwh   busbar total  → LCOE denominator
  delivered_meter_mwh meter total   → savings & landed tariff
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.energy.plant_engine import PlantEngine


class EnergyProjection:
    """
    Runs a per-year full plant simulation to produce an accurate 25-year
    energy delivery projection.

    Parameters
    ----------
    config        : FullConfig
    data          : dict — time-series data (solar_cuf, wind_cuf, load_profile, …)
    year1_results : dict — output of Year1Engine.evaluate(); must contain
                    ``sim_params`` (stored automatically by Year1Engine).

    The ``solar_capacity_mw``, ``wind_capacity_mw``, and ``loss_factor``
    keyword arguments are accepted for call-site backward compatibility
    but are not used — all required values come through sim_params.
    """

    def __init__(
        self,
        config:        FullConfig,
        data:          dict[str, Any],
        year1_results: dict[str, Any],
        # Kept for backward-compatible call sites; values come from sim_params.
        solar_capacity_mw: float | None = None,
        wind_capacity_mw:  float | None = None,
        loss_factor:       float | None = None,
    ) -> None:
        if "sim_params" not in year1_results:
            raise KeyError(
                "'sim_params' not found in year1_results. "
                "Ensure Year1Engine.evaluate() produced this result — "
                "direct PlantEngine.simulate() outputs are not sufficient."
            )

        self._sim_params   = year1_results["sim_params"]
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

        # Year-1 scalar totals — used by the fast (scalar-scaling) path.
        self._solar_1     = float(np.sum(year1_results["solar_direct_pre"]))
        self._wind_1      = float(np.sum(year1_results["wind_direct_pre"]))
        self._battery_1   = float(np.sum(year1_results["discharge_pre"]))
        self._loss_factor = self._sim_params["loss_factor"]

        # One PlantEngine instance, reused across all 25 year simulations.
        self._plant = PlantEngine(config, data)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_curve(path: Path, column: str) -> dict[int, float]:
        """Load a degradation CSV into a {year: value} dict."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        if "year" not in df.columns:
            raise ValueError(f"'year' column not found in {path}")
        col = column.lower()
        if col not in df.columns:
            raise ValueError(f"'{column}' column not found in {path}")

        return dict(zip(df["year"].astype(int), df[col]))

    # ─────────────────────────────────────────────────────────────────────────

    def project(self, fast_mode: bool = False) -> dict[str, np.ndarray]:
        """
        Return annual energy totals across the 25-year project lifetime.

        Parameters
        ----------
        fast_mode : bool, default False
            When True, applies degradation factors directly to Year-1 scalar
            totals (fast, suitable for solver trial ranking).
            When False, re-runs the full 8760-hour dispatch for each year
            (accurate, used for final reporting and dashboard).

        Returns
        -------
        dict
            solar_direct_mwh    : np.ndarray  shape (project_life,)  busbar, pre-loss
            wind_direct_mwh     : np.ndarray  shape (project_life,)  busbar, pre-loss
            battery_mwh         : np.ndarray  shape (project_life,)  busbar, pre-loss
            delivered_pre_mwh   : np.ndarray  busbar total  (LCOE denominator)
            delivered_meter_mwh : np.ndarray  at client meter (savings & landed tariff)
        """
        if fast_mode:
            return self._project_fast()
        return self._project_full()

    # ─────────────────────────────────────────────────────────────────────────

    def _project_fast(self) -> dict[str, np.ndarray]:
        """
        Fast path: scale Year-1 scalar totals by annual degradation factors.
        Runs in microseconds. Used during solver trials for ranking only.
        """
        solar_arr   = np.zeros(self._project_life)
        wind_arr    = np.zeros(self._project_life)
        battery_arr = np.zeros(self._project_life)
        pre_arr     = np.zeros(self._project_life)
        meter_arr   = np.zeros(self._project_life)

        for i, year in enumerate(range(1, self._project_life + 1)):
            s = self._solar_1   * self._solar_eff.get(year, 1.0)
            w = self._wind_1    * self._wind_eff.get(year, 1.0)
            b = self._battery_1 * self._soh.get(year, 1.0)

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = s + w + b
            meter_arr[i]   = (s + w + b) * self._loss_factor

        return {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,
            "delivered_meter_mwh":  meter_arr,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _project_full(self) -> dict[str, np.ndarray]:
        """
        Full path: re-simulate each of the 25 project years with that year's
        degraded plant capacities. Used for final reporting and dashboard.

        For each year t, the simulation receives:
          • solar_capacity_mw × solar_eff[t]   (degraded AC solar capacity)
          • wind_capacity_mw  × wind_eff[t]    (degraded wind capacity)
          • bess_soh_factor = soh[t]           (scales BESS energy & power caps)
          • all other params unchanged (C-rates, PPA cap, dispatch rules, …)
        """
        sp = self._sim_params

        solar_arr   = np.zeros(self._project_life)
        wind_arr    = np.zeros(self._project_life)
        battery_arr = np.zeros(self._project_life)
        pre_arr     = np.zeros(self._project_life)
        meter_arr   = np.zeros(self._project_life)

        for i, year in enumerate(range(1, self._project_life + 1)):
            solar_eff = self._solar_eff.get(year, 1.0)
            wind_eff  = self._wind_eff.get(year, 1.0)
            soh       = self._soh.get(year, 1.0)

            yr = self._plant.simulate(
                solar_capacity_mw  = sp["solar_capacity_mw"] * solar_eff,
                wind_capacity_mw   = sp["wind_capacity_mw"]  * wind_eff,
                bess_containers    = sp["bess_containers"],
                bess_soh_factor    = soh,
                charge_c_rate      = sp["charge_c_rate"],
                discharge_c_rate   = sp["discharge_c_rate"],
                ppa_capacity_mw    = sp["ppa_capacity_mw"],
                dispatch_priority  = sp["dispatch_priority"],
                bess_charge_source = sp["bess_charge_source"],
                loss_factor        = sp["loss_factor"],
            )

            s = float(np.sum(yr["solar_direct_pre"]))
            w = float(np.sum(yr["wind_direct_pre"]))
            b = float(np.sum(yr["discharge_pre"]))

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = s + w + b
            meter_arr[i]   = (s + w + b) * sp["loss_factor"]

        return {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,    # busbar — LCOE denominator
            "delivered_meter_mwh":  meter_arr,  # at meter — savings calc
        }