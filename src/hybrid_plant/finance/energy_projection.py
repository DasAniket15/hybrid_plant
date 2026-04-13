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

Degradation + augmentation model per year t
────────────────────────────────────────────
  Solar AC capacity   = base_solar_mw  × solar_eff[t]   (efficiency curve)
  Wind capacity       = base_wind_mw   × wind_eff[t]    (efficiency curve)
  BESS energy cap     = blended effective capacity across all cohorts
                        (managed by AugmentationEngine.compute_effective)
  Power caps, BESS    = C-rate × degraded energy cap    (inside PlantEngine)

CUF-based augmentation trigger (full mode only)
────────────────────────────────────────────────
  For each year ≥ 2 (when augmentation is enabled):
    1. Simulate with current cohort fleet.
    2. Compute plant CUF = busbar_MWh / (ppa_MW × 8760) × 100.
    3. If CUF < trigger_cuf_percent × year1_cuf, OR gap trigger fires:
         • Compute new containers needed to restore BESS capacity.
         • Add cohort, re-simulate this year.

  This requires at most one re-simulation per augmentation year and keeps
  the algorithm deterministic (no iteration).

Outputs
───────
  solar_direct_mwh    annual solar direct delivery (busbar, pre-loss)
  wind_direct_mwh     annual wind direct delivery  (busbar, pre-loss)
  battery_mwh         annual BESS discharge        (busbar, pre-loss)
  delivered_pre_mwh   busbar total  → LCOE denominator
  delivered_meter_mwh meter total   → savings & landed tariff
  cuf_per_year        plant CUF %   → augmentation dashboard & trigger tracing
  augmentation_result dict          → passed on to OpexModel & dashboard
"""

from __future__ import annotations

import math
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
    aug_engine    : AugmentationEngine instance (or None).
                    When provided, the full-mode projection applies CUF-based
                    dynamic augmentation; the fast-mode projection uses
                    ``aug_engine.passthrough_result()``.
    """

    def __init__(
        self,
        config:        FullConfig,
        data:          dict[str, Any],
        year1_results: dict[str, Any],
        # Legacy keyword kept for backward-compatible call sites (unused).
        solar_capacity_mw:  float | None = None,
        wind_capacity_mw:   float | None = None,
        loss_factor:        float | None = None,
        # Augmentation: pass an AugmentationEngine instance.
        aug_engine:          Any | None = None,
        # Deprecated: old dict-based augmentation_result (ignored; use aug_engine).
        augmentation_result: dict | None = None,
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

        # AugmentationEngine instance (or None for legacy call sites).
        self._aug = aug_engine

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
            cuf_per_year        : np.ndarray  plant CUF % per year (0 in fast mode)
            augmentation_result : dict | None  full augmentation schedule dict
        """
        if fast_mode:
            return self._project_fast()
        return self._project_full()

    # ─────────────────────────────────────────────────────────────────────────

    def _project_fast(self) -> dict[str, np.ndarray]:
        """
        Fast path: scale Year-1 scalar totals by annual degradation factors.
        Runs in microseconds. Used during solver trials for ranking only.

        No CUF-based augmentation is applied in fast mode — augmentation
        would require full 8760-h simulation per year.  Returns a passthrough
        augmentation result (no events) so downstream consumers receive a
        valid dict without conditional logic.
        """
        aug = self._aug
        initial_containers = self._sim_params["bess_containers"]

        # Passthrough result: no augmentation events, plain SOH per year.
        aug_result = (
            aug.passthrough_result(initial_containers)
            if aug is not None
            else None
        )

        # Pre-compute Year-1 capacity factor for battery scaling.
        if aug_result is not None:
            _y1_eff_n   = aug_result["effective_containers_per_year"][1]
            _y1_eff_soh = aug_result["effective_soh_per_year"][1]
            _y1_aug_cap = _y1_eff_n * _y1_eff_soh
        else:
            _y1_aug_cap = None

        solar_arr   = np.zeros(self._project_life)
        wind_arr    = np.zeros(self._project_life)
        battery_arr = np.zeros(self._project_life)
        pre_arr     = np.zeros(self._project_life)
        meter_arr   = np.zeros(self._project_life)

        for i, year in enumerate(range(1, self._project_life + 1)):
            s = self._solar_1 * self._solar_eff.get(year, 1.0)
            w = self._wind_1  * self._wind_eff.get(year, 1.0)

            if _y1_aug_cap is not None and _y1_aug_cap > 0:
                yr_cap = (
                    aug_result["effective_containers_per_year"][year]
                    * aug_result["effective_soh_per_year"][year]
                )
                b = self._battery_1 * (yr_cap / _y1_aug_cap)
            else:
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
            "cuf_per_year":         np.zeros(self._project_life),
            "augmentation_result":  aug_result,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _project_full(self) -> dict[str, np.ndarray]:
        """
        Full path: re-simulate each of the 25 project years with that year's
        degraded plant capacities.  When augmentation is enabled, checks the
        plant CUF after each simulation and, if the trigger fires, adds a new
        cohort and re-simulates that year.

        Algorithm (per year t)
        ──────────────────────
          1. Compute blended effective containers / SOH from current cohorts.
          2. Simulate 8760 hours → compute CUF.
          3. If t ≥ 2 and enabled:
               a. Check ``CUF < trigger_cuf_percent/100 × year1_cuf``
                  OR gap trigger (max_gap_years override).
               b. If triggered: add new cohort, recompute effective, re-simulate.
          4. Record energy totals, CUF, cohort state.
        """
        sp  = self._sim_params
        aug = self._aug   # AugmentationEngine | None
        ppa_mw = sp["ppa_capacity_mw"]

        # ── Augmentation tracking ─────────────────────────────────────────────
        cohorts: list[tuple[int, int]] = [(1, sp["bess_containers"])]
        last_aug_year = 1
        year1_cuf: float | None          = None
        year1_bess_eff_mwh: float | None = None

        aug_purchase_opex:            dict[int, float] = {}
        containers_added_per_year:    dict[int, int]   = {}
        aug_years:                    list[int]        = []
        effective_containers_per_year: dict[int, int]  = {}
        effective_soh_per_year:        dict[int, float] = {}
        total_installed_mwh_per_year:  dict[int, float] = {}

        # ── Output arrays ─────────────────────────────────────────────────────
        solar_arr   = np.zeros(self._project_life)
        wind_arr    = np.zeros(self._project_life)
        battery_arr = np.zeros(self._project_life)
        pre_arr     = np.zeros(self._project_life)
        meter_arr   = np.zeros(self._project_life)
        cuf_arr     = np.zeros(self._project_life)

        # ── Shared simulate helper (avoids repeating keyword list) ────────────
        def _simulate(total_n: int, eff_soh: float, sol_eff: float, wnd_eff: float):
            return self._plant.simulate(
                solar_capacity_mw  = sp["solar_capacity_mw"] * sol_eff,
                wind_capacity_mw   = sp["wind_capacity_mw"]  * wnd_eff,
                bess_containers    = total_n,
                bess_soh_factor    = eff_soh,
                charge_c_rate      = sp["charge_c_rate"],
                discharge_c_rate   = sp["discharge_c_rate"],
                ppa_capacity_mw    = ppa_mw,
                dispatch_priority  = sp["dispatch_priority"],
                bess_charge_source = sp["bess_charge_source"],
                loss_factor        = sp["loss_factor"],
            )

        def _cuf_from_result(yr_result) -> float:
            busbar = (
                float(np.sum(yr_result["solar_direct_pre"]))
                + float(np.sum(yr_result["wind_direct_pre"]))
                + float(np.sum(yr_result["discharge_pre"]))
            )
            return busbar / (ppa_mw * 8760) * 100 if ppa_mw > 0 else 0.0

        # ── Main year loop ────────────────────────────────────────────────────
        for i, year in enumerate(range(1, self._project_life + 1)):
            solar_eff = self._solar_eff.get(year, 1.0)
            wind_eff  = self._wind_eff.get(year, 1.0)

            # Compute blended effective values for current cohort fleet.
            if aug is not None:
                eff_mwh, total_n, eff_soh, installed = aug.compute_effective(year, cohorts)
            else:
                total_n   = sp["bess_containers"]
                eff_soh   = self._soh.get(year, 1.0)
                eff_mwh   = total_n * eff_soh  # proxy (container_size cancels out)
                installed = float(total_n)      # proxy

            yr      = _simulate(total_n, eff_soh, solar_eff, wind_eff)
            year_cuf = _cuf_from_result(yr)

            # ── CUF-based augmentation trigger (full mode, year ≥ 2) ──────────
            if aug is not None and aug.enabled:
                if year == 1:
                    year1_cuf          = year_cuf
                    year1_bess_eff_mwh = eff_mwh
                elif year1_cuf is not None:
                    trigger_cuf_abs = aug.trigger_cuf_percent / 100.0 * year1_cuf
                    gap_trigger = (
                        aug.max_gap_years is not None
                        and (year - last_aug_year) >= aug.max_gap_years
                    )
                    if year_cuf < trigger_cuf_abs or gap_trigger:
                        # How many new containers to restore BESS capacity?
                        target_bess = aug.restore_pct / 100.0 * year1_bess_eff_mwh
                        deficit     = max(target_bess - eff_mwh, 0.0)
                        new_soh     = aug.soh_curve.get(1, 1.0)
                        new_n       = max(
                            math.ceil(deficit / (aug.container_size * new_soh)),
                            aug.min_containers,
                        )

                        # Record augmentation event
                        aug_purchase_opex[year]       = new_n * aug.container_size * aug.cost_per_mwh
                        containers_added_per_year[year] = new_n
                        aug_years.append(year)
                        cohorts.append((year, new_n))
                        last_aug_year = year

                        # Recompute blended effective and re-simulate this year
                        eff_mwh, total_n, eff_soh, installed = aug.compute_effective(year, cohorts)
                        yr       = _simulate(total_n, eff_soh, solar_eff, wind_eff)
                        year_cuf = _cuf_from_result(yr)

            # ── Record per-year cohort state ──────────────────────────────────
            if year not in containers_added_per_year:
                containers_added_per_year[year]     = 0
            effective_containers_per_year[year]     = total_n
            effective_soh_per_year[year]            = eff_soh
            total_installed_mwh_per_year[year]      = installed

            # ── Record energy totals ──────────────────────────────────────────
            s = float(np.sum(yr["solar_direct_pre"]))
            w = float(np.sum(yr["wind_direct_pre"]))
            b = float(np.sum(yr["discharge_pre"]))

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = s + w + b
            meter_arr[i]   = (s + w + b) * sp["loss_factor"]
            cuf_arr[i]     = year_cuf

        # ── Build augmentation result dict ────────────────────────────────────
        if aug is not None:
            aug_result: dict | None = {
                "enabled":                       aug.enabled,
                "cohorts":                       cohorts,
                "augmentation_years":            aug_years,
                "containers_added_per_year":     containers_added_per_year,
                "effective_containers_per_year": effective_containers_per_year,
                "effective_soh_per_year":        effective_soh_per_year,
                "total_installed_mwh_per_year":  total_installed_mwh_per_year,
                "augmentation_purchase_opex":    aug_purchase_opex,
            }
        else:
            aug_result = None

        return {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,       # busbar — LCOE denominator
            "delivered_meter_mwh":  meter_arr,     # at meter — savings calc
            "cuf_per_year":         cuf_arr,
            "augmentation_result":  aug_result,
        }
