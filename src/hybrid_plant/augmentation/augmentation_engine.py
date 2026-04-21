"""
augmentation_engine.py
──────────────────────
Top-level facade for the BESS Augmentation Engine.

This module is the single entry point used by:
  • oversize_optimizer.find_optimal_oversize()  (post-processing sweep)
  • run_model.__main__                          (post-processing baseline best)
  • Dashboard rendering                         (section 6b)

AugmentationEngine holds shared, expensive-to-construct state that is
reused across all evaluate_scenario() calls:
  • config, data, loss_factor
  • shared PlantEngine (already constructed by Year1Engine)
  • SOH curve, solar/wind efficiency curves
  • trigger_threshold_cuf — fixed from Pass-1 baseline

Each call to evaluate_scenario() is stateless from the caller's perspective;
internally it constructs a fresh LifecycleSimulator and CohortRegistry.

Oversizing
──────────
evaluate_scenario() accepts an optional ``initial_containers`` parameter.
When provided, it overrides params["bess_containers"] as the starting cohort
size.  params is never mutated.  This allows the oversize sweep to explore
B* + extra candidates without altering the solver's chosen params.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf, year1_busbar_mwh
from hybrid_plant.augmentation.lifecycle_simulator import LifecycleSimulator
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import LAKH_TO_RS
from hybrid_plant._paths import find_project_root

logger = logging.getLogger(__name__)


def _load_curve(path: Path, column: str) -> dict[int, float]:
    """Load a degradation CSV into a {year: value} dict."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return dict(zip(df["year"].astype(int), df[column.lower()]))


class AugmentationEngine:
    """
    Top-level facade for the augmentation feature.

    Parameters
    ----------
    config                : FullConfig
    data                  : dict — time-series data from data_loader
    energy_engine         : Year1Engine — provides the shared PlantEngine
    soh_curve             : dict[int, float] — {year: soh_fraction}
    trigger_threshold_cuf : float — pre-oversize Year-1 Plant CUF (Pass-1 output);
                            acts as the hard floor that triggers augmentation and
                            the restoration target for the k-search.
    """

    def __init__(
        self,
        config:                FullConfig,
        data:                  dict[str, Any],
        energy_engine:         Any,
        soh_curve:             dict[int, float],
        trigger_threshold_cuf: float,
    ) -> None:
        self._config     = config
        self._data       = data
        self._engine     = energy_engine
        self._soh_curve  = soh_curve
        self._threshold  = trigger_threshold_cuf
        self._loss_factor = float(energy_engine.grid.loss_factor)

        root = find_project_root()
        self._solar_eff = _load_curve(
            root / config.project["generation"]["solar"]["degradation"]["file"],
            "efficiency",
        )
        self._wind_eff = _load_curve(
            root / config.project["generation"]["wind"]["degradation"]["file"],
            "efficiency",
        )

        # Build a reusable LifecycleSimulator template
        self._lc_kwargs = dict(
            config          = config,
            plant_engine    = energy_engine.plant,
            soh_curve       = soh_curve,
            solar_eff_curve = self._solar_eff,
            wind_eff_curve  = self._wind_eff,
            loss_factor     = self._loss_factor,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_scenario(
        self,
        params:              dict[str, Any],
        initial_containers:  int | None = None,
        fast_mode:           bool = False,
        max_events_override: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the augmentation lifecycle for one scenario and return a finance
        result dict compatible with FinanceEngine.evaluate() output, augmented
        with an ``'augmentation'`` sub-dict.

        Parameters
        ----------
        params             : decision variables — same dict used by SolverEngine
        initial_containers : starting BESS container count.  When provided,
                             overrides params["bess_containers"] as the initial
                             cohort size (e.g. for oversize sweep: B* + extra).
                             params is never mutated.  Default None → use
                             params["bess_containers"].
        fast_mode          : passed through to LifecycleSimulator
        max_events_override: passed through to LifecycleSimulator; overrides
                             config max_augmentation_events (0 = no events,
                             large int = unlimited).

        Returns
        -------
        dict — same structure as FinanceEngine.evaluate() return value, plus:
            finance["augmentation"] = {
                "trigger_threshold_cuf": float,
                "event_log": list[dict],
                "cuf_series": list[float],
                "cohort_snapshot": list[dict],
                "cohort_capacity_timeline": dict[int, list[float]],
                "total_lump_cost_rs": float,
                "total_om_cost_rs": float,
                "initial_containers": int,
                "total_containers_added": int,
                "n_events": int,
            }
        """
        from hybrid_plant.finance.finance_engine import FinanceEngine

        # Resolve initial container count — do NOT mutate params
        if initial_containers is None:
            initial_containers = params["bess_containers"]

        # ── Step 1: Year-1 simulation with the (possibly oversized) container count
        year1 = self._engine.evaluate(
            solar_capacity_mw  = params["solar_capacity_mw"],
            wind_capacity_mw   = params["wind_capacity_mw"],
            bess_containers    = initial_containers,
            charge_c_rate      = params["charge_c_rate"],
            discharge_c_rate   = params["discharge_c_rate"],
            ppa_capacity_mw    = params["ppa_capacity_mw"],
            dispatch_priority  = params["dispatch_priority"],
            bess_charge_source = params["bess_charge_source"],
        )

        # ── Step 2: Lifecycle simulation ───────────────────────────────────
        simulator = LifecycleSimulator(**self._lc_kwargs)
        lc_result = simulator.simulate(
            params                  = params,
            initial_containers      = initial_containers,
            trigger_threshold_cuf   = self._threshold,
            fast_mode               = fast_mode,
            max_events_override     = max_events_override,
        )

        # ── Step 3: Build combined OPEX augmentation series ────────────────
        opex_aug_combined = [
            lump + om
            for lump, om in zip(
                lc_result.opex_augmentation_lump,
                lc_result.opex_augmentation_om,
            )
        ]

        # ── Step 4: Finance pipeline with overrides ────────────────────────
        finance_engine = FinanceEngine(self._config, self._data)
        finance = finance_engine.evaluate(
            year1_results             = year1,
            solar_capacity_mw         = params["solar_capacity_mw"],
            wind_capacity_mw          = params["wind_capacity_mw"],
            ppa_capacity_mw           = params["ppa_capacity_mw"],
            fast_mode                 = fast_mode,
            energy_projection_override = lc_result.energy_projection,
            opex_augmentation_series  = opex_aug_combined,
        )

        # ── Step 5: Attach augmentation metadata ──────────────────────────
        total_added = sum(e["k_containers"] for e in lc_result.event_log)
        finance["augmentation"] = {
            "trigger_threshold_cuf":    self._threshold,
            "event_log":                lc_result.event_log,
            "cuf_series":               lc_result.cuf_series,
            "cohort_snapshot":          lc_result.cohort_snapshot,
            "cohort_capacity_timeline": lc_result.cohort_capacity_timeline,
            "opex_augmentation_lump":   lc_result.opex_augmentation_lump,
            "opex_augmentation_om":     lc_result.opex_augmentation_om,
            "total_lump_cost_rs":       sum(lc_result.opex_augmentation_lump),
            "total_om_cost_rs":         float(np.sum(lc_result.opex_augmentation_om)),
            "initial_containers":       initial_containers,
            "total_containers_added":   total_added,
            "n_events":                 len(lc_result.event_log),
        }

        return {"year1": year1, "finance": finance}
