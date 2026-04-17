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
  • payback_filter closure — per-event economic gate (built at init if
    pass1_lcoe is supplied and payback_filter.enabled = true in bess.yaml)

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
                            acts as the hard floor that triggers augmentation.
    pass1_lcoe            : float | None — Pass-1 LCOE in Rs/kWh, used to build the
                            payback filter proxy rate.  If None (default), the payback
                            filter is disabled regardless of bess.yaml config.
    """

    def __init__(
        self,
        config:                FullConfig,
        data:                  dict[str, Any],
        energy_engine:         Any,
        soh_curve:             dict[int, float],
        trigger_threshold_cuf: float,
        pass1_lcoe:            float | None = None,
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

        # Build the optional per-event payback filter
        self._payback_filter = self._build_payback_filter(
            config, soh_curve, pass1_lcoe
        )

        # Build a reusable LifecycleSimulator template (event_filter injected here)
        self._lc_kwargs = dict(
            config          = config,
            plant_engine    = energy_engine.plant,
            soh_curve       = soh_curve,
            solar_eff_curve = self._solar_eff,
            wind_eff_curve  = self._wind_eff,
            loss_factor     = self._loss_factor,
            event_filter    = self._payback_filter,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_scenario(
        self,
        params:             dict[str, Any],
        initial_containers: int | None = None,
        fast_mode:          bool = False,
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

        Returns
        -------
        dict — same structure as FinanceEngine.evaluate() return value, plus:
            finance["augmentation"] = {
                "trigger_threshold_cuf": float,
                "restoration_target_cuf": float,
                "event_log": list[dict],
                "skipped_event_log": list[dict],
                "cuf_series": list[float],
                "cohort_snapshot": list[dict],
                "cohort_capacity_timeline": dict[int, list[float]],
                "total_lump_cost_rs": float,
                "total_om_cost_rs": float,
                "initial_containers": int,
                "total_containers_added": int,
                "n_events": int,
                "n_skipped": int,
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

        # ── Step 2: Restoration target = Y1 CUF with initial_containers ───
        # Post-oversizing this will be strictly > trigger_threshold_cuf,
        # giving real degradation headroom before the first trigger fires.
        restoration_target_cuf = compute_plant_cuf(
            year1_busbar_mwh(year1),
            params["ppa_capacity_mw"],
        )

        # ── Step 3: Lifecycle simulation ───────────────────────────────────
        simulator = LifecycleSimulator(**self._lc_kwargs)
        lc_result = simulator.simulate(
            params                  = params,
            initial_containers      = initial_containers,
            trigger_threshold_cuf   = self._threshold,
            restoration_target_cuf  = restoration_target_cuf,
            fast_mode               = fast_mode,
        )

        # ── Step 4: Build combined OPEX augmentation series ────────────────
        opex_aug_combined = [
            lump + om
            for lump, om in zip(
                lc_result.opex_augmentation_lump,
                lc_result.opex_augmentation_om,
            )
        ]

        # ── Step 5: Finance pipeline with overrides ────────────────────────
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

        # ── Step 6: Attach augmentation metadata ──────────────────────────
        total_added = sum(e["k_containers"] for e in lc_result.event_log)
        finance["augmentation"] = {
            "trigger_threshold_cuf":    self._threshold,
            "restoration_target_cuf":   restoration_target_cuf,
            "event_log":                lc_result.event_log,
            "skipped_event_log":        lc_result.skipped_event_log,
            "cuf_series":               lc_result.cuf_series,
            "adjusted_target_series":   lc_result.adjusted_target_series,
            "cohort_snapshot":          lc_result.cohort_snapshot,
            "cohort_capacity_timeline": lc_result.cohort_capacity_timeline,
            "opex_augmentation_lump":   lc_result.opex_augmentation_lump,
            "opex_augmentation_om":     lc_result.opex_augmentation_om,
            "total_lump_cost_rs":       sum(lc_result.opex_augmentation_lump),
            "total_om_cost_rs":         float(np.sum(lc_result.opex_augmentation_om)),
            "initial_containers":       initial_containers,
            "total_containers_added":   total_added,
            "n_events":                 len(lc_result.event_log),
            "n_skipped":                len(lc_result.skipped_event_log),
        }

        return {"year1": year1, "finance": finance}

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_payback_filter(
        config:     Any,
        soh_curve:  dict[int, float],
        pass1_lcoe: float | None,
    ):
        """
        Construct a per-event payback filter closure, or return None if the
        filter is disabled (either by config or by pass1_lcoe being absent).

        The closure signature is: (event_info: dict) -> bool
          Returns True  → fire the event
          Returns False → skip the event (logged at INFO level by the caller)

        Payback test (section 2.4 of the redesign spec)
        ────────────────────────────────────────────────
          proxy_rate = (discom_tariff − pass1_lcoe) × 1000  [Rs/MWh]
          For t = event_year … project_life:
              value_t   = delta_busbar × bess_decay(age_offset) × proxy_rate
              cost_t    = annual_om
              df        = 1 / (1 + wacc)^(t − event_year)
          total_value = Σ value_t × df
          total_cost  = lump_cost + Σ annual_om × df
          pass = total_value > total_cost
        """
        aug_cfg = config.bess["bess"]["augmentation"]
        pf_cfg  = aug_cfg.get("payback_filter", {})
        if not bool(pf_cfg.get("enabled", False)):
            return None
        if pass1_lcoe is None:
            return None

        from hybrid_plant.finance.savings_model import SavingsModel
        from hybrid_plant.finance.lcoe_model import LCOEModel
        from hybrid_plant.data_loader import operating_value

        discom_tariff  = SavingsModel._weighted_discom_tariff(config)
        wacc           = LCOEModel(config).wacc
        container_size = float(config.bess["bess"]["container"]["size_mwh"])
        cost_per_mwh   = float(aug_cfg["cost_per_mwh"])
        om_rs_per_mwh  = float(config.finance["opex"]["bess"]["rate_lakh_per_mwh"]) * LAKH_TO_RS
        project_life   = int(config.project["project"]["project_life_years"])
        proxy_rate     = (discom_tariff - pass1_lcoe) * 1000.0   # Rs/MWh

        def payback_filter(event_info: dict) -> bool:
            event_year   = event_info["year"]
            k            = event_info["k_containers"]
            pre_busbar   = event_info["pre_event_busbar_mwh"]
            post_busbar  = event_info["post_event_busbar_mwh"]

            lump_cost    = k * container_size * cost_per_mwh
            annual_om    = k * container_size * om_rs_per_mwh
            delta_busbar = post_busbar - pre_busbar   # MWh incremental in event year

            if proxy_rate <= 0.0:
                # Augmentation has no economic value at the margin — always skip
                logger.info(
                    "Payback filter Y%d: proxy_rate=%.4f Rs/MWh <= 0 — skipping event.",
                    event_year, proxy_rate,
                )
                return False

            total_value_rs = 0.0
            total_cost_rs  = lump_cost    # lump paid at t=event_year, df=1.0

            for t in range(event_year, project_life + 1):
                age_offset = t - event_year + 1        # 1 at event_year (fresh)
                bess_decay = operating_value(soh_curve, age_offset)
                df         = 1.0 / (1.0 + wacc) ** (t - event_year)
                total_value_rs += delta_busbar * bess_decay * proxy_rate * df
                total_cost_rs  += annual_om * df

            passes = total_value_rs > total_cost_rs
            if not passes:
                logger.info(
                    "Payback filter Y%d: SKIPPED. k=%d, lump=%.0f Rs, "
                    "value=%.0f Rs, cost=%.0f Rs, remaining=%d yrs.",
                    event_year, k, lump_cost,
                    total_value_rs, total_cost_rs,
                    project_life - event_year + 1,
                )
            return passes

        return payback_filter