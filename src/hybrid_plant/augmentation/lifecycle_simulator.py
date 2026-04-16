"""
lifecycle_simulator.py
──────────────────────
25-year cohort-evolving BESS lifecycle simulation.

Role in the two-pass system
────────────────────────────
LifecycleSimulator is the per-scenario workhorse.  Given a set of plant
parameters (solar, wind, BESS containers, C-rates, PPA) plus the fixed
trigger threshold from Pass 1, it:

  1. Simulates each project year with the correct blended SOH for all
     active cohorts.
  2. Detects when Plant CUF falls below ``trigger_threshold_cuf``.
  3. At each triggered year, searches for the smallest k containers that
     restore CUF back to ``restoration_target_cuf`` (this scenario's own
     Year-1 CUF), then continues searching in case larger k improves NPV.
  4. Returns per-year energy arrays compatible with EnergyProjection
     outputs, plus augmentation OPEX series, an event log, and metadata
     for the dashboard.

fast_mode contract
──────────────────
  fast_mode=True  (used inside Pass 2 Optuna trials):
      Per-year Plant CUF is approximated by scaling an "anchor" CUF using
      the ratio of current blended SOH to anchor blended SOH:

          cuf_t ≈ cuf_anchor × (blended_soh_t / blended_soh_anchor)

      The anchor is the last year for which we ran a real simulation
      (initially Year 1; updated to the post-event year after each event).
      Energy totals are also approximated by scaling Year-1 actuals.
      Per-year plant_engine.simulate() calls occur only for Year 1 and
      immediately after each augmentation event.

  fast_mode=False (used for final best-result reporting):
      Full plant_engine.simulate() call every year with that year's exact
      blended SOH.  Accurate non-linear interactions are captured.

Augmentation cost accounting
─────────────────────────────
  Lump-sum procurement (event year only):
      cost_rs = k × container_size_mwh × cost_per_mwh_rs

  Recurring O&M (event year through Year 25):
      annual_om_rs = k × container_size_mwh × bess_om_rate_rs_per_mwh

  Both are injected into opex_projection via FinanceEngine's override
  mechanism.  CAPEX, debt, and depreciation schedules are never changed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybrid_plant.augmentation.cohort import CohortRegistry
from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import LAKH_TO_RS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LifecycleResult:
    """
    Output of a single LifecycleSimulator.simulate() run.

    All list/array fields have length == project_life (25).

    Attributes
    ----------
    energy_projection          : dict — same keys as EnergyProjection.project():
                                  solar_direct_mwh, wind_direct_mwh, battery_mwh,
                                  delivered_pre_mwh, delivered_meter_mwh
    opex_augmentation_lump     : list[float] — lump-sum procurement cost per year (Rs)
    opex_augmentation_om       : list[float] — cumulative recurring O&M from new
                                  cohorts active in each year (Rs)
    event_log                  : list[dict]  — one record per augmentation event
    cuf_series                 : list[float] — Plant CUF (%) for each year
    cohort_snapshot            : list[dict]  — final cohort list (for dashboard)
    cohort_capacity_timeline   : dict[int, list[float]] — per-cohort effective MWh/yr
    """

    energy_projection:         dict[str, np.ndarray]
    opex_augmentation_lump:    list[float]
    opex_augmentation_om:      list[float]
    event_log:                 list[dict[str, Any]]
    cuf_series:                list[float]
    cohort_snapshot:           list[dict[str, Any]]
    cohort_capacity_timeline:  dict[int, list[float]]


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class LifecycleSimulator:
    """
    Simulates a 25-year cohort-evolving lifecycle for one plant scenario.

    Parameters
    ----------
    config         : FullConfig
    plant_engine   : PlantEngine — shared simulation instance
    soh_curve      : dict[int, float] — {year: soh_fraction} from bess_soh_curve.csv
    solar_eff_curve: dict[int, float] — {year: eff_fraction}
    wind_eff_curve : dict[int, float] — {year: eff_fraction}
    loss_factor    : float — grid loss factor (from GridInterface)
    """

    def __init__(
        self,
        config:          FullConfig,
        plant_engine:    Any,
        soh_curve:       dict[int, float],
        solar_eff_curve: dict[int, float],
        wind_eff_curve:  dict[int, float],
        loss_factor:     float,
    ) -> None:
        self._config         = config
        self._plant          = plant_engine
        self._soh_curve      = soh_curve
        self._solar_eff      = solar_eff_curve
        self._wind_eff       = wind_eff_curve
        self._loss_factor    = loss_factor

        bess_cfg              = config.bess["bess"]
        aug_cfg               = bess_cfg["augmentation"]
        self._container_size  = float(bess_cfg["container"]["size_mwh"])
        self._project_life    = int(config.project["project"]["project_life_years"])
        self._cost_per_mwh    = float(aug_cfg["cost_per_mwh"])
        self._min_k           = int(aug_cfg.get("minimum_augmentation_containers", 1))
        self._max_k           = int(aug_cfg.get("max_augmentation_containers_per_event", 50))

        # BESS O&M rate (Rs per MWh) — no escalation, from finance.yaml
        bess_om_lakh_per_mwh   = config.finance["opex"]["bess"]["rate_lakh_per_mwh"]
        self._bess_om_rate_rs  = float(bess_om_lakh_per_mwh) * LAKH_TO_RS

    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        params:                  dict[str, Any],
        initial_containers:      int,
        trigger_threshold_cuf:   float,
        restoration_target_cuf:  float,
        fast_mode:               bool = False,
    ) -> LifecycleResult:
        """
        Run the full 25-year augmentation lifecycle for one scenario.

        Parameters
        ----------
        params                 : solar_mw, wind_mw, ppa_mw, c-rates, dispatch settings
        initial_containers     : BESS container count at project start
        trigger_threshold_cuf  : CUF floor from Pass-1 baseline (fixed across all trials)
        restoration_target_cuf : This scenario's own Year-1 CUF (per-trial target)
        fast_mode              : True → approximate CUF and energy; False → full re-sim

        Returns
        -------
        LifecycleResult
        """
        registry  = CohortRegistry(initial_containers)
        project_life  = self._project_life
        container_size = self._container_size

        # Accumulators
        solar_arr   = np.zeros(project_life)
        wind_arr    = np.zeros(project_life)
        battery_arr = np.zeros(project_life)
        pre_arr     = np.zeros(project_life)
        meter_arr   = np.zeros(project_life)
        cuf_series: list[float]  = []
        event_log:  list[dict]   = []

        # Augmentation OPEX accumulators (length = project_life)
        opex_lump = [0.0] * project_life   # one-time procurement in event year
        opex_om   = [0.0] * project_life   # recurring O&M from NEW cohorts only

        # fast_mode anchor state — updated after each event
        anchor_cuf = 0.0
        anchor_soh = 0.0
        anchor_y1_sim: dict[str, Any] | None = None  # Year-1 or post-event full sim

        # ── Year 1: always a full real simulation ──────────────────────────
        n0, soh0  = registry.to_plant_params(1, container_size, self._soh_curve)
        solar_eff1 = self._solar_eff.get(1, 1.0)
        wind_eff1  = self._wind_eff.get(1, 1.0)

        yr1 = self._plant.simulate(
            solar_capacity_mw  = params["solar_capacity_mw"] * solar_eff1,
            wind_capacity_mw   = params["wind_capacity_mw"]  * wind_eff1,
            bess_containers    = n0,
            charge_c_rate      = params["charge_c_rate"],
            discharge_c_rate   = params["discharge_c_rate"],
            ppa_capacity_mw    = params["ppa_capacity_mw"],
            dispatch_priority  = params["dispatch_priority"],
            bess_charge_source = params["bess_charge_source"],
            loss_factor        = self._loss_factor,
            bess_soh_factor    = soh0,
        )
        s1 = float(np.sum(yr1["solar_direct_pre"]))
        w1 = float(np.sum(yr1["wind_direct_pre"]))
        b1 = float(np.sum(yr1["discharge_pre"]))
        solar_arr[0]   = s1
        wind_arr[0]    = w1
        battery_arr[0] = b1
        pre_arr[0]     = s1 + w1 + b1
        meter_arr[0]   = (s1 + w1 + b1) * self._loss_factor

        cuf_y1 = compute_plant_cuf(self._plant, params, n0, soh0)
        cuf_series.append(cuf_y1)
        anchor_cuf     = cuf_y1
        anchor_soh     = soh0
        anchor_y1_sim  = yr1

        # ── Years 2–25 ────────────────────────────────────────────────────
        for i, year in enumerate(range(2, project_life + 1)):
            year_idx = year - 1  # 0-based position: year 2 → idx 1, year 3 → idx 2, …
            n_cont, blended_soh = registry.to_plant_params(year, container_size, self._soh_curve)
            solar_eff = self._solar_eff.get(year, 1.0)
            wind_eff  = self._wind_eff.get(year, 1.0)

            if fast_mode:
                # ── Approximate CUF via SOH ratio ─────────────────────────
                cuf_t = anchor_cuf * (blended_soh / anchor_soh) if anchor_soh > 0 else 0.0
            else:
                # ── Full re-simulation ────────────────────────────────────
                yr = self._plant.simulate(
                    solar_capacity_mw  = params["solar_capacity_mw"] * solar_eff,
                    wind_capacity_mw   = params["wind_capacity_mw"]  * wind_eff,
                    bess_containers    = n_cont,
                    charge_c_rate      = params["charge_c_rate"],
                    discharge_c_rate   = params["discharge_c_rate"],
                    ppa_capacity_mw    = params["ppa_capacity_mw"],
                    dispatch_priority  = params["dispatch_priority"],
                    bess_charge_source = params["bess_charge_source"],
                    loss_factor        = self._loss_factor,
                    bess_soh_factor    = blended_soh,
                )
                solar_arr[year_idx]   = float(np.sum(yr["solar_direct_pre"]))
                wind_arr[year_idx]    = float(np.sum(yr["wind_direct_pre"]))
                battery_arr[year_idx] = float(np.sum(yr["discharge_pre"]))
                pre_arr[year_idx]     = solar_arr[year_idx] + wind_arr[year_idx] + battery_arr[year_idx]
                meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor
                cuf_t          = compute_plant_cuf(self._plant, params, n_cont, blended_soh)

            cuf_series.append(cuf_t)

            # ── Augmentation check (never in Year 1) ──────────────────────
            if year > 1 and cuf_t < trigger_threshold_cuf:
                best_k, post_event_cuf = self._find_best_k(
                    params, year, registry, trigger_threshold_cuf, restoration_target_cuf,
                    solar_eff, wind_eff,
                )

                # Register the event cohort
                registry.add(install_year=year, containers=best_k)

                # Compute costs
                new_mwh     = best_k * container_size
                lump_cost   = new_mwh * self._cost_per_mwh
                annual_om   = new_mwh * self._bess_om_rate_rs

                # Lump-sum in event year only
                opex_lump[year_idx] = lump_cost

                # Recurring O&M from this new cohort: event_year → year 25
                for om_i in range(year_idx, project_life):
                    opex_om[om_i] += annual_om

                event_log.append({
                    "year":           year,
                    "trigger_cuf":    cuf_t,
                    "target_cuf":     restoration_target_cuf,
                    "post_event_cuf": post_event_cuf,
                    "k_containers":   best_k,
                    "lump_cost_rs":   lump_cost,
                })

                # ── Re-anchor CUF and re-compute energy for this year ─────
                n_post, soh_post = registry.to_plant_params(year, container_size, self._soh_curve)
                anchor_cuf = post_event_cuf
                anchor_soh = soh_post

                # Energy re-sim after augmentation (always, even in fast_mode)
                yr_post = self._plant.simulate(
                    solar_capacity_mw  = params["solar_capacity_mw"] * solar_eff,
                    wind_capacity_mw   = params["wind_capacity_mw"]  * wind_eff,
                    bess_containers    = n_post,
                    charge_c_rate      = params["charge_c_rate"],
                    discharge_c_rate   = params["discharge_c_rate"],
                    ppa_capacity_mw    = params["ppa_capacity_mw"],
                    dispatch_priority  = params["dispatch_priority"],
                    bess_charge_source = params["bess_charge_source"],
                    loss_factor        = self._loss_factor,
                    bess_soh_factor    = soh_post,
                )
                solar_arr[year_idx]   = float(np.sum(yr_post["solar_direct_pre"]))
                wind_arr[year_idx]    = float(np.sum(yr_post["wind_direct_pre"]))
                battery_arr[year_idx] = float(np.sum(yr_post["discharge_pre"]))
                pre_arr[year_idx]     = solar_arr[year_idx] + wind_arr[year_idx] + battery_arr[year_idx]
                meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor
                # Update CUF entry for this year to the post-event value
                cuf_series[-1] = post_event_cuf

            elif fast_mode:
                # fast_mode energy approximation (no event year)
                # Scale Year-1 totals by degradation factor ratios
                solar_arr[year_idx]   = float(np.sum(anchor_y1_sim["solar_direct_pre"])) * solar_eff / self._solar_eff.get(1, 1.0)
                wind_arr[year_idx]    = float(np.sum(anchor_y1_sim["wind_direct_pre"]))  * wind_eff  / self._wind_eff.get(1, 1.0)
                battery_arr[year_idx] = float(np.sum(anchor_y1_sim["discharge_pre"]))    * blended_soh / self._soh_curve.get(1, 1.0)
                pre_arr[year_idx]     = solar_arr[year_idx] + wind_arr[year_idx] + battery_arr[year_idx]
                meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor

        energy_projection = {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,
            "delivered_meter_mwh":  meter_arr,
        }

        cohort_snap     = registry.snapshot()
        capacity_timeline = registry.cohort_capacity_timeline(
            project_life, container_size, self._soh_curve
        )

        return LifecycleResult(
            energy_projection        = energy_projection,
            opex_augmentation_lump   = opex_lump,
            opex_augmentation_om     = opex_om,
            event_log                = event_log,
            cuf_series               = cuf_series,
            cohort_snapshot          = cohort_snap,
            cohort_capacity_timeline = capacity_timeline,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_best_k(
        self,
        params:                  dict[str, Any],
        event_year:              int,
        registry:                CohortRegistry,
        trigger_threshold_cuf:   float,
        restoration_target_cuf:  float,
        solar_eff:               float,
        wind_eff:                float,
    ) -> tuple[int, float]:
        """
        Search for the best number of containers to add at an event.

        Strategy (§3.7 of the implementation brief):
          1. Try k = min_k first.
          2. If post-event CUF >= restoration_target_cuf: this k satisfies the
             hard constraint.  Continue trying larger k in case economics improve.
          3. Larger k is accepted while the post-event CUF improves (greedy
             stopping: first k where improvement stops or k >= max_k).
          4. If no k restores CUF to target: warn and use largest k tried.

        Returns
        -------
        (best_k, post_event_cuf_with_best_k)
        """
        container_size = self._container_size
        best_k         = self._min_k
        best_post_cuf  = 0.0
        cuf_satisfied  = False
        prev_post_cuf  = -1.0

        for k in range(self._min_k, self._max_k + 1):
            # Temporarily compute post-event params without mutating registry
            trial_registry = CohortRegistry(registry.cohorts[0].containers)
            for cohort in registry.cohorts[1:]:
                trial_registry.add(cohort.install_year, cohort.containers)
            trial_registry.add(event_year, k)

            n_trial, soh_trial = trial_registry.to_plant_params(
                event_year, container_size, self._soh_curve
            )

            post_cuf = compute_plant_cuf(self._plant, params, n_trial, soh_trial)

            if k == self._min_k:
                best_k        = k
                best_post_cuf = post_cuf
                cuf_satisfied = post_cuf >= restoration_target_cuf
                prev_post_cuf = post_cuf
                continue

            # Accept larger k if CUF is still improving
            if post_cuf > prev_post_cuf:
                best_k        = k
                best_post_cuf = post_cuf
                prev_post_cuf = post_cuf
                if post_cuf >= restoration_target_cuf:
                    cuf_satisfied = True
                if k == self._max_k:
                    break
                continue
            else:
                # CUF stopped improving — check if we've already satisfied target
                if best_post_cuf >= restoration_target_cuf:
                    break  # good: stop with current best_k
                # Not satisfied yet but CUF regressed — keep best_k so far, stop
                break

        if k == self._max_k and best_post_cuf < restoration_target_cuf:
            warnings.warn(
                f"Augmentation Year {event_year}: max_k={self._max_k} containers tried "
                f"but post-event CUF {best_post_cuf:.2f}% < target {restoration_target_cuf:.2f}%. "
                "Using best k found. Check inputs for pathological configuration.",
                stacklevel=4,
            )

        return best_k, best_post_cuf
