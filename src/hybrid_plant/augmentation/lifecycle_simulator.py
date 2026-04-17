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
     active cohorts AND year-t solar/wind operating efficiencies.
  2. Detects when Plant CUF falls below ``trigger_threshold_cuf`` by more
     than ``trigger_tolerance_pp`` percentage points (suppresses float-
     noise triggers at the margin).
  3. At each triggered year, searches for the smallest k containers that
     restore CUF back to the **adjusted** restoration target
     (Y1 CUF × solar_eff_ratio × wind_eff_ratio — honest about what
     augmentation can actually recover), then continues searching in case
     larger k improves NPV.
  4. Returns per-year energy arrays compatible with EnergyProjection
     outputs, plus augmentation OPEX series, an event log, and metadata
     for the dashboard.

CUF formula (blended, single source of truth)
──────────────────────────────────────────────
    CUF_t (%) = busbar_t_MWh / (PPA_MW × 8760) × 100

This is the naive formula from ``cuf_evaluator.compute_plant_cuf``.  It
responds to all three degradation sources (solar, wind, BESS) simultaneously
because busbar_t is computed with year-t operating values for all three.

Restoration target (adjusted)
──────────────────────────────
For year t, the adjusted target is:

    target_t = Y1_CUF × operating_value(solar_eff, t) × operating_value(wind_eff, t)
              [weighted by the Y1 share of solar vs wind contribution]

Rationale: augmentation can restore BESS capacity but cannot replace
degraded solar panels or wind turbines.  A fixed target equal to Y1 CUF
is structurally unreachable in late years once solar/wind have aged,
which would cause the k-search to fruitlessly hit max_k and warn every
year.  The adjusted target reflects what augmentation CAN reach.

Warning behaviour: warn only if post-event CUF < adjusted_target_t
(i.e. "we failed to restore even what was achievable").  Events that
reach adjusted_target but not the hard Y1 threshold fire silently —
this is expected in late years and not a bug.

Hard requirement: when augmentation is enabled, the post-event CUF
should not fall below ``trigger_threshold_cuf`` in any year.  If the
adjusted target permits this, it means solar+wind alone have degraded
past the threshold and no amount of augmentation can help.  That
condition is flagged separately with a "threshold unreachable" warning.

fast_mode contract
──────────────────
  fast_mode=True  (used inside Pass 2 Optuna trials):
      Per-year Plant CUF is approximated from scalar-scaled energy:

          busbar_t ≈ anchor_busbar × (solar_eff[t] × solar_share_anchor
                                     + wind_eff[t]  × wind_share_anchor
                                     + soh[t]       × battery_share_anchor)

      The anchor is the last year for which we ran a real simulation
      (initially Year 1; updated to the post-event year after each event).
      Per-year plant_engine.simulate() calls occur only for Year 1 and
      immediately after each augmentation event.

  fast_mode=False (used for final best-result reporting):
      Full plant_engine.simulate() call every year with that year's exact
      blended SOH and solar/wind operating efficiencies.  Captures non-
      linear interactions between degraded streams.

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
from dataclasses import dataclass
from typing import Any

import numpy as np

from hybrid_plant.augmentation.cohort import CohortRegistry
from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import HOURS_PER_YEAR, LAKH_TO_RS
from hybrid_plant.data_loader import operating_value

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
    event_log                  : list[dict]  — one record per fired augmentation event
    skipped_event_log          : list[dict]  — one record per event candidate that was
                                  suppressed by the payback filter (for dashboard visibility)
    cuf_series                 : list[float] — Plant CUF (%) for each year
    adjusted_target_series     : list[float] — adjusted restoration target (%) per year
    cohort_snapshot            : list[dict]  — final cohort list (for dashboard)
    cohort_capacity_timeline   : dict[int, list[float]] — per-cohort effective MWh/yr
    """

    energy_projection:         dict[str, np.ndarray]
    opex_augmentation_lump:    list[float]
    opex_augmentation_om:      list[float]
    event_log:                 list[dict[str, Any]]
    skipped_event_log:         list[dict[str, Any]]
    cuf_series:                list[float]
    adjusted_target_series:    list[float]
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
        event_filter:    Any | None = None,
    ) -> None:
        self._config         = config
        self._plant          = plant_engine
        self._soh_curve      = soh_curve
        self._solar_eff      = solar_eff_curve
        self._wind_eff       = wind_eff_curve
        self._loss_factor    = loss_factor
        self._event_filter   = event_filter

        bess_cfg              = config.bess["bess"]
        aug_cfg               = bess_cfg["augmentation"]
        self._container_size  = float(bess_cfg["container"]["size_mwh"])
        self._project_life    = int(config.project["project"]["project_life_years"])
        self._cost_per_mwh    = float(aug_cfg["cost_per_mwh"])
        self._min_k           = int(aug_cfg.get("minimum_augmentation_containers", 1))
        self._max_k           = int(aug_cfg.get("max_augmentation_containers_per_event", 50))
        self._trigger_tol_pp  = float(aug_cfg.get("trigger_tolerance_pp", 0.0))

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
        restoration_target_cuf : This scenario's own Year-1 CUF (basis for adjusted target)
        fast_mode              : True → approximate; False → full re-sim each year

        Returns
        -------
        LifecycleResult
        """
        registry        = CohortRegistry(initial_containers)
        project_life    = self._project_life
        container_size  = self._container_size
        ppa_mw          = float(params["ppa_capacity_mw"])
        eps             = self._trigger_tol_pp

        # Accumulators
        solar_arr   = np.zeros(project_life)
        wind_arr    = np.zeros(project_life)
        battery_arr = np.zeros(project_life)
        pre_arr     = np.zeros(project_life)
        meter_arr   = np.zeros(project_life)
        cuf_series:             list[float]  = []
        adjusted_target_series: list[float]  = []
        event_log:              list[dict]   = []

        # Augmentation OPEX accumulators (length = project_life)
        opex_lump = [0.0] * project_life
        opex_om   = [0.0] * project_life

        skipped_event_log: list[dict] = []

        # ── Year 1: always a full real simulation ──────────────────────────
        # New convention: operating value = 1.0 for year 1 (fresh plant).
        n0, soh0    = registry.to_plant_params(1, container_size, self._soh_curve)
        solar_eff1  = operating_value(self._solar_eff, 1)    # = 1.0
        wind_eff1   = operating_value(self._wind_eff, 1)     # = 1.0

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
            bess_soh_factor    = soh0,   # = 1.0 for fresh cohort age 1
        )
        s1 = float(np.sum(yr1["solar_direct_pre"]))
        w1 = float(np.sum(yr1["wind_direct_pre"]))
        b1 = float(np.sum(yr1["discharge_pre"]))
        solar_arr[0]   = s1
        wind_arr[0]    = w1
        battery_arr[0] = b1
        pre_arr[0]     = s1 + w1 + b1
        meter_arr[0]   = pre_arr[0] * self._loss_factor

        cuf_y1 = compute_plant_cuf(pre_arr[0], ppa_mw)
        cuf_series.append(cuf_y1)
        adjusted_target_series.append(restoration_target_cuf)  # Y1 target = Y1 itself

        # Fast-mode anchor state — tracks the last "true" simulation point
        # and the Year-1 share of each stream for scalar scaling.
        anchor_year          = 1
        anchor_solar_busbar  = s1
        anchor_wind_busbar   = w1
        anchor_battery_busbar = b1
        anchor_solar_eff     = solar_eff1
        anchor_wind_eff      = wind_eff1
        anchor_soh           = soh0

        # ── Years 2–25 ────────────────────────────────────────────────────
        for year in range(2, project_life + 1):
            year_idx  = year - 1
            n_cont, blended_soh = registry.to_plant_params(year, container_size, self._soh_curve)
            solar_eff = operating_value(self._solar_eff, year)
            wind_eff  = operating_value(self._wind_eff,  year)

            # Adjusted restoration target for this year — the Y1 CUF that's
            # achievable given CURRENT solar/wind operating values.  (BESS
            # factor is 1.0 because augmentation CAN restore it.)
            # We scale Y1 CUF by the share-weighted combination of solar/wind
            # degradation, using Y1 contribution shares.
            y1_total = s1 + w1 + b1
            if y1_total > 0:
                solar_share = s1 / y1_total
                wind_share  = w1 / y1_total
                battery_share = b1 / y1_total
                # Degradation factor applied to each stream for this year
                deg_factor = (
                    solar_share   * solar_eff
                    + wind_share  * wind_eff
                    + battery_share * 1.0    # BESS fully restorable by aug
                )
                adjusted_target = restoration_target_cuf * deg_factor
            else:
                adjusted_target = restoration_target_cuf
            adjusted_target_series.append(adjusted_target)

            # ── Compute per-year energy & CUF ─────────────────────────────
            if fast_mode:
                # Scalar-scale anchor busbar by per-stream operating-value ratios
                solar_t = anchor_solar_busbar  * (solar_eff / anchor_solar_eff) if anchor_solar_eff > 0 else 0.0
                wind_t  = anchor_wind_busbar   * (wind_eff  / anchor_wind_eff)  if anchor_wind_eff  > 0 else 0.0
                batt_t  = anchor_battery_busbar * (blended_soh / anchor_soh)    if anchor_soh > 0 else 0.0
                solar_arr[year_idx]   = solar_t
                wind_arr[year_idx]    = wind_t
                battery_arr[year_idx] = batt_t
                pre_arr[year_idx]     = solar_t + wind_t + batt_t
                meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor
                cuf_t = compute_plant_cuf(pre_arr[year_idx], ppa_mw)
            else:
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
                pre_arr[year_idx]     = (solar_arr[year_idx] + wind_arr[year_idx]
                                         + battery_arr[year_idx])
                meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor
                cuf_t = compute_plant_cuf(pre_arr[year_idx], ppa_mw)

            cuf_series.append(cuf_t)

            # ── Augmentation check (with tolerance) ───────────────────────
            if cuf_t < (trigger_threshold_cuf - eps):
                best_k, post_event_cuf, post_event_sim = self._find_best_k(
                    params, year, registry, solar_eff, wind_eff,
                    adjusted_target, trigger_threshold_cuf, ppa_mw,
                )

                # Compute post-event busbar for the payback filter
                if post_event_sim is not None:
                    post_event_busbar = (
                        float(np.sum(post_event_sim["solar_direct_pre"]))
                        + float(np.sum(post_event_sim["wind_direct_pre"]))
                        + float(np.sum(post_event_sim["discharge_pre"]))
                    )
                else:
                    post_event_busbar = pre_arr[year_idx]

                # ── Payback filter (optional) ──────────────────────────────
                event_info = {
                    "year":                  year,
                    "trigger_cuf":           cuf_t,
                    "adjusted_target":       adjusted_target,
                    "hard_threshold":        trigger_threshold_cuf,
                    "post_event_cuf":        post_event_cuf,
                    "k_containers":          best_k,
                    "pre_event_busbar_mwh":  pre_arr[year_idx],
                    "post_event_busbar_mwh": post_event_busbar,
                }

                should_fire = (
                    self._event_filter is None or self._event_filter(event_info)
                )

                if not should_fire:
                    # Log skipped event for dashboard visibility; do NOT mutate state
                    logger.info(
                        "Augmentation Year %d: event SKIPPED by payback filter "
                        "(k=%d, post_event_cuf=%.4f%%).",
                        year, best_k, post_event_cuf,
                    )
                    skipped_event_log.append({
                        "year":             year,
                        "trigger_cuf":      cuf_t,
                        "adjusted_target":  adjusted_target,
                        "hard_threshold":   trigger_threshold_cuf,
                        "post_event_cuf":   post_event_cuf,
                        "reached_adjusted": post_event_cuf >= adjusted_target - 1e-9,
                        "reached_hard":     post_event_cuf >= trigger_threshold_cuf - 1e-9,
                        "k_containers":     0,
                        "lump_cost_rs":     0.0,
                        "skipped_by_filter": True,
                    })
                else:
                    # ── Register the event cohort and account for costs ────
                    registry.add(install_year=year, containers=best_k)

                    new_mwh   = best_k * container_size
                    lump_cost = new_mwh * self._cost_per_mwh
                    annual_om = new_mwh * self._bess_om_rate_rs
                    opex_lump[year_idx] = lump_cost
                    for om_i in range(year_idx, project_life):
                        opex_om[om_i] += annual_om

                    event_log.append({
                        "year":               year,
                        "trigger_cuf":        cuf_t,
                        "adjusted_target":    adjusted_target,
                        "hard_threshold":     trigger_threshold_cuf,
                        "post_event_cuf":     post_event_cuf,
                        "reached_adjusted":   post_event_cuf >= adjusted_target - 1e-9,
                        "reached_hard":       post_event_cuf >= trigger_threshold_cuf - 1e-9,
                        "k_containers":       best_k,
                        "lump_cost_rs":       lump_cost,
                    })

                    # ── Update this year's energy to post-event values ────
                    if post_event_sim is not None:
                        solar_arr[year_idx]   = float(np.sum(post_event_sim["solar_direct_pre"]))
                        wind_arr[year_idx]    = float(np.sum(post_event_sim["wind_direct_pre"]))
                        battery_arr[year_idx] = float(np.sum(post_event_sim["discharge_pre"]))
                        pre_arr[year_idx]     = (solar_arr[year_idx] + wind_arr[year_idx]
                                                 + battery_arr[year_idx])
                        meter_arr[year_idx]   = pre_arr[year_idx] * self._loss_factor
                        cuf_series[-1]        = post_event_cuf

                        # Re-anchor fast_mode state to this post-event year
                        n_post, soh_post = registry.to_plant_params(year, container_size,
                                                                    self._soh_curve)
                        anchor_year           = year
                        anchor_solar_busbar   = solar_arr[year_idx]
                        anchor_wind_busbar    = wind_arr[year_idx]
                        anchor_battery_busbar = battery_arr[year_idx]
                        anchor_solar_eff      = solar_eff
                        anchor_wind_eff       = wind_eff
                        anchor_soh            = soh_post

        energy_projection = {
            "solar_direct_mwh":     solar_arr,
            "wind_direct_mwh":      wind_arr,
            "battery_mwh":          battery_arr,
            "delivered_pre_mwh":    pre_arr,
            "delivered_meter_mwh":  meter_arr,
        }

        cohort_snap = registry.snapshot()
        capacity_timeline = registry.cohort_capacity_timeline(
            project_life, container_size, self._soh_curve
        )

        return LifecycleResult(
            energy_projection        = energy_projection,
            opex_augmentation_lump   = opex_lump,
            opex_augmentation_om     = opex_om,
            event_log                = event_log,
            skipped_event_log        = skipped_event_log,
            cuf_series               = cuf_series,
            adjusted_target_series   = adjusted_target_series,
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
        solar_eff:               float,
        wind_eff:                float,
        adjusted_target_cuf:     float,
        hard_threshold_cuf:      float,
        ppa_mw:                  float,
    ) -> tuple[int, float, dict | None]:
        """
        Find the **minimum** k in [min_k, max_k] such that the post-event
        CUF meets or exceeds ``adjusted_target_cuf``.

        Strategy
        ────────
        1. Iterate k = min_k, min_k+1, …, max_k.
        2. Early-exit at the first k whose post-event CUF >= adjusted_target.
        3. Track the best (highest-CUF) result seen so far in case the target
           is never reached — if the loop exhausts max_k, return max_k with a
           warning.  max_k is a pure safety cap, NOT a target.

        Warnings
        ────────
        • best post-CUF < adjusted_target after full search → WARN (failed
          to restore even the achievable CUF; check max_k or config).
        • best post-CUF < hard_threshold but >= adjusted_target → INFO
          (structural solar/wind degradation; not a BESS problem).

        Returns
        -------
        (best_k, post_event_cuf_with_best_k, post_event_sim_dict)
        """
        container_size = self._container_size
        best_k         = self._min_k
        best_post_cuf  = -1.0
        best_sim: dict | None = None

        for k in range(self._min_k, self._max_k + 1):
            # Build trial registry with this candidate k
            trial_registry = CohortRegistry(registry.cohorts[0].containers)
            for cohort in registry.cohorts[1:]:
                trial_registry.add(cohort.install_year, cohort.containers)
            trial_registry.add(event_year, k)

            n_trial, soh_trial = trial_registry.to_plant_params(
                event_year, container_size, self._soh_curve
            )

            # Full dispatch simulation for this k
            sim = self._plant.simulate(
                solar_capacity_mw  = params["solar_capacity_mw"] * solar_eff,
                wind_capacity_mw   = params["wind_capacity_mw"]  * wind_eff,
                bess_containers    = n_trial,
                charge_c_rate      = params["charge_c_rate"],
                discharge_c_rate   = params["discharge_c_rate"],
                ppa_capacity_mw    = params["ppa_capacity_mw"],
                dispatch_priority  = params["dispatch_priority"],
                bess_charge_source = params["bess_charge_source"],
                loss_factor        = self._loss_factor,
                bess_soh_factor    = soh_trial,
            )
            busbar = (float(np.sum(sim["solar_direct_pre"]))
                      + float(np.sum(sim["wind_direct_pre"]))
                      + float(np.sum(sim["discharge_pre"])))
            post_cuf = compute_plant_cuf(busbar, ppa_mw)

            # Always track the best-so-far (fallback when target is unreachable)
            if post_cuf > best_post_cuf:
                best_k        = k
                best_post_cuf = post_cuf
                best_sim      = sim

            # ── Early exit: minimum k that meets the adjusted target ──────
            if post_cuf >= adjusted_target_cuf - 1e-9:
                best_k        = k
                best_post_cuf = post_cuf
                best_sim      = sim
                break
        else:
            # Loop exhausted max_k without meeting the adjusted target
            warnings.warn(
                f"Augmentation Year {event_year}: k-search exhausted max_k={self._max_k} "
                f"without reaching adjusted target {adjusted_target_cuf:.4f}%. "
                f"Best post-event CUF was {best_post_cuf:.4f}%. "
                f"Returning k={best_k} (best found). Check "
                f"max_augmentation_containers_per_event in bess.yaml.",
                stacklevel=4,
            )

        # ── Post-search warnings ──────────────────────────────────────────
        if best_post_cuf < adjusted_target_cuf - 1e-6:
            # Only warn if not already warned by the loop-exhausted path above
            if best_k < self._max_k:
                warnings.warn(
                    f"Augmentation Year {event_year}: best k={best_k} reached "
                    f"post-event CUF {best_post_cuf:.4f}% but adjusted target "
                    f"is {adjusted_target_cuf:.4f}%. Augmentation failed to "
                    f"restore even the achievable CUF.",
                    stacklevel=4,
                )
        elif best_post_cuf < hard_threshold_cuf - 1e-6:
            # Reached adjusted target but still below hard threshold — structural
            # solar/wind degradation; not fixable by BESS augmentation alone.
            logger.info(
                "Augmentation Year %d: post-event CUF %.4f%% is below "
                "hard threshold %.4f%% but meets adjusted target "
                "%.4f%% (solar/wind degradation, not a BESS issue).",
                event_year, best_post_cuf, hard_threshold_cuf,
                adjusted_target_cuf,
            )

        return best_k, best_post_cuf, best_sim