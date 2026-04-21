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
  3. At each triggered year (up to ``max_augmentation_events`` total),
     searches for the smallest k containers that restore CUF back to
     the hard threshold, then continues searching in case larger k
     improves NPV.
  4. Returns per-year energy arrays compatible with EnergyProjection
     outputs, plus augmentation OPEX series, an event log, and metadata
     for the dashboard.

CUF formula (blended, single source of truth)
──────────────────────────────────────────────
    CUF_t (%) = busbar_t_MWh / (PPA_MW × 8760) × 100

This is the naive formula from ``cuf_evaluator.compute_plant_cuf``.  It
responds to all three degradation sources (solar, wind, BESS) simultaneously
because busbar_t is computed with year-t operating values for all three.

Augmentation target
───────────────────
k-search finds the minimum k that restores CUF to ``trigger_threshold_cuf``
(the hard floor from the Pass-1 baseline).  At most ``max_augmentation_events``
events fire across the 25-year life; after that limit is reached, no further
augmentation occurs.

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
        self._trigger_tol_pp  = float(aug_cfg.get("trigger_tolerance_pp", 0.0))
        self._max_events      = int(aug_cfg.get("max_augmentation_events", 3))

        # BESS O&M rate (Rs per MWh) — no escalation, from finance.yaml
        bess_om_lakh_per_mwh   = config.finance["opex"]["bess"]["rate_lakh_per_mwh"]
        self._bess_om_rate_rs  = float(bess_om_lakh_per_mwh) * LAKH_TO_RS

    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        params:                dict[str, Any],
        initial_containers:    int,
        trigger_threshold_cuf: float,
        fast_mode:             bool = False,
        max_events_override:   int | None = None,
    ) -> LifecycleResult:
        """
        Run the full 25-year augmentation lifecycle for one scenario.

        Parameters
        ----------
        params                 : solar_mw, wind_mw, ppa_mw, c-rates, dispatch settings
        initial_containers     : BESS container count at project start
        trigger_threshold_cuf  : CUF floor from Pass-1 baseline (fixed across all trials).
                                 Augmentation fires when CUF drops below this.
                                 k-search restores CUF back to this level.
        fast_mode              : True → approximate; False → full re-sim each year
        max_events_override    : When provided, overrides the config max_augmentation_events.
                                 Pass 0 to suppress all events (CUF constraint feasibility
                                 check), or a large int for effectively unlimited events.

        Returns
        -------
        LifecycleResult
        """
        registry        = CohortRegistry(initial_containers)
        project_life    = self._project_life
        container_size  = self._container_size
        ppa_mw          = float(params["ppa_capacity_mw"])
        eps             = self._trigger_tol_pp
        max_events      = max_events_override if max_events_override is not None else self._max_events

        # Accumulators
        solar_arr   = np.zeros(project_life)
        wind_arr    = np.zeros(project_life)
        battery_arr = np.zeros(project_life)
        pre_arr     = np.zeros(project_life)
        meter_arr   = np.zeros(project_life)
        cuf_series: list[float] = []
        event_log:  list[dict]  = []

        # Augmentation OPEX accumulators (length = project_life)
        opex_lump = [0.0] * project_life
        opex_om   = [0.0] * project_life

        events_used = 0
        # max_events resolved above (override or config value)

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

        # Fast-mode anchor state — tracks the last "true" simulation point
        anchor_solar_busbar   = s1
        anchor_wind_busbar    = w1
        anchor_battery_busbar = b1
        anchor_solar_eff      = solar_eff1
        anchor_wind_eff       = wind_eff1
        anchor_soh            = soh0

        # ── Years 2–25 ────────────────────────────────────────────────────
        for year in range(2, project_life + 1):
            year_idx  = year - 1
            n_cont, blended_soh = registry.to_plant_params(year, container_size, self._soh_curve)
            solar_eff = operating_value(self._solar_eff, year)
            wind_eff  = operating_value(self._wind_eff,  year)

            # ── Compute per-year energy & CUF ─────────────────────────────
            if fast_mode:
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

            # ── Augmentation check ─────────────────────────────────────────
            if cuf_t < (trigger_threshold_cuf - eps) and events_used < max_events:
                best_k, post_event_cuf, post_event_sim = self._find_best_k(
                    params, year, registry, solar_eff, wind_eff,
                    trigger_threshold_cuf, ppa_mw,
                )

                # Register the event cohort and account for costs
                registry.add(install_year=year, containers=best_k)
                events_used += 1

                new_mwh   = best_k * container_size
                lump_cost = new_mwh * self._cost_per_mwh
                annual_om = new_mwh * self._bess_om_rate_rs
                opex_lump[year_idx] = lump_cost
                for om_i in range(year_idx, project_life):
                    opex_om[om_i] += annual_om

                event_log.append({
                    "year":           year,
                    "trigger_cuf":    cuf_t,
                    "hard_threshold": trigger_threshold_cuf,
                    "post_event_cuf": post_event_cuf,
                    "reached_hard":   post_event_cuf >= trigger_threshold_cuf - 1e-9,
                    "k_containers":   best_k,
                    "lump_cost_rs":   lump_cost,
                })

                # Update this year's energy to post-event values
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
            cuf_series               = cuf_series,
            cohort_snapshot          = cohort_snap,
            cohort_capacity_timeline = capacity_timeline,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fast_future_min_cuf(
        self,
        trial_registry:     CohortRegistry,
        anchor_year:        int,
        anchor_solar:       float,
        anchor_wind:        float,
        anchor_battery:     float,
        anchor_solar_eff:   float,
        anchor_wind_eff:    float,
        anchor_blended_soh: float,
        ppa_mw:             float,
    ) -> float:
        """
        Return the minimum projected CUF for years anchor_year+1 … project_life,
        using the same fast-mode linear-scaling approximation as simulate().

        This is cheap (pure arithmetic — no plant.simulate() calls) and used by
        _find_best_k to check whether a candidate k sustains CUF through the
        end of the project without another augmentation event.
        """
        min_cuf = float("inf")
        container_size = self._container_size
        for future_year in range(anchor_year + 1, self._project_life + 1):
            fs   = operating_value(self._solar_eff, future_year)
            fw   = operating_value(self._wind_eff,  future_year)
            _, fsoh = trial_registry.to_plant_params(future_year, container_size, self._soh_curve)
            solar_t = anchor_solar   * (fs   / anchor_solar_eff)   if anchor_solar_eff   > 0 else 0.0
            wind_t  = anchor_wind    * (fw   / anchor_wind_eff)    if anchor_wind_eff    > 0 else 0.0
            batt_t  = anchor_battery * (fsoh / anchor_blended_soh) if anchor_blended_soh > 0 else 0.0
            cuf_t   = compute_plant_cuf(solar_t + wind_t + batt_t, ppa_mw)
            if cuf_t < min_cuf:
                min_cuf = cuf_t
        return min_cuf

    def _find_best_k(
        self,
        params:               dict[str, Any],
        event_year:           int,
        registry:             CohortRegistry,
        solar_eff:            float,
        wind_eff:             float,
        hard_threshold_cuf:   float,
        ppa_mw:               float,
    ) -> tuple[int, float, dict | None]:
        """
        Find the **minimum** k in [min_k, max_k] such that:

        Gate 1 — post-event CUF at event_year >= hard_threshold_cuf.
        Gate 2 — projected CUF for ALL years event_year+1 … project_life
                 >= hard_threshold_cuf (fast-mode approximation).

        Strategy
        ────────
        Phase 1 (linear scan, Gate 1):
            Iterate k = min_k … max_k running a full plant.simulate() per k.
            Cache results.  Stop at the first k that passes Gate 1 (k_min1).
            If no k passes Gate 1, return the best-CUF k found with a warning.

        Phase 2 (binary search, Gate 2):
            Among k_min1 … max_k, binary-search for the smallest k whose
            fast-mode future lookahead also stays >= threshold.
            If even max_k fails Gate 2, return max_k with an informational
            warning — a future event will handle the remaining shortfall.

        Returns
        -------
        (best_k, post_event_cuf_with_best_k, post_event_sim_dict)
        """
        container_size = self._container_size

        # ── helpers ───────────────────────────────────────────────────────────

        def _build_registry(k: int) -> CohortRegistry:
            tr = CohortRegistry(registry.cohorts[0].containers)
            for cohort in registry.cohorts[1:]:
                tr.add(cohort.install_year, cohort.containers)
            tr.add(event_year, k)
            return tr

        def _run_sim(k: int) -> tuple[float, float, dict, CohortRegistry]:
            """Full plant sim for candidate k; returns (post_cuf, soh, sim, registry)."""
            tr = _build_registry(k)
            n_trial, soh_trial = tr.to_plant_params(event_year, container_size, self._soh_curve)
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
            busbar   = (float(np.sum(sim["solar_direct_pre"]))
                        + float(np.sum(sim["wind_direct_pre"]))
                        + float(np.sum(sim["discharge_pre"])))
            post_cuf = compute_plant_cuf(busbar, ppa_mw)
            return post_cuf, soh_trial, sim, tr

        def _future_ok(k: int, soh_trial: float, sim: dict, tr: CohortRegistry) -> bool:
            """Gate 2: fast-mode lookahead for years event_year+1 … project_life."""
            if event_year >= self._project_life:
                return True   # no future years to check
            anchor_solar   = float(np.sum(sim["solar_direct_pre"]))
            anchor_wind    = float(np.sum(sim["wind_direct_pre"]))
            anchor_battery = float(np.sum(sim["discharge_pre"]))
            future_min = self._fast_future_min_cuf(
                trial_registry     = tr,
                anchor_year        = event_year,
                anchor_solar       = anchor_solar,
                anchor_wind        = anchor_wind,
                anchor_battery     = anchor_battery,
                anchor_solar_eff   = solar_eff,
                anchor_wind_eff    = wind_eff,
                anchor_blended_soh = soh_trial,
                ppa_mw             = ppa_mw,
            )
            return future_min >= hard_threshold_cuf - 1e-9

        # ── Phase 1: linear scan to find k_min1 (Gate 1) ─────────────────────
        sim_cache: dict[int, tuple[float, float, dict, CohortRegistry]] = {}
        best_k_g1     = self._min_k
        best_cuf_g1   = -1.0
        best_sim_g1:  dict | None = None
        k_min1: int | None = None

        for k in range(self._min_k, self._max_k + 1):
            post_cuf, soh_trial, sim, tr = _run_sim(k)
            sim_cache[k] = (post_cuf, soh_trial, sim, tr)
            if post_cuf > best_cuf_g1:
                best_k_g1, best_cuf_g1, best_sim_g1 = k, post_cuf, sim
            if post_cuf >= hard_threshold_cuf - 1e-9:
                k_min1 = k
                break

        # Gate 1 failed entirely — return the highest-CUF result with a warning
        if k_min1 is None:
            warnings.warn(
                f"Augmentation Year {event_year}: k-search exhausted max_k={self._max_k} "
                f"without reaching hard threshold {hard_threshold_cuf:.4f}% at the event year. "
                f"Best post-event CUF was {best_cuf_g1:.4f}%. "
                f"Returning k={best_k_g1} (best found). "
                f"Consider raising max_augmentation_containers_per_event in bess.yaml.",
                stacklevel=4,
            )
            return best_k_g1, best_cuf_g1, best_sim_g1

        # ── Phase 2: binary search k_min1 … max_k for Gate 2 ─────────────────
        # First check whether max_k itself passes Gate 2; if not, return it and warn.
        post_cuf_max, soh_max, sim_max, tr_max = (
            sim_cache[self._max_k]
            if self._max_k in sim_cache
            else _run_sim(self._max_k)
        )
        sim_cache[self._max_k] = (post_cuf_max, soh_max, sim_max, tr_max)

        if not _future_ok(self._max_k, soh_max, sim_max, tr_max):
            warnings.warn(
                f"Augmentation Year {event_year}: even max_k={self._max_k} cannot sustain "
                f"CUF >= {hard_threshold_cuf:.4f}% through end of project. "
                f"A future augmentation event will be needed.",
                stacklevel=4,
            )
            return self._max_k, post_cuf_max, sim_max

        # Binary search: smallest k in [k_min1, max_k] that passes Gate 2.
        lo, hi = k_min1, self._max_k
        while lo < hi:
            mid = (lo + hi) // 2
            if mid not in sim_cache:
                sim_cache[mid] = _run_sim(mid)
            pc_mid, soh_mid, sim_mid, tr_mid = sim_cache[mid]
            if _future_ok(mid, soh_mid, sim_mid, tr_mid):
                hi = mid
            else:
                lo = mid + 1

        best_pc, best_soh, best_sim_final, _ = sim_cache[lo]
        return lo, best_pc, best_sim_final
