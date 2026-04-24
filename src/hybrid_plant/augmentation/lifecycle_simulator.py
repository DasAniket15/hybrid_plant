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
     runs ``_find_best_horizon_k`` to select the minimum k containers
     that keeps CUF above the hard floor for the longest achievable
     coverage horizon, subject to an economic diminishing-returns guard.
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
``_find_best_horizon_k`` determines at each breach year N:

  Phase 1 — For each candidate k in [min_k, max_k], run one full plant sim
             at year N to obtain post-event anchor busbars, then use cheap
             fast-mode scalar projection for years N+1..N+H_cap to find
             H_max(k): the longest horizon over which this k keeps CUF
             above ``hard_threshold + safety_margin_pp``.

  Phase 2 — Pareto reduction: for each distinct H value seen, keep only
             the minimum k that achieves it.

  Phase 3 — Economic guard: compute discounted cost-per-coverage-year for
             each candidate H.  Reject any H whose ratio to the baseline
             (H=1) exceeds ``coverage_cost_ratio_cap`` — prevents absurdly
             expensive long-horizon purchases.

  Phase 4 — Pick H* = maximum remaining viable horizon; k* = minimum k
             achieving H*.

Years inside [event_year .. event_year + H* - 1] are skipped by the
trigger check (CUF compliance is guaranteed by the horizon search).  This
structurally prevents micro-augmentation top-ups within the verified window.

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
from hybrid_plant.constants import LAKH_TO_RS
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

        # Horizon-based sizing parameters
        self._safety_margin_pp   = float(aug_cfg.get("safety_margin_pp",       0.05))
        self._horizon_cap        = int(  aug_cfg.get("horizon_cap_years",       self._project_life))
        self._coverage_ratio_cap = float(aug_cfg.get("coverage_cost_ratio_cap", 3.0))

        # BESS O&M rate (Rs per MWh) — no escalation, from finance.yaml
        bess_om_lakh_per_mwh   = config.finance["opex"]["bess"]["rate_lakh_per_mwh"]
        self._bess_om_rate_rs  = float(bess_om_lakh_per_mwh) * LAKH_TO_RS

        # WACC — used for discounted cost-per-coverage-year comparison.
        # Computed from finance.yaml financing block; falls back to 10.5 % if
        # the config structure differs.
        try:
            fin = config.finance["financing"]
            d   = float(fin.get("debt_percent",               70))   / 100.0
            rd  = float(fin["debt"]["interest_rate_percent"])         / 100.0
            tc  = float(fin.get("corporate_tax_rate_percent", 25.17)) / 100.0
            re  = float(fin["equity"]["return_on_equity_percent"])    / 100.0
            self._wacc = d * rd * (1.0 - tc) + (1.0 - d) * re
        except (KeyError, TypeError):
            self._wacc = 0.105   # reasonable fallback (~10.5 %)

    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        params:                dict[str, Any],
        initial_containers:    int,
        trigger_threshold_cuf: float,
        fast_mode:             bool = False,
        max_events_override:   int | None = None,
        cuf_buffer_pp:         float = 0.0,
    ) -> LifecycleResult:
        """
        Run the full 25-year augmentation lifecycle for one scenario.

        Parameters
        ----------
        params                 : solar_mw, wind_mw, ppa_mw, c-rates, dispatch settings
        initial_containers     : BESS container count at project start
        trigger_threshold_cuf  : CUF floor from Pass-1 baseline (fixed across all trials).
                                 Augmentation fires when CUF drops below this.
                                 The horizon search restores to this + safety_margin_pp.
        fast_mode              : True → approximate; False → full re-sim each year
        max_events_override    : When provided, overrides the config max_augmentation_events.
                                 Pass 0 to suppress all events (CUF constraint feasibility
                                 check), or a large int for effectively unlimited events.
        cuf_buffer_pp          : Deprecated.  Previously used as a solver-tuned restoration
                                 buffer.  Replaced by the horizon-based sizing logic and
                                 ``safety_margin_pp`` in bess.yaml.  A non-zero value emits
                                 a DeprecationWarning and is silently ignored.

        Returns
        -------
        LifecycleResult
        """
        if cuf_buffer_pp != 0.0:
            warnings.warn(
                "cuf_buffer_pp is deprecated and has no effect.  "
                "Horizon-based sizing now handles multi-year coverage headroom "
                "automatically.  Set safety_margin_pp in bess.yaml instead.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        # skip_until_year: after a horizon event at year N with H years of
        # coverage, years N..N+H-1 are structurally above threshold (verified
        # by _find_best_horizon_k).  The trigger check is suppressed for
        # those years; energy arrays are still populated normally.
        skip_until_year: int = 1

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
            # Skip trigger inside a verified horizon window.
            if year <= skip_until_year:
                continue

            if cuf_t < (trigger_threshold_cuf - eps) and events_used < max_events:
                best_k, horizon_H, post_event_cuf, post_event_sim = \
                    self._find_best_horizon_k(
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
                    "year":            year,
                    "trigger_cuf":     cuf_t,
                    "hard_threshold":  trigger_threshold_cuf,
                    "post_event_cuf":  post_event_cuf,
                    "reached_hard":    post_event_cuf >= trigger_threshold_cuf - 1e-9,
                    "k_containers":    best_k,
                    "lump_cost_rs":    lump_cost,
                    "horizon_years":   horizon_H,
                    "skip_until_year": year + horizon_H,
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

                # Mark the verified coverage window — no triggers until horizon expires.
                # Coverage spans: event_year, event_year+1, ..., event_year+horizon_H-1
                skip_until_year = year + horizon_H - 1

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

    def _find_best_horizon_k(
        self,
        params:             dict[str, Any],
        event_year:         int,
        registry:           CohortRegistry,
        solar_eff_N:        float,
        wind_eff_N:         float,
        hard_threshold_cuf: float,
        ppa_mw:             float,
    ) -> tuple[int, int, float, dict | None]:
        """
        Find the minimum k containers that covers the longest achievable CUF
        horizon starting at ``event_year``, subject to an economic guard.

        Algorithm (four phases)
        ───────────────────────
        Phase 1 — k-scan: for each k in [min_k, max_k]:
            • Run one full plant sim at event_year with k new containers (anchor).
            • Use fast-mode scalar projection for subsequent years to find
              H_max(k): the number of consecutive years (starting at event_year)
              where CUF >= hard_threshold + safety_margin_pp.
            • Exit early when H_max reaches H_cap or plateaus for 3 consecutive k.

        Phase 2 — Pareto reduction:
            • For each distinct horizon H seen, keep only the minimum k.

        Phase 3 — Economic guard:
            • Compute discounted cost-per-coverage-year for each H.
            • Reject any H whose ratio to the H=1 baseline exceeds
              ``coverage_cost_ratio_cap``.

        Phase 4 — Select:
            • H* = longest viable horizon; k* = minimum k achieving H*.

        Returns
        -------
        (best_k, horizon_H, post_event_cuf, post_event_sim)
        """
        container_size  = self._container_size
        project_life    = self._project_life
        safety_margin   = self._safety_margin_pp
        target_cuf      = hard_threshold_cuf + safety_margin
        remaining_life  = project_life - event_year + 1
        H_cap           = min(self._horizon_cap, remaining_life)
        wacc            = self._wacc

        # ── Inner helpers ─────────────────────────────────────────────────────

        def _build_trial_registry(k: int) -> CohortRegistry:
            tr = CohortRegistry(registry.cohorts[0].containers)
            for cohort in registry.cohorts[1:]:
                tr.add(cohort.install_year, cohort.containers)
            tr.add(event_year, k)
            return tr

        def _anchor_sim(k: int) -> tuple[float, float, dict, CohortRegistry]:
            """Full plant sim at event_year for candidate k."""
            tr         = _build_trial_registry(k)
            n_t, soh_t = tr.to_plant_params(event_year, container_size, self._soh_curve)
            sim = self._plant.simulate(
                solar_capacity_mw  = params["solar_capacity_mw"] * solar_eff_N,
                wind_capacity_mw   = params["wind_capacity_mw"]  * wind_eff_N,
                bess_containers    = n_t,
                charge_c_rate      = params["charge_c_rate"],
                discharge_c_rate   = params["discharge_c_rate"],
                ppa_capacity_mw    = params["ppa_capacity_mw"],
                dispatch_priority  = params["dispatch_priority"],
                bess_charge_source = params["bess_charge_source"],
                loss_factor        = self._loss_factor,
                bess_soh_factor    = soh_t,
            )
            busbar   = (float(np.sum(sim["solar_direct_pre"]))
                        + float(np.sum(sim["wind_direct_pre"]))
                        + float(np.sum(sim["discharge_pre"])))
            post_cuf = compute_plant_cuf(busbar, ppa_mw)
            return post_cuf, soh_t, sim, tr

        def _forward_horizon(
            tr:           CohortRegistry,
            anchor_solar: float,
            anchor_wind:  float,
            anchor_batt:  float,
            soh_N:        float,
        ) -> int:
            """
            Fast-mode scalar projection forward from event_year anchor.
            Returns the number of consecutive years (including event_year itself)
            for which CUF >= target_cuf.
            """
            H = 1   # event_year already confirmed above target_cuf
            while H < H_cap:
                fy = event_year + H
                _, soh_fy    = tr.to_plant_params(fy, container_size, self._soh_curve)
                solar_eff_fy = operating_value(self._solar_eff, fy)
                wind_eff_fy  = operating_value(self._wind_eff,  fy)
                proj = (
                    (anchor_solar * solar_eff_fy / solar_eff_N if solar_eff_N > 0 else 0.0)
                    + (anchor_wind  * wind_eff_fy  / wind_eff_N  if wind_eff_N  > 0 else 0.0)
                    + (anchor_batt  * soh_fy       / soh_N       if soh_N       > 0 else 0.0)
                )
                if compute_plant_cuf(proj, ppa_mw) < target_cuf - 1e-9:
                    break
                H += 1
            return H

        def _total_discounted_cost(k: int) -> float:
            """Discounted lump PV + O&M PV for k containers installed at event_year."""
            mwh     = k * container_size
            lump_pv = mwh * self._cost_per_mwh / (1.0 + wacc) ** (event_year - 1)
            om_pv   = sum(
                mwh * self._bess_om_rate_rs / (1.0 + wacc) ** (y - 1)
                for y in range(event_year, project_life + 1)
            )
            return lump_pv + om_pv

        # ── Phase 1: build (k → H_max) map ───────────────────────────────────
        k_data:       dict[int, tuple[int, float, dict]] = {}
        fallback_k:   int        = self._min_k
        fallback_cuf: float      = -1.0
        fallback_sim: dict | None = None

        best_H_seen   = 0
        plateau_count = 0

        for k in range(self._min_k, self._max_k + 1):
            post_cuf_N, soh_N, sim_N, tr = _anchor_sim(k)

            # Track global best for fallback (highest post-event CUF found)
            if post_cuf_N > fallback_cuf:
                fallback_k, fallback_cuf, fallback_sim = k, post_cuf_N, sim_N

            if post_cuf_N < target_cuf - 1e-9:
                # This k doesn't cover year N — keep scanning; give up after 5 misses
                if best_H_seen == 0 and k >= self._min_k + 5:
                    break
                continue

            # Compute forward horizon from this k's anchor busbars
            anchor_solar = float(np.sum(sim_N["solar_direct_pre"]))
            anchor_wind  = float(np.sum(sim_N["wind_direct_pre"]))
            anchor_batt  = float(np.sum(sim_N["discharge_pre"]))
            H = _forward_horizon(tr, anchor_solar, anchor_wind, anchor_batt, soh_N)

            k_data[k] = (H, post_cuf_N, sim_N)

            # Early exit when full-project coverage is achieved
            if H >= H_cap:
                break

            # Monotonicity exit: H has plateaued for 3 consecutive k steps
            if H > best_H_seen:
                best_H_seen   = H
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count >= 3:
                    break

        # ── Fallback: no k reached target CUF even at event year ─────────────
        if not k_data:
            warnings.warn(
                f"Augmentation Year {event_year}: k-search exhausted max_k={self._max_k} "
                f"without reaching target CUF {target_cuf:.4f}% "
                f"(hard_threshold={hard_threshold_cuf:.4f}%, "
                f"safety_margin={safety_margin:.2f}pp). "
                f"Best post-event CUF was {fallback_cuf:.4f}%. "
                f"Returning k={fallback_k} (best found). "
                f"Consider raising max_augmentation_containers_per_event or "
                f"reducing safety_margin_pp in bess.yaml.",
                stacklevel=4,
            )
            return fallback_k, 1, fallback_cuf, fallback_sim

        # ── Phase 2: Pareto reduction ─────────────────────────────────────────
        # For each distinct H value, keep only the minimum k that achieves it.
        pareto: dict[int, tuple[int, float, dict]] = {}
        for k in sorted(k_data):
            H, post_cuf, sim = k_data[k]
            if H not in pareto:
                pareto[H] = (k, post_cuf, sim)

        # ── Phase 3: Economic diminishing-returns guard ───────────────────────
        baseline_H           = min(pareto)
        baseline_k, _, _     = pareto[baseline_H]
        baseline_cost        = _total_discounted_cost(baseline_k)
        baseline_cpy         = baseline_cost / baseline_H if baseline_H > 0 else 1.0

        viable: dict[int, tuple[int, float, dict]] = {}
        for H, (k, post_cuf, sim) in pareto.items():
            cost = _total_discounted_cost(k)
            cpy  = cost / H if H > 0 else float("inf")
            if cpy <= self._coverage_ratio_cap * baseline_cpy:
                viable[H] = (k, post_cuf, sim)

        if not viable:
            viable = pareto   # safety: economic guard eliminated all — use full pareto

        # ── Phase 4: H* = longest viable horizon, k* = min k for H* ─────────
        H_star                     = max(viable)
        k_star, cuf_star, sim_star = viable[H_star]

        logger.debug(
            "AugmentationEngine Year %d: k*=%d  H*=%d  post_cuf=%.4f%%  "
            "k_candidates=%d  pareto_H=%s",
            event_year, k_star, H_star, cuf_star,
            len(k_data), sorted(pareto),
        )

        return k_star, H_star, cuf_star, sim_star
