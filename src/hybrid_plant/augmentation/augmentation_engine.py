"""
augmentation_engine.py
──────────────────────
Phase-2 joint Solar + BESS augmentation lifecycle optimizer.

Both asset classes are augmented in an event-based, forward-looking manner:
  - No upfront oversizing at Year 1 for either asset.
  - The optimizer chooses *when* to deploy new capacity (year ≥ 2) and *how
    much* to add, evaluating the full project lifetime at every trial.
  - Each new solar cohort and each new BESS cohort ages independently from
    the year it is deployed.

Multi-Cohort BESS Model
───────────────────────
  total_mwh(t) = (base + x0) × size × SOH(t)
               + Σ_i  k_i × size × SOH(t − y_i + 1)   [for y_i ≤ t]

  effective_soh(t) = total_mwh(t) / (total_containers(t) × size)

Multi-Cohort Solar Model
────────────────────────
  total_solar_ac_mw(t) = base_solar_mw × solar_eff[t]
                        + Σ_j  (mwp_j / dc_ac_ratio) × solar_eff[t − s_j + 1]
                                                        [for s_j ≤ t]

  Each cohort degrades from its own deployment year (age 1 = first year of
  operation), so a cohort deployed at Year 5 uses solar_eff[1] in Year 5,
  solar_eff[2] in Year 6, etc.

CUF (delivery-based)
─────────────────────
  CUF_t = delivered_pre_mwh[t] / (ppa_capacity_mw × hours_per_year) × 100

Objective (pure NPV, no penalties)
──────────────────────────────────
  Score = Savings_NPV_gain
        − PV(BESS augmentation CAPEX)   [x0 at Year 1 + future events]
        − PV(Solar augmentation CAPEX)  [future events only]

Decision Variables (Optuna)
───────────────────────────
  x0                      : int, extra BESS containers at Year 1
  bess_year_{i}, bess_k_{i} : int × N_max  — BESS event year and containers
  solar_year_{j}, solar_mwp_{j} : int + float × N_solar_max — solar event year and DC MWp
    (only sampled when solar_augmentation is enabled in config)

Forward-Looking Design
──────────────────────
  Each Optuna trial evaluates the full 25-year CUF trajectory of a proposed
  schedule.  The optimizer therefore naturally discovers forward-looking,
  frequency-minimizing solutions rather than year-by-year greedy fixes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from hybrid_plant._paths import find_project_root
from hybrid_plant.augmentation.augmentation_pre_analysis import PreAnalysisResult
from hybrid_plant.augmentation.augmentation_result import AugmentationResult
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.data_loader import operating_value
from hybrid_plant.energy.plant_engine import PlantEngine
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.savings_model import SavingsModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Minimum solar MWp per event to be treated as "active" (avoids floating noise).
_SOLAR_MWP_THRESHOLD = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_same_year_int(events: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge BESS augmentation events on the same year by summing containers."""
    merged: dict[int, int] = {}
    for year, k in events:
        merged[year] = merged.get(year, 0) + k
    return sorted(merged.items())


def _merge_same_year_float(events: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Merge solar augmentation events on the same year by summing DC MWp."""
    merged: dict[int, float] = {}
    for year, mwp in events:
        merged[year] = merged.get(year, 0.0) + mwp
    return sorted(merged.items())


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationEngine:
    """
    Phase-2 joint Solar + BESS augmentation lifecycle optimizer.

    Parameters
    ----------
    pre        : PreAnalysisResult — output of AugmentationPreAnalysis.run().
    sim_params : dict — Phase-1 baseline simulation parameters
                 (ppa_capacity_mw, solar_capacity_mw, wind_capacity_mw,
                 bess_containers, charge_c_rate, discharge_c_rate,
                 dispatch_priority, bess_charge_source, loss_factor, …).
    config     : FullConfig.
    data       : dict[str, Any] — time-series arrays from data_loader,
                 needed by PlantEngine.
    """

    def __init__(
        self,
        pre:        PreAnalysisResult,
        sim_params: dict[str, Any],
        config:     FullConfig,
        data:       dict[str, Any],
    ) -> None:
        self._pre        = pre
        self._sim_params = sim_params
        self._config     = config
        self._data       = data

        # ── Project-level params ──────────────────────────────────────────────
        self._project_life   = config.project["project"]["project_life_years"]
        self._hours_per_year = config.project["simulation"].get("hours_per_year", 8760)

        # ── Finance params ────────────────────────────────────────────────────
        self._wacc = sim_params.get("wacc")
        if self._wacc is None:
            self._wacc = LCOEModel(config).wacc

        self._discom_tariff = sim_params.get("discom_tariff")
        if self._discom_tariff is None:
            self._discom_tariff = SavingsModel._weighted_discom_tariff(config)

        # ── Solar economics ───────────────────────────────────────────────────
        capex_cfg = config.finance["capex"]
        self._solar_capex_per_mwp = float(capex_cfg["solar"]["cost_per_mwp"])
        # ac_dc_ratio in finance.yaml is DC/AC ratio (DC MWp per AC MW).
        self._dc_ac_ratio = float(capex_cfg["solar"]["ac_dc_ratio"])

        # ── BESS constants ────────────────────────────────────────────────────
        bess_cfg = config.bess["bess"]
        self._container_size   = float(bess_cfg["container"]["size_mwh"])
        self._aug_cost_per_mwh = float(bess_cfg["augmentation"]["cost_per_mwh"])

        # ── Augmentation optimizer config ─────────────────────────────────────
        self._aug_cfg = config.augmentation.get("augmentation_optimizer", {})

        # ── Degradation curves ────────────────────────────────────────────────
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
            root / bess_cfg["degradation"]["file"],
            column="soh",
        )

        # ── PlantEngine (single instance, reused across all simulations) ──────
        self._plant = PlantEngine(config, data)

        # ── Baseline (no augmentation) projection ─────────────────────────────
        self._baseline_proj = self._simulate_schedule(x0=0, bess_events=[], solar_events=[])

        self._n_feasible = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_curve(path: Path, column: str) -> dict[int, float]:
        """Load a degradation CSV into a {year: value} dict."""
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return dict(zip(df["year"].astype(int), df[column.lower()]))

    def _cohort_soh(self, cohort_age: int) -> float:
        """SOH for a BESS cohort that is `cohort_age` years old (age 1 = first year)."""
        return operating_value(self._soh, cohort_age)

    def _bess_cohort_capacity(
        self,
        year:            int,
        base_containers: int,
        x0:              int,
        bess_events:     list[tuple[int, int]],
    ) -> tuple[int, float]:
        """
        Compute total active BESS containers and effective SOH factor for a project year.

        Returns
        -------
        (total_containers, effective_soh)
        """
        size = self._container_size

        # Cohort 0: base containers + upfront x0, deployed at Year 1
        initial    = base_containers + x0
        total_mwh  = initial * size * self._cohort_soh(year)
        total_cont = initial

        for ev_year, ev_k in bess_events:
            if ev_k > 0 and year >= ev_year:
                age = year - ev_year + 1
                total_mwh  += ev_k * size * self._cohort_soh(age)
                total_cont += ev_k

        if total_cont == 0:
            return 0, 1.0

        effective_soh = total_mwh / (total_cont * size)
        return total_cont, effective_soh

    def _solar_cohort_capacity(
        self,
        year:         int,
        base_solar_mw: float,
        solar_events:  list[tuple[int, float]],
    ) -> float:
        """
        Compute total solar AC MW for a project year, summing across all cohorts.

        The base cohort uses solar_eff[year] (age = year).
        Each augmentation cohort uses solar_eff[year − cohort_year + 1] (own age).
        """
        total_ac = base_solar_mw * operating_value(self._solar_eff, year)

        for ev_year, ev_mwp in solar_events:
            if ev_mwp > _SOLAR_MWP_THRESHOLD and year >= ev_year:
                cohort_age = year - ev_year + 1
                total_ac += (ev_mwp / self._dc_ac_ratio) * operating_value(
                    self._solar_eff, cohort_age
                )

        return total_ac

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _simulate_schedule(
        self,
        x0:          int,
        bess_events: list[tuple[int, int]],
        solar_events: list[tuple[int, float]],
    ) -> dict[str, np.ndarray]:
        """
        Simulate all project years with multi-cohort BESS and solar capacity.

        Parameters
        ----------
        x0           : extra BESS containers added at Year 1.
        bess_events  : list of (year, k) BESS augmentation events.
        solar_events : list of (year, mwp) solar augmentation events.

        Returns
        -------
        dict of per-year arrays:
            solar_direct_mwh, wind_direct_mwh, battery_mwh,
            delivered_pre_mwh, delivered_meter_mwh, cuf_series.
        """
        sp          = self._sim_params
        life        = self._project_life
        ppa_cap     = sp["ppa_capacity_mw"]
        base_cont   = sp["bess_containers"]
        base_solar  = sp["solar_capacity_mw"]
        loss_factor = sp["loss_factor"]

        solar_arr   = np.zeros(life)
        wind_arr    = np.zeros(life)
        battery_arr = np.zeros(life)
        pre_arr     = np.zeros(life)
        meter_arr   = np.zeros(life)
        cuf_arr     = np.zeros(life)

        for i, year in enumerate(range(1, life + 1)):
            wind_eff  = operating_value(self._wind_eff, year)
            total_cont, eff_soh = self._bess_cohort_capacity(
                year, base_cont, x0, bess_events
            )
            solar_ac_mw = self._solar_cohort_capacity(year, base_solar, solar_events)
            wind_ac_mw  = sp["wind_capacity_mw"] * wind_eff

            yr = self._plant.simulate(
                solar_capacity_mw  = solar_ac_mw,
                wind_capacity_mw   = wind_ac_mw,
                bess_containers    = total_cont,
                bess_soh_factor    = eff_soh,
                charge_c_rate      = sp["charge_c_rate"],
                discharge_c_rate   = sp["discharge_c_rate"],
                ppa_capacity_mw    = ppa_cap,
                dispatch_priority  = sp["dispatch_priority"],
                bess_charge_source = sp["bess_charge_source"],
                loss_factor        = loss_factor,
            )

            s = float(np.sum(yr["solar_direct_pre"]))
            w = float(np.sum(yr["wind_direct_pre"]))
            b = float(np.sum(yr["discharge_pre"]))
            total_pre = s + w + b

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = total_pre
            meter_arr[i]   = total_pre * loss_factor
            cuf_arr[i]     = total_pre / (ppa_cap * self._hours_per_year) * 100

        return {
            "solar_direct_mwh":    solar_arr,
            "wind_direct_mwh":     wind_arr,
            "battery_mwh":         battery_arr,
            "delivered_pre_mwh":   pre_arr,
            "delivered_meter_mwh": meter_arr,
            "cuf_series":          cuf_arr,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Cost computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_yearly_costs(
        self,
        x0:          int,
        bess_events: list[tuple[int, int]],
        solar_events: list[tuple[int, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return per-year BESS and solar augmentation CAPEX arrays (Rs), shape (project_life,).

        x0 BESS CAPEX is charged at Year 1 (index 0).
        Each event's CAPEX is charged at its event year.
        """
        life        = self._project_life
        size        = self._container_size
        cpm         = self._aug_cost_per_mwh
        bess_costs  = np.zeros(life)
        solar_costs = np.zeros(life)

        # ── BESS ──────────────────────────────────────────────────────────────
        if x0 > 0:
            bess_costs[0] += x0 * size * cpm

        for ev_year, ev_k in bess_events:
            if ev_k > 0:
                idx = ev_year - 1
                if 0 <= idx < life:
                    bess_costs[idx] += ev_k * size * cpm

        # ── Solar ─────────────────────────────────────────────────────────────
        for ev_year, ev_mwp in solar_events:
            if ev_mwp > _SOLAR_MWP_THRESHOLD:
                idx = ev_year - 1
                if 0 <= idx < life:
                    solar_costs[idx] += ev_mwp * self._solar_capex_per_mwp

        return bess_costs, solar_costs

    def _pv(self, annual_series: np.ndarray) -> float:
        """NPV of an annual series (series[0] = Year 1, discounted at t=1)."""
        wacc = self._wacc
        return sum(
            v / (1 + wacc) ** (t + 1)
            for t, v in enumerate(annual_series)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Optuna objective
    # ─────────────────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        PENALTY = -1e15
        bounds   = self._pre.search_bounds
        N_max    = self._pre.N_max
        N_solar  = self._pre.N_solar_max

        # ── BESS decision variables ────────────────────────────────────────────
        x0 = trial.suggest_int("x0", 0, bounds.x0_max)

        raw_bess: list[tuple[int, int]] = []
        for i in range(N_max):
            y = trial.suggest_int(f"bess_year_{i}", bounds.year_min, bounds.year_max)
            k = trial.suggest_int(f"bess_k_{i}",    0,               bounds.k_max)
            raw_bess.append((y, k))

        active_bess = [(y, k) for y, k in raw_bess if k > 0]
        bess_events = _merge_same_year_int(active_bess)

        # ── Solar decision variables ──────────────────────────────────────────
        raw_solar: list[tuple[int, float]] = []
        if bounds.solar_event_max_mwp > 0:
            for j in range(N_solar):
                y   = trial.suggest_int(f"solar_year_{j}", bounds.year_min, bounds.year_max)
                mwp = trial.suggest_float(f"solar_mwp_{j}", 0.0, bounds.solar_event_max_mwp)
                raw_solar.append((y, mwp))

        active_solar = [(y, m) for y, m in raw_solar if m > _SOLAR_MWP_THRESHOLD]
        solar_events = _merge_same_year_float(active_solar)

        # ── Simulate & score ──────────────────────────────────────────────────
        try:
            proj = self._simulate_schedule(x0, bess_events, solar_events)

            # Hard constraint: CUF must meet the fixed floor every year
            if np.any(proj["cuf_series"] < self._pre.fixed_cuf_floor):
                return PENALTY

            # Savings NPV gain: extra meter delivery × DISCOM tariff (Rs/kWh)
            delta_meter      = (
                proj["delivered_meter_mwh"]
                - self._baseline_proj["delivered_meter_mwh"]
            )
            delta_savings    = delta_meter * 1000.0 * self._discom_tariff
            savings_npv_gain = self._pv(delta_savings)

            # PV of BESS and solar augmentation CAPEX
            bess_costs, solar_costs = self._compute_yearly_costs(x0, bess_events, solar_events)
            pv_bess  = self._pv(bess_costs)
            pv_solar = self._pv(solar_costs)

            score = savings_npv_gain - pv_bess - pv_solar
            self._n_feasible += 1
            return score

        except Exception as exc:
            warnings.warn(f"Augmentation trial {trial.number} failed: {exc}")
            return PENALTY

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, n_trials: int | None = None) -> AugmentationResult:
        """
        Run the joint Solar + BESS augmentation optimisation.

        Parameters
        ----------
        n_trials : override augmentation.yaml n_trials if provided.
        """
        cfg    = self._aug_cfg
        solver = cfg.get("solver", {})
        trials = n_trials or int(solver.get("n_trials", 500))
        seed   = int(solver.get("random_seed", 42))

        self._n_feasible = 0
        N_max   = self._pre.N_max
        N_solar = self._pre.N_solar_max
        bounds  = self._pre.search_bounds

        study = optuna.create_study(
            direction = "maximize",
            sampler   = optuna.samplers.TPESampler(seed=seed),
        )
        try:
            study.optimize(self._objective, n_trials=trials, show_progress_bar=True)
        except KeyboardInterrupt:
            feasible = [t for t in study.trials if t.value is not None and t.value > -1e14]
            if not feasible:
                raise RuntimeError(
                    "Augmentation solver interrupted with no feasible solution found."
                ) from None

        if study.best_trial.value is None or study.best_trial.value <= -1e14:
            raise RuntimeError(
                f"No CUF-compliant augmentation schedule found in {trials} trials. "
                "Try increasing augmentation_optimizer.solver.n_trials or "
                "reviewing the CUF floor and search bounds."
            )

        best = study.best_trial.params

        # ── Reconstruct optimal BESS schedule ─────────────────────────────────
        x0_opt = int(best["x0"])

        raw_bess_opt: list[tuple[int, int]] = []
        for i in range(N_max):
            y = int(best[f"bess_year_{i}"])
            k = int(best[f"bess_k_{i}"])
            raw_bess_opt.append((y, k))

        active_bess_opt = [(y, k) for y, k in raw_bess_opt if k > 0]
        bess_events_opt = _merge_same_year_int(active_bess_opt)

        # ── Reconstruct optimal solar schedule ────────────────────────────────
        raw_solar_opt: list[tuple[int, float]] = []
        if bounds.solar_event_max_mwp > 0:
            for j in range(N_solar):
                y   = int(best[f"solar_year_{j}"])
                mwp = float(best[f"solar_mwp_{j}"])
                raw_solar_opt.append((y, mwp))

        active_solar_opt = [(y, m) for y, m in raw_solar_opt if m > _SOLAR_MWP_THRESHOLD]
        solar_events_opt = _merge_same_year_float(active_solar_opt)

        # ── Full re-simulation of optimal schedule ────────────────────────────
        proj_opt = self._simulate_schedule(x0_opt, bess_events_opt, solar_events_opt)
        bess_costs, solar_costs = self._compute_yearly_costs(
            x0_opt, bess_events_opt, solar_events_opt
        )

        delta_meter   = (
            proj_opt["delivered_meter_mwh"]
            - self._baseline_proj["delivered_meter_mwh"]
        )
        delta_savings    = delta_meter * 1000.0 * self._discom_tariff
        savings_npv_gain = self._pv(delta_savings)
        pv_bess_cost     = self._pv(bess_costs)
        pv_solar_cost    = self._pv(solar_costs)
        final_score      = savings_npv_gain - pv_bess_cost - pv_solar_cost

        active_bess_out = [
            {"year": y, "containers": k}
            for y, k in bess_events_opt
            if k > 0
        ]
        active_solar_out = [
            {"year": y, "mwp": round(m, 2)}
            for y, m in solar_events_opt
            if m > _SOLAR_MWP_THRESHOLD
        ]

        return AugmentationResult(
            initial_extra_containers  = x0_opt,
            bess_augmentation_events  = active_bess_out,
            solar_augmentation_events = active_solar_out,
            cuf_floor_fixed_pct       = self._pre.fixed_cuf_floor,
            cuf_series                = proj_opt["cuf_series"],
            baseline_cuf_series       = self._pre.baseline_cuf_series,
            n_max                     = N_max,
            shortfall_windows         = self._pre.shortfall_windows,
            yearly_bess_aug_costs     = bess_costs,
            yearly_solar_aug_costs    = solar_costs,
            yearly_delta_savings      = delta_savings,
            total_pv_bess_aug_cost    = pv_bess_cost,
            total_pv_solar_aug_cost   = pv_solar_cost,
            savings_npv_gain          = savings_npv_gain,
            final_score               = final_score,
            n_trials                  = len(study.trials),
            n_feasible                = self._n_feasible,
        )
