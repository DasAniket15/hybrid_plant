"""
augmentation_engine.py
──────────────────────
Phase-2 BESS + solar augmentation lifecycle optimizer (v4 redesign).

Takes a PreAnalysisResult (containing the fixed contractual CUF floor,
N_max, shortfall windows, and search bounds) plus the Phase-1 sim params
and searches for the most economical mix of:

  - upfront solar oversizing  (s0, extra DC MWp)
  - upfront BESS oversizing   (x0, extra containers)
  - up to N_max BESS augmentation events (year_i, k_i)

that keeps annual plant CUF ≥ the fixed Year-1 floor across all project
years.

Multi-Cohort BESS Model
───────────────────────
  Each cohort ages independently using the SOH curve indexed from its
  deployment year.

  total_mwh(t) = (base + x0) × size × SOH(t)
               + Σ_i  k_i × size × SOH(t − y_i + 1)   [for y_i ≤ t]

  Effective SOH passed to PlantEngine:
    effective_soh(t) = total_mwh(t) / (total_containers(t) × size)

CUF (delivery-based)
─────────────────────
  CUF_t = delivered_pre_mwh[t] / (ppa_capacity_mw × hours_per_year) × 100

Objective (pure NPV, no penalties)
──────────────────────────────────
  Score = Savings_NPV_gain
        − PV(BESS augmentation CAPEX)
        − PV(solar oversizing CAPEX)   (charged at Year 1)
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_same_year(events: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge augmentation events on the same year by summing their containers."""
    merged: dict[int, int] = {}
    for year, k in events:
        merged[year] = merged.get(year, 0) + k
    return sorted(merged.items())


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationEngine:
    """
    Phase-2 BESS + solar augmentation lifecycle optimizer.

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
        self._wacc          = config.finance["financing"].get("wacc")
        # WACC may be stored elsewhere in the config; fall back to a computed
        # value attached to sim_params if not in finance config.
        if self._wacc is None:
            self._wacc = sim_params.get("wacc")
        if self._wacc is None:
            # Last-resort: compute from financing structure (debt/equity weighted).
            fin = config.finance["financing"]
            d   = fin["debt_percent"] / 100.0
            e   = fin["equity_percent"] / 100.0
            kd  = fin["debt"]["interest_rate_percent"] / 100.0
            ke  = fin["equity"]["return_on_equity_percent"] / 100.0
            tax = fin["corporate_tax_rate_percent"] / 100.0
            self._wacc = d * kd * (1 - tax) + e * ke

        self._discom_tariff = sim_params.get("discom_tariff")
        if self._discom_tariff is None:
            # fallback: read from config if present
            self._discom_tariff = (
                config.finance.get("savings", {}).get("discom_tariff")
                or config.project.get("savings", {}).get("discom_tariff")
                or 0.0
            )

        # ── Solar oversizing economics ────────────────────────────────────────
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
        self._max_soh_year = max(self._soh)

        # ── PlantEngine (single instance, reused across all simulations) ──────
        self._plant = PlantEngine(config, data)

        # ── Baseline (no-augmentation, no-oversizing) projection ──────────────
        self._baseline_proj = self._simulate_schedule(s0=0.0, x0=0, events=[])

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
        """SOH for a cohort that is `cohort_age` years old (age 1 = first year of operation)."""
        return operating_value(self._soh, cohort_age)

    def _cohort_capacity(
        self,
        year:            int,
        base_containers: int,
        x0:              int,
        events:          list[tuple[int, int]],
    ) -> tuple[int, float]:
        """
        Compute total active containers and effective SOH factor for a project year.

        Returns
        -------
        (total_containers, effective_soh)
            effective_soh is in [0, 1]; represents weighted average SOH
            across all deployed cohorts.
        """
        size = self._container_size

        # Cohort 0: base containers + upfront oversizing, deployed at Year 1
        initial = base_containers + x0
        total_mwh = initial * size * self._cohort_soh(year)
        total_containers = initial

        # Future augmentation cohorts
        for ev_year, ev_k in events:
            if ev_k > 0 and year >= ev_year:
                age = year - ev_year + 1
                total_mwh += ev_k * size * self._cohort_soh(age)
                total_containers += ev_k

        if total_containers == 0:
            return 0, 1.0

        effective_soh = total_mwh / (total_containers * size)
        return total_containers, effective_soh

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _simulate_schedule(
        self,
        s0:     float,
        x0:     int,
        events: list[tuple[int, int]],
    ) -> dict[str, np.ndarray]:
        """
        Simulate all project years with multi-cohort BESS capacity and optional
        solar oversizing.

        Parameters
        ----------
        s0     : extra solar DC MWp deployed at Year 1.
        x0     : extra BESS containers added at Year 1.
        events : list of (year, k) augmentation events.

        Returns
        -------
        dict of per-year arrays: solar_direct_mwh, wind_direct_mwh, battery_mwh,
        delivered_pre_mwh, delivered_meter_mwh, cuf_series.
        """
        sp          = self._sim_params
        life        = self._project_life
        ppa_cap     = sp["ppa_capacity_mw"]
        base_cont   = sp["bess_containers"]
        loss_factor = sp["loss_factor"]

        # Convert s0 (extra DC MWp) to extra AC MW via the DC/AC ratio.
        s0_extra_ac_mw = s0 / self._dc_ac_ratio if self._dc_ac_ratio else 0.0

        solar_arr   = np.zeros(life)
        wind_arr    = np.zeros(life)
        battery_arr = np.zeros(life)
        pre_arr     = np.zeros(life)
        meter_arr   = np.zeros(life)
        cuf_arr     = np.zeros(life)

        for i, year in enumerate(range(1, life + 1)):
            solar_eff = operating_value(self._solar_eff, year)
            wind_eff  = operating_value(self._wind_eff,  year)
            total_cont, eff_soh = self._cohort_capacity(year, base_cont, x0, events)

            solar_ac_mw = (sp["solar_capacity_mw"] + s0_extra_ac_mw) * solar_eff
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
    # Objective components
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_yearly_aug_costs(
        self,
        x0:     int,
        events: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Return per-year BESS augmentation CAPEX as an Rs array, shape (project_life,).

        x0 CAPEX is charged at Year 1 (index 0); each event's CAPEX at its event year.
        """
        life  = self._project_life
        size  = self._container_size
        cpm   = self._aug_cost_per_mwh
        costs = np.zeros(life)

        if x0 > 0:
            costs[0] += x0 * size * cpm

        for ev_year, ev_k in events:
            if ev_k > 0:
                idx = ev_year - 1
                if 0 <= idx < life:
                    costs[idx] += ev_k * size * cpm

        return costs

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
        bounds  = self._pre.search_bounds
        N_max   = self._pre.N_max

        # ── Decision variables ────────────────────────────────────────────────
        s0 = (
            trial.suggest_float("s0", 0.0, bounds.s0_max)
            if bounds.s0_max > 0
            else 0.0
        )
        x0 = trial.suggest_int("x0", 0, bounds.x0_max)

        raw_events: list[tuple[int, int]] = []
        for i in range(N_max):
            y = trial.suggest_int(f"year_{i}", bounds.year_min, bounds.year_max)
            k = trial.suggest_int(f"k_{i}",    0,               bounds.k_max)
            raw_events.append((y, k))

        active = [(y, k) for y, k in raw_events if k > 0]
        active.sort(key=lambda e: e[0])
        events = _merge_same_year(active)

        # ── Simulate & score ──────────────────────────────────────────────────
        try:
            proj = self._simulate_schedule(s0, x0, events)

            # Hard constraint: fixed CUF floor must hold every year
            if np.any(proj["cuf_series"] < self._pre.fixed_cuf_floor):
                return PENALTY

            # Savings NPV gain: extra meter delivery × DISCOM tariff (Rs/kWh)
            delta_meter   = (
                proj["delivered_meter_mwh"]
                - self._baseline_proj["delivered_meter_mwh"]
            )
            delta_savings = delta_meter * 1000.0 * self._discom_tariff
            savings_npv_gain = self._pv(delta_savings)

            # PV of BESS augmentation costs
            aug_costs       = self._compute_yearly_aug_costs(x0, events)
            pv_bess_aug_cost = self._pv(aug_costs)

            # PV of solar oversizing CAPEX (charged at Year 1, t=1)
            pv_solar_oversize_cost = (
                (s0 * self._solar_capex_per_mwp) / (1 + self._wacc)
                if s0 > 0 else 0.0
            )

            score = (
                savings_npv_gain
                - pv_bess_aug_cost
                - pv_solar_oversize_cost
            )

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
        Run the augmentation optimisation and return a structured result.

        Parameters
        ----------
        n_trials : override augmentation.yaml n_trials if provided.
        """
        cfg     = self._aug_cfg
        solver  = cfg.get("solver", {})
        trials  = n_trials or int(solver.get("n_trials", 500))
        seed    = int(solver.get("random_seed", 42))

        self._n_feasible = 0
        N_max  = self._pre.N_max
        bounds = self._pre.search_bounds

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

        # ── Reconstruct best schedule ─────────────────────────────────────────
        s0_opt = float(best["s0"]) if "s0" in best else 0.0
        x0_opt = int(best["x0"])

        raw_events_opt: list[tuple[int, int]] = []
        for i in range(N_max):
            y = int(best[f"year_{i}"])
            k = int(best[f"k_{i}"])
            raw_events_opt.append((y, k))

        active_opt = [(y, k) for y, k in raw_events_opt if k > 0]
        active_opt.sort(key=lambda e: e[0])
        events_opt = _merge_same_year(active_opt)

        # ── Full re-simulation of optimal schedule ────────────────────────────
        proj_opt  = self._simulate_schedule(s0_opt, x0_opt, events_opt)
        aug_costs = self._compute_yearly_aug_costs(x0_opt, events_opt)

        delta_meter   = (
            proj_opt["delivered_meter_mwh"]
            - self._baseline_proj["delivered_meter_mwh"]
        )
        delta_savings = delta_meter * 1000.0 * self._discom_tariff

        pv_bess_cost     = self._pv(aug_costs)
        pv_solar_cost    = (
            (s0_opt * self._solar_capex_per_mwp) / (1 + self._wacc)
            if s0_opt > 0 else 0.0
        )
        savings_npv_gain = self._pv(delta_savings)
        final_score      = savings_npv_gain - pv_bess_cost - pv_solar_cost

        active_events = [
            {"year": y, "containers": k}
            for y, k in events_opt
        ]

        return AugmentationResult(
            initial_extra_containers = x0_opt,
            s0_extra_mwp             = s0_opt,
            augmentation_events      = active_events,
            cuf_floor_fixed_pct      = self._pre.fixed_cuf_floor,
            cuf_series               = proj_opt["cuf_series"],
            baseline_cuf_series      = self._pre.baseline_cuf_series,
            n_max                    = N_max,
            shortfall_windows        = self._pre.shortfall_windows,
            yearly_aug_costs         = aug_costs,
            yearly_delta_savings     = delta_savings,
            total_pv_aug_cost        = pv_bess_cost,
            pv_solar_oversize_cost   = pv_solar_cost,
            savings_npv_gain         = savings_npv_gain,
            final_score              = final_score,
            n_trials                 = len(study.trials),
            n_feasible               = self._n_feasible,
        )
