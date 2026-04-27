"""
augmentation_engine.py
──────────────────────
Phase-2 BESS augmentation lifecycle optimizer.

Takes the Phase-1 solver's optimal plant configuration as fixed, then searches
for the most economical BESS augmentation schedule that keeps annual plant CUF
≥ the Year-1 contractual floor for all 25 project years.

Decision Variables
──────────────────
  x0          : int  — extra BESS containers added at Year 1 (upfront oversizing)
  (y1, k1)    : (int, int) — augmentation event 1: year y1, add k1 containers
  (y2, k2)    : (int, int) — augmentation event 2: year y2 ≥ y1, add k2 containers
  (y3, k3)    : (int, int) — augmentation event 3: year y3 ≥ y2, add k3 containers

  If k_i = 0, event i is unused.  Years are sorted ascending.

Multi-Cohort BESS Model
───────────────────────
  Each cohort ages independently using the SOH curve indexed from its deployment year.

  total_mwh(t) = (base + x0) × size × SOH(t)
               + Σ_i  k_i × size × SOH(t − y_i + 1)   [for y_i ≤ t]

  Effective SOH passed to PlantEngine:
    effective_soh(t) = total_mwh(t) / (total_containers(t) × size)

CUF (delivery-based)
─────────────────────
  CUF_t = delivered_pre_mwh[t] / (ppa_capacity_mw × hours_per_year) × 100

  The floor is derived from Phase-1 Year-1 delivered energy — same formula,
  fully consistent across all years.

Objective
─────────
  Score = Savings_NPV_gain
        − PV(augmentation CAPEX + O&M)
        − Penalty_events
        − Penalty_late
        − Penalty_oversize
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from hybrid_plant._paths import find_project_root
from hybrid_plant.augmentation.augmentation_result import AugmentationResult
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.energy.plant_engine import PlantEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AugmentationEngine:
    """
    Phase-2 BESS augmentation lifecycle optimizer.

    Parameters
    ----------
    config      : FullConfig
    data        : dict  — time-series arrays from data_loader
    base_result : SolverResult  — output of Phase-1 SolverEngine.run()
    """

    def __init__(
        self,
        config:      FullConfig,
        data:        dict[str, Any],
        base_result: Any,           # SolverResult (avoid circular import)
    ) -> None:
        self._config = config
        self._data   = data

        # ── Phase-1 baseline params ───────────────────────────────────────────
        fi = base_result.full_result["finance"]
        self._sim_params   = base_result.full_result["year1"]["sim_params"]
        self._project_life = config.project["project"]["project_life_years"]
        self._hours_per_year = config.project["simulation"].get("hours_per_year", 8760)

        self._wacc          = fi["wacc"]
        self._discom_tariff = fi["savings_breakdown"]["discom_tariff"]   # Rs/kWh

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

        # ── CUF floor: delivery-based CUF from Phase-1 Year-1 result ─────────
        ep = fi["energy_projection"]
        ppa_cap = self._sim_params["ppa_capacity_mw"]
        self._cuf_floor = (
            ep["delivered_pre_mwh"][0] / (ppa_cap * self._hours_per_year) * 100
        )

        # ── Baseline (no-augmentation) projection — computed once ─────────────
        self._baseline_proj = self._simulate_lifecycle(x0=0, events=[])

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
        return self._soh.get(cohort_age, self._soh[self._max_soh_year])

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

    def _simulate_lifecycle(
        self,
        x0:     int,
        events: list[tuple[int, int]],
    ) -> dict[str, np.ndarray]:
        """
        Simulate all project years with multi-cohort BESS capacity.

        Returns per-year arrays: solar_direct_mwh, wind_direct_mwh, battery_mwh,
        delivered_pre_mwh, delivered_meter_mwh, cuf_series.
        """
        sp          = self._sim_params
        life        = self._project_life
        ppa_cap     = sp["ppa_capacity_mw"]
        base_cont   = sp["bess_containers"]
        loss_factor = sp["loss_factor"]

        solar_arr   = np.zeros(life)
        wind_arr    = np.zeros(life)
        battery_arr = np.zeros(life)
        pre_arr     = np.zeros(life)
        meter_arr   = np.zeros(life)
        cuf_arr     = np.zeros(life)

        for i, year in enumerate(range(1, life + 1)):
            solar_eff = self._solar_eff.get(year, 1.0)
            wind_eff  = self._wind_eff.get(year, 1.0)
            total_cont, eff_soh = self._cohort_capacity(year, base_cont, x0, events)

            yr = self._plant.simulate(
                solar_capacity_mw  = sp["solar_capacity_mw"] * solar_eff,
                wind_capacity_mw   = sp["wind_capacity_mw"]  * wind_eff,
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
        Return per-year augmentation CAPEX as an Rs array, shape (project_life,).

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

    def _compute_score(
        self,
        proj:     dict[str, np.ndarray],
        x0:       int,
        events:   list[tuple[int, int]],
        aug_costs: np.ndarray,
    ) -> float:
        """Compute the full objective score for a candidate augmentation schedule."""
        cfg  = self._aug_cfg
        wacc = self._wacc

        # 1. Savings NPV gain: additional meter delivery × DISCOM tariff (Rs/kWh)
        delta_meter = proj["delivered_meter_mwh"] - self._baseline_proj["delivered_meter_mwh"]
        # MWh → kWh × Rs/kWh = Rs
        delta_savings = delta_meter * 1000.0 * self._discom_tariff
        savings_npv_gain = self._pv(delta_savings)

        # 2. PV of augmentation costs
        pv_aug_cost = self._pv(aug_costs)

        # 3. Penalty: number of active augmentation events
        n_events = sum(1 for _, k in events if k > 0)
        if x0 > 0:
            n_events += 1   # oversizing also counts as an "event" for sparsity
        lambda1      = float(cfg.get("penalties", {}).get("lambda1", 5_000_000))
        penalty_events = lambda1 * n_events

        # 4. Penalty: late-life augmentation
        late_start = int(cfg.get("late_penalty_start_year", 21))
        lambda2    = float(cfg.get("penalties", {}).get("lambda2", 2_000_000))
        penalty_late = 0.0
        for ev_year, ev_k in events:
            if ev_k > 0 and ev_year >= late_start:
                penalty_late += lambda2 * (ev_year - late_start + 1) * ev_k

        # 5. Penalty: excessive upfront oversizing
        oversize_limit = int(cfg.get("reasonable_oversize_limit", 5))
        lambda3         = float(cfg.get("penalties", {}).get("lambda3", 1_000_000))
        penalty_oversize = lambda3 * max(0, x0 - oversize_limit)

        score = (
            savings_npv_gain
            - pv_aug_cost
            - penalty_events
            - penalty_late
            - penalty_oversize
        )
        return score

    # ─────────────────────────────────────────────────────────────────────────
    # Optuna objective
    # ─────────────────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        PENALTY = -1e15
        cfg     = self._aug_cfg

        max_x0  = int(cfg.get("max_extra_containers",       10))
        max_k   = int(cfg.get("max_augmentation_containers", 20))

        x0 = trial.suggest_int("x0", 0, max_x0)

        # Sample three event years independently from [2, 25], then sort to enforce ordering.
        # Sorting after sampling is standard for ordered-variable problems in Optuna —
        # the TPE sampler still learns effectively because the sorted mapping is deterministic.
        raw_y1 = trial.suggest_int("y1_raw", 2, 25)
        raw_y2 = trial.suggest_int("y2_raw", 2, 25)
        raw_y3 = trial.suggest_int("y3_raw", 2, 25)
        y1, y2, y3 = sorted([raw_y1, raw_y2, raw_y3])

        k1 = trial.suggest_int("k1", 0, max_k)
        k2 = trial.suggest_int("k2", 0, max_k)
        k3 = trial.suggest_int("k3", 0, max_k)

        events = [(y1, k1), (y2, k2), (y3, k3)]

        try:
            proj = self._simulate_lifecycle(x0, events)

            # Hard constraint: CUF floor must hold every year
            if np.any(proj["cuf_series"] < self._cuf_floor):
                return PENALTY

            aug_costs = self._compute_yearly_aug_costs(x0, events)
            score     = self._compute_score(proj, x0, events, aug_costs)

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

        best  = study.best_trial.params
        max_k = int(cfg.get("max_augmentation_containers", 20))

        x0_opt = best["x0"]
        raw_years = sorted([best["y1_raw"], best["y2_raw"], best["y3_raw"]])
        y1_opt, y2_opt, y3_opt = raw_years
        k1_opt = best["k1"]
        k2_opt = best["k2"]
        k3_opt = best["k3"]

        events_opt = [(y1_opt, k1_opt), (y2_opt, k2_opt), (y3_opt, k3_opt)]

        # Full re-simulation of optimal schedule
        proj_opt  = self._simulate_lifecycle(x0_opt, events_opt)
        aug_costs = self._compute_yearly_aug_costs(x0_opt, events_opt)

        delta_meter   = proj_opt["delivered_meter_mwh"] - self._baseline_proj["delivered_meter_mwh"]
        delta_savings = delta_meter * 1000.0 * self._discom_tariff

        pv_aug_cost      = self._pv(aug_costs)
        savings_npv_gain = self._pv(delta_savings)
        final_score      = self._compute_score(proj_opt, x0_opt, events_opt, aug_costs)

        active_events = [
            {"year": y, "containers": k}
            for y, k in events_opt
            if k > 0
        ]

        return AugmentationResult(
            cuf_floor_pct             = self._cuf_floor,
            initial_extra_containers  = x0_opt,
            augmentation_events       = active_events,
            cuf_series                = proj_opt["cuf_series"],
            baseline_cuf_series       = self._baseline_proj["cuf_series"],
            yearly_aug_costs          = aug_costs,
            yearly_delta_savings      = delta_savings,
            total_pv_aug_cost         = pv_aug_cost,
            savings_npv_gain          = savings_npv_gain,
            final_score               = final_score,
            n_trials                  = len(study.trials),
            n_feasible                = self._n_feasible,
        )
