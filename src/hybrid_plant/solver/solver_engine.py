"""
solver_engine.py
────────────────
Optimisation layer — wraps Optuna TPE to search the decision-variable space
defined in ``solver.yaml``.

Objective   : maximise client savings NPV vs. 100 % DISCOM baseline
Algorithm   : Tree-structured Parzen Estimator (TPE)
Current scope variables
    solar_capacity_mw   continuous
    wind_capacity_mw    continuous
    ppa_capacity_mw     continuous
    bess_containers     integer
Future-scope variables are fixed at their ``fixed_value`` from ``solver.yaml``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

from hybrid_plant.config_loader import FullConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """Structured output of a completed optimisation run."""

    best_params:            dict[str, Any]
    best_savings_npv:       float
    best_year1_savings:     float
    best_lcoe:              float
    best_landed_tariff_y1:  float
    all_trials:             pd.DataFrame   # feasible trials, ranked by savings_npv_cr
    full_result:            dict[str, Any] # raw engine outputs for best solution
    n_trials_completed:     int
    n_trials_feasible:      int


# ─────────────────────────────────────────────────────────────────────────────
# Solver engine
# ─────────────────────────────────────────────────────────────────────────────

class SolverEngine:
    """
    Wraps Optuna to search the hybrid plant decision-variable space.

    Parameters
    ----------
    config         : FullConfig
    data           : dict  — from data_loader
    energy_engine  : Year1Engine
    finance_engine : FinanceEngine

    Usage
    -----
        solver = SolverEngine(config, data, energy_engine, finance_engine)
        result = solver.run()
    """

    # Fallback fixed values for future-scope variables
    _FUTURE_FIXED: dict[str, Any] = {
        "dispatch_priority":  "solar_first",
        "bess_charge_source": "solar_only",
    }

    def __init__(
        self,
        config:         FullConfig,
        data:           dict[str, Any],
        energy_engine:  Any,
        finance_engine: Any,
    ) -> None:
        self._config          = config
        self._data            = data
        self._energy_engine   = energy_engine
        self._finance_engine  = finance_engine

        sv = config.solver["solver"]
        self._solver_cfg = sv
        self._dv         = sv["decision_variables"]

        # Override fixed values from YAML where present (future-scope variables only)
        for key, yaml_key in [
            ("dispatch_priority",  "dispatch_priority"),
            ("bess_charge_source", "bess_charge_source"),
        ]:
            fv = self._dv.get(yaml_key, {}).get("fixed_value")
            if fv is not None:
                self._FUTURE_FIXED[key] = fv

        self._n_trials    = int(sv.get("n_trials", 300))
        self._n_jobs      = int(sv.get("n_jobs", 1))
        self._random_seed = int(sv.get("random_seed", 42))
        self._fast_mode   = bool(sv.get("fast_mode", True))

        self._trial_log: list[dict[str, Any]] = []

    # ── Search space ──────────────────────────────────────────────────────────

    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Map an Optuna trial to a complete parameter set."""
        dv = self._dv

        def _istep(key: str) -> int:
            """Return int step, defaulting to 1 — guards against YAML null/zero."""
            v = dv[key].get("step")
            return int(v) if v is not None and int(v) > 0 else 1

        def _is_current(key: str) -> bool:
            """True if the key exists in dv AND scope == 'current'."""
            return key in dv and dv[key].get("scope", "future") == "current"

        solar_mw = trial.suggest_float(
            "solar_capacity_mw",
            dv["solar_capacity_mw"]["min"],
            dv["solar_capacity_mw"]["max"],
        )
        wind_mw = trial.suggest_float(
            "wind_capacity_mw",
            dv["wind_capacity_mw"]["min"],
            dv["wind_capacity_mw"]["max"],
        )
        ppa_mw = trial.suggest_float(
            "ppa_capacity_mw",
            dv["ppa_capacity_mw"]["min"],
            dv["ppa_capacity_mw"]["max"],
        )
        bess_containers = trial.suggest_int(
            "bess_containers",
            dv["bess_containers"]["min"],
            dv["bess_containers"]["max"],
            step=_istep("bess_containers"),
        )

        # C-rates: optimised when scope=current in solver.yaml; fixed otherwise.
        # C-rates are sampled continuously (no step) to avoid a known Optuna
        # TPE numerical issue where small step sizes over narrow ranges cause
        # the truncated-normal computation to loop near float epsilon.
        # The solver result is a real-valued optimum; 0.01 discretisation
        # adds no physical value for a continuous parameter like C-rate.
        if _is_current("bess_charge_c_rate"):
            charge_c = trial.suggest_float(
                "charge_c_rate",
                dv["bess_charge_c_rate"]["min"],
                dv["bess_charge_c_rate"]["max"],
            )
        else:
            fv = dv.get("bess_charge_c_rate", {}).get("fixed_value")
            charge_c = float(fv) if fv is not None else 0.5

        if _is_current("bess_discharge_c_rate"):
            discharge_c = trial.suggest_float(
                "discharge_c_rate",
                dv["bess_discharge_c_rate"]["min"],
                dv["bess_discharge_c_rate"]["max"],
            )
        else:
            fv = dv.get("bess_discharge_c_rate", {}).get("fixed_value")
            discharge_c = float(fv) if fv is not None else 0.5

        return {
            "solar_capacity_mw":  solar_mw,
            "wind_capacity_mw":   wind_mw,
            "ppa_capacity_mw":    ppa_mw,
            "bess_containers":    bess_containers,
            "charge_c_rate":      charge_c,
            "discharge_c_rate":   discharge_c,
            **self._FUTURE_FIXED,
        }

    # ── Single evaluation ─────────────────────────────────────────────────────

    def _evaluate(self, params: dict[str, Any], fast_mode: bool = False) -> dict[str, Any]:
        """Run energy + finance engines for a given parameter set."""
        year1 = self._energy_engine.evaluate(
            solar_capacity_mw  = params["solar_capacity_mw"],
            wind_capacity_mw   = params["wind_capacity_mw"],
            bess_containers    = params["bess_containers"],
            charge_c_rate      = params["charge_c_rate"],
            discharge_c_rate   = params["discharge_c_rate"],
            ppa_capacity_mw    = params["ppa_capacity_mw"],
            dispatch_priority  = params["dispatch_priority"],
            bess_charge_source = params["bess_charge_source"],
        )
        finance = self._finance_engine.evaluate(
            year1_results     = year1,
            solar_capacity_mw = params["solar_capacity_mw"],
            wind_capacity_mw  = params["wind_capacity_mw"],
            ppa_capacity_mw   = params["ppa_capacity_mw"],
            fast_mode         = fast_mode,
        )
        return {"year1": year1, "finance": finance}

    # ── Constraint check ──────────────────────────────────────────────────────

    def _is_feasible(self, finance: dict[str, Any]) -> bool:
        """Return True if savings_npv meets the configured minimum."""
        cfg = self._solver_cfg.get("constraints", {})
        npv_constraint = cfg.get("minimum_savings_npv", {})
        if npv_constraint.get("enabled", True):
            if finance["savings_npv"] < npv_constraint.get("min_value", 0):
                return False
        return True

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        PENALTY = -1e15
        params  = self._suggest(trial)

        try:
            result  = self._evaluate(params, fast_mode=self._fast_mode)
            finance = result["finance"]
            year1   = result["year1"]
            feasible    = self._is_feasible(finance)
            savings_npv = finance["savings_npv"]

            self._trial_log.append({
                "trial_number":         trial.number,
                "feasible":             feasible,
                "solar_capacity_mw":    params["solar_capacity_mw"],
                "wind_capacity_mw":     params["wind_capacity_mw"],
                "ppa_capacity_mw":      params["ppa_capacity_mw"],
                "bess_containers":      params["bess_containers"],
                "charge_c_rate":        params["charge_c_rate"],
                "discharge_c_rate":     params["discharge_c_rate"],
                "bess_mwh":             float(year1["energy_capacity_mwh"]),
                "savings_npv_cr":       round(savings_npv / 1e7, 4),
                "annual_savings_y1_cr": round(finance["annual_savings_year1"] / 1e7, 4),
                "lcoe":                 round(finance["lcoe_inr_per_kwh"], 4),
                "landed_tariff_y1":     round(finance["landed_tariff_series"][0], 4),
                "meter_mwh_y1":         round(float(finance["energy_projection"]["delivered_meter_mwh"][0]), 2),
                "curtailment_mwh":      round(float(np.sum(year1["curtailment_pre"])), 2),
            })

            return savings_npv if feasible else PENALTY

        except Exception as exc:
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            self._trial_log.append({
                "trial_number":      trial.number,
                "feasible":          False,
                "solar_capacity_mw": params["solar_capacity_mw"],
                "wind_capacity_mw":  params["wind_capacity_mw"],
                "ppa_capacity_mw":   params["ppa_capacity_mw"],
                "bess_containers":   params["bess_containers"],
                "error":             str(exc),
            })
            return PENALTY

    # ── Public run interface ──────────────────────────────────────────────────

    def run(
        self,
        n_trials:      int | None = None,
        n_jobs:        int | None = None,
        show_progress: bool       = True,
    ) -> SolverResult:
        """
        Execute the optimisation study.

        Parameters
        ----------
        n_trials      : override solver.yaml n_trials if provided
        n_jobs        : override solver.yaml n_jobs if provided
        show_progress : display Optuna progress bar

        Returns
        -------
        SolverResult
        """
        n_trials = n_trials or self._n_trials
        n_jobs   = n_jobs   or self._n_jobs

        study = optuna.create_study(
            direction = "maximize",
            sampler   = optuna.samplers.TPESampler(seed=self._random_seed),
        )
        try:
            study.optimize(
                self._objective,
                n_trials          = n_trials,
                n_jobs            = n_jobs,
                show_progress_bar = show_progress,
            )
        except KeyboardInterrupt:
            # On Windows, QuickEdit Mode can send SIGINT on a terminal click.
            # Catch it gracefully and return the best solution found so far,
            # provided at least one feasible trial completed.
            completed = len(study.trials)
            feasible  = [t for t in study.trials if t.value is not None and t.value > -1e14]
            if not feasible:
                raise RuntimeError(
                    f"Solver interrupted after {completed} trials with no feasible solution found."
                ) from None
            print(f"\n  [Solver] Interrupted after {completed} trials — "
                  f"returning best of {len(feasible)} feasible trials.")

        # Re-run best to get full result dict
        best_params = {
            **study.best_trial.params,
            **self._FUTURE_FIXED,
        }
        full_result = self._evaluate(best_params)
        finance     = full_result["finance"]
        year1       = full_result["year1"]

        trials_df = pd.DataFrame(self._trial_log)
        if not trials_df.empty and "savings_npv_cr" in trials_df.columns:
            trials_df = (
                trials_df[trials_df["feasible"] == True]
                .sort_values("savings_npv_cr", ascending=False)
                .reset_index(drop=True)
            )

        n_feasible = int(trials_df.shape[0]) if not trials_df.empty else 0

        return SolverResult(
            best_params           = best_params,
            best_savings_npv      = finance["savings_npv"],
            best_year1_savings    = finance["annual_savings_year1"],
            best_lcoe             = finance["lcoe_inr_per_kwh"],
            best_landed_tariff_y1 = finance["landed_tariff_series"][0],
            all_trials            = trials_df,
            full_result           = full_result,
            n_trials_completed    = n_trials,
            n_trials_feasible     = n_feasible,
        )