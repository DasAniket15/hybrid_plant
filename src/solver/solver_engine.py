"""
solver_engine.py
────────────────
Optimization layer for the hybrid RE plant model.

Algorithm : Optuna TPE (Tree-structured Parzen Estimator)
Objective : Maximize client savings NPV vs. 100% DISCOM baseline
Scope     : solar_mw (continuous), wind_mw (continuous),
            ppa_mw (continuous), bess_containers (integer)

All other decision variables are fixed at their `fixed_value` from
solver.yaml and are clearly marked as future scope there.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    best_params:            dict
    best_savings_npv:       float
    best_year1_savings:     float
    best_lcoe:              float
    best_landed_tariff_y1:  float
    all_trials:             pd.DataFrame      # ranked, all completed trials
    full_result:            dict              # raw engine outputs for best solution
    n_trials_completed:     int
    n_trials_feasible:      int


# ─────────────────────────────────────────────────────────────────────────────
# Solver engine
# ─────────────────────────────────────────────────────────────────────────────

class SolverEngine:
    """
    Wraps Optuna to search the decision-variable space defined in solver.yaml.

    Usage
    -----
        solver  = SolverEngine(config, data, energy_engine, finance_engine)
        result  = solver.run()
        print(result.best_params)
    """

    # ── Fixed values for future-scope variables ──────────────────────────────
    # Read directly from solver.yaml fixed_value fields; hardcoded here as
    # fallback defaults in case the YAML is missing a key.
    _FUTURE_FIXED = {
        "charge_c_rate":      1.0,
        "discharge_c_rate":   1.0,
        "dispatch_priority":  "solar_first",
        "bess_charge_source": "solar_only",
    }

    def __init__(self, config, data, energy_engine, finance_engine):
        self.config          = config
        self.data            = data
        self.energy_engine   = energy_engine
        self.finance_engine  = finance_engine

        sv = config.solver["solver"]
        self.solver_cfg      = sv
        self.dv              = sv["decision_variables"]

        # Override fixed values from YAML where present
        for key, yaml_key in [
            ("charge_c_rate",      "bess_charge_c_rate"),
            ("discharge_c_rate",   "bess_discharge_c_rate"),
            ("dispatch_priority",  "dispatch_priority"),
            ("bess_charge_source", "bess_charge_source"),
        ]:
            fv = self.dv.get(yaml_key, {}).get("fixed_value")
            if fv is not None:
                self._FUTURE_FIXED[key] = fv

        self.n_trials    = int(sv.get("n_trials", 300))
        self.n_jobs      = int(sv.get("n_jobs", 1))
        self.random_seed = int(sv.get("random_seed", 42))

        # Internal trial log
        self._trial_log: list[dict] = []

    # ── Search space ─────────────────────────────────────────────────────────

    def _suggest(self, trial: optuna.Trial) -> dict:
        """Map an Optuna trial to a full set of engine kwargs."""

        dv = self.dv

        solar_mw = trial.suggest_float(
            "solar_capacity_mw",
            dv["solar_capacity_mw"]["min"],
            dv["solar_capacity_mw"]["max"],
            step=dv["solar_capacity_mw"].get("step"),
        )
        wind_mw = trial.suggest_float(
            "wind_capacity_mw",
            dv["wind_capacity_mw"]["min"],
            dv["wind_capacity_mw"]["max"],
            step=dv["wind_capacity_mw"].get("step"),
        )
        ppa_mw = trial.suggest_float(
            "ppa_capacity_mw",
            dv["ppa_capacity_mw"]["min"],
            dv["ppa_capacity_mw"]["max"],
            step=dv["ppa_capacity_mw"].get("step"),
        )
        bess_containers = trial.suggest_int(
            "bess_containers",
            dv["bess_containers"]["min"],
            dv["bess_containers"]["max"],
            step=dv["bess_containers"].get("step", 1),
        )

        return {
            # current-scope
            "solar_capacity_mw":  solar_mw,
            "wind_capacity_mw":   wind_mw,
            "ppa_capacity_mw":    ppa_mw,
            "bess_containers":    bess_containers,
            # future-scope (fixed)
            "charge_c_rate":      self._FUTURE_FIXED["charge_c_rate"],
            "discharge_c_rate":   self._FUTURE_FIXED["discharge_c_rate"],
            "dispatch_priority":  self._FUTURE_FIXED["dispatch_priority"],
            "bess_charge_source": self._FUTURE_FIXED["bess_charge_source"],
        }

    # ── Single evaluation ─────────────────────────────────────────────────────

    def _evaluate(self, params: dict) -> dict[str, Any]:
        """
        Run energy + finance engines for a given parameter set.
        Returns a dict with all results, or raises on simulation error.
        """
        year1 = self.energy_engine.evaluate(
            solar_capacity_mw  = params["solar_capacity_mw"],
            wind_capacity_mw   = params["wind_capacity_mw"],
            bess_containers    = params["bess_containers"],
            charge_c_rate      = params["charge_c_rate"],
            discharge_c_rate   = params["discharge_c_rate"],
            ppa_capacity_mw    = params["ppa_capacity_mw"],
            dispatch_priority  = params["dispatch_priority"],
            bess_charge_source = params["bess_charge_source"],
        )

        finance = self.finance_engine.evaluate(
            year1_results     = year1,
            solar_capacity_mw = params["solar_capacity_mw"],
            wind_capacity_mw  = params["wind_capacity_mw"],
            ppa_capacity_mw   = params["ppa_capacity_mw"],
        )

        return {"year1": year1, "finance": finance}

    # ── Constraint check ──────────────────────────────────────────────────────

    def _is_feasible(self, finance: dict) -> bool:
        """
        Currently enforces:  savings_npv > 0  (hybrid beats DISCOM baseline)
        """
        constraints = self.solver_cfg.get("constraints", {})
        if constraints.get("minimum_savings_npv", {}).get("enabled", True):
            min_npv = constraints["minimum_savings_npv"].get("min_value", 0)
            if finance["savings_npv"] < min_npv:
                return False
        return True

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Returns savings_npv for feasible trials.
        Infeasible / errored trials return a large negative penalty so Optuna
        can still learn directionally from them.
        """
        PENALTY = -1e15

        params = self._suggest(trial)

        try:
            result  = self._evaluate(params)
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
        n_trials:    int | None = None,
        n_jobs:      int | None = None,
        show_progress: bool = True,
    ) -> SolverResult:
        """
        Execute the optimization study.

        Parameters
        ----------
        n_trials        : override solver.yaml n_trials if provided
        n_jobs          : override solver.yaml n_jobs if provided
        show_progress   : print a progress bar via optuna's tqdm callback

        Returns
        -------
        SolverResult
        """
        n_trials = n_trials or self.n_trials
        n_jobs   = n_jobs   or self.n_jobs

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)

        study = optuna.create_study(
            direction = "maximize",
            sampler   = sampler,
        )

        callbacks = []
        if show_progress:
            try:
                from optuna.progress_bar import _ProgressBar
                callbacks.append(optuna.study._optimize._ProgressBarCallback(n_trials))
            except Exception:
                pass  # progress bar optional

        study.optimize(
            self._objective,
            n_trials  = n_trials,
            n_jobs    = n_jobs,
            callbacks = callbacks if callbacks else None,
            show_progress_bar = show_progress,
        )

        # ── Build results ─────────────────────────────────────────────────────

        best_params_raw = study.best_trial.params
        best_params = {
            **best_params_raw,
            # restore fixed future-scope values for completeness
            "charge_c_rate":      self._FUTURE_FIXED["charge_c_rate"],
            "discharge_c_rate":   self._FUTURE_FIXED["discharge_c_rate"],
            "dispatch_priority":  self._FUTURE_FIXED["dispatch_priority"],
            "bess_charge_source": self._FUTURE_FIXED["bess_charge_source"],
        }

        # Re-run best to get full result dict
        full_result = self._evaluate(best_params)

        finance = full_result["finance"]
        year1   = full_result["year1"]

        # Build ranked trial dataframe
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