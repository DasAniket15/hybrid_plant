"""
augmentation_solver.py
──────────────────────
Optuna-based solver that jointly optimises Year-1 BESS oversizing and the
resulting lifecycle augmentation economics.

Problem statement
─────────────────
Pass-1 selects B* containers to maximise Year-1 savings NPV with no lifecycle
model.  The oversize dimension — how many extra containers to add upfront —
and the number of augmentation events that are economically justified are
jointly determined here.

Constraint
──────────
CUF must NEVER drop below the baseline (Pass-1 Year-1) CUF at any year across
the 25-year lifecycle.  This is guaranteed by the augmentation trigger
mechanism; any trial where min(cuf_series) < baseline_cuf is penalised with
PENALTY = -1e15 and excluded from the Optuna best.

Objective
─────────
Maximise savings_npv (Rs), identical to the Pass-1 objective.  The solver
naturally avoids:
  • Very high extra: upfront CAPEX grows, NPV falls.
  • Very low extra: more augmentation events needed → higher lifecycle OPEX.

Algorithm
─────────
Single Optuna TPE study; decision variable: extra ∈ [0, max_extra] (integer).

  _objective(trial):
      extra  = trial.suggest_int("extra", 0, max_extra)
      result = engine.evaluate_scenario(
                   initial_containers  = B* + extra,
                   fast_mode           = True,          # speed in trials
                   max_events_override = BIG_EVENTS,    # effectively unlimited
               )
      if min(cuf_series) < baseline_cuf - tol: return PENALTY
      return savings_npv

After convergence, re-run the best extra in full mode (fast_mode=False) for
accurate finance output.

Configuration (bess.yaml → bess.augmentation.solver)
─────────────────────────────────────────────────────
  max_extra_containers  : int   = 200   upper search bound for extra
  n_trials              : int   = 100   Optuna trial budget
  max_events_override   : int   = 20    ceiling for augmentation events per lifecycle
  seed                  : int   = 42    reproducibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

PENALTY: float = -1e15
BIG_EVENTS: int = 999   # "unlimited" events sentinel passed to lifecycle simulator


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentationSolveResult:
    """
    Output of AugmentationSolver.solve().

    Mirrors OversizeResult's interface so run_model.py needs minimal changes.

    Attributes
    ----------
    best_extra              : int   — extra containers above B* chosen by solver
    best_initial_containers : int   — B* + best_extra (total Year-1 cohort)
    best_result             : dict  — full evaluate_scenario() output (full mode)
    sweep_log               : list[dict] — one entry per Optuna trial:
                                  {extra, initial_containers, npv, n_events,
                                   total_aug_cost_rs, feasible}
    n_trials_completed      : int   — actual trials run
    """
    best_extra:              int
    best_initial_containers: int
    best_result:             dict[str, Any]
    sweep_log:               list[dict[str, Any]] = field(default_factory=list)
    n_trials_completed:      int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationSolver:
    """
    Optuna TPE solver for joint oversize + lifecycle economics optimisation.

    Parameters
    ----------
    augmentation_engine   : AugmentationEngine — constructed from Pass-1 result
    base_params           : dict — C* from Pass 1 (solver best_params)
    baseline_cuf          : float — Pass-1 Year-1 CUF; hard CUF floor
    config                : FullConfig — project configuration
    """

    def __init__(
        self,
        augmentation_engine: Any,
        base_params:         dict[str, Any],
        baseline_cuf:        float,
        config:              Any,
    ) -> None:
        self._engine       = augmentation_engine
        self._base_params  = base_params
        self._base_cont    = int(base_params["bess_containers"])
        self._baseline_cuf = baseline_cuf

        aug_cfg            = config.bess["bess"]["augmentation"]
        solver_cfg         = aug_cfg.get("solver", {})
        self._max_extra    = int(solver_cfg.get("max_extra_containers", 200))
        self._n_trials     = int(solver_cfg.get("n_trials", 100))
        self._max_events   = int(solver_cfg.get("max_events_override", 20))
        self._seed         = int(solver_cfg.get("seed", 42))
        self._tol          = float(aug_cfg.get("trigger_tolerance_pp", 0.05))

        self._sweep_log: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────

    def solve(self) -> AugmentationSolveResult:
        """
        Run the Optuna study and return the best (extra, lifecycle) result.

        Returns
        -------
        AugmentationSolveResult
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction = "maximize",
            sampler   = TPESampler(seed=self._seed),
        )
        study.optimize(self._objective, n_trials=self._n_trials)

        best_extra = int(study.best_params["extra"])
        logger.info(
            "AugmentationSolver: Optuna done. best_extra=%d  best_initial=%d  "
            "best_npv=%.2f Cr  trials=%d",
            best_extra,
            self._base_cont + best_extra,
            study.best_value / 1e7,
            len(study.trials),
        )

        # Final accurate re-run with full mode
        logger.info("AugmentationSolver: re-running best extra=%d in full mode …", best_extra)
        best_result = self._engine.evaluate_scenario(
            params              = self._base_params,
            initial_containers  = self._base_cont + best_extra,
            fast_mode           = False,
            max_events_override = self._max_events,
        )

        return AugmentationSolveResult(
            best_extra              = best_extra,
            best_initial_containers = self._base_cont + best_extra,
            best_result             = best_result,
            sweep_log               = self._sweep_log,
            n_trials_completed      = len(study.trials),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: suggest extra, evaluate, return NPV or PENALTY."""
        extra  = trial.suggest_int("extra", 0, self._max_extra)
        result = self._engine.evaluate_scenario(
            params              = self._base_params,
            initial_containers  = self._base_cont + extra,
            fast_mode           = True,
            max_events_override = self._max_events,
        )

        fi        = result["finance"]
        aug       = fi.get("augmentation", {})
        cuf_series = aug.get("cuf_series", [])
        npv       = fi["savings_npv"]

        # Hard CUF constraint: reject trials where CUF dips below floor
        feasible = (
            len(cuf_series) > 0
            and min(cuf_series) >= self._baseline_cuf - self._tol
        )

        self._sweep_log.append({
            "extra":              extra,
            "initial_containers": self._base_cont + extra,
            "npv":                npv,
            "n_events":           aug.get("n_events", 0),
            "total_aug_cost_rs":  (aug.get("total_lump_cost_rs", 0.0)
                                   + aug.get("total_om_cost_rs", 0.0)),
            "feasible":           feasible,
            "min_cuf":            min(cuf_series) if cuf_series else None,
        })

        if not feasible:
            logger.debug(
                "AugmentationSolver trial %d: extra=%d  infeasible "
                "(min_cuf=%.4f < baseline_cuf=%.4f)",
                trial.number, extra,
                min(cuf_series) if cuf_series else float("nan"),
                self._baseline_cuf,
            )
            return PENALTY

        logger.debug(
            "AugmentationSolver trial %d: extra=%d  npv=%.2f Cr  n_events=%d",
            trial.number, extra, npv / 1e7, aug.get("n_events", 0),
        )
        return npv
