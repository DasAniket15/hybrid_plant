"""
augmentation_result.py
──────────────────────
Result container for AugmentationEngine.run().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationResult:
    """
    Structured output of a completed augmentation optimisation run.

    Solar and BESS augmentation are both event-based; neither asset is
    oversized at Year 1.  Each solar cohort has its own age-indexed
    degradation curve, identical in spirit to the multi-cohort BESS model.

    Attributes
    ----------
    initial_extra_containers  : x0 — extra BESS containers added at Year 1 beyond base.
    bess_augmentation_events  : [{year, containers}, ...] for active BESS events (k_i > 0).
    solar_augmentation_events : [{year, mwp}, ...] for active solar events (MWp > threshold).

    cuf_floor_fixed_pct       : Fixed Year-1 CUF floor (%), the contractual minimum.
    cuf_series                : Annual plant CUF (%) over project life for optimal schedule.
    baseline_cuf_series       : Annual CUF without any augmentation (for comparison).
    n_max                     : N_max (BESS events) from pre-analysis.
    shortfall_windows         : [(start_yr, end_yr), ...] from pre-analysis.

    yearly_bess_aug_costs     : BESS augmentation CAPEX per project year (Rs), shape (life,).
    yearly_solar_aug_costs    : Solar augmentation CAPEX per project year (Rs), shape (life,).
    yearly_delta_savings      : Extra savings vs. no-augmentation baseline per year (Rs).
    total_pv_bess_aug_cost    : NPV of all BESS augmentation CAPEX (Rs).
    total_pv_solar_aug_cost   : NPV of all solar augmentation CAPEX (Rs).
    savings_npv_gain          : NPV gain in client savings from augmented delivery (Rs).
    final_score               : Net objective = savings_npv_gain − bess_capex − solar_capex (Rs).

    n_trials                  : Total Optuna trials run.
    n_feasible                : Feasible trials (CUF-compliant) found.
    """

    # --- schedule ---
    initial_extra_containers:  int
    bess_augmentation_events:  list[dict]   # [{year, containers}, ...]
    solar_augmentation_events: list[dict]   # [{year, mwp}, ...]

    # --- CUF ---
    cuf_floor_fixed_pct:       float
    cuf_series:                np.ndarray   # shape (project_life,)
    baseline_cuf_series:       np.ndarray   # shape (project_life,)
    n_max:                     int
    shortfall_windows:         list[tuple[int, int]]

    # --- economics ---
    yearly_bess_aug_costs:     np.ndarray   # shape (project_life,)  Rs
    yearly_solar_aug_costs:    np.ndarray   # shape (project_life,)  Rs
    yearly_delta_savings:      np.ndarray   # shape (project_life,)  Rs
    total_pv_bess_aug_cost:    float        # Rs
    total_pv_solar_aug_cost:   float        # Rs
    savings_npv_gain:          float        # Rs
    final_score:               float        # Rs  (net objective)

    # --- solver stats ---
    n_trials:                  int
    n_feasible:                int
