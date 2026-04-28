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

    Attributes
    ----------
    initial_extra_containers : x0 — extra containers added at Year 1 beyond the base.
    s0_extra_mwp           : Extra solar DC MWp deployed at Year 1 (0.0 if none).
    augmentation_events    : List of {year, containers} dicts for active events (k_i > 0).
    cuf_floor_fixed_pct    : Fixed Year-1 CUF floor (%), not solar-adjusted.
    cuf_series             : Annual plant CUF (%) over 25 years for the optimal schedule.
    baseline_cuf_series    : Annual CUF without any augmentation (for comparison).
    n_max                  : N_max from pre-analysis.
    shortfall_windows      : List of (start_yr, end_yr) tuples from pre-analysis.
    yearly_aug_costs       : Augmentation CAPEX charged each project year (Rs).
    yearly_delta_savings   : Extra savings vs. no-augmentation baseline per year (Rs).
    total_pv_aug_cost      : NPV of all augmentation costs (Rs).
    pv_solar_oversize_cost : PV of solar oversizing capex (Rs).
    savings_npv_gain       : NPV gain in client savings from augmented delivery (Rs).
    final_score            : Net objective value = savings_npv_gain - pv_bess_capex - pv_solar_capex (Rs).
    n_trials               : Total Optuna trials run.
    n_feasible             : Feasible trials (CUF-compliant) found.
    """

    # --- schedule ---
    initial_extra_containers:   int
    s0_extra_mwp:               float
    augmentation_events:        list[dict]         # [{year, containers}, ...]

    # --- CUF ---
    cuf_floor_fixed_pct:        float              # fixed Year-1 CUF floor (%)
    cuf_series:                 np.ndarray         # shape (project_life,)
    baseline_cuf_series:        np.ndarray         # shape (project_life,)
    n_max:                      int
    shortfall_windows:          list[tuple[int, int]]  # [(start_yr, end_yr), ...]

    # --- economics ---
    yearly_aug_costs:           np.ndarray         # shape (project_life,)  Rs
    yearly_delta_savings:       np.ndarray         # shape (project_life,)  Rs
    total_pv_aug_cost:          float              # Rs
    pv_solar_oversize_cost:     float              # Rs
    savings_npv_gain:           float              # Rs
    final_score:                float              # Rs (net objective = savings - capex)

    # --- solver stats ---
    n_trials:                   int
    n_feasible:                 int
