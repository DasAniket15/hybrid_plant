"""
augmentation_result.py
──────────────────────
Result container for AugmentationEngine.run().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AugmentationResult:
    """
    Structured output of a completed augmentation optimisation run.

    Attributes
    ----------
    cuf_floor_pct          : Contractual CUF floor derived from Phase-1 Year-1 delivery.
    initial_extra_containers : x0 — extra containers added at Year 1 beyond the base.
    augmentation_events    : List of {year, containers} dicts for active events (k_i > 0).
    cuf_series             : Annual plant CUF (%) over 25 years for the optimal schedule.
    baseline_cuf_series    : Annual CUF without any augmentation (for comparison).
    yearly_aug_costs       : Augmentation CAPEX charged each project year (Rs).
    yearly_delta_savings   : Extra savings vs. no-augmentation baseline per year (Rs).
    total_pv_aug_cost      : NPV of all augmentation costs (Rs).
    savings_npv_gain       : NPV gain in client savings from augmented delivery (Rs).
    final_score            : Objective value = savings_npv_gain - pv_aug_cost - penalties.
    n_trials               : Total Optuna trials run.
    n_feasible             : Feasible trials (CUF-compliant) found.
    """

    cuf_floor_pct:              float        # year-1 base CUF floor (%)
    cuf_floor_per_year:         np.ndarray   # solar-adjusted per-year floor, shape (project_life,)
    initial_extra_containers:   int
    augmentation_events:        list[dict]         # [{year, containers}, ...]
    cuf_series:                 np.ndarray         # shape (project_life,)
    baseline_cuf_series:        np.ndarray         # shape (project_life,)
    yearly_aug_costs:           np.ndarray         # shape (project_life,)  Rs
    yearly_delta_savings:       np.ndarray         # shape (project_life,)  Rs
    total_pv_aug_cost:          float              # Rs
    savings_npv_gain:           float              # Rs
    final_score:                float
    n_trials:                   int
    n_feasible:                 int
