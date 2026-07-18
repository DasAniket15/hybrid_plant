"""
optimise/build.py
─────────────────
Assembles a Pyomo ConcreteModel from OptModelConfig + OptParams.

Horizon toggle (D6) — the single place the mode is selected:
  ``build_model(opt_cfg, ...)`` dispatches on ``opt_cfg.horizon``:
    "single" → build_single_year_model  (8760-h representative year)
    "full"   → build_full_model         (25 × 8760-h continuous horizon)

One set of index-generic constraint builders serves both modes; they receive a
``TimeContext`` carrying the per-timestep hour map and degradation factors.

Fixed-sizing mode
─────────────────
``fixed_sizing`` pins any subset of {S, W, P, nb}; with all four set the model
is a pure LP (integer nb pinned).  Used for validation in Steps 2–5.

Objective
─────────
``objective='savings_npv'`` (default) — §2.4 ToD-aware objective
  (single: D_s/D_w/D_b factored form; full: explicit df[year] double sum).
``objective='maximize_re'``           — placeholder maximise-RE-delivery
  (Step 2 physics validation only).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.constraints.allocation import add_allocation_constraints
from hybrid_plant.optimise.constraints.balance import add_balance_constraints
from hybrid_plant.optimise.constraints.optional import add_optional_constraints
from hybrid_plant.optimise.constraints.ppa import add_ppa_constraint
from hybrid_plant.optimise.constraints.soc import add_soc_constraints
from hybrid_plant.optimise.objective import (
    add_maximize_re_delivery_objective,
    add_savings_npv_objective,
    add_savings_npv_objective_full,
)
from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import add_sets, build_time_context
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars


def build_model(
    opt_cfg:      OptModelConfig,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
    objective:    str = "savings_npv",
) -> pyo.ConcreteModel:
    """
    Build the model for the horizon selected in ``opt_cfg`` (the D6 toggle).
    """
    if opt_cfg.horizon == "single":
        return build_single_year_model(opt_cfg, params, fixed_sizing, objective)
    if opt_cfg.horizon == "full":
        return build_full_model(opt_cfg, params, fixed_sizing, objective)
    raise ValueError(f"Unknown horizon: {opt_cfg.horizon!r}. Use 'single' or 'full'.")


def _assemble_common(
    name:         str,
    horizon:      str,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None,
) -> tuple[pyo.ConcreteModel, Any]:
    """Build sets, variables, and C1–C10 (shared by both horizons)."""
    tc = build_time_context(params, horizon)

    m = pyo.ConcreteModel(name=name)
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, params, fixed_sizing)
    add_dispatch_vars(m, params)

    add_allocation_constraints(m, params, tc)   # C1, C2
    add_balance_constraints(m, params, tc)      # C3  (C4 via domain)
    add_ppa_constraint(m, params, tc)           # C5
    add_soc_constraints(m, params, tc)          # C6–C10

    return m, tc


def build_single_year_model(
    opt_cfg:      OptModelConfig,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
    objective:    str = "savings_npv",
) -> pyo.ConcreteModel:
    """
    Build the single-year LP / MILP (8760 h).  Degradation enters the objective
    via D_s/D_w/D_b; capacity bounds use fresh (Year-1) capacity.
    """
    m, tc = _assemble_common("hybrid_plant_single_year", "single", params, fixed_sizing)

    if objective == "savings_npv":
        add_savings_npv_objective(m, params)
    elif objective == "maximize_re":
        add_maximize_re_delivery_objective(m, params)
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Use 'savings_npv' or 'maximize_re'.")

    add_optional_constraints(m, params, tc)   # §3.6 PPA toggles (no-op if all off)

    return m


def build_full_model(
    opt_cfg:      OptModelConfig,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
    objective:    str = "savings_npv",
) -> pyo.ConcreteModel:
    """
    Build the full 25-year MILP (25 × 8760 h).  Degradation enters the capacity
    bounds (C1′/C2′/C8′–C10′); the objective discounts each year by df[year]
    with one continuous SOC chain across year boundaries (§3.4, §3.5).
    """
    m, tc = _assemble_common("hybrid_plant_full_horizon", "full", params, fixed_sizing)

    if objective == "savings_npv":
        add_savings_npv_objective_full(m, params, tc, scale=opt_cfg.scale_money)
    elif objective == "maximize_re":
        add_maximize_re_delivery_objective(m, params)
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Use 'savings_npv' or 'maximize_re'.")

    add_optional_constraints(m, params, tc)   # §3.6 PPA toggles (no-op if all off)

    return m
