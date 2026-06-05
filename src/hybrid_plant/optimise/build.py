"""
optimise/build.py
─────────────────
Assembles a Pyomo ConcreteModel from OptModelConfig + OptParams.

Horizon toggle (D6)
───────────────────
``OptModelConfig.horizon == "single"`` — one representative year (8760 h).
``OptModelConfig.horizon == "full"``   — full 25 × 8760 h (Step 5).

Fixed-sizing mode
─────────────────
``fixed_sizing`` dict pins any subset of {S, W, P, nb} to given values.
Used in Steps 2 (dispatch-only validation) and 3 (finance-parity check).
With all four keys set the model is a pure LP (integer nb is pinned).

Objective
─────────
``objective='savings_npv'``  (default) — use the §2.4 ToD-aware objective.
``objective='maximize_re'``            — placeholder maximise-RE-delivery
                                         (used in Step 2 for physics comparison).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.constraints.allocation import add_allocation_constraints
from hybrid_plant.optimise.constraints.balance import add_balance_constraints
from hybrid_plant.optimise.constraints.ppa import add_ppa_constraint
from hybrid_plant.optimise.constraints.soc import add_soc_constraints
from hybrid_plant.optimise.objective import (
    add_maximize_re_delivery_objective,
    add_savings_npv_objective,
)
from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import add_sets
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars

_HOURS_PER_YEAR: int = 8760


def build_single_year_model(
    opt_cfg:      OptModelConfig,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
    objective:    str = "savings_npv",
) -> pyo.ConcreteModel:
    """
    Build the single-year LP / MILP.

    Parameters
    ----------
    opt_cfg      : OptModelConfig
    params       : OptParams (all numeric data)
    fixed_sizing : optional dict pinning sizing vars (keys: S, W, P, nb).
                   When all four provided, the model is a pure LP.
    objective    : ``"savings_npv"``  — ToD-aware §2.4 objective (default).
                   ``"maximize_re"``  — maximize total RE meter delivery
                                        (Step 2 physics validation only).

    Returns
    -------
    pyo.ConcreteModel
        Fully assembled model ready for ``solve.py``.
    """
    m = pyo.ConcreteModel(name="hybrid_plant_single_year")

    add_sets(m, n_hours=_HOURS_PER_YEAR, horizon="single", project_life=params.project_life)
    add_sizing_vars(m, params, fixed_sizing)
    add_dispatch_vars(m, params)

    add_allocation_constraints(m, params)   # C1, C2
    add_balance_constraints(m, params)      # C3  (C4 via domain)
    add_ppa_constraint(m, params)           # C5
    add_soc_constraints(m, params)          # C6–C10

    if objective == "savings_npv":
        add_savings_npv_objective(m, params)
    elif objective == "maximize_re":
        add_maximize_re_delivery_objective(m, params)
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Use 'savings_npv' or 'maximize_re'.")

    return m
