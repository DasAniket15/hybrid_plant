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
Step 2 uses a placeholder "maximize RE meter delivery" objective so the LP
has a unique optimum suitable for physics comparison.  The real savings_npv
objective is wired in Step 3 via ``objective.py``.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.constraints.allocation import add_allocation_constraints
from hybrid_plant.optimise.constraints.balance import add_balance_constraints
from hybrid_plant.optimise.constraints.ppa import add_ppa_constraint
from hybrid_plant.optimise.constraints.soc import add_soc_constraints
from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import add_sets
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars

# Hours per simulated year (always 8760 in Phase 1).
_HOURS_PER_YEAR: int = 8760


def build_single_year_model(
    opt_cfg:      OptModelConfig,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
) -> pyo.ConcreteModel:
    """
    Build the single-year LP / MILP.

    Parameters
    ----------
    opt_cfg      : OptModelConfig
    params       : OptParams (all numeric data)
    fixed_sizing : optional dict pinning sizing vars.
                   Keys: ``"S"``, ``"W"``, ``"P"``, ``"nb"`` (any subset).
                   When all four are provided, the model is a pure LP.

    Returns
    -------
    pyo.ConcreteModel
        Fully assembled model ready for ``solve.py``.
        Objective: placeholder maximise-RE-delivery (replaced in Step 3).
    """
    m = pyo.ConcreteModel(name="hybrid_plant_single_year")

    # ── Sets ─────────────────────────────────────────────────────────────────
    add_sets(m, n_hours=_HOURS_PER_YEAR, horizon="single", project_life=params.project_life)

    # ── Variables ─────────────────────────────────────────────────────────────
    add_sizing_vars(m, params, fixed_sizing)
    add_dispatch_vars(m, params)

    # ── Constraints C1–C10 ────────────────────────────────────────────────────
    add_allocation_constraints(m, params)   # C1, C2
    add_balance_constraints(m, params)      # C3  (C4 via domain)
    add_ppa_constraint(m, params)           # C5
    add_soc_constraints(m, params)          # C6–C10

    # ── Placeholder objective (Step 2) ────────────────────────────────────────
    # Maximise total RE meter delivery.  This gives a physics-sensible dispatch
    # for Layer-1 comparison and is replaced by the savings_npv objective in
    # Step 3.
    lf    = params.lf
    eta_d = params.eta_d
    m.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=lf * pyo.quicksum(
            m.sd[h] + m.wd[h] + eta_d * m.dis[h]
            for h in m.H
        ),
    )

    return m
