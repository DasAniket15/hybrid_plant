"""
optimise/variables.py
─────────────────────
Sizing and hourly dispatch variable declarations for the Pyomo model.

Variable conventions (design §2.3)
───────────────────────────────────
Sizing (scalar, all ≥ 0):
  S   — AC solar capacity (MW)
  W   — wind capacity (MW)
  P   — PPA export cap (MW)
  nb  — BESS container count (non-negative integer, the sole integrality)
  E_b — derived expression: nb × cs (MWh); not a free variable

Hourly dispatch (indexed over H, all ≥ 0):
  sd    — solar → client load direct, busbar basis (MWh)
  wd    — wind  → client load direct, busbar basis (MWh)
  chg   — solar → battery, busbar basis pre charge-efficiency (MWh)
  dis   — energy removed from SOC, pre discharge-efficiency (MWh);
          busbar discharge delivery = η_d × dis
  soc   — state of charge at END of each hour (MWh)
  ddraw — residual DISCOM draw at the meter (MWh)

All dispatch vars use NonNegativeReals; C4 (ddraw ≥ 0) is therefore
implicit in the domain declaration.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_sizing_vars(
    model:        pyo.ConcreteModel,
    params:       OptParams,
    fixed_sizing: dict[str, Any] | None = None,
) -> None:
    """
    Declare S, W, P (continuous) and nb (integer) with bounds from *params*.

    Parameters
    ----------
    model        : ConcreteModel (modified in-place)
    params       : OptParams
    fixed_sizing : optional dict with keys ``"S"``, ``"W"``, ``"P"``, ``"nb"``.
                   Any key present pins that variable to the given value
                   (used in Steps 2 and 3 for fixed-sizing validation).
    """
    model.S  = pyo.Var(domain=pyo.NonNegativeReals,    bounds=(params.s_min,  params.s_max))
    model.W  = pyo.Var(domain=pyo.NonNegativeReals,    bounds=(params.w_min,  params.w_max))
    model.P  = pyo.Var(domain=pyo.NonNegativeReals,    bounds=(params.p_min,  params.p_max))
    model.nb = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(params.nb_min, params.nb_max))

    if fixed_sizing is not None:
        if "S"  in fixed_sizing: model.S.fix(float(fixed_sizing["S"]))
        if "W"  in fixed_sizing: model.W.fix(float(fixed_sizing["W"]))
        if "P"  in fixed_sizing: model.P.fix(float(fixed_sizing["P"]))
        if "nb" in fixed_sizing: model.nb.fix(int(fixed_sizing["nb"]))

    # E_b is a Pyomo Expression (linear in nb) so constraints are identical
    # whether sizing is fixed or free.
    model.E_b = pyo.Expression(expr=model.nb * params.cs)


def add_dispatch_vars(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Declare hourly dispatch variables indexed over ``model.H``.

    Parameters
    ----------
    model  : ConcreteModel with ``H`` set already attached
    params : OptParams (unused directly here; kept for signature consistency)
    """
    H = model.H
    model.sd    = pyo.Var(H, domain=pyo.NonNegativeReals)
    model.wd    = pyo.Var(H, domain=pyo.NonNegativeReals)
    model.chg   = pyo.Var(H, domain=pyo.NonNegativeReals)
    model.dis   = pyo.Var(H, domain=pyo.NonNegativeReals)
    model.soc   = pyo.Var(H, domain=pyo.NonNegativeReals)
    model.ddraw = pyo.Var(H, domain=pyo.NonNegativeReals)

    # D5 charge-source split: chg_w = wind-sourced portion of the total charge
    # chg (solar portion = chg - chg_w).  Only created when the battery may
    # charge from wind; solar_only (default) keeps the base model unchanged.
    if params.bess_charge_source != "solar_only":
        model.chg_w = pyo.Var(H, domain=pyo.NonNegativeReals)
