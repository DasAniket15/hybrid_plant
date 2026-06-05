"""
optimise/constraints/balance.py
────────────────────────────────
C3  lf·(sd[h] + wd[h] + η_d·dis[h]) + ddraw[h] = load[h]
C4  ddraw[h] ≥ 0   (implicit in NonNegativeReals domain of ddraw)

D13 export future-proofing: add exp[h] ≥ 0 to the LHS; C3 becomes
    lf·(sd + wd + η_d·dis) + ddraw − exp = load.
    SOC/discharge structures are unchanged.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_balance_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Attach C3 to *model* in-place.  C4 is enforced by variable domain.

    Parameters
    ----------
    model  : ConcreteModel with H, load/lf/eta_d available via params,
             and sd/wd/dis/ddraw vars attached
    params : OptParams
    """
    lf    = params.lf
    eta_d = params.eta_d
    load  = params.load

    @model.Constraint(model.H)
    def c3_load_balance(m, h: int) -> pyo.ConstraintData:
        return (
            lf * (m.sd[h] + m.wd[h] + eta_d * m.dis[h]) + m.ddraw[h]
            == float(load[h])
        )
