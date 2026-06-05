"""
optimise/constraints/balance.py
────────────────────────────────
C3  lf·(sd[t] + wd[t] + η_d·dis[t] − n_b·aux_pc) + ddraw[t] = load[hour(t)]
C4  ddraw[t] ≥ 0   (implicit in NonNegativeReals domain of ddraw)

Index-generic over the flat time index (single or full horizon).  Load is the
same 8760-profile every year, so it is indexed by hour-of-year.

Aux treatment (D8 — energy-level)
───────────────────────────────────
Aux is consumed at the plant busbar, *before* grid export.  It reduces the net
busbar available for transport → lf applies to (busbar − aux); aux carries no
grid loss of its own and no wheeling / electricity-tax.  See objective.py for
the matching cost term.

D13 export future-proofing: add exp[t] ≥ 0 to the LHS; C3 becomes
    lf·(sd + wd + η_d·dis − n_b·aux_pc) + ddraw − exp = load.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext


def add_balance_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
) -> None:
    """Attach C3 to *model* in-place.  C4 is enforced by variable domain."""
    lf    = params.lf
    eta_d = params.eta_d
    load  = params.load
    aux   = params.aux_pc   # MWh/h per container

    @model.Constraint(model.H)
    def c3_load_balance(m, t: int) -> pyo.ConstraintData:
        return (
            lf * (m.sd[t] + m.wd[t] + eta_d * m.dis[t] - m.nb * aux)
            + m.ddraw[t]
            == float(load[tc.hour_of[t]])
        )
