"""
optimise/constraints/balance.py
────────────────────────────────
C3  lf·(sd[h] + wd[h] + η_d·dis[h] − n_b·aux_pc) + ddraw[h] = load[h]
C4  ddraw[h] ≥ 0   (implicit in NonNegativeReals domain of ddraw)

Aux treatment (D8 — energy-level)
───────────────────────────────────
Aux is consumed at the plant busbar, *before* the RE energy is exported
through the grid.  It therefore:
  • reduces the net busbar available for transport → lf applies to
    (busbar − aux), not to aux independently
  • does not incur grid losses on the aux energy itself
  • does not attract wheeling or electricity-tax charges

The n_b·aux_pc term sits inside the lf bracket so the constraint reads:
  net meter RE delivered + DISCOM draw = client load
where "net meter RE" = lf × (busbar generation − aux consumed at plant).

D13 export future-proofing: add exp[h] ≥ 0 to the LHS; C3 becomes
    lf·(sd + wd + η_d·dis − n_b·aux_pc) + ddraw − exp = load.
    SOC/dispatch structures are unchanged.
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
    model  : ConcreteModel with H set, nb var, E_b expression,
             and sd/wd/dis/ddraw vars attached
    params : OptParams
    """
    lf    = params.lf
    eta_d = params.eta_d
    load  = params.load
    aux   = params.aux_pc   # MWh/h per container

    @model.Constraint(model.H)
    def c3_load_balance(m, h: int) -> pyo.ConstraintData:
        return (
            lf * (m.sd[h] + m.wd[h] + eta_d * m.dis[h] - m.nb * aux)
            + m.ddraw[h]
            == float(load[h])
        )
