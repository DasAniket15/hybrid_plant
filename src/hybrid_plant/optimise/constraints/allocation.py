"""
optimise/constraints/allocation.py
───────────────────────────────────
C1  sd[h] + chg[h] ≤ S · cuf_s[h]   (solar serves direct + charge; D5 source)
C2  wd[h]          ≤ W · cuf_w[h]   (wind serves direct only in Phase 1)

D5 future-proofing: a charge_w[h] term and a source-selection parameter slot
into C1/C2 with no structural change to these constraints.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_allocation_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Attach C1 and C2 to *model* in-place.

    Parameters
    ----------
    model  : ConcreteModel with H set, S/W vars, sd/wd/chg vars already attached
    params : OptParams supplying cuf_s and cuf_w arrays
    """
    cuf_s = params.cuf_s
    cuf_w = params.cuf_w

    @model.Constraint(model.H)
    def c1_solar_alloc(m, h: int) -> pyo.ConstraintData:
        return m.sd[h] + m.chg[h] <= m.S * float(cuf_s[h])

    @model.Constraint(model.H)
    def c2_wind_alloc(m, h: int) -> pyo.ConstraintData:
        return m.wd[h] <= m.W * float(cuf_w[h])
