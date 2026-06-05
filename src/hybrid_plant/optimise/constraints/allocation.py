"""
optimise/constraints/allocation.py
───────────────────────────────────
C1  sd[t] + chg[t] ≤ S · deg_s[t] · cuf_s[hour(t)]   (solar: direct + charge)
C2  wd[t]          ≤ W · deg_w[t] · cuf_w[hour(t)]   (wind: direct only, Phase 1)

Index-generic over the flat time index (single or full horizon).  The
degradation factor deg_*[t] is 1.0 in single mode (degradation enters the
objective there) and d_*[year(t)] in full mode (§3.5 C1′/C2′).

D5 future-proofing: a charge_w[t] term and a source-selection parameter slot
into C1/C2 with no structural change.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext


def add_allocation_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
) -> None:
    """Attach C1 and C2 to *model* in-place."""
    cuf_s = params.cuf_s
    cuf_w = params.cuf_w

    @model.Constraint(model.H)
    def c1_solar_alloc(m, t: int) -> pyo.ConstraintData:
        coef = float(tc.deg_s[t]) * float(cuf_s[tc.hour_of[t]])
        return m.sd[t] + m.chg[t] <= m.S * coef

    @model.Constraint(model.H)
    def c2_wind_alloc(m, t: int) -> pyo.ConstraintData:
        coef = float(tc.deg_w[t]) * float(cuf_w[tc.hour_of[t]])
        return m.wd[t] <= m.W * coef
