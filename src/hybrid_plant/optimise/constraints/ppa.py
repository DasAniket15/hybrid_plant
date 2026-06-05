"""
optimise/constraints/ppa.py
────────────────────────────
C5  sd[t] + wd[t] + η_d·dis[t] ≤ P

The LHS is total busbar delivery before the loss factor (= PlantEngine's
export[t]).  The PPA export cap P is a contracted limit that does not degrade,
so this constraint is identical in single and full horizon modes; the
TimeContext is accepted for a uniform builder signature but not used here.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext


def add_ppa_constraint(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
) -> None:
    """Attach C5 to *model* in-place."""
    eta_d = params.eta_d

    @model.Constraint(model.H)
    def c5_ppa_cap(m, t: int) -> pyo.ConstraintData:
        return m.sd[t] + m.wd[t] + eta_d * m.dis[t] <= m.P
