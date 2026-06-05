"""
optimise/constraints/ppa.py
────────────────────────────
C5  sd[h] + wd[h] + η_d·dis[h] ≤ P

The LHS is total busbar delivery before the loss factor — equivalent to
PlantEngine's export[h].  P is the contracted PPA export cap (MW = MWh/h).
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_ppa_constraint(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Attach C5 to *model* in-place.

    Parameters
    ----------
    model  : ConcreteModel with H, P var, sd/wd/dis vars attached
    params : OptParams supplying eta_d
    """
    eta_d = params.eta_d

    @model.Constraint(model.H)
    def c5_ppa_cap(m, h: int) -> pyo.ConstraintData:
        return m.sd[h] + m.wd[h] + eta_d * m.dis[h] <= m.P
