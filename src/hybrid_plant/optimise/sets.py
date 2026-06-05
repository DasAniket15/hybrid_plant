"""
optimise/sets.py
────────────────
Pyomo set construction for the hybrid plant optimisation model.

Sets
────
  H      — integer hours 0 … n_hours−1 (single-year: 0…8759)
  Y      — project years 1 … project_life (full-horizon mode only)
"""

from __future__ import annotations

import pyomo.environ as pyo


def add_sets(
    model:        pyo.ConcreteModel,
    n_hours:      int,
    horizon:      str,
    project_life: int,
) -> None:
    """
    Attach set declarations to *model* in-place.

    Parameters
    ----------
    model        : Pyomo ConcreteModel (modified in-place)
    n_hours      : number of hours (8760 for annual)
    horizon      : ``"single"`` | ``"full"``
    project_life : project lifetime in years
    """
    model.H = pyo.RangeSet(0, n_hours - 1)

    if horizon == "full":
        model.Y = pyo.RangeSet(1, project_life)
