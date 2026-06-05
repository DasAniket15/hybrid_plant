"""
optimise/sets.py
────────────────
Time-index construction for the hybrid plant optimisation model.

Both horizon modes use a single flat dispatch index ``H`` = 0 … n_steps−1:
  • single mode: n_steps = 8760            (one representative year)
  • full mode:   n_steps = 8760 × 25       (continuous 25-year horizon)

A flat index makes the SOC recursion identical in both modes —
``soc[t] = soc[t-1] + η_c·chg[t] − dis[t]`` with ``soc[-1] = 0`` — so the
year-to-year SOC carryover (design §3.4) is automatic, with no special-casing.

The ``TimeContext`` carries the per-timestep lookups that *do* differ by mode:
  • ``hour_of[t]`` — maps a flat step to its hour-of-year (0…8759) for
    indexing the 8760-length cuf / load / tod parameter arrays
  • ``year_of[t]`` — 1-indexed project year (1…project_life)
  • ``deg_s/deg_w/deg_b[t]`` — per-timestep degradation factor applied to the
    capacity bounds (C1′, C2′, C8′–C10′).  All 1.0 in single mode (degradation
    enters the single-mode *objective* via D_s/D_w/D_b instead).
  • ``disc[t]`` — per-timestep discount factor df[year_of[t]] used by the
    full-mode objective.  Unused (NaN) in single mode, which discounts via
    the precomputed D_s/D_w/D_b sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.constants import HOURS_PER_YEAR
from hybrid_plant.optimise.params import OptParams


@dataclass(frozen=True)
class TimeContext:
    """Per-timestep index maps and coefficients for one horizon mode."""

    horizon: str
    n_steps: int
    hour_of: np.ndarray   # [n_steps] int, values in 0 … 8759
    year_of: np.ndarray   # [n_steps] int, values in 1 … project_life
    deg_s:   np.ndarray   # [n_steps] float, solar degradation per step
    deg_w:   np.ndarray   # [n_steps] float, wind degradation
    deg_b:   np.ndarray   # [n_steps] float, BESS SOH
    disc:    np.ndarray   # [n_steps] float, df[year_of[t]] (full mode only)


def build_time_context(params: OptParams, horizon: str) -> TimeContext:
    """
    Construct the ``TimeContext`` for the requested horizon.

    Parameters
    ----------
    params  : OptParams (supplies project_life, d_s/d_w/d_b, df)
    horizon : ``"single"`` | ``"full"``

    Returns
    -------
    TimeContext
    """
    H = HOURS_PER_YEAR

    if horizon == "single":
        n = H
        return TimeContext(
            horizon="single",
            n_steps=n,
            hour_of=np.arange(H, dtype=int),
            year_of=np.ones(H, dtype=int),
            deg_s=np.ones(H),
            deg_w=np.ones(H),
            deg_b=np.ones(H),
            disc=np.full(H, np.nan),   # not used: single objective uses D_s/D_w/D_b
        )

    if horizon == "full":
        Y = params.project_life
        n = H * Y
        return TimeContext(
            horizon="full",
            n_steps=n,
            hour_of=np.tile(np.arange(H, dtype=int), Y),
            year_of=np.repeat(np.arange(1, Y + 1, dtype=int), H),
            deg_s=np.repeat(params.d_s, H),
            deg_w=np.repeat(params.d_w, H),
            deg_b=np.repeat(params.d_b, H),
            disc=np.repeat(params.df, H),
        )

    raise ValueError(f"Unknown horizon: {horizon!r}. Use 'single' or 'full'.")


def add_sets(model: pyo.ConcreteModel, n_steps: int) -> None:
    """
    Attach the flat dispatch index ``H`` = 0 … n_steps−1 to *model* in-place.
    """
    model.H = pyo.RangeSet(0, n_steps - 1)
