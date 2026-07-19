"""
optimise/constraints/allocation.py
───────────────────────────────────
C1  sd[t] + chg[t] ≤ S · deg_s[t] · cuf_s[hour(t)]   (solar: direct + charge)
C2  wd[t]          ≤ W · deg_w[t] · cuf_w[hour(t)]   (wind: direct only, Phase 1)

Index-generic over the flat time index (single or full horizon).  The
degradation factor deg_*[t] is 1.0 in single mode (degradation enters the
objective there) and d_*[year(t)] in full mode (§3.5 C1′/C2′).

D5 charge source (bess_charge_source)
─────────────────────────────────────
``solar_only`` (default, Phase 1): the whole charge is solar, so chg lives in
C1 and C2 is wind-direct only — the form above, unchanged.

``wind_only`` / ``solar_and_wind``: the total charge is split into a wind
portion ``chg_w`` (created in variables.py) and a solar portion ``chg − chg_w``:

  C1'  sd[t] + (chg[t] − chg_w[t]) ≤ S·deg_s[t]·cuf_s[hour(t)]
  C2'  wd[t] + chg_w[t]            ≤ W·deg_w[t]·cuf_w[hour(t)]
  Csrc chg_w[t] ≤ chg[t]                       (solar portion ≥ 0; solar_and_wind)
       chg_w[t] = chg[t]                        (wind_only: no solar charging)

The total ``chg`` is unchanged, so SOC dynamics (C6), the power cap (C9), and
the objective are all identical across sources.
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
    """Attach C1 and C2 (and the D5 split, if enabled) to *model* in-place."""
    cuf_s = params.cuf_s
    cuf_w = params.cuf_w
    source = params.bess_charge_source

    if source == "solar_only":
        @model.Constraint(model.H)
        def c1_solar_alloc(m, t: int) -> pyo.ConstraintData:
            coef = float(tc.deg_s[t]) * float(cuf_s[tc.hour_of[t]])
            return m.sd[t] + m.chg[t] <= m.S * coef

        @model.Constraint(model.H)
        def c2_wind_alloc(m, t: int) -> pyo.ConstraintData:
            coef = float(tc.deg_w[t]) * float(cuf_w[tc.hour_of[t]])
            return m.wd[t] <= m.W * coef
        return

    # ── Split formulation (wind_only / solar_and_wind) ────────────────────────
    @model.Constraint(model.H)
    def c1_solar_alloc(m, t: int) -> pyo.ConstraintData:
        coef = float(tc.deg_s[t]) * float(cuf_s[tc.hour_of[t]])
        return m.sd[t] + (m.chg[t] - m.chg_w[t]) <= m.S * coef

    @model.Constraint(model.H)
    def c2_wind_alloc(m, t: int) -> pyo.ConstraintData:
        coef = float(tc.deg_w[t]) * float(cuf_w[tc.hour_of[t]])
        return m.wd[t] + m.chg_w[t] <= m.W * coef

    if source == "wind_only":
        @model.Constraint(model.H)
        def c_charge_source(m, t: int) -> pyo.ConstraintData:
            return m.chg_w[t] == m.chg[t]           # no solar charging
    else:  # solar_and_wind
        @model.Constraint(model.H)
        def c_charge_source(m, t: int) -> pyo.ConstraintData:
            return m.chg_w[t] <= m.chg[t]           # solar portion ≥ 0
