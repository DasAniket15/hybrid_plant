"""
optimise/constraints/soc.py
────────────────────────────
C6   soc[t] = soc[t−1] + η_c·chg[t] − dis[t]      (soc[−1] = 0, commissioning-zero)
C8   soc[t] ≤ E_b · deg_b[t]
C9   chg[t] ≤ crc · E_b · deg_b[t]                  (charge power cap, D3)
C10  dis[t] ≤ crd · E_b · deg_b[t]                  (discharge power cap, D3)

Flat-index SOC chain (single & full horizon)
─────────────────────────────────────────────
With a flat time index, C6 is identical in both modes.  In full mode the
recursion runs continuously across year boundaries, so the design §3.4 carryover
soc[y,0] = soc[y−1, H_last] is automatic — no special boundary constraint.
soc[−1] = 0 sets the commissioning-zero start; the terminal SOC is free.

deg_b[t] is 1.0 in single mode and d_b[year(t)] in full mode (§3.5 C8′–C10′),
so the degraded energy capacity and power caps are handled index-generically.

Aux treatment (D8 — energy-level)
──────────────────────────────────
Aux is NOT in the SOC balance; it is netted at the energy level in C3
(balance.py).  This keeps C6 LP-feasible at commissioning-zero and avoids any
min(aux, SOC) non-linearity.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext


def add_soc_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
) -> None:
    """Attach C6, C8, C9, C10 to *model* in-place (C7 encoded as soc[-1]=0)."""
    eta_c = params.eta_c
    crc   = params.crc
    crd   = params.crd

    @model.Constraint(model.H)
    def c6_soc_dynamics(m, t: int) -> pyo.ConstraintData:
        soc_prev = 0.0 if t == 0 else m.soc[t - 1]
        return m.soc[t] == soc_prev + eta_c * m.chg[t] - m.dis[t]

    @model.Constraint(model.H)
    def c8_soc_cap(m, t: int) -> pyo.ConstraintData:
        return m.soc[t] <= m.E_b * float(tc.deg_b[t])

    @model.Constraint(model.H)
    def c9_charge_power_cap(m, t: int) -> pyo.ConstraintData:
        return m.chg[t] <= crc * m.E_b * float(tc.deg_b[t])

    @model.Constraint(model.H)
    def c10_discharge_power_cap(m, t: int) -> pyo.ConstraintData:
        return m.dis[t] <= crd * m.E_b * float(tc.deg_b[t])
