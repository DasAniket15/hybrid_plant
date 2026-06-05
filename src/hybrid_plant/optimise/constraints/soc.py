"""
optimise/constraints/soc.py
────────────────────────────
C6   soc[h] = soc[h−1] + η_c·chg[h] − dis[h]      (interior hours)
C7   initial SOC = 0  (commissioning-zero; encoded as soc_prev = 0 for h=0)
C8   soc[h] ≤ E_b
C9   chg[h] ≤ crc·E_b                               (charge power cap, D3)
C10  dis[h] ≤ crd·E_b                               (discharge power cap, D3)

SOC convention
──────────────
soc[h] is the state of charge at the END of hour h (after charge/discharge
in that hour).  For h = 0 the implicit prior state is 0 (C7).

Aux treatment (D8 — energy-level)
──────────────────────────────────
Auxiliary consumption is NOT in the SOC balance.  Instead it is netted off
at the energy level in C3 (balance.py): aux reduces the net busbar exported
through the grid, so the client's effective DISCOM draw increases by
lf × n_b × aux_pc per hour.  This keeps C6 LP-feasible at commissioning-zero
and avoids any min(aux, SOC) non-linearity.  The resulting divergence from
PlantEngine's DC-drain aux is quantified in the Layer-1 validation test.

Full-mode (§3.5)
────────────────
Degraded E_b per year is handled in build.py by passing a scaled E_b expression;
this module is index-generic.
"""

from __future__ import annotations

import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_soc_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Attach C6–C10 to *model* in-place.

    Parameters
    ----------
    model  : ConcreteModel with H, E_b expression, chg/dis/soc vars attached
    params : OptParams supplying eta_c, eta_d, crc, crd
    """
    eta_c = params.eta_c
    crc   = params.crc
    crd   = params.crd

    @model.Constraint(model.H)
    def c6_soc_dynamics(m, h: int) -> pyo.ConstraintData:
        # C7 encoded here: soc[-1] = 0 (commissioning-zero prior state)
        soc_prev = 0.0 if h == 0 else m.soc[h - 1]
        return m.soc[h] == soc_prev + eta_c * m.chg[h] - m.dis[h]

    @model.Constraint(model.H)
    def c8_soc_cap(m, h: int) -> pyo.ConstraintData:
        return m.soc[h] <= m.E_b

    @model.Constraint(model.H)
    def c9_charge_power_cap(m, h: int) -> pyo.ConstraintData:
        return m.chg[h] <= crc * m.E_b

    @model.Constraint(model.H)
    def c10_discharge_power_cap(m, h: int) -> pyo.ConstraintData:
        return m.dis[h] <= crd * m.E_b
