"""
optimise/objective.py
─────────────────────
Pyomo savings_npv objective (design §2.4, single-year factored form).

Derivation summary
──────────────────
Starting from the full 25-year NPV:

  savings_npv = Σ_t df[t] · [baseline_t − discom_cost_t − re_payment_t
                               − capacity_charges_t − wheeling_t − tax_t
                               − aux_cost_t]

Because load[h] − ddraw[h,t] = re_meter[h,t] (no export; C4):

  baseline_t − discom_cost_t = Σ_h re_meter[h,t] · tod[h] · 1000

Cost-plus cancellation (Finding A):

  Σ_t df[t] · re_payment_t = NPV(financing) + NPV(opex)

In single-year mode the year-sum and hour-sum separate per stream because
degradation is hour-independent:

  Σ_t df[t] · Σ_h re_meter[h,t] · tod[h] · 1000
      = lf · 1000 · [ (Σ_h tod[h]·sd[h])·D_s
                    + (Σ_h tod[h]·wd[h])·D_w
                    + (Σ_h tod[h]·η_d·dis[h])·D_b ]

Combining avoided DISCOM (hourly ToD) and grid charges (wheel+tax) into a
single net-tod coefficient array keeps the expression compact:

  net_tod[h] = tod[h] − (wheel + tax)  [INR/kWh]

Aux (D8) — grid-fed, constant per container:

  NPV(aux) = n_b · aux_pc · Σ_h tod[h] · 1000 · A_N

The complete single-year objective is:

  max savings_npv =
      lf · 1000 · [D_s·Σ_h net_tod[h]·sd[h]
                 + D_w·Σ_h net_tod[h]·wd[h]
                 + D_b·Σ_h net_tod[h]·η_d·dis[h]]
    − total_capex · Φ                              # NPV(financing)
    − NPV(opex)                                    # incl. insurance
    − cap_rate · P · 12 · A_N                      # NPV(capacity charges)
    − n_b · aux_annual_rate · A_N                  # NPV(aux)

Every coefficient is a Python scalar (or float-cast numpy value); all
expressions are linear in the Pyomo decision variables.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams


def add_savings_npv_objective(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Attach (or replace) the savings_npv maximisation objective on *model*.

    Parameters
    ----------
    model  : ConcreteModel with sets, variables, and E_b expression attached
    params : OptParams
    """
    # Remove any existing objective before adding the new one
    if hasattr(model, "obj"):
        model.del_component(model.obj)

    lf    = params.lf
    eta_d = params.eta_d
    D_s   = params.D_s
    D_w   = params.D_w
    D_b   = params.D_b

    # ── Net ToD coefficient per hour ─────────────────────────────────────────
    # net_tod[h] = tod[h] − (wheel + tax)  [INR/kWh]
    # Multiplying energy (MWh) by net_tod × 1000 gives INR, discounted via D_*.
    net_tod = params.tod - (params.wheel + params.tax)   # numpy array (8760,)

    coef_sd  = lf * 1000.0 * D_s * net_tod              # per sd[h]  (8760,)
    coef_wd  = lf * 1000.0 * D_w * net_tod              # per wd[h]
    coef_dis = lf * 1000.0 * D_b * net_tod * eta_d      # per dis[h]

    # ── Revenue minus grid charges (§2.4 first two lines) ────────────────────
    rev_minus_gc = pyo.quicksum(
        float(coef_sd[h])  * model.sd[h]
        + float(coef_wd[h])  * model.wd[h]
        + float(coef_dis[h]) * model.dis[h]
        for h in model.H
    )

    # ── total_capex (linear expression in S, W, E_b) ─────────────────────────
    total_capex = (
        model.S   * params.ac_dc * params.solar_rate
        + model.W   * params.wind_rate
        + model.E_b * params.bess_rate
        + params.trans_fixed
    )

    # ── NPV(financing) ────────────────────────────────────────────────────────
    npv_financing = total_capex * params.phi

    # ── NPV(opex) including insurance (§2.5) ─────────────────────────────────
    solar_dc = model.S * params.ac_dc
    npv_opex = (
        solar_dc  * (params.solar_om_rate * params.G_solar_om
                     + params.solar_trans_om_rate * params.A_N)
        + model.W   * (params.wind_om_rate * params.G_wind_om
                       + params.wind_trans_om_rate * params.A_N)
        + model.E_b * params.bess_om_rate * params.A_N
        + params.land_lease_monthly * 12.0 * params.G_land   # constant
        + total_capex * params.insurance_pct * params.A_N
    )

    # ── NPV(capacity charges) ─────────────────────────────────────────────────
    npv_cap = params.cap_rate * model.P * 12.0 * params.A_N

    # ── NPV(aux impact) — energy-level netting (D8) ──────────────────────────
    # Aux is consumed at the plant busbar; it reduces net RE meter delivery by
    # lf × aux_pc per container per hour.  No wheeling/tax on aux (it is never
    # transported); grid loss does not apply to the aux energy itself.
    # Rate per container per year = lf × aux_pc × Σ_h net_tod[h] × 1000
    # (net_tod already excludes wheel+tax — the loss is priced at the
    # delivered-RE rate, not the full DISCOM tariff).
    aux_net_rate = params.lf * params.aux_pc * float(np.sum(net_tod)) * 1000.0
    npv_aux = model.nb * aux_net_rate * params.A_N

    # ── Objective ─────────────────────────────────────────────────────────────
    model.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=rev_minus_gc - npv_financing - npv_opex - npv_cap - npv_aux,
    )


def add_savings_npv_objective_full(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     "object",
) -> None:
    """
    Full 25-year savings_npv objective (design §2.4 full-mode double sum).

    Revenue is summed explicitly over the 25 × 8760 horizon; per-year
    degradation enters through the capacity bounds (C1′/C2′/C8′–C10′, see the
    constraint modules), NOT through D_s/D_w/D_b.  Each timestep is discounted
    by df[year(t)] (carried in ``tc.disc``).

        revenue = Σ_t disc[t]·lf·1000·net_tod[hour(t)]·(sd[t] + wd[t] + η_d·dis[t])

    The cost side (financing, opex, capacity, aux) is identical to the
    single-year objective — it depends only on sizing and the precomputed
    annuity factors.  Aux is undegraded, so it discounts at A_N.

    All coefficients are premultiplied by _S = 1e-7 (INR → scaled units) so
    HiGHS sees a well-conditioned objective range.  model._obj_scale is set so
    solve.py can recover original INR values without knowing the horizon mode.

    Parameters
    ----------
    model  : ConcreteModel with sets, variables, E_b expression attached
    params : OptParams
    tc     : TimeContext (supplies hour_of and disc per timestep)
    """
    if hasattr(model, "obj"):
        model.del_component(model.obj)

    # Scale factor: keeps objective coefficients in [~1e-3, ~1e2] instead of
    # [5e2, 9e7].  Matches OptModelConfig.scale_money (Step 7 will wire it).
    _S = 1e-7
    # Bypass Pyomo's Block.__setattr__ (which traverses all ~3M components on
    # the full model to validate/register the attribute — catastrophically slow).
    # object.__setattr__ sets a plain Python attribute directly on the instance.
    object.__setattr__(model, "_obj_scale", _S)

    lf    = params.lf
    eta_d = params.eta_d
    net_tod = params.tod - (params.wheel + params.tax)   # (8760,)

    # Per-timestep revenue coefficient: _S · disc[t] · lf · 1000 · net_tod[hour(t)]
    # Premultiplied as a numpy float array — pyo.quicksum sees flat float × LinearExpr.
    base_coef = _S * tc.disc * lf * 1000.0 * net_tod[tc.hour_of]   # (n_steps,)

    rev = pyo.quicksum(
        float(base_coef[t]) * (model.sd[t] + model.wd[t] + eta_d * model.dis[t])
        for t in model.H
    )

    # ── Cost side: all scalar coefficients premultiplied by _S ───────────────
    total_capex = (
        model.S   * params.ac_dc * params.solar_rate
        + model.W   * params.wind_rate
        + model.E_b * params.bess_rate
        + params.trans_fixed
    )
    npv_financing = total_capex * (params.phi * _S)

    solar_dc = model.S * params.ac_dc
    npv_opex = (
        solar_dc  * ((params.solar_om_rate * params.G_solar_om
                      + params.solar_trans_om_rate * params.A_N) * _S)
        + model.W   * ((params.wind_om_rate * params.G_wind_om
                        + params.wind_trans_om_rate * params.A_N) * _S)
        + model.E_b * (params.bess_om_rate * params.A_N * _S)
        + params.land_lease_monthly * 12.0 * params.G_land * _S
        + total_capex * (params.insurance_pct * params.A_N * _S)
    )
    npv_cap = params.cap_rate * model.P * (12.0 * params.A_N * _S)

    # Aux (energy-level, undegraded → discounts at A_N)
    aux_net_rate = lf * params.aux_pc * float(np.sum(net_tod)) * 1000.0
    npv_aux = model.nb * (aux_net_rate * params.A_N * _S)

    model.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=rev - npv_financing - npv_opex - npv_cap - npv_aux,
    )


def add_maximize_re_delivery_objective(
    model:  pyo.ConcreteModel,
    params: OptParams,
) -> None:
    """
    Placeholder objective: maximise total RE meter delivery.
    Used in Step 2 validation; replaced by savings_npv in Step 3.
    """
    if hasattr(model, "obj"):
        model.del_component(model.obj)

    lf    = params.lf
    eta_d = params.eta_d
    model.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=lf * pyo.quicksum(
            model.sd[h] + model.wd[h] + eta_d * model.dis[h]
            for h in model.H
        ),
    )
