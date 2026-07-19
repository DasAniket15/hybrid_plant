"""
optimise/constraints/optional.py
─────────────────────────────────
Toggleable PPA-contract constraints (design §3.6).

Each constraint is switched on/off by its ``enabled`` flag in ``solver.yaml``
(→ ``OptParams.opt_constraints``).  When all are disabled — the default — this
adds nothing, so the base model is byte-for-byte the Steps 2–5 model.

All constraints are index-generic linear expressions over the flat time index
``model.H``, so one builder serves both the single-year and full-horizon
models.  Annual (per-year) quantities are scaled by the number of horizon years
(1 for single, 25 for full) via ``TimeContext.year_of``, so a per-year
contractual floor keeps its meaning in both modes.

Constraints
───────────
plant_cuf              min% ≤ Σ busbar_export / (P · n_steps) · 100 ≤ max%
                       busbar_export[t] = sd[t] + wd[t] + η_d·dis[t]  (= the
                       PPA-capped plant export, matching ppa.py / PlantEngine).
minimum_bess_capacity  E_b (= nb·cs, Year-1 SOH) ≥ min_mwh
minimum_bess_discharge Σ η_d·dis ≥ min_annual_mwh · n_years   (busbar MWh)
re_penetration         min% ≤ Σ(load − ddraw) / Σ load · 100 ≤ max%
                       meter RE delivery = load[hour] − ddraw  (from C3).

Step 6b — Required (real India PPA / grid clauses):
peak_supply_obligation Σ_{peak}(load − ddraw) ≥ min% · Σ_{peak} load
                       RTC / FDRE guaranteed supply in defined peak blocks
                       (annual tier — the primary compliance measure).
peak_supply_monthly    per (year, month): Σ_{peak∈month}(load − ddraw)
                         ≥ min% · Σ_{peak∈month} load
                       Monthly tier — real contracts pair an annual floor with a
                       lower monthly floor so a good annual average cannot mask a
                       bad month.  Per-interval shortfalls are handled financially
                       by hourly_re_penetration_penalty, not as a hard gate.
peak_bess_discharge    Σ_{peak} η_d·dis ≥ min_annual_mwh · n_years
                       Firm BESS dispatch commitment in peak hours.
poi_capacity           sd[t] + wd[t] + η_d·dis[t] ≤ poi_mw   (per hour)
                       Physical CTU/STU connection limit, distinct from PPA cap.
sanctioned_demand      ddraw[t] ≤ demand_mw   (per hour)
                       Client's sanctioned grid connection ceiling.

Step 6b — easy Nice-to-have:
min_grid_drawal        Σ ddraw ≥ min_annual_mwh · n_years
                       Contractual minimum grid offtake (CSS / demand charge).
energy_purchase_cap    Σ(load − ddraw) ≤ max_annual_mwh · n_years
                       Buyer-side annual energy (MU) spend ceiling.
land_area              S·acre_s + W·acre_w ≤ available_acres
                       Finite site area (~4–5 acre/MW solar, ~0.5 acre/MW wind).

Not added here
──────────────
minimum_savings_npv  — report-only viability gate (handoff §9).  Enforcing it
  as a hard LP row would couple the row to the objective expression and its
  horizon-dependent 1e-7 scaling; instead it is checked post-solve in
  report/verify.  Its config still rides on ``opt_constraints`` for that check.
hourly_re_penetration_penalty — a *soft* objective term, not a hard row; it
  belongs in the objective and is a Step 6 follow-up.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext

_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)   # 365-day year


def _month_of_hour(n_hours: int = 8760) -> np.ndarray:
    """Map hour-of-year (0…8759) → month index 0…11 (non-leap calendar)."""
    out = np.full(n_hours, 11, dtype=int)
    h = 0
    for m, days in enumerate(_MONTH_DAYS):
        span = days * 24
        out[h:min(h + span, n_hours)] = m
        h += span
        if h >= n_hours:
            break
    return out


def add_optional_constraints(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
) -> None:
    """
    Attach the enabled toggleable constraints to *model* in-place.

    No-op when every toggle is disabled (the default), leaving the base model
    unchanged.
    """
    cfg     = params.opt_constraints
    eta_d   = params.eta_d
    n_years = int(tc.year_of.max())   # 1 (single) or project_life (full)

    # ── plant_cuf ─────────────────────────────────────────────────────────────
    # Plant CUF band on busbar export vs the contracted PPA rating over the
    # horizon.  Linear: RHS is (pct/100)·P·n_steps (P a var, rest constant).
    if cfg.plant_cuf_enabled:
        busbar = pyo.quicksum(
            model.sd[t] + model.wd[t] + eta_d * model.dis[t] for t in model.H
        )
        denom = model.P * tc.n_steps          # P · hours-in-horizon
        if cfg.plant_cuf_min_pct > 0.0:
            model.opt_plant_cuf_min = pyo.Constraint(
                expr=busbar >= (cfg.plant_cuf_min_pct / 100.0) * denom
            )
        if cfg.plant_cuf_max_pct < 100.0:
            model.opt_plant_cuf_max = pyo.Constraint(
                expr=busbar <= (cfg.plant_cuf_max_pct / 100.0) * denom
            )

    # ── minimum_bess_capacity ─────────────────────────────────────────────────
    # Physical floor on installed BESS energy (Year-1 SOH).  E_b = nb·cs.
    if cfg.min_bess_capacity_enabled and cfg.min_bess_capacity_mwh > 0.0:
        model.opt_min_bess_capacity = pyo.Constraint(
            expr=model.E_b >= cfg.min_bess_capacity_mwh
        )

    # ── minimum_bess_discharge ────────────────────────────────────────────────
    # Contractual throughput floor on busbar discharge energy.  Per-year floor
    # scaled by n_years so the horizon total is consistent in both modes.
    if cfg.min_bess_discharge_enabled and cfg.min_bess_discharge_annual_mwh > 0.0:
        discharge = pyo.quicksum(eta_d * model.dis[t] for t in model.H)
        model.opt_min_bess_discharge = pyo.Constraint(
            expr=discharge >= cfg.min_bess_discharge_annual_mwh * n_years
        )

    # ── re_penetration ────────────────────────────────────────────────────────
    # Band on the fraction of client load met by RE at the meter.  Σ meter
    # delivery = Σ load − Σ ddraw; Σ load is a constant, so both bounds are
    # linear in ddraw.  Horizon-agnostic (a ratio of like-scaled sums).
    if cfg.re_penetration_enabled:
        total_load = float(np.sum(params.load[tc.hour_of]))
        ddraw_sum  = pyo.quicksum(model.ddraw[t] for t in model.H)
        meter      = total_load - ddraw_sum
        if cfg.re_penetration_min_pct > 0.0:
            model.opt_re_penetration_min = pyo.Constraint(
                expr=meter >= (cfg.re_penetration_min_pct / 100.0) * total_load
            )
        if cfg.re_penetration_max_pct < 100.0:
            model.opt_re_penetration_max = pyo.Constraint(
                expr=meter <= (cfg.re_penetration_max_pct / 100.0) * total_load
            )

    # hour-of-day for peak-window matching (hour_of is hour-of-year 0…8759)
    hod = tc.hour_of % 24

    # ── peak_supply_obligation (RTC / FDRE) ───────────────────────────────────
    # RE meter delivery must cover ≥ min% of load within the peak hour set.
    if cfg.peak_supply_enabled and cfg.peak_supply_hours and cfg.peak_supply_min_pct > 0.0:
        peak_idx   = np.nonzero(np.isin(hod, np.asarray(cfg.peak_supply_hours)))[0]
        load_peak  = float(np.sum(params.load[tc.hour_of[peak_idx]]))
        meter_peak = load_peak - pyo.quicksum(model.ddraw[int(t)] for t in peak_idx)
        model.opt_peak_supply_min = pyo.Constraint(
            expr=meter_peak >= (cfg.peak_supply_min_pct / 100.0) * load_peak
        )

    # ── peak_supply_monthly (monthly tier of the peak obligation) ─────────────
    # RTC / FDRE contracts commonly pair an annual floor with a lower monthly
    # floor, so a good annual average cannot mask a bad month.  One constraint
    # per (year, month) that contains peak hours.
    if (cfg.peak_supply_monthly_enabled and cfg.peak_supply_monthly_hours
            and cfg.peak_supply_monthly_min_pct > 0.0):
        moy = _month_of_hour(len(params.load))[tc.hour_of]        # month per step
        pk  = np.isin(hod, np.asarray(cfg.peak_supply_monthly_hours))
        key = (tc.year_of - 1) * 12 + moy                          # distinct month
        frac = cfg.peak_supply_monthly_min_pct / 100.0

        groups: dict[int, np.ndarray] = {}
        loads:  dict[int, float] = {}
        for k in np.unique(key[pk]):
            idx = np.nonzero(pk & (key == k))[0]
            lo  = float(np.sum(params.load[tc.hour_of[idx]]))
            if lo > 0.0:
                groups[int(k)] = idx
                loads[int(k)]  = lo

        if groups:
            def _monthly(m, k: int) -> pyo.ConstraintData:
                idx = groups[k]
                meter = loads[k] - pyo.quicksum(m.ddraw[int(t)] for t in idx)
                return meter >= frac * loads[k]

            model.opt_peak_supply_monthly = pyo.Constraint(sorted(groups), rule=_monthly)

    # ── peak_bess_discharge (firm dispatch commitment) ────────────────────────
    if cfg.peak_discharge_enabled and cfg.peak_discharge_hours and cfg.peak_discharge_annual_mwh > 0.0:
        peak_idx  = np.nonzero(np.isin(hod, np.asarray(cfg.peak_discharge_hours)))[0]
        discharge = pyo.quicksum(eta_d * model.dis[int(t)] for t in peak_idx)
        model.opt_peak_discharge = pyo.Constraint(
            expr=discharge >= cfg.peak_discharge_annual_mwh * n_years
        )

    # ── poi_capacity (physical grid connection cap) ───────────────────────────
    if cfg.poi_enabled and cfg.poi_mw > 0.0:
        poi_mw = cfg.poi_mw

        @model.Constraint(model.H)
        def opt_poi_cap(m, t: int) -> pyo.ConstraintData:
            return m.sd[t] + m.wd[t] + eta_d * m.dis[t] <= poi_mw

    # ── sanctioned_demand (client grid ceiling) ───────────────────────────────
    if cfg.sanctioned_demand_enabled and cfg.sanctioned_demand_mw > 0.0:
        demand_mw = cfg.sanctioned_demand_mw

        @model.Constraint(model.H)
        def opt_sanctioned_demand(m, t: int) -> pyo.ConstraintData:
            return m.ddraw[t] <= demand_mw

    # ── min_grid_drawal (contractual minimum offtake) ─────────────────────────
    if cfg.min_grid_drawal_enabled and cfg.min_grid_drawal_annual_mwh > 0.0:
        model.opt_min_grid_drawal = pyo.Constraint(
            expr=pyo.quicksum(model.ddraw[t] for t in model.H)
            >= cfg.min_grid_drawal_annual_mwh * n_years
        )

    # ── energy_purchase_cap (buyer annual MU ceiling) ─────────────────────────
    if cfg.energy_purchase_cap_enabled and cfg.energy_purchase_cap_annual_mwh > 0.0:
        total_load = float(np.sum(params.load[tc.hour_of]))
        meter      = total_load - pyo.quicksum(model.ddraw[t] for t in model.H)
        model.opt_energy_purchase_cap = pyo.Constraint(
            expr=meter <= cfg.energy_purchase_cap_annual_mwh * n_years
        )

    # ── land_area (finite site) ───────────────────────────────────────────────
    if cfg.land_area_enabled and cfg.land_available_acres > 0.0:
        model.opt_land_area = pyo.Constraint(
            expr=model.S * cfg.land_solar_acre_per_mw
            + model.W * cfg.land_wind_acre_per_mw
            <= cfg.land_available_acres
        )

    # ── strict_charge_discharge (D7 exclusivity, big-M binary) ────────────────
    # Efficiency losses already deter simultaneous charge+discharge, but a
    # degenerate optimum can still return both (verify surfaces this).  This
    # toggle enforces it hard with a per-hour binary u: u=1 -> only charge,
    # u=0 -> only discharge.  It turns the model into a genuine MILP (a binary
    # per timestep), so enable only when full exclusivity is required.
    if cfg.strict_cd_enabled:
        m_chg = params.crc * params.nb_max * params.cs   # >= max chg (C9 bound)
        m_dis = params.crd * params.nb_max * params.cs   # >= max dis (C10 bound)
        model.u_cd = pyo.Var(model.H, domain=pyo.Binary)

        @model.Constraint(model.H)
        def opt_strict_charge(m, t: int) -> pyo.ConstraintData:
            return m.chg[t] <= m_chg * m.u_cd[t]

        @model.Constraint(model.H)
        def opt_strict_discharge(m, t: int) -> pyo.ConstraintData:
            return m.dis[t] <= m_dis * (1 - m.u_cd[t])
