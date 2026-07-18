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
