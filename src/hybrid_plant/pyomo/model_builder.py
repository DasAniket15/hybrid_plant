"""
model_builder.py
────────────────
Constructs a Pyomo ConcreteModel implementing the 25-year MILP described in
the spec (sections 2–7).

Call build_model(params) to get a model ready to pass to solver_runner.py.
Augmentation variables are included only when the corresponding
aug_*_enabled flags are True in params.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo


def build_model(params: dict[str, Any]) -> pyo.ConcreteModel:
    """
    Construct the full Pyomo MILP.

    Parameters
    ----------
    params : dict from parameter_builder.build_parameters()

    Returns
    -------
    pyomo.environ.ConcreteModel — unsolved model
    """
    Y = params["project_life_years"]
    T = params["hours_per_year"]

    m = pyo.ConcreteModel()

    # ─────────────────────────────────────────────────────────────────────────
    # §2  Sets
    # ─────────────────────────────────────────────────────────────────────────
    m.Y = pyo.RangeSet(1, Y)
    m.T = pyo.RangeSet(1, T)

    # ─────────────────────────────────────────────────────────────────────────
    # §4a  Initial sizing variables
    # ─────────────────────────────────────────────────────────────────────────
    m.solar_mw_0 = pyo.Var(
        bounds=(params["solar_mw_min"], params["solar_mw_max"]),
        domain=pyo.NonNegativeReals,
    )
    m.wind_mw_0 = pyo.Var(
        bounds=(params["wind_mw_min"], params["wind_mw_max"]),
        domain=pyo.NonNegativeReals,
    )
    m.bess_cap_0 = pyo.Var(
        bounds=(params["bess_cap_min"], params["bess_cap_max"]),
        domain=pyo.NonNegativeReals,
    )
    m.ppa_mw = pyo.Var(
        bounds=(params["ppa_mw_min"], params["ppa_mw_max"]),
        domain=pyo.NonNegativeReals,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §4b  C-rate variables (initial cohort)
    # ─────────────────────────────────────────────────────────────────────────
    m.c_rate_charge_0 = pyo.Var(
        bounds=(params["c_rate_min"], params["c_rate_max"]),
        domain=pyo.NonNegativeReals,
    )
    m.c_rate_discharge_0 = pyo.Var(
        bounds=(params["c_rate_min"], params["c_rate_max"]),
        domain=pyo.NonNegativeReals,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §4c  McCormick auxiliary variables (initial cohort)
    # ─────────────────────────────────────────────────────────────────────────
    bess_cap_max = params["bess_cap_max"]
    m.z_charge_0 = pyo.Var(
        bounds=(0.0, bess_cap_max),
        domain=pyo.NonNegativeReals,
    )
    m.z_discharge_0 = pyo.Var(
        bounds=(0.0, bess_cap_max),
        domain=pyo.NonNegativeReals,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §4d / §4e  Augmentation variables
    # ─────────────────────────────────────────────────────────────────────────
    aug_solar  = params["aug_solar_enabled"]
    aug_wind   = params["aug_wind_enabled"]
    aug_bess   = params["aug_bess_enabled"]
    d_solar_max = params["delta_solar_mw_max"]
    d_wind_max  = params["delta_wind_mw_max"]
    d_bess_max  = params["delta_bess_cap_max"]

    if aug_solar:
        m.delta_solar_mw = pyo.Var(m.Y, bounds=(0.0, d_solar_max), domain=pyo.NonNegativeReals)
        m.y_solar        = pyo.Var(m.Y, domain=pyo.Binary)
    if aug_wind:
        m.delta_wind_mw  = pyo.Var(m.Y, bounds=(0.0, d_wind_max),  domain=pyo.NonNegativeReals)
        m.y_wind         = pyo.Var(m.Y, domain=pyo.Binary)
    if aug_bess:
        m.delta_bess_cap = pyo.Var(m.Y, bounds=(0.0, d_bess_max),  domain=pyo.NonNegativeReals)
        m.y_bess         = pyo.Var(m.Y, domain=pyo.Binary)
        m.c_rate_charge_aug    = pyo.Var(m.Y, bounds=(0.0, params["c_rate_max"]), domain=pyo.NonNegativeReals)
        m.c_rate_discharge_aug = pyo.Var(m.Y, bounds=(0.0, params["c_rate_max"]), domain=pyo.NonNegativeReals)
        m.z_charge_aug    = pyo.Var(m.Y, bounds=(0.0, d_bess_max), domain=pyo.NonNegativeReals)
        m.z_discharge_aug = pyo.Var(m.Y, bounds=(0.0, d_bess_max), domain=pyo.NonNegativeReals)

    # ─────────────────────────────────────────────────────────────────────────
    # §4f  Dispatch variables
    # ─────────────────────────────────────────────────────────────────────────
    m.gen_solar   = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.gen_wind    = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.charge      = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.discharge   = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.curtail     = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.soc         = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)
    m.discom      = pyo.Var(m.Y, m.T, domain=pyo.NonNegativeReals)

    # ─────────────────────────────────────────────────────────────────────────
    # Local aliases
    # ─────────────────────────────────────────────────────────────────────────
    solar_eff   = params["solar_eff"]
    wind_eff    = params["wind_eff"]
    bess_soh    = params["bess_soh"]
    aug_bess_s  = params["aug_bess_soh"]
    solar_cuf   = params["solar_cuf"]   # 0-indexed numpy array
    wind_cuf    = params["wind_cuf"]
    load_mwh    = params["load_mwh"]
    tariff      = params["tariff_array"]
    loss_factor = params["loss_factor"]
    charge_eff  = params["charge_eff"]
    disc_eff    = params["discharge_eff"]
    container_sz = params["container_size_mwh"]
    aux_hr      = params["aux_mwh_per_hour"]

    # ─────────────────────────────────────────────────────────────────────────
    # §5a  Cumulative capacity expressions
    # ─────────────────────────────────────────────────────────────────────────
    def _solar_mw(y):
        base = m.solar_mw_0
        if aug_solar:
            return base + sum(m.delta_solar_mw[s] for s in range(1, y + 1))
        return base

    def _wind_mw(y):
        base = m.wind_mw_0
        if aug_wind:
            return base + sum(m.delta_wind_mw[s] for s in range(1, y + 1))
        return base

    def _bess_cap(y):
        base = m.bess_cap_0
        if aug_bess:
            return base + sum(m.delta_bess_cap[s] for s in range(1, y + 1))
        return base

    # ─────────────────────────────────────────────────────────────────────────
    # §5b  Effective (degraded) capacity
    # ─────────────────────────────────────────────────────────────────────────
    def _eff_solar_mw(y):
        expr = m.solar_mw_0 * solar_eff[y]
        if aug_solar and y >= 2:
            expr = expr + sum(m.delta_solar_mw[s] * params["aug_solar_eff"][y - s + 1]
                              for s in range(1, y))
        return expr

    def _eff_wind_mw(y):
        expr = m.wind_mw_0 * wind_eff[y]
        if aug_wind and y >= 2:
            expr = expr + sum(m.delta_wind_mw[s] * params["aug_wind_eff"][y - s + 1]
                              for s in range(1, y))
        return expr

    def _eff_bess_cap(y):
        expr = m.bess_cap_0 * bess_soh[y]
        if aug_bess and y >= 2:
            expr = expr + sum(m.delta_bess_cap[s] * aug_bess_s[y - s + 1]
                              for s in range(1, y))
        return expr

    # ─────────────────────────────────────────────────────────────────────────
    # §5c  Effective BESS power caps (via z-variables)
    # ─────────────────────────────────────────────────────────────────────────
    def _charge_pw(y):
        expr = bess_soh[y] * m.z_charge_0
        if aug_bess and y >= 2:
            expr = expr + sum(aug_bess_s[y - s + 1] * m.z_charge_aug[s]
                              for s in range(1, y))
        return expr

    def _discharge_pw(y):
        expr = bess_soh[y] * m.z_discharge_0
        if aug_bess and y >= 2:
            expr = expr + sum(aug_bess_s[y - s + 1] * m.z_discharge_aug[s]
                              for s in range(1, y))
        return expr

    # ─────────────────────────────────────────────────────────────────────────
    # §5d  Aux BESS consumption per hour
    # ─────────────────────────────────────────────────────────────────────────
    def _aux_mwh(y):
        return aux_hr * (_eff_bess_cap(y) / container_sz)

    # ─────────────────────────────────────────────────────────────────────────
    # §5f  CAPEX
    # ─────────────────────────────────────────────────────────────────────────
    solar_dc_mwp_expr = m.solar_mw_0 * params["ac_dc_ratio"]
    CAPEX = (
        solar_dc_mwp_expr         * params["capex_solar_rs_per_mwp"]
        + m.wind_mw_0              * params["capex_wind_rs_per_mw"]
        + m.bess_cap_0             * params["capex_bess_rs_per_mwh"]
        + params["transmission_capex_rs"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §5g  Debt service
    # ─────────────────────────────────────────────────────────────────────────
    debt_amount = CAPEX * params["debt_frac"]
    annuity     = params["annuity_factor"]
    debt_tenure = params["debt_tenure"]

    def _debt_service(y):
        return debt_amount * annuity if y <= debt_tenure else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # §5h  ROE payment
    # ─────────────────────────────────────────────────────────────────────────
    equity_amount = CAPEX * params["equity_frac"]
    roe_payment   = equity_amount * params["roe"]

    # ─────────────────────────────────────────────────────────────────────────
    # §5i  OPEX
    # ─────────────────────────────────────────────────────────────────────────
    def _opex(y):
        solar_dc_y = _solar_mw(y) * params["ac_dc_ratio"]
        base = (
            solar_dc_y   * params["solar_om_rate_rs_per_mwp"]  * params["solar_om_esc_factor"][y]
            + _wind_mw(y)  * params["wind_om_rate_rs_per_mw"]    * params["wind_om_esc_factor"][y]
            + params["land_lease_base_rs_per_yr"]                * params["land_lease_esc_factor"][y]
            + _bess_cap(y) * params["bess_om_rate_rs_per_mwh"]
            + solar_dc_y   * params["solar_trans_om_rs_per_mwp"]
            + _wind_mw(y)  * params["wind_trans_om_rs_per_mw"]
            + params["insurance_frac"] * CAPEX
        )
        if not aug_solar and not aug_wind and not aug_bess:
            return base

        # Augmentation CAPEX expensed in event year
        aug_capex_y = 0.0
        if aug_solar:
            aug_capex_y = aug_capex_y + m.delta_solar_mw[y] * params["aug_capex_solar_rs_per_mw"]
        if aug_wind:
            aug_capex_y = aug_capex_y + m.delta_wind_mw[y]  * params["aug_capex_wind_rs_per_mw"]
        if aug_bess:
            aug_capex_y = aug_capex_y + m.delta_bess_cap[y] * params["aug_capex_bess_rs_per_mwh"]

        # Augmentation O&M on all prior+current augmented capacity
        aug_om_y = 0.0
        if aug_solar:
            aug_om_y = aug_om_y + sum(m.delta_solar_mw[s] for s in range(1, y + 1)) * params["aug_om_solar_rs_per_mw_yr"]
        if aug_wind:
            aug_om_y = aug_om_y + sum(m.delta_wind_mw[s]  for s in range(1, y + 1)) * params["aug_om_wind_rs_per_mw_yr"]
        if aug_bess:
            aug_om_y = aug_om_y + sum(m.delta_bess_cap[s] for s in range(1, y + 1)) * params["aug_om_bess_rs_per_mwh_yr"]

        return base + aug_capex_y + aug_om_y

    # ─────────────────────────────────────────────────────────────────────────
    # §5k  Revenue
    # ─────────────────────────────────────────────────────────────────────────
    def _direct(y, t):
        # direct[y,t] = gen_solar + gen_wind - charge - curtail
        return m.gen_solar[y, t] + m.gen_wind[y, t] - m.charge[y, t] - m.curtail[y, t]

    # ─────────────────────────────────────────────────────────────────────────
    # §6a  Generation bounds
    # ─────────────────────────────────────────────────────────────────────────
    def _gen_solar_ub(m_, y, t):
        # solar_cuf is 0-indexed numpy array; t is 1-indexed
        return m_.gen_solar[y, t] <= _eff_solar_mw(y) * float(solar_cuf[t - 1])

    def _gen_wind_ub(m_, y, t):
        return m_.gen_wind[y, t] <= _eff_wind_mw(y) * float(wind_cuf[t - 1])

    m.con_gen_solar_ub = pyo.Constraint(m.Y, m.T, rule=_gen_solar_ub)
    m.con_gen_wind_ub  = pyo.Constraint(m.Y, m.T, rule=_gen_wind_ub)

    # ─────────────────────────────────────────────────────────────────────────
    # §6a-extra  PPA capacity == installed AC capacity (solar + wind)
    # Capacity charges scale with ppa_mw; without this the solver exploits the
    # unconstrained upper-bound to eliminate the PPA cap constraint for free.
    # ─────────────────────────────────────────────────────────────────────────
    m.con_ppa_equals_installed = pyo.Constraint(
        expr=m.ppa_mw == m.solar_mw_0 + m.wind_mw_0
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §6b  Non-negative direct (gen ≥ charge + curtail)
    # ─────────────────────────────────────────────────────────────────────────
    def _direct_nonneg(m_, y, t):
        return m_.gen_solar[y, t] + m_.gen_wind[y, t] - m_.charge[y, t] - m_.curtail[y, t] >= 0.0

    m.con_direct_nonneg = pyo.Constraint(m.Y, m.T, rule=_direct_nonneg)

    # ─────────────────────────────────────────────────────────────────────────
    # §6c  PPA cap
    # ─────────────────────────────────────────────────────────────────────────
    def _ppa_cap(m_, y, t):
        return (
            m_.gen_solar[y, t] + m_.gen_wind[y, t] - m_.charge[y, t] - m_.curtail[y, t]
            + m_.discharge[y, t] * disc_eff
            <= m_.ppa_mw
        )

    m.con_ppa_cap = pyo.Constraint(m.Y, m.T, rule=_ppa_cap)

    # ─────────────────────────────────────────────────────────────────────────
    # §6d  Load balance at client meter
    # ─────────────────────────────────────────────────────────────────────────
    def _load_balance(m_, y, t):
        return (
            (m_.gen_solar[y, t] + m_.gen_wind[y, t] - m_.charge[y, t] - m_.curtail[y, t]
             + m_.discharge[y, t] * disc_eff) * loss_factor
            + m_.discom[y, t]
            == float(load_mwh[t - 1])
        )

    m.con_load_balance = pyo.Constraint(m.Y, m.T, rule=_load_balance)

    # ─────────────────────────────────────────────────────────────────────────
    # §6e  BESS charge/discharge power bounds
    # ─────────────────────────────────────────────────────────────────────────
    def _charge_ub(m_, y, t):
        return m_.charge[y, t] <= _charge_pw(y)

    def _discharge_ub(m_, y, t):
        return m_.discharge[y, t] <= _discharge_pw(y)

    m.con_charge_ub    = pyo.Constraint(m.Y, m.T, rule=_charge_ub)
    m.con_discharge_ub = pyo.Constraint(m.Y, m.T, rule=_discharge_ub)

    # ─────────────────────────────────────────────────────────────────────────
    # §6f  SOC energy capacity bound
    # ─────────────────────────────────────────────────────────────────────────
    def _soc_ub(m_, y, t):
        return m_.soc[y, t] <= _eff_bess_cap(y)

    m.con_soc_ub = pyo.Constraint(m.Y, m.T, rule=_soc_ub)

    # ─────────────────────────────────────────────────────────────────────────
    # §6g  SOC dynamics
    # ─────────────────────────────────────────────────────────────────────────
    soc_init_y1 = 0.0

    def _soc_dynamics(m_, y, t):
        aux_y = _aux_mwh(y)
        if y == 1 and t == 1:
            prev_soc = soc_init_y1
        elif t == 1:
            prev_soc = m_.soc[y - 1, T]
        else:
            prev_soc = m_.soc[y, t - 1]
        return (
            m_.soc[y, t]
            == prev_soc
            + m_.charge[y, t] * charge_eff
            - m_.discharge[y, t]
            - aux_y
        )

    m.con_soc_dynamics = pyo.Constraint(m.Y, m.T, rule=_soc_dynamics)

    # ─────────────────────────────────────────────────────────────────────────
    # §6h  Augmentation linking (big-M)
    # ─────────────────────────────────────────────────────────────────────────
    if aug_solar:
        def _aug_solar_link(m_, s):
            return m_.delta_solar_mw[s] <= d_solar_max * m_.y_solar[s]
        m.con_aug_solar_link = pyo.Constraint(m.Y, rule=_aug_solar_link)

    if aug_wind:
        def _aug_wind_link(m_, s):
            return m_.delta_wind_mw[s] <= d_wind_max * m_.y_wind[s]
        m.con_aug_wind_link = pyo.Constraint(m.Y, rule=_aug_wind_link)

    if aug_bess:
        def _aug_bess_link(m_, s):
            return m_.delta_bess_cap[s] <= d_bess_max * m_.y_bess[s]
        m.con_aug_bess_link = pyo.Constraint(m.Y, rule=_aug_bess_link)

    # ─────────────────────────────────────────────────────────────────────────
    # §6i  CUF maintenance (busbar_mwh[y] >= busbar_mwh[1] for y >= 2)
    # ─────────────────────────────────────────────────────────────────────────
    if params.get("cuf_maintenance_enabled", True) and Y >= 2:
        def _busbar_y(y_val):
            return sum(
                m.gen_solar[y_val, t] + m.gen_wind[y_val, t]
                - m.charge[y_val, t] - m.curtail[y_val, t]
                + m.discharge[y_val, t] * disc_eff
                for t in range(1, T + 1)
            )

        def _cuf_maintain(m_, y):
            return _busbar_y(y) >= _busbar_y(1)

        m.con_cuf_maintain = pyo.Constraint(
            pyo.RangeSet(2, Y),
            rule=_cuf_maintain,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §6j  McCormick envelope constraints — initial cohort
    # ─────────────────────────────────────────────────────────────────────────
    eL = params["bess_cap_min"]
    eU = params["bess_cap_max"]

    # z_charge_0 = c_rate_charge_0 × bess_cap_0
    m.mc_zc0_1 = pyo.Constraint(expr=m.z_charge_0 >= m.bess_cap_0 + eU * m.c_rate_charge_0 - eU)
    m.mc_zc0_2 = pyo.Constraint(expr=m.z_charge_0 <= m.bess_cap_0 + eL * m.c_rate_charge_0 - eL)
    m.mc_zc0_3 = pyo.Constraint(expr=m.z_charge_0 <= eU * m.c_rate_charge_0)

    # z_discharge_0 = c_rate_discharge_0 × bess_cap_0
    m.mc_zd0_1 = pyo.Constraint(expr=m.z_discharge_0 >= m.bess_cap_0 + eU * m.c_rate_discharge_0 - eU)
    m.mc_zd0_2 = pyo.Constraint(expr=m.z_discharge_0 <= m.bess_cap_0 + eL * m.c_rate_discharge_0 - eL)
    m.mc_zd0_3 = pyo.Constraint(expr=m.z_discharge_0 <= eU * m.c_rate_discharge_0)

    # ─────────────────────────────────────────────────────────────────────────
    # §6j  McCormick envelope constraints — augmented cohorts
    # ─────────────────────────────────────────────────────────────────────────
    if aug_bess:
        eL_aug = 0.0
        eU_aug = params["delta_bess_cap_max"]

        def _mc_zca_1(m_, s):
            return m_.z_charge_aug[s] >= m_.delta_bess_cap[s] + eU_aug * m_.c_rate_charge_aug[s] - eU_aug
        def _mc_zca_2(m_, s):
            return m_.z_charge_aug[s] <= m_.delta_bess_cap[s]
        def _mc_zca_3(m_, s):
            return m_.z_charge_aug[s] <= eU_aug * m_.c_rate_charge_aug[s]

        def _mc_zda_1(m_, s):
            return m_.z_discharge_aug[s] >= m_.delta_bess_cap[s] + eU_aug * m_.c_rate_discharge_aug[s] - eU_aug
        def _mc_zda_2(m_, s):
            return m_.z_discharge_aug[s] <= m_.delta_bess_cap[s]
        def _mc_zda_3(m_, s):
            return m_.z_discharge_aug[s] <= eU_aug * m_.c_rate_discharge_aug[s]

        m.con_mc_zca_1 = pyo.Constraint(m.Y, rule=_mc_zca_1)
        m.con_mc_zca_2 = pyo.Constraint(m.Y, rule=_mc_zca_2)
        m.con_mc_zca_3 = pyo.Constraint(m.Y, rule=_mc_zca_3)
        m.con_mc_zda_1 = pyo.Constraint(m.Y, rule=_mc_zda_1)
        m.con_mc_zda_2 = pyo.Constraint(m.Y, rule=_mc_zda_2)
        m.con_mc_zda_3 = pyo.Constraint(m.Y, rule=_mc_zda_3)

    # ─────────────────────────────────────────────────────────────────────────
    # §7  Objective: maximise discounted NPV (revenue - opex - debt - roe)
    # ─────────────────────────────────────────────────────────────────────────
    disc = params["discount_factor"]

    def _revenue(y):
        return sum(
            float(tariff[t - 1])
            * (m.gen_solar[y, t] + m.gen_wind[y, t] - m.charge[y, t] - m.curtail[y, t]
               + m.discharge[y, t] * disc_eff)
            * loss_factor
            for t in range(1, T + 1)
        )

    def _obj_rule(m_):
        return sum(
            disc[y] * (_revenue(y) - _opex(y) - _debt_service(y) - roe_payment)
            for y in range(1, Y + 1)
        )

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.maximize)

    return m
