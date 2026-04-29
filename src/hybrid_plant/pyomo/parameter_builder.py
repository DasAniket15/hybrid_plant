"""
parameter_builder.py
────────────────────
Loads all YAML configs and CSV degradation/profile data into flat Python
dicts that are passed directly to model_builder.py.

No Pyomo objects are created here — pure Python/NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import (
    CRORE_TO_RS,
    LAKH_TO_RS,
    MONTHS_PER_YEAR,
    PERCENT_TO_DECIMAL,
)
from hybrid_plant.data_loader import load_timeseries_data, operating_value
from hybrid_plant.energy.grid_interface import GridInterface


def build_parameters(config: FullConfig) -> dict[str, Any]:
    """
    Build and return the complete flat parameter dict for the Pyomo model.

    Returns
    -------
    dict with keys matching all §3 parameter names from the spec.
    """
    data = load_timeseries_data(config)
    return _build_from_data(config, data)


def build_parameters_with_data(config: FullConfig, data: dict[str, Any]) -> dict[str, Any]:
    """Build parameters when timeseries data is already loaded."""
    return _build_from_data(config, data)


def _build_from_data(config: FullConfig, data: dict[str, Any]) -> dict[str, Any]:
    p: dict[str, Any] = {}

    project_cfg   = config.project["project"]
    sim_cfg       = config.project["simulation"]
    bess_cfg      = config.bess["bess"]
    fin_cfg       = config.finance
    reg_cfg       = config.regulatory["regulatory"]
    tariff_cfg    = config.tariffs["discom"]
    solver_cfg    = config.solver

    # ── §3a  Hourly profiles ──────────────────────────────────────────────────
    p["solar_cuf"]  = data["solar_cuf"]           # shape (8760,), fraction
    p["wind_cuf"]   = data["wind_cuf"]            # shape (8760,), fraction
    p["load_mwh"]   = data["load_profile"]        # shape (8760,), MWh — already converted in data_loader

    # ── §3b  Degradation curves ───────────────────────────────────────────────
    years        = project_cfg["project_life_years"]
    hours        = sim_cfg["hours_per_year"]

    solar_deg_df = data["solar_degradation_curve"]
    wind_deg_df  = data["wind_degradation_curve"]
    bess_soh_df  = data["bess_soh_curve"]

    solar_deg_curve = {int(row.iloc[0]): float(row.iloc[1]) for _, row in solar_deg_df.iterrows()}
    wind_deg_curve  = {int(row.iloc[0]): float(row.iloc[1]) for _, row in wind_deg_df.iterrows()}
    bess_soh_curve  = {int(row.iloc[0]): float(row.iloc[1]) for _, row in bess_soh_df.iterrows()}

    # solar_eff[y], wind_eff[y], bess_soh[y] indexed 1..years
    p["solar_eff"]  = {y: operating_value(solar_deg_curve, y) for y in range(1, years + 1)}
    p["wind_eff"]   = {y: operating_value(wind_deg_curve,  y) for y in range(1, years + 1)}
    p["bess_soh"]   = {y: operating_value(bess_soh_curve,  y) for y in range(1, years + 1)}

    # aug_*_eff[a] = same curves indexed by cohort age a = 1..years
    p["aug_solar_eff"] = p["solar_eff"]   # same dict; age indexing identical to year indexing
    p["aug_wind_eff"]  = p["wind_eff"]
    p["aug_bess_soh"]  = p["bess_soh"]

    # ── §3c  BESS hardware ────────────────────────────────────────────────────
    container = bess_cfg["container"]
    eff       = bess_cfg["efficiency"]

    p["container_size_mwh"]  = float(container["size_mwh"])
    p["aux_mwh_per_hour"]    = float(container["auxiliary_consumption_mwh_per_day"]) / 24.0
    p["charge_eff"]          = float(eff["charge_efficiency"])
    p["discharge_eff"]       = float(eff["discharge_efficiency"])

    # ── §3d  Grid loss factor ─────────────────────────────────────────────────
    p["loss_factor"] = GridInterface(config).loss_factor

    # ── §3e  Financial parameters ─────────────────────────────────────────────
    capex_cfg = fin_cfg["capex"]
    fin       = fin_cfg["financing"]
    debt      = fin["debt"]
    equity    = fin["equity"]

    p["capex_solar_rs_per_mwp"]  = float(capex_cfg["solar"]["cost_per_mwp"])
    p["ac_dc_ratio"]             = float(capex_cfg["solar"]["ac_dc_ratio"])
    p["capex_wind_rs_per_mw"]    = float(capex_cfg["wind"]["cost_per_mw"])
    p["capex_bess_rs_per_mwh"]   = float(capex_cfg["bess"]["cost_per_mwh"])
    p["transmission_capex_rs"]   = (
        float(capex_cfg["transmission"]["length_km"])
        * float(capex_cfg["transmission"]["cost_per_km"])
    )

    p["debt_frac"]    = fin["debt_percent"]   * PERCENT_TO_DECIMAL
    p["equity_frac"]  = fin["equity_percent"] * PERCENT_TO_DECIMAL
    p["debt_rate"]    = debt["interest_rate_percent"] * PERCENT_TO_DECIMAL
    p["debt_tenure"]  = int(debt["tenure_years"])
    p["roe"]          = equity["return_on_equity_percent"] * PERCENT_TO_DECIMAL
    p["tax_rate"]     = fin["corporate_tax_rate_percent"]  * PERCENT_TO_DECIMAL

    wacc = (
        p["debt_frac"]   * p["debt_rate"] * (1.0 - p["tax_rate"])
        + p["equity_frac"] * p["roe"]
    )
    p["wacc"] = wacc

    r  = p["debt_rate"]
    n  = p["debt_tenure"]
    annuity = r * (1 + r) ** n / ((1 + r) ** n - 1)
    p["annuity_factor"] = annuity

    p["discount_factor"] = {y: 1.0 / (1.0 + wacc) ** y for y in range(1, years + 1)}

    # ── §3f  OPEX parameters ──────────────────────────────────────────────────
    opex_cfg = fin_cfg["opex"]

    p["solar_om_rate_rs_per_mwp"]  = opex_cfg["solar"]["rate_lakh_per_mwp"]       * LAKH_TO_RS
    p["solar_om_esc"]              = opex_cfg["solar"]["escalation_percent"]       * PERCENT_TO_DECIMAL
    p["wind_om_rate_rs_per_mw"]    = opex_cfg["wind"]["rate_lakh_per_mw"]          * LAKH_TO_RS
    p["wind_om_esc"]               = opex_cfg["wind"]["escalation_percent"]        * PERCENT_TO_DECIMAL
    p["land_lease_base_rs_per_yr"] = (
        opex_cfg["land_lease"]["base_monthly_cost_crore"] * CRORE_TO_RS * MONTHS_PER_YEAR
    )
    p["land_lease_esc"]            = opex_cfg["land_lease"]["escalation_percent"]  * PERCENT_TO_DECIMAL
    p["bess_om_rate_rs_per_mwh"]   = opex_cfg["bess"]["rate_lakh_per_mwh"]         * LAKH_TO_RS
    p["solar_trans_om_rs_per_mwp"] = opex_cfg["solar_transmission"]["rate_lakh_per_mwp"] * LAKH_TO_RS
    p["wind_trans_om_rs_per_mw"]   = opex_cfg["wind_transmission"]["rate_lakh_per_mw"]   * LAKH_TO_RS
    p["insurance_frac"]            = opex_cfg["insurance"]["percent_of_total_capex"] * PERCENT_TO_DECIMAL

    p["solar_om_esc_factor"]   = {y: (1 + p["solar_om_esc"])   ** (y - 1) for y in range(1, years + 1)}
    p["wind_om_esc_factor"]    = {y: (1 + p["wind_om_esc"])    ** (y - 1) for y in range(1, years + 1)}
    p["land_lease_esc_factor"] = {y: (1 + p["land_lease_esc"]) ** (y - 1) for y in range(1, years + 1)}

    # ── §3g  Tariff parameters ────────────────────────────────────────────────
    # Build tariff[t] for t in 1..8760, Rs/MWh
    # With ht_lt_split=0: 100% LT
    split = reg_cfg["connection"]["ht_lt_split_percent"] * PERCENT_TO_DECIMAL

    def _rate_array(side_key: str) -> np.ndarray:
        rate = np.zeros(8760)
        for period_name, period in tariff_cfg[side_key]["tod_periods"].items():
            # tariffs.yaml hours are 1-indexed; hod = hour - 1 (0-indexed)
            for h in period["hours"]:
                hod = h - 1   # 0-indexed hour-of-day
                # Set for all occurrences of this hod across 8760 hours
                for t_idx in range(hod, 8760, 24):
                    rate[t_idx] = period["rate_inr_per_kwh"] * 1000.0  # Rs/MWh
        return rate

    lt_rate = _rate_array("lt")
    ht_rate = _rate_array("ht")
    # tariff[t] as 0-indexed numpy array; model uses 1-indexed but we store 0-indexed internally
    p["tariff_array"] = split * ht_rate + (1.0 - split) * lt_rate  # shape (8760,), Rs/MWh

    # ── §3h  Regulatory charges ───────────────────────────────────────────────
    reg_ch    = fin_cfg["regulatory_charges"]
    lt_ch     = reg_ch["lt"]
    # With split=0, use LT only
    p["capacity_charge_rs_per_mw_per_month"] = (
        lt_ch["ctu_charge_inr_per_mw_per_month"]
        + lt_ch["stu_charge_inr_per_mw_per_month"]
        + lt_ch["sldc_charge_inr_per_mw_per_month"]
    )
    p["wheeling_rs_per_kwh"]        = lt_ch["wheeling_charge_inr_per_kwh"]
    p["electricity_tax_rs_per_kwh"] = lt_ch["electricity_tax_inr_per_kwh"]
    p["banking_rs_per_kwh"]         = lt_ch["banking_charge_inr_per_kwh"]

    # ── §3i  Solver variable bounds ───────────────────────────────────────────
    dv = solver_cfg["solver"]["decision_variables"]

    p["solar_mw_min"]  = float(dv["solar_capacity_mw"]["min"])
    p["solar_mw_max"]  = float(dv["solar_capacity_mw"]["max"])
    p["wind_mw_min"]   = float(dv["wind_capacity_mw"]["min"])
    p["wind_mw_max"]   = float(dv["wind_capacity_mw"]["max"])
    p["ppa_mw_min"]    = float(dv["ppa_capacity_mw"]["min"])
    p["ppa_mw_max"]    = float(dv["ppa_capacity_mw"]["max"])

    containers_min     = float(dv["bess_containers"]["min"])
    containers_max     = float(dv["bess_containers"]["max"])
    p["bess_cap_min"]  = containers_min * p["container_size_mwh"]
    p["bess_cap_max"]  = containers_max * p["container_size_mwh"]

    p["c_rate_min"]    = float(dv["bess_charge_c_rate"]["min"])
    p["c_rate_max"]    = float(dv["bess_charge_c_rate"]["max"])

    # ── §3j  Augmentation parameters ─────────────────────────────────────────
    pyomo_cfg = solver_cfg.get("pyomo", {})
    aug_cfg   = pyomo_cfg.get("augmentation", {})

    p["aug_solar_enabled"] = aug_cfg.get("solar", {}).get("enabled", False)
    p["aug_wind_enabled"]  = aug_cfg.get("wind",  {}).get("enabled", False)
    p["aug_bess_enabled"]  = aug_cfg.get("bess",  {}).get("enabled", False)

    p["aug_capex_solar_rs_per_mw"]  = float(aug_cfg.get("solar", {}).get("capex_rs_per_mw",  p["capex_solar_rs_per_mwp"]))
    p["aug_capex_wind_rs_per_mw"]   = float(aug_cfg.get("wind",  {}).get("capex_rs_per_mw",  p["capex_wind_rs_per_mw"]))
    p["aug_capex_bess_rs_per_mwh"]  = float(aug_cfg.get("bess",  {}).get("capex_rs_per_mwh", p["capex_bess_rs_per_mwh"]))

    p["aug_om_solar_rs_per_mw_yr"]  = float(aug_cfg.get("solar", {}).get("om_rs_per_mw_per_year",  1.0e5))
    p["aug_om_wind_rs_per_mw_yr"]   = float(aug_cfg.get("wind",  {}).get("om_rs_per_mw_per_year",  1.2e6))
    p["aug_om_bess_rs_per_mwh_yr"]  = float(aug_cfg.get("bess",  {}).get("om_rs_per_mwh_per_year", 6.0e4))

    p["delta_solar_mw_max"] = p["solar_mw_max"]
    p["delta_wind_mw_max"]  = p["wind_mw_max"]
    p["delta_bess_cap_max"] = p["bess_cap_max"]

    # ── Scalar counts ─────────────────────────────────────────────────────────
    p["project_life_years"] = years
    p["hours_per_year"]     = hours

    # ── Pyomo solver settings ─────────────────────────────────────────────────
    p["pyomo_solver"]       = pyomo_cfg.get("solver", "highs")
    p["time_limit_seconds"] = pyomo_cfg.get("time_limit_seconds", 7200)
    p["mip_gap"]            = pyomo_cfg.get("mip_gap", 0.005)
    p["cuf_maintenance_enabled"] = pyomo_cfg.get(
        "cuf_maintenance_enabled",
        p["aug_solar_enabled"] or p["aug_wind_enabled"] or p["aug_bess_enabled"],
    )

    # average DISCOM tariff (load-weighted) for savings calculations
    load_arr  = p["load_mwh"]
    tariff_rs = p["tariff_array"]
    total_load = float(np.sum(load_arr))
    p["discom_tariff_avg_rs_per_mwh"] = (
        float(np.sum(tariff_rs * load_arr)) / total_load if total_load > 0 else 0.0
    )
    p["discom_tariff_avg_rs_per_kwh"] = p["discom_tariff_avg_rs_per_mwh"] / 1000.0

    return p
