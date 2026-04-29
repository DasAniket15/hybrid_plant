"""
post_solve.py
─────────────
Extracts solution arrays from a solved Pyomo model and computes all
post-solve quantities described in §9 of the spec:
  energy, CUF, augmentation schedule, CAPEX, LCOE, landed tariff,
  client savings, C-rate back-calculation, BESS container counts.

Returns a PostSolveResult dataclass suitable for the run_model.py dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.finance.capex_model import CapexModel
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.landed_tariff_model import LandedTariffModel
from hybrid_plant.finance.savings_model import SavingsModel
from hybrid_plant.finance.opex_model import OpexModel


@dataclass
class PostSolveResult:
    # Sizing
    sizing:              dict[str, Any]       = field(default_factory=dict)
    # CAPEX
    capex_result:        dict[str, Any]       = field(default_factory=dict)
    # 25-year arrays (all indexed year 1-25)
    busbar_mwh:          list[float]          = field(default_factory=list)
    meter_mwh:           list[float]          = field(default_factory=list)
    discom_draw_mwh:     list[float]          = field(default_factory=list)
    curtailment_mwh:     list[float]          = field(default_factory=list)
    charge_total_mwh:    list[float]          = field(default_factory=list)
    discharge_total_mwh: list[float]          = field(default_factory=list)
    plant_cuf_pct:       list[float]          = field(default_factory=list)
    opex_by_year:        list[float]          = field(default_factory=list)
    opex_breakdown:      list[dict[str, Any]] = field(default_factory=list)
    # Finance
    lcoe:                float                = 0.0
    lcoe_breakdown:      dict[str, Any]       = field(default_factory=dict)
    landed_tariff:       list[float]          = field(default_factory=list)   # Rs/kWh
    landed_tariff_breakdown: dict[str, Any]   = field(default_factory=dict)
    savings_npv:         float                = 0.0
    savings_breakdown:   dict[str, Any]       = field(default_factory=dict)
    wacc:                float                = 0.0
    # Year-1 energy slice (for dashboard Section 3)
    year1_energy:        dict[str, Any]       = field(default_factory=dict)
    # Augmentation
    aug_events:          list[dict[str, Any]] = field(default_factory=list)
    # C-rate back-calculated
    c_rate_charge_initial:    float           = 0.0
    c_rate_discharge_initial: float           = 0.0


def compute_post_solve(
    model:  pyo.ConcreteModel,
    params: dict[str, Any],
    config: FullConfig,
    data:   dict[str, Any],
) -> PostSolveResult:
    """
    Parameters
    ----------
    model   : solved ConcreteModel
    params  : flat parameter dict from parameter_builder
    config  : FullConfig
    data    : timeseries data dict from load_timeseries_data

    Returns
    -------
    PostSolveResult
    """
    Y          = params["project_life_years"]
    T          = params["hours_per_year"]
    disc_eff   = params["discharge_eff"]
    charge_eff = params["charge_eff"]
    loss_factor = params["loss_factor"]
    container_sz = params["container_size_mwh"]
    aug_bess    = params["aug_bess_enabled"]

    # ─────────────────────────────────────────────────────────────────────────
    # Extract scalar sizing variables
    # ─────────────────────────────────────────────────────────────────────────
    solar_mw_0 = float(pyo.value(model.solar_mw_0))
    wind_mw_0  = float(pyo.value(model.wind_mw_0))
    bess_cap_0 = float(pyo.value(model.bess_cap_0))
    ppa_mw     = float(pyo.value(model.ppa_mw))
    z_charge_0    = float(pyo.value(model.z_charge_0))
    z_discharge_0 = float(pyo.value(model.z_discharge_0))

    bess_containers_initial = round(bess_cap_0 / container_sz) if container_sz > 0 else 0

    # ─────────────────────────────────────────────────────────────────────────
    # §9a  Energy quantities per year
    # ─────────────────────────────────────────────────────────────────────────
    busbar_mwh_arr       = []
    meter_mwh_arr        = []
    discom_draw_arr      = []
    curtailment_arr      = []
    charge_total_arr     = []
    discharge_total_arr  = []
    load_mwh             = data["load_profile"]   # shape (8760,)
    annual_load_mwh      = float(np.sum(load_mwh))

    for y in range(1, Y + 1):
        busbar_y = 0.0
        discom_y = 0.0
        curtail_y = 0.0
        charge_y = 0.0
        discharge_y = 0.0

        for t in range(1, T + 1):
            gen_s  = float(pyo.value(model.gen_solar[y, t]))
            gen_w  = float(pyo.value(model.gen_wind[y, t]))
            ch     = float(pyo.value(model.charge[y, t]))
            disc   = float(pyo.value(model.discharge[y, t]))
            curt   = float(pyo.value(model.curtail[y, t]))
            dis_v  = float(pyo.value(model.discom[y, t]))

            direct_yt   = gen_s + gen_w - ch - curt
            busbar_yt   = direct_yt + disc * disc_eff
            busbar_y   += busbar_yt
            discom_y   += dis_v
            curtail_y  += curt
            charge_y   += ch * charge_eff
            discharge_y += disc * disc_eff

        busbar_mwh_arr.append(busbar_y)
        meter_mwh_arr.append(busbar_y * loss_factor)
        discom_draw_arr.append(discom_y)
        curtailment_arr.append(curtail_y)
        charge_total_arr.append(charge_y)
        discharge_total_arr.append(discharge_y)

    # ─────────────────────────────────────────────────────────────────────────
    # §9b  Plant CUF per year
    # ─────────────────────────────────────────────────────────────────────────
    plant_cuf_arr = [
        (busbar_mwh_arr[y - 1] / (ppa_mw * T) * 100.0) if ppa_mw > 0 else 0.0
        for y in range(1, Y + 1)
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # §9d  CAPEX via existing CapexModel
    # ─────────────────────────────────────────────────────────────────────────
    capex_result = CapexModel(config).compute(
        solar_capacity_mw        = solar_mw_0,
        wind_capacity_mw         = wind_mw_0,
        bess_energy_capacity_mwh = bess_cap_0,
    )
    total_capex = capex_result["total_capex"]

    # ─────────────────────────────────────────────────────────────────────────
    # OPEX via existing OpexModel (initial capacity basis)
    # ─────────────────────────────────────────────────────────────────────────
    opex_by_year, opex_breakdown = OpexModel(config).compute(
        solar_capacity_mw = solar_mw_0,
        wind_capacity_mw  = wind_mw_0,
        bess_energy_mwh   = bess_cap_0,
        total_capex       = total_capex,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # §9e  LCOE via existing LCOEModel
    # ─────────────────────────────────────────────────────────────────────────
    lcoe_model  = LCOEModel(config)
    lcoe_result = lcoe_model.compute(
        total_capex                  = total_capex,
        opex_projection              = opex_by_year,
        busbar_energy_mwh_projection = np.array(busbar_mwh_arr),
    )
    lcoe         = lcoe_result["lcoe_inr_per_kwh"]
    wacc         = lcoe_result["wacc"]

    # ─────────────────────────────────────────────────────────────────────────
    # §9f  Landed tariff via existing LandedTariffModel
    # ─────────────────────────────────────────────────────────────────────────
    lt_model  = LandedTariffModel(config)
    lt_result = lt_model.compute(
        lcoe_inr_per_kwh             = lcoe,
        ppa_capacity_mw              = ppa_mw,
        busbar_energy_mwh_projection = busbar_mwh_arr,
        meter_energy_mwh_projection  = meter_mwh_arr,
    )
    landed_tariff = lt_result["landed_tariff_series"]

    # ─────────────────────────────────────────────────────────────────────────
    # §9g  Client savings via existing SavingsModel
    # ─────────────────────────────────────────────────────────────────────────
    sv_model  = SavingsModel(config, data)
    sv_result = sv_model.compute(
        landed_tariff_series        = landed_tariff,
        meter_energy_mwh_projection = meter_mwh_arr,
        wacc                        = wacc,
    )
    savings_npv = sv_result["savings_npv"]

    # ─────────────────────────────────────────────────────────────────────────
    # §9c  Augmentation event schedule
    # ─────────────────────────────────────────────────────────────────────────
    aug_events: list[dict[str, Any]] = []
    if params["aug_solar_enabled"] or params["aug_wind_enabled"] or aug_bess:
        for s in range(1, Y + 1):
            d_solar = float(pyo.value(model.delta_solar_mw[s]))  if params["aug_solar_enabled"] else 0.0
            d_wind  = float(pyo.value(model.delta_wind_mw[s]))   if params["aug_wind_enabled"]  else 0.0
            d_bess  = float(pyo.value(model.delta_bess_cap[s]))  if aug_bess                    else 0.0
            y_solar = float(pyo.value(model.y_solar[s]))         if params["aug_solar_enabled"] else 0.0
            y_wind  = float(pyo.value(model.y_wind[s]))          if params["aug_wind_enabled"]  else 0.0
            y_bess  = float(pyo.value(model.y_bess[s]))          if aug_bess                    else 0.0

            if y_solar > 0.5 or y_wind > 0.5 or y_bess > 0.5:
                event: dict[str, Any] = {
                    "year":              s,
                    "solar_mw_added":    d_solar,
                    "wind_mw_added":     d_wind,
                    "bess_mwh_added":    d_bess,
                    "capex_expensed_rs": (
                        d_solar * params["aug_capex_solar_rs_per_mw"]
                        + d_wind  * params["aug_capex_wind_rs_per_mw"]
                        + d_bess  * params["aug_capex_bess_rs_per_mwh"]
                    ),
                }
                if aug_bess and d_bess > 1e-6:
                    event["c_rate_charge"]    = float(pyo.value(model.c_rate_charge_aug[s]))
                    event["c_rate_discharge"] = float(pyo.value(model.c_rate_discharge_aug[s]))
                aug_events.append(event)

    # ─────────────────────────────────────────────────────────────────────────
    # §9h  C-rate back-calculation
    # ─────────────────────────────────────────────────────────────────────────
    c_rate_charge_init    = z_charge_0    / bess_cap_0 if bess_cap_0 > 1e-6 else 0.0
    c_rate_discharge_init = z_discharge_0 / bess_cap_0 if bess_cap_0 > 1e-6 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # §9i  BESS container counts
    # ─────────────────────────────────────────────────────────────────────────
    bess_containers_aug: dict[int, int] = {}
    if aug_bess:
        for s in range(1, Y + 1):
            d_bess = float(pyo.value(model.delta_bess_cap[s]))
            bess_containers_aug[s] = round(d_bess / container_sz) if container_sz > 0 else 0

    bess_containers_total = {
        y: bess_containers_initial + sum(bess_containers_aug.get(s, 0) for s in range(1, y + 1))
        for y in range(1, Y + 1)
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Sizing dict for dashboard Section 1
    # ─────────────────────────────────────────────────────────────────────────
    sizing = {
        "solar_mw_0":               solar_mw_0,
        "wind_mw_0":                wind_mw_0,
        "bess_cap_0_mwh":           bess_cap_0,
        "bess_containers_initial":  bess_containers_initial,
        "ppa_mw":                   ppa_mw,
        "c_rate_charge_initial":    c_rate_charge_init,
        "c_rate_discharge_initial": c_rate_discharge_init,
        "bess_containers_total":    bess_containers_total,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Year-1 energy slice for dashboard Section 3
    # ─────────────────────────────────────────────────────────────────────────
    year1_energy = _build_year1_slice(model, params, loss_factor, disc_eff, charge_eff)
    year1_energy["loss_factor"] = loss_factor

    return PostSolveResult(
        sizing               = sizing,
        capex_result         = capex_result,
        busbar_mwh           = busbar_mwh_arr,
        meter_mwh            = meter_mwh_arr,
        discom_draw_mwh      = discom_draw_arr,
        curtailment_mwh      = curtailment_arr,
        charge_total_mwh     = charge_total_arr,
        discharge_total_mwh  = discharge_total_arr,
        plant_cuf_pct        = plant_cuf_arr,
        opex_by_year         = opex_by_year,
        opex_breakdown       = opex_breakdown,
        lcoe                 = lcoe,
        lcoe_breakdown       = lcoe_result,
        landed_tariff        = landed_tariff,
        landed_tariff_breakdown = lt_result,
        savings_npv          = savings_npv,
        savings_breakdown    = sv_result,
        wacc                 = wacc,
        year1_energy         = year1_energy,
        aug_events           = aug_events,
        c_rate_charge_initial    = c_rate_charge_init,
        c_rate_discharge_initial = c_rate_discharge_init,
    )


def _build_year1_slice(
    model:      pyo.ConcreteModel,
    params:     dict[str, Any],
    loss_factor: float,
    disc_eff:   float,
    charge_eff: float,
) -> dict[str, Any]:
    """Extract year-1 hourly arrays for the Section 3 energy balance."""
    T = params["hours_per_year"]
    y = 1

    solar_direct_pre  = np.array([float(pyo.value(model.gen_solar[y, t]))  for t in range(1, T + 1)])
    wind_direct_pre   = np.array([float(pyo.value(model.gen_wind[y, t]))   for t in range(1, T + 1)])
    charge_pre        = np.array([float(pyo.value(model.charge[y, t]))     for t in range(1, T + 1)])
    discharge_pre_mw  = np.array([float(pyo.value(model.discharge[y, t]))  for t in range(1, T + 1)])
    curtailment_pre   = np.array([float(pyo.value(model.curtail[y, t]))    for t in range(1, T + 1)])
    soc_arr           = np.array([float(pyo.value(model.soc[y, t]))        for t in range(1, T + 1)])

    discharge_pre = discharge_pre_mw * disc_eff   # post-efficiency (MW delivered to busbar)

    return {
        "solar_direct_pre":    solar_direct_pre,
        "wind_direct_pre":     wind_direct_pre,
        "charge_pre":          charge_pre,
        "discharge_pre":       discharge_pre,
        "curtailment_pre":     curtailment_pre,
        "soc":                 soc_arr,
        "bess_end_soc_mwh":    float(soc_arr[-1]),
        "solar_direct_meter":  solar_direct_pre * loss_factor,
        "wind_direct_meter":   wind_direct_pre  * loss_factor,
        "discharge_meter":     discharge_pre    * loss_factor,
        "charge_loss":         charge_pre * (1.0 - charge_eff),
        "discharge_loss":      discharge_pre_mw * (1.0 - disc_eff),
        "aux_loss":            np.zeros(T),    # aux already deducted via SOC dynamics
        "energy_capacity_mwh": params["bess_cap_min"],   # placeholder — use sizing dict
        "charge_power_mw":     0.0,            # placeholder — use sizing dict
        "discharge_power_mw":  0.0,
    }
