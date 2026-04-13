"""
run_model.py
────────────
Master entry point for the hybrid RE plant model.

Pipeline
────────
  1. Load config + data
  2. Run SolverEngine  (Optuna TPE, n_trials from solver.yaml)
  3. Extract best solution
  4. Print 7-section dashboard to stdout
  5. Save model_output.png (4-panel figure) to outputs/
  6. Save day250_dispatch.png (BESS dispatch diagnostic) to outputs/

Run
───
  python -m hybrid_plant.run_model
"""

from __future__ import annotations

import logging
from pathlib import Path

import math
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import load_config
from hybrid_plant.constants import CRORE_TO_RS
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.grid_interface import GridInterface
from hybrid_plant.energy.plant_engine import PlantEngine
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine
from hybrid_plant.solver.solver_engine import SolverEngine

logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def cr(v: float) -> float:
    """Rs → Rs Crore, 4 dp."""
    return round(float(v) / CRORE_TO_RS, 4)

def pct(v: float) -> float:
    return round(float(v) * 100, 2)

def sep(title: str = "", width: int = 68) -> None:
    if title:
        pad = width - len(title) - 13
        print(f"\n{'─'*10} {title} {'─'*max(pad, 2)}")
    else:
        print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Derived metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_payback_year(annual_savings: list[float]) -> int | None:
    """First year where cumulative savings turns positive."""
    cumulative = 0.0
    for yr, s in enumerate(annual_savings, start=1):
        cumulative += s
        if cumulative > 0:
            return yr
    return None


def compute_cuf(busbar_mwh_y1: float, capacity_mw: float, hours: int = 8760) -> float | None:
    """CUF = annual busbar MWh / (capacity_mw × 8760), as percent."""
    if capacity_mw <= 0:
        return None
    return busbar_mwh_y1 / (capacity_mw * hours) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard sections
# ─────────────────────────────────────────────────────────────────────────────

def print_section1(params, y1, fi):
    sep("SECTION 1 — OPTIMAL CONFIGURATION")
    cap = fi["capex"]
    print(f"\n  {'Solar capacity (AC MW)':<38} : {round(params['solar_capacity_mw'], 2)}")
    print(f"  {'Solar capacity (DC MWp)':<38} : {round(cap['solar_dc_mwp'], 2)}")
    ac_dc = round(cap['solar_dc_mwp'] / params['solar_capacity_mw'], 4) if params['solar_capacity_mw'] > 0 else "N/A"
    print(f"  {'AC/DC ratio':<38} : {ac_dc}")
    print(f"  {'Wind capacity (MW)':<38} : {round(params['wind_capacity_mw'], 2)}")
    print(f"  {'BESS containers':<38} : {params['bess_containers']}")
    print(f"  {'BESS energy capacity (MWh)':<38} : {round(float(y1['energy_capacity_mwh']), 2)}")
    print(f"  {'BESS charge power (MW)':<38} : {round(float(y1['charge_power_mw']), 2)}")
    print(f"  {'BESS discharge power (MW)':<38} : {round(float(y1['discharge_power_mw']), 2)}")
    print(f"  {'BESS charge C-rate':<38} : {round(params['charge_c_rate'], 4)}")
    print(f"  {'BESS discharge C-rate':<38} : {round(params['discharge_c_rate'], 4)}")
    print(f"  {'PPA capacity (MW)':<38} : {round(params['ppa_capacity_mw'], 2)}")
    print(f"  {'Dispatch priority':<38} : {params['dispatch_priority']}")
    print(f"  {'BESS charge source':<38} : {params['bess_charge_source']}")


def print_section2(fi):
    sep("SECTION 2 — CAPITAL & FINANCING")
    cap = fi["capex"]
    lcd = fi["lcoe_breakdown"]
    print(f"\n  {'── CAPEX (Rs Crore)'}")
    print(f"  {'Solar':<38} : {cr(cap['solar_capex'])}")
    print(f"  {'Wind':<38} : {cr(cap['wind_capex'])}")
    print(f"  {'BESS':<38} : {cr(cap['bess_capex'])}")
    print(f"  {'Transmission':<38} : {cr(cap['transmission_capex'])}")
    print(f"  {'Total CAPEX':<38} : {cr(cap['total_capex'])}")
    print(f"\n  {'── FINANCING'}")
    print(f"  {'Debt (Rs Crore)':<38} : {cr(lcd['debt_amount'])}")
    print(f"  {'Equity (Rs Crore)':<38} : {cr(lcd['equity_amount'])}")
    print(f"  {'WACC':<38} : {pct(fi['wacc'])} %")
    print(f"  {'Annual EMI (Rs Crore)':<38} : {cr(lcd['emi'])}")
    print(f"  {'Annual ROE payment (Rs Crore)':<38} : {cr(lcd['roe_schedule'][0])}")


def print_section3(params, y1, fi, data, energy_engine):
    sep("SECTION 3 — YEAR-1 ENERGY BALANCE")

    solar_direct_pre = float(np.sum(y1["solar_direct_pre"]))
    wind_direct_pre  = float(np.sum(y1["wind_direct_pre"]))
    discharge_pre    = float(np.sum(y1["discharge_pre"]))
    curtailment      = float(np.sum(y1["curtailment_pre"]))
    total_busbar     = solar_direct_pre + wind_direct_pre + discharge_pre

    solar_direct_m  = float(np.sum(y1["solar_direct_meter"]))
    wind_direct_m   = float(np.sum(y1["wind_direct_meter"]))
    discharge_m     = float(np.sum(y1["discharge_meter"]))
    total_meter     = solar_direct_m + wind_direct_m + discharge_m

    annual_load_mwh = float(np.sum(data["load_profile"]))
    discom_draw_mwh = max(annual_load_mwh - total_meter, 0)
    sv              = fi["savings_breakdown"]
    discom_cost_y1  = discom_draw_mwh * 1000 * sv["discom_tariff"]
    re_penetration  = total_meter / annual_load_mwh * 100

    raw_solar = float(np.sum(params["solar_capacity_mw"] * data["solar_cuf"]))
    raw_wind  = float(np.sum(params["wind_capacity_mw"]  * data["wind_cuf"]))

    y1_lf1 = energy_engine.plant.simulate(
        solar_capacity_mw  = params["solar_capacity_mw"],
        wind_capacity_mw   = params["wind_capacity_mw"],
        bess_containers    = params["bess_containers"],
        charge_c_rate      = params["charge_c_rate"],
        discharge_c_rate   = params["discharge_c_rate"],
        ppa_capacity_mw    = params["ppa_capacity_mw"],
        dispatch_priority  = params["dispatch_priority"],
        bess_charge_source = params["bess_charge_source"],
        loss_factor        = 1.0,
    )
    plant_cuf_num = float(np.sum(
        np.minimum(
            y1_lf1["plant_export_pre"] + y1_lf1["curtailment_pre"],
            params["ppa_capacity_mw"],
        )
    ))

    solar_cuf = compute_cuf(raw_solar, params["solar_capacity_mw"])
    wind_cuf  = compute_cuf(raw_wind,  params["wind_capacity_mw"])
    plant_cuf = compute_cuf(plant_cuf_num, params["ppa_capacity_mw"])
    loss_factor = float(y1["loss_factor"])

    print(f"\n  {'── BUSBAR (Pre-Loss)'}")
    print(f"  {'Solar Direct (MWh)':<38} : {round(solar_direct_pre, 1)}")
    print(f"  {'Wind Direct (MWh)':<38} : {round(wind_direct_pre, 1)}")
    print(f"  {'BESS Discharge (MWh)':<38} : {round(discharge_pre, 1)}")
    print(f"  {'Total Busbar Delivery (MWh)':<38} : {round(total_busbar, 1)}")
    print(f"  {'Excess Energy (MWh)':<38} : {round(curtailment, 1)}")
    print(f"\n  {'── METER (Post-Loss)'}")
    print(f"  {'Solar Direct at Meter (MWh)':<38} : {round(solar_direct_m, 1)}")
    print(f"  {'Wind Direct at Meter (MWh)':<38} : {round(wind_direct_m, 1)}")
    print(f"  {'BESS Discharge at Meter (MWh)':<38} : {round(discharge_m, 1)}")
    print(f"  {'Total RE at Meter (MWh)':<38} : {round(total_meter, 1)}")
    print(f"  {'Grid Loss Factor':<38} : {round(loss_factor, 4)}")
    print(f"\n  {'── LOAD & DISCOM DEPENDENCY'}")
    print(f"  {'Annual Load (MWh)':<38} : {round(annual_load_mwh, 1)}")
    print(f"  {'DISCOM Draw (MWh)':<38} : {round(discom_draw_mwh, 1)}")
    print(f"  {'DISCOM Cost Year-1 (Rs Crore)':<38} : {cr(discom_cost_y1)}")
    print(f"  {'RE Penetration (%)':<38} : {round(re_penetration, 2)} %")
    print(f"\n  {'── CAPACITY UTILISATION (CUF)'}")
    print(f"  {'Solar CUF (%)':<38} : {round(solar_cuf, 2) if solar_cuf else 'N/A'}")
    print(f"  {'Wind CUF (%)':<38} : {round(wind_cuf, 2) if wind_cuf else 'N/A'}")
    print(f"  {'Plant CUF (%) [busbar / PPA×8760]':<38} : {round(plant_cuf, 2) if plant_cuf else 'N/A'}")


def print_section4(fi):
    sep("SECTION 4 — LCOE & TARIFF BUILD-UP")
    lcd  = fi["lcoe_breakdown"]
    lt   = fi["landed_tariff_breakdown"]
    lts  = fi["landed_tariff_series"]
    sv   = fi["savings_breakdown"]
    lcoe = fi["lcoe_inr_per_kwh"]

    wheeling = lt.get("wheeling_per_kwh", 0.0)
    elec_tax = lt.get("electricity_tax_per_kwh", 0.0)
    banking  = lt.get("banking_per_kwh", 0.0)

    cap_series = lt.get("capacity_charge_per_kwh_series", None)
    if cap_series is not None:
        cap_y1, cap_y25 = cap_series[0], cap_series[-1]
    else:
        energy_charge = wheeling + elec_tax + banking
        cap_y1  = lts[0]  - lcoe - energy_charge
        cap_y25 = lts[-1] - lcoe - energy_charge

    print(f"\n  {'── LCOE BUILD-UP (NPV basis)'}")
    print(f"  {'NPV Total Cost (Rs Crore)':<38} : {cr(lcd['npv_total_cost'])}")
    print(f"    {'↳ Interest':<36} : {cr(lcd['npv_interest'])}")
    print(f"    {'↳ Principal':<36} : {cr(lcd['npv_principal'])}")
    print(f"    {'↳ ROE':<36} : {cr(lcd['npv_roe'])}")
    print(f"    {'↳ OPEX':<36} : {cr(lcd['npv_opex'])}")
    print(f"  {'NPV Busbar Energy (Bn kWh)':<38} : {round(lcd['npv_energy_kwh'] / 1e9, 4)}")
    print(f"  {'LCOE (Rs/kWh)':<38} : {round(lcoe, 4)}")
    print(f"\n  {'── LANDED TARIFF BUILD-UP (Year 1)'}")
    print(f"  {'LCOE (busbar)':<38} : {round(lcoe, 4)}")
    print(f"  {'Wheeling charge (Rs/kWh)':<38} : {round(wheeling, 4)}")
    print(f"  {'Electricity tax (Rs/kWh)':<38} : {round(elec_tax, 4)}")
    print(f"  {'Banking charge (Rs/kWh)':<38} : {round(banking, 4)}")
    print(f"  {'Capacity charge Y1 (Rs/kWh)':<38} : {round(cap_y1, 4)}")
    print(f"  {'Capacity charge Y25 (Rs/kWh)':<38} : {round(cap_y25, 4)}")
    print(f"  {'Landed Tariff Year 1 (Rs/kWh)':<38} : {round(lts[0], 4)}")
    print(f"  {'Landed Tariff Year 25 (Rs/kWh)':<38} : {round(lts[-1], 4)}")
    print(f"  {'DISCOM Tariff wt-avg (Rs/kWh)':<38} : {round(sv['discom_tariff'], 4)}")
    print(f"  {'Saving vs DISCOM Y1 (Rs/kWh)':<38} : {round(sv['discom_tariff'] - lts[0], 4)}")


def print_section5(fi):
    sep("SECTION 5 — CLIENT SAVINGS")
    sv             = fi["savings_breakdown"]
    lts            = fi["landed_tariff_series"]
    annual_savings = sv["annual_savings"]
    baseline       = sv["baseline_annual_cost"]
    savings_y1     = annual_savings[0]
    savings_pct    = savings_y1 / baseline * 100
    payback_yr     = compute_payback_year(annual_savings)
    cum_savings_25 = sum(annual_savings)

    print(f"\n  {'Baseline Annual Cost (Rs Crore)':<38} : {cr(baseline)}")
    print(f"  {'Hybrid Cost Year-1 (Rs Crore)':<38} : {cr(sv['annual_hybrid_cost'][0])}")
    print(f"  {'Savings Year-1 (Rs Crore)':<38} : {cr(savings_y1)}")
    print(f"  {'Savings as % of Baseline':<38} : {round(savings_pct, 2)} %")
    print(f"  {'Savings NPV (Rs Crore)':<38} : {cr(fi['savings_npv'])}")
    print(f"  {'Cumulative Savings 25 Yr (Rs Cr)':<38} : {cr(cum_savings_25)}")
    print(f"  {'Payback Year':<38} : {'Year ' + str(payback_yr) if payback_yr else 'Beyond project life'}")


def print_section6(fi):
    sep("SECTION 6 — OPEX BREAKDOWN (Rs Crore)")
    ob_y1  = fi["opex_breakdown"][0]
    ob_y25 = fi["opex_breakdown"][-1]
    cols = [
        ("Solar O&M",              "solar_om"),
        ("Wind O&M",               "wind_om"),
        ("BESS O&M",               "bess_om"),
        ("Solar Transmission O&M", "solar_transmission_om"),
        ("Wind Transmission O&M",  "wind_transmission_om"),
        ("Land Lease",             "land_lease"),
        ("Insurance",              "insurance"),
        ("Augmentation Purchase",  "augmentation_purchase"),
    ]
    print(f"\n  {'Component':<36}  {'Year 1':>10}  {'Year 25':>10}  {'Δ':>10}")
    sep()
    for label, key in cols:
        v1  = cr(ob_y1.get(key, 0.0))
        v25 = cr(ob_y25.get(key, 0.0))
        print(f"  {label:<36}  {v1:>10}  {v25:>10}  {round(v25 - v1, 4):>+10}")
    sep()
    print(f"  {'Total OPEX':<36}  {cr(ob_y1['total']):>10}  {cr(ob_y25['total']):>10}  {round(cr(ob_y25['total'])-cr(ob_y1['total']),4):>+10}")
    print(f"  {'  BESS Installed (MWh)':<36}  {ob_y1.get('bess_installed_mwh', 0.0):>10.2f}  {ob_y25.get('bess_installed_mwh', 0.0):>10.2f}  {'':>10}")


def print_section8(fi, config, n_opt: int | None = None, n_extra: int = 0):
    sep("SECTION 8 — BESS AUGMENTATION")
    aug     = fi["augmentation"]
    aug_cfg = config.bess["bess"]["augmentation"]
    wacc    = fi["wacc"]

    enabled      = aug["enabled"]
    cohorts      = aug["cohorts"]
    aug_years    = aug["augmentation_years"]
    eff_cont     = aug["effective_containers_per_year"]
    eff_soh      = aug["effective_soh_per_year"]
    inst_mwh     = aug["total_installed_mwh_per_year"]
    purch_opex   = aug["augmentation_purchase_opex"]
    added        = aug["containers_added_per_year"]
    cont_size    = float(config.bess["bess"]["container"]["size_mwh"])
    project_life = config.project["project"]["project_life_years"]

    # CUF per year from energy projection (all zeros in fast mode)
    cuf_per_year = fi["energy_projection"].get("cuf_per_year", None)

    # ── Config summary ────────────────────────────────────────────────────────
    print(f"\n  {'Augmentation enabled':<42} : {'Yes' if enabled else 'No'}")
    if not enabled:
        print(f"  (Augmentation is disabled — no events will occur)\n")
        return

    trigger_cuf_pct = float(aug_cfg.get("trigger_cuf_percent", 95.0))
    restore_pct     = float(aug_cfg.get("restore_to_percent_of_year1", 100.0))
    max_gap         = aug_cfg.get("max_gap_years")
    cost_per_mwh    = float(aug_cfg.get("cost_per_mwh", 0.60e7))
    oversize_pct    = float(aug_cfg.get("oversize_percent", 0.0))
    y1_eff_mwh      = eff_cont[1] * cont_size * eff_soh[1]

    y1_cuf     = float(cuf_per_year[0]) if cuf_per_year is not None and len(cuf_per_year) > 0 else None
    trig_cuf   = trigger_cuf_pct / 100.0 * y1_cuf if y1_cuf is not None else None

    print(f"  {'Trigger metric':<42} : Plant CUF %")
    print(f"  {'Trigger threshold':<42} : {trigger_cuf_pct:.1f} % of Year-1 CUF"
          + (f"  →  abs. {trig_cuf:.3f} %" if trig_cuf is not None else ""))
    print(f"  {'Year-1 plant CUF':<42} : "
          + (f"{y1_cuf:.3f} %" if y1_cuf is not None else "N/A (fast mode)"))
    print(f"  {'Restore target (BESS capacity)':<42} : {restore_pct:.1f} % of Year-1 BESS eff. MWh")
    print(f"  {'Max gap override':<42} : {str(max_gap) + ' years' if max_gap else 'None (threshold-only)'}")
    print(f"  {'Cost per MWh augmented':<42} : Rs {cr(cost_per_mwh):.4f} Cr/MWh")
    print(f"  {'Year-1 effective BESS capacity':<42} : {y1_eff_mwh:.2f} MWh  "
          f"({cohorts[0][1]} containers × {cont_size} MWh × SOH {eff_soh[1]:.4f})")

    # ── Oversizing summary ────────────────────────────────────────────────────
    if oversize_pct > 0 and n_opt is not None:
        n_oversize = n_opt + n_extra
        print(f"\n  {'── YEAR-1 OVERSIZING'}")
        print(f"  {'Solver-optimal containers (Phase 1)':<42} : {n_opt}")
        print(f"  {'Oversize percent':<42} : {oversize_pct:.1f} %")
        print(f"  {'Extra containers added':<42} : {n_extra}")
        print(f"  {'Final Year-1 containers (Phase 2)':<42} : {n_oversize}")
        print(f"  {'Extra BESS capacity added':<42} : {n_extra * cont_size:.2f} MWh  (nameplate)")

    # ── Event summary ────────────────────────────────────────────────────────
    n_events     = len(aug_years)
    total_added  = sum(n for _, n in cohorts[1:])
    total_inst   = inst_mwh[project_life]
    total_cont   = eff_cont[project_life]
    total_spend  = sum(purch_opex.values())
    npv_spend    = sum(v / (1 + wacc) ** t for t, v in purch_opex.items())

    print(f"\n  {'Number of augmentation events':<42} : {n_events}")
    print(f"  {'Total containers added (augmentation)':<42} : {total_added}")
    print(f"  {f'Total installed capacity (Yr {project_life})':<42} : "
          f"{total_inst:.2f} MWh  ({total_cont} containers)")
    print(f"  {'Total augmentation spend (undiscounted)':<42} : {cr(total_spend):.4f} Rs Crore")
    print(f"  {'Total augmentation spend (NPV)':<42} : {cr(npv_spend):.4f} Rs Crore")

    ob = fi["opex_breakdown"]
    bess_om_y1  = ob[0]["bess_om"]
    bess_om_y25 = ob[-1]["bess_om"]
    bess_om_pct = (bess_om_y25 - bess_om_y1) / bess_om_y1 * 100 if bess_om_y1 > 0 else 0.0
    print(f"  {'BESS O&M Year 1 → Year 25':<42} : "
          f"{cr(bess_om_y1):.4f} Cr → {cr(bess_om_y25):.4f} Cr  ({bess_om_pct:+.1f}%)")

    # ── Cohort summary table ─────────────────────────────────────────────────
    print(f"\n  {'── COHORT SUMMARY'}")
    print(f"  {'Cohort':>7}  {'Start Yr':>9}  {'Containers':>11}  "
          f"{'Nominal MWh':>12}  {'Purchase (Rs Cr)':>17}")
    sep()
    for idx, (start_yr, n_c) in enumerate(cohorts):
        nom_mwh   = n_c * cont_size
        cost_str  = f"{cr(nom_mwh * cost_per_mwh):>17.4f}" if idx > 0 else f"{'—  (initial)':>17}"
        print(f"  {idx:>7}  {start_yr:>9}  {n_c:>11}  {nom_mwh:>12.2f}  {cost_str}")
    sep()
    total_nom = sum(n * cont_size for _, n in cohorts)
    print(f"  {'Total':>7}  {'':>9}  {sum(n for _, n in cohorts):>11}  "
          f"{total_nom:>12.2f}  {cr(total_spend):>17.4f}")

    # ── Year-by-year profile ─────────────────────────────────────────────────
    print(f"\n  {'── YEAR-BY-YEAR PROFILE'}")
    has_cuf = cuf_per_year is not None and np.any(cuf_per_year > 0)
    if has_cuf:
        hdr = (
            f"  {'Yr':>3}  {'Containers':>11}  {'Installed MWh':>14}  "
            f"{'Eff SOH':>8}  {'Eff MWh':>9}  {'CUF%':>7}  {'Purchase (Rs Cr)':>17}  {'':>8}"
        )
    else:
        hdr = (
            f"  {'Yr':>3}  {'Containers':>11}  {'Installed MWh':>14}  "
            f"{'Eff SOH':>8}  {'Eff MWh':>9}  {'Purchase (Rs Cr)':>17}  {'':>8}"
        )
    print(f"\n{hdr}")
    sep()
    for yr in range(1, project_life + 1):
        n_c       = eff_cont[yr]
        inst      = inst_mwh[yr]
        soh_v     = eff_soh[yr]
        eff_mwh_v = n_c * cont_size * soh_v
        purch     = purch_opex.get(yr, 0.0)
        aug_flag  = "  ◄ AUG" if yr in aug_years else ""
        purch_str = f"{cr(purch):>17.4f}" if purch > 0 else f"{'—':>17}"
        cuf_v     = float(cuf_per_year[yr - 1]) if has_cuf else None
        trig_flag = ""
        if trig_cuf is not None and cuf_v is not None and yr > 1 and yr not in aug_years:
            trig_flag = "  (below trig)" if cuf_v < trig_cuf else ""
        if has_cuf:
            print(
                f"  {yr:>3}  {n_c:>11}  {inst:>14.2f}  "
                f"{soh_v:>8.4f}  {eff_mwh_v:>9.2f}  {cuf_v:>7.3f}  {purch_str}  {aug_flag}{trig_flag}"
            )
        else:
            print(
                f"  {yr:>3}  {n_c:>11}  {inst:>14.2f}  "
                f"{soh_v:>8.4f}  {eff_mwh_v:>9.2f}  {purch_str}  {aug_flag}"
            )
    sep()


def print_section7(fi):
    sep("SECTION 7 — 25-YEAR PROJECTIONS")
    ep            = fi["energy_projection"]
    sv            = fi["savings_breakdown"]
    lts           = fi["landed_tariff_series"]
    annual_savings = sv["annual_savings"]
    cumulative    = 0.0

    hdr = (
        f"  {'Yr':>3}  {'Busbar MWh':>11}  {'Meter MWh':>10}  "
        f"{'OPEX (Cr)':>10}  {'Landed ₹/kWh':>13}  "
        f"{'Savings (Cr)':>13}  {'Cum Savings (Cr)':>17}"
    )
    print(f"\n{hdr}")
    sep()
    for y in range(25):
        cumulative += annual_savings[y]
        print(
            f"  {y+1:>3}  "
            f"{ep['delivered_pre_mwh'][y]:>11.1f}  "
            f"{ep['delivered_meter_mwh'][y]:>10.1f}  "
            f"{fi['opex_projection'][y]/CRORE_TO_RS:>10.4f}  "
            f"{lts[y]:>13.4f}  "
            f"{annual_savings[y]/CRORE_TO_RS:>13.4f}  "
            f"{cumulative/CRORE_TO_RS:>17.4f}"
        )
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(params, y1, fi, data, output_path: Path, augmentation_years: list | None = None) -> None:
    ep   = fi["energy_projection"]
    sv   = fi["savings_breakdown"]
    lts  = fi["landed_tariff_series"]

    years          = np.arange(1, 26)
    annual_savings = np.array(sv["annual_savings"])
    cumulative     = np.cumsum(annual_savings) / CRORE_TO_RS

    solar_mwh  = ep["solar_direct_mwh"]
    wind_mwh   = ep["wind_direct_mwh"]
    bess_mwh   = ep["battery_mwh"]
    meter_mwh  = ep["delivered_meter_mwh"]
    load_mwh   = float(np.sum(data["load_profile"]))
    discom_mwh = np.maximum(load_mwh - meter_mwh, 0)

    ob          = fi["opex_breakdown"]
    opex_solar  = np.array([x["solar_om"]              for x in ob]) / CRORE_TO_RS
    opex_wind   = np.array([x["wind_om"]               for x in ob]) / CRORE_TO_RS
    opex_bess   = np.array([x["bess_om"]               for x in ob]) / CRORE_TO_RS
    opex_trans  = np.array([x["solar_transmission_om"] + x["wind_transmission_om"] for x in ob]) / CRORE_TO_RS
    opex_land   = np.array([x["land_lease"]             for x in ob]) / CRORE_TO_RS
    opex_ins    = np.array([x["insurance"]              for x in ob]) / CRORE_TO_RS
    opex_aug    = np.array([x.get("augmentation_purchase", 0.0) for x in ob]) / CRORE_TO_RS

    aug_label = ""
    if augmentation_years:
        aug_label = f"  |  {len(augmentation_years)} Augmentation Events (Yrs {', '.join(str(y) for y in augmentation_years)})"

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Hybrid RE Plant — Model Output Dashboard\n"
        f"Solar {round(params['solar_capacity_mw'],1)} MW  |  "
        f"Wind {round(params['wind_capacity_mw'],1)} MW  |  "
        f"BESS {round(float(y1['energy_capacity_mwh']),1)} MWh  |  "
        f"PPA {round(params['ppa_capacity_mw'],1)} MW"
        f"{aug_label}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    C  = {"solar":"#f4a832","wind":"#4c9be8","bess":"#66bb6a",
          "discom":"#ef5350","savings":"#26a69a","cumul":"#ab47bc"}

    # Panel 1: Annual savings + cumulative
    ax1 = fig.add_subplot(gs[0, 0])
    bar_colors = [C["savings"] if s >= 0 else C["discom"] for s in annual_savings]
    ax1.bar(years, annual_savings / CRORE_TO_RS, color=bar_colors, alpha=0.8, label="Annual Savings")
    ax1r = ax1.twinx()
    ax1r.plot(years, cumulative, color=C["cumul"], linewidth=2, marker="o", markersize=3, label="Cumulative")
    ax1r.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_title("Annual & Cumulative Savings", fontweight="bold")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Annual Savings (Rs Crore)")
    ax1r.set_ylabel("Cumulative Savings (Rs Crore)", color=C["cumul"])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25)

    # Panel 2: Landed tariff vs DISCOM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, lts, color="#1f77b4", linewidth=2, label="Landed Tariff")
    ax2.axhline(sv["discom_tariff"], color=C["discom"], linewidth=1.5, linestyle="--",
                label=f"DISCOM Avg (Rs {round(sv['discom_tariff'],2)})")
    ax2.fill_between(years, lts, sv["discom_tariff"],
                     where=[l < sv["discom_tariff"] for l in lts],
                     alpha=0.15, color=C["savings"], label="Savings band")
    ax2.set_title("Landed Tariff vs DISCOM Tariff", fontweight="bold")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Rs / kWh")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    # Panel 3: Energy mix stacked area
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.stackplot(years, solar_mwh/1e3, wind_mwh/1e3, bess_mwh/1e3, discom_mwh/1e3,
                  labels=["Solar Direct","Wind Direct","BESS Discharge","DISCOM Draw"],
                  colors=[C["solar"],C["wind"],C["bess"],C["discom"]], alpha=0.85)
    ax3.set_title("Energy Mix Over 25 Years", fontweight="bold")
    ax3.set_xlabel("Year"); ax3.set_ylabel("Energy (GWh)")
    ax3.legend(fontsize=8, loc="lower left"); ax3.grid(True, alpha=0.25)

    # Panel 4: OPEX stack + augmentation spikes
    ax4 = fig.add_subplot(gs[1, 1])
    stack_base = ax4.stackplot(
        years, opex_solar, opex_wind, opex_bess, opex_trans, opex_land, opex_ins,
        labels=["Solar O&M", "Wind O&M", "BESS O&M", "Transmission O&M", "Land Lease", "Insurance"],
        colors=["#fdd835", "#42a5f5", "#66bb6a", "#ab47bc", "#ff7043", "#78909c"], alpha=0.85,
    )
    # Augmentation purchase spikes — bold orange bars on top of the stack
    opex_regular = opex_solar + opex_wind + opex_bess + opex_trans + opex_land + opex_ins
    if np.any(opex_aug > 0):
        ax4.bar(years, opex_aug, bottom=opex_regular, color="#FF6D00", alpha=0.9,
                width=0.6, label="Augmentation Purchase", zorder=5)
        for yr_idx, val in enumerate(opex_aug):
            if val > 0:
                ax4.annotate(
                    f"+{val:.2f}Cr",
                    xy=(years[yr_idx], opex_regular[yr_idx] + val),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color="#BF360C", fontweight="bold",
                )
    ax4.set_title("OPEX Stack Over 25 Years", fontweight="bold")
    ax4.set_xlabel("Year"); ax4.set_ylabel("OPEX (Rs Crore)")
    ax4.legend(fontsize=7, loc="upper left"); ax4.grid(True, alpha=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard plot saved → {output_path}")
    plt.close()



# ─────────────────────────────────────────────────────────────────────────────
# BESS Augmentation profile plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_augmentation(fi, config, output_path: Path) -> None:
    """
    Six-panel BESS augmentation figure.

    Layout (GridSpec 4×2):
      Row 0 (full width) — Panel 1: Stacked cohort effective capacity
      Row 1 left         — Panel 2: SOH curves (blended + per-cohort)
      Row 1 right        — Panel 3: OPEX profile with augmentation spikes
      Row 2 left         — Panel 4: Container fleet step chart
      Row 2 right        — Panel 5: BESS installed MWh & O&M cost growth
      Row 3 (full width) — Panel 6: Plant CUF % trajectory with trigger line
    """
    from hybrid_plant._paths import find_project_root
    import pandas as pd

    aug          = fi["augmentation"]
    if not aug["enabled"]:
        return

    cohorts      = aug["cohorts"]
    aug_years    = aug["augmentation_years"]
    eff_cont     = aug["effective_containers_per_year"]
    eff_soh      = aug["effective_soh_per_year"]
    inst_mwh     = aug["total_installed_mwh_per_year"]
    purch_opex   = aug["augmentation_purchase_opex"]
    cont_size    = float(config.bess["bess"]["container"]["size_mwh"])
    project_life = config.project["project"]["project_life_years"]
    aug_cfg      = config.bess["bess"]["augmentation"]
    trigger_cuf_pct = float(aug_cfg.get("trigger_cuf_percent", 95.0))
    restore_pct     = float(aug_cfg.get("restore_to_percent_of_year1", 100.0))

    # CUF per year (all zeros when run in fast mode — panel 6 will be blank)
    cuf_arr_raw = fi["energy_projection"].get("cuf_per_year", None)
    cuf_arr     = (
        np.asarray(cuf_arr_raw, dtype=float)
        if cuf_arr_raw is not None
        else np.zeros(project_life)
    )

    # Load SOH curve
    root     = find_project_root()
    soh_df   = pd.read_csv(root / config.bess["bess"]["degradation"]["file"])
    soh_df.columns = soh_df.columns.str.strip().str.lower()
    soh_curve = dict(zip(soh_df["year"].astype(int), soh_df["soh"]))

    years = np.arange(1, project_life + 1)

    # ── Per-cohort effective MWh (for stacked area) ───────────────────────────
    cohort_eff = np.zeros((len(cohorts), project_life))
    for c_idx, (start_yr, n_c) in enumerate(cohorts):
        for y_idx, yr in enumerate(years):
            age     = int(yr) - start_yr + 1
            age     = max(1, min(age, project_life))
            soh_val = soh_curve.get(age, soh_curve[project_life])
            cohort_eff[c_idx, y_idx] = n_c * cont_size * soh_val

    # ── Per-cohort own SOH trajectory (for SOH panel) ────────────────────────
    cohort_soh_traj = np.zeros((len(cohorts), project_life))
    for c_idx, (start_yr, _) in enumerate(cohorts):
        for y_idx, yr in enumerate(years):
            age     = int(yr) - start_yr + 1
            age     = max(1, min(age, project_life))
            cohort_soh_traj[c_idx, y_idx] = soh_curve.get(age, soh_curve[project_life])

    # ── Reference lines ───────────────────────────────────────────────────────
    y1_eff_mwh = cohort_eff[:, 0].sum()
    target_mwh = restore_pct / 100.0 * y1_eff_mwh
    blended_soh = np.array([eff_soh[yr] for yr in years])

    # CUF trigger references (absolute values)
    y1_cuf      = float(cuf_arr[0]) if np.any(cuf_arr > 0) else None
    trig_cuf    = trigger_cuf_pct / 100.0 * y1_cuf if y1_cuf is not None else None

    # ── OPEX data ─────────────────────────────────────────────────────────────
    ob         = fi["opex_breakdown"]
    opex_solar = np.array([x["solar_om"]              for x in ob]) / CRORE_TO_RS
    opex_wind  = np.array([x["wind_om"]               for x in ob]) / CRORE_TO_RS
    opex_bess  = np.array([x["bess_om"]               for x in ob]) / CRORE_TO_RS
    opex_trans = np.array([x["solar_transmission_om"] + x["wind_transmission_om"] for x in ob]) / CRORE_TO_RS
    opex_land  = np.array([x["land_lease"]             for x in ob]) / CRORE_TO_RS
    opex_ins   = np.array([x["insurance"]              for x in ob]) / CRORE_TO_RS
    opex_aug_v = np.array([x.get("augmentation_purchase", 0.0) for x in ob]) / CRORE_TO_RS
    opex_reg   = opex_solar + opex_wind + opex_bess + opex_trans + opex_land + opex_ins
    bess_om_yr = np.array([x["bess_om"] for x in ob]) / CRORE_TO_RS
    inst_arr   = np.array([inst_mwh[yr] for yr in years])

    # ── Cohort color palette (green gradient, darkening per cohort) ───────────
    greens = [
        "#a5d6a7", "#66bb6a", "#388e3c", "#1b5e20",
        "#004d00", "#002600", "#001400",
    ]
    def cohort_color(idx):
        return greens[min(idx, len(greens) - 1)]

    # ── Figure layout ─────────────────────────────────────────────────────────
    cuf_label = (
        f"  |  Trigger {trigger_cuf_pct:.0f}% of Y1 CUF"
        + (f" ({trig_cuf:.2f}%)" if trig_cuf is not None else "")
    )
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        f"BESS Augmentation Profile  —  "
        f"{len(aug_years)} event(s)"
        f"{cuf_label}  |  "
        f"Restore {restore_pct:.0f}% BESS  |  "
        f"Final fleet: {eff_cont[project_life]} containers  /  "
        f"{inst_mwh[project_life]:.1f} MWh installed",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.3, 1.0, 1.0, 0.9],
        hspace=0.48, wspace=0.35,
    )

    def _add_aug_vlines(ax, ymax_frac=0.97):
        """Draw dashed vertical lines + year labels at each augmentation year."""
        ylim = ax.get_ylim()
        span = ylim[1] - ylim[0]
        for yr in aug_years:
            ax.axvline(yr, color="#FF6D00", lw=1.0, ls="--", alpha=0.6, zorder=2)
            ax.text(
                yr + 0.15,
                ylim[0] + span * ymax_frac,
                f"Yr {yr}",
                fontsize=6.5, color="#BF360C", va="top", rotation=90, alpha=0.8,
            )

    # ── Panel 1 — Stacked cohort capacity (full width) ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.stackplot(
        years,
        *[cohort_eff[c] for c in range(len(cohorts))],
        labels=[f"Cohort {c} (Yr {cohorts[c][0]}, {cohorts[c][1]} cont.)"
                for c in range(len(cohorts))],
        colors=[cohort_color(c) for c in range(len(cohorts))],
        alpha=0.88,
    )
    ax1.axhline(target_mwh, color="#1565c0", lw=1.4, ls="--",
                label=f"Restore target ({restore_pct:.0f}% = {target_mwh:.1f} MWh)")
    ax1.set_title(
        "BESS Effective Capacity by Cohort  [Trigger is CUF-based — see Panel 6]",
        fontweight="bold", fontsize=11,
    )
    ax1.set_xlabel("Project Year")
    ax1.set_ylabel("Effective Energy Capacity (MWh)")
    ax1.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.2)
    _add_aug_vlines(ax1)

    # ── Panel 2 — SOH curves ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for c_idx, (start_yr, _) in enumerate(cohorts):
        ax2.plot(
            years, cohort_soh_traj[c_idx],
            lw=1.2, ls="--", alpha=0.55,
            color=cohort_color(c_idx),
            label=f"Cohort {c_idx} (Yr {start_yr})",
        )
    ax2.plot(years, blended_soh, color="#212121", lw=2.2, label="Blended effective SOH", zorder=5)
    ax2.set_title("State of Health — Cohort Trajectories", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Project Year")
    ax2.set_ylabel("SOH")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.2)
    _add_aug_vlines(ax2)

    # ── Panel 3 — OPEX profile with augmentation spikes ──────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.stackplot(
        years, opex_solar, opex_wind, opex_bess, opex_trans, opex_land, opex_ins,
        labels=["Solar O&M", "Wind O&M", "BESS O&M", "Trans O&M", "Land Lease", "Insurance"],
        colors=["#fdd835", "#42a5f5", "#66bb6a", "#ab47bc", "#ff7043", "#78909c"],
        alpha=0.82,
    )
    if np.any(opex_aug_v > 0):
        ax3.bar(years, opex_aug_v, bottom=opex_reg, color="#FF6D00", alpha=0.92,
                width=0.6, label="Augmentation Purchase", zorder=5)
        for yr_idx, val in enumerate(opex_aug_v):
            if val > 0:
                ax3.annotate(
                    f"{val:.2f}Cr",
                    xy=(years[yr_idx], opex_reg[yr_idx] + val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#BF360C", fontweight="bold",
                )
    ax3.set_title("Annual OPEX incl. Augmentation Purchases", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Project Year")
    ax3.set_ylabel("Rs Crore / Year")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, alpha=0.2)

    # ── Panel 4 — Container fleet growth ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    cont_arr = np.array([eff_cont[yr] for yr in years])
    ax4.step(years, cont_arr, where="post", color="#2e7d32", lw=2.2, label="Total containers", zorder=4)
    ax4.fill_between(years, cont_arr, step="post", color="#66bb6a", alpha=0.25)
    for yr in aug_years:
        y_idx = yr - 1
        ax4.scatter(yr, cont_arr[y_idx], marker="D", s=60, color="#FF6D00",
                    zorder=6, label="Augmentation event" if yr == aug_years[0] else "")
        ax4.annotate(
            f"+{aug['containers_added_per_year'][yr]}",
            xy=(yr, cont_arr[y_idx]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color="#BF360C", fontweight="bold",
        )
    ax4.set_title("BESS Fleet Growth (Container Count)", fontweight="bold", fontsize=10)
    ax4.set_xlabel("Project Year")
    ax4.set_ylabel("Total Containers")
    ax4.yaxis.get_major_locator().set_params(integer=True)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # ── Panel 5 — Installed capacity & BESS O&M ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5b = ax5.twinx()
    ax5.bar(years, inst_arr, color="#90caf9", alpha=0.75, label="Installed MWh (nameplate)", zorder=3)
    ax5b.plot(years, bess_om_yr, color="#1565c0", lw=2.0, marker="o",
              markersize=3.5, label="BESS O&M cost (Rs Cr)", zorder=4)
    for yr in aug_years:
        ax5.axvline(yr, color="#FF6D00", lw=0.9, ls="--", alpha=0.55, zorder=2)
    ax5.set_title("BESS Installed Capacity & Annual O&M", fontweight="bold", fontsize=10)
    ax5.set_xlabel("Project Year")
    ax5.set_ylabel("Installed MWh (nameplate)", color="#546e7a")
    ax5b.set_ylabel("BESS O&M (Rs Crore / Year)", color="#1565c0")
    ax5b.tick_params(axis="y", labelcolor="#1565c0")
    h_bar, l_bar = ax5.get_legend_handles_labels()
    h_line, l_line = ax5b.get_legend_handles_labels()
    ax5.legend(h_bar + h_line, l_bar + l_line, fontsize=7.5, loc="upper left")
    ax5.grid(True, alpha=0.2)

    # ── Panel 6 — Plant CUF % trajectory (full width) ────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    has_cuf = np.any(cuf_arr > 0)
    if has_cuf:
        ax6.plot(years, cuf_arr, color="#0288d1", lw=2.2, marker="o",
                 markersize=3.5, label="Plant CUF (%)", zorder=4)
        if y1_cuf is not None:
            ax6.axhline(y1_cuf, color="#2e7d32", lw=1.3, ls="--",
                        label=f"Year-1 CUF ({y1_cuf:.2f}%)", zorder=3)
        if trig_cuf is not None:
            ax6.axhline(trig_cuf, color="#e53935", lw=1.4, ls=":",
                        label=f"Trigger CUF ({trigger_cuf_pct:.0f}% × Y1 = {trig_cuf:.2f}%)", zorder=3)
            ax6.fill_between(years, 0, trig_cuf, alpha=0.06, color="#e53935",
                             label="Below-trigger zone")
        # Mark augmentation years
        for yr in aug_years:
            y_idx = yr - 1
            ax6.scatter(yr, cuf_arr[y_idx], marker="^", s=70, color="#FF6D00",
                        zorder=6, label="Aug. event" if yr == aug_years[0] else "")
            ax6.annotate(
                f"Yr {yr}\n{cuf_arr[y_idx]:.2f}%",
                xy=(yr, cuf_arr[y_idx]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7, color="#BF360C", fontweight="bold",
            )
        ax6.set_ylim(bottom=max(cuf_arr.min() * 0.92, 0))
    else:
        ax6.text(
            0.5, 0.5, "CUF data not available (fast mode run)",
            ha="center", va="center", transform=ax6.transAxes,
            fontsize=11, color="#888888",
        )
    ax6.set_title("Plant CUF % Trajectory — Trigger & Augmentation Events", fontweight="bold", fontsize=11)
    ax6.set_xlabel("Project Year")
    ax6.set_ylabel("Plant CUF (%)")
    ax6.legend(fontsize=8, loc="lower left")
    ax6.grid(True, alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Augmentation profile plot saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Day-250 dispatch diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_day250(params: dict, config, data: dict, output_path: Path) -> None:
    """
    Four-panel BESS dispatch diagnostic for Day 250.

    Panels
    ──────
    1. Load coverage  — solar direct, BESS discharge, DISCOM shortfall (stacked)
    2. Charge / discharge flows (busbar)
    3. SOC trajectory with rsrv_evening, rsrv_morning_next, rsrv_fwd_evening
    4. DISCOM tariff rate per hour (context)
    """
    DAY = 250

    C_SOLAR    = "#F4A623"
    C_DISC     = "#4A90D9"
    C_CHG      = "#7ED321"
    C_SHORTFL  = "#D0021B"
    C_SOC      = "#9B59B6"
    C_RSRV_E   = "#E74C3C"
    C_RSRV_M   = "#3498DB"
    C_RSRV_FWD = "#E67E22"

    PERIOD_ALPHA  = 0.10
    PERIOD_COLORS = {
        "morning_peak":  "#FF6B6B",
        "solar_offpeak": "#FFD93D",
        "evening_peak":  "#FF6B6B",
        "normal":        "#AAAAAA",
    }
    RATES = {
        "morning_peak":  9.182,
        "solar_offpeak": 8.027,
        "evening_peak":  9.182,
        "normal":        8.687,
    }

    lf  = GridInterface(config).loss_factor
    pe  = PlantEngine(config, data)
    res = pe.simulate(loss_factor=lf, **params)
    load = data["load_profile"]

    s = DAY * 24
    hods = np.arange(24)

    solar_d = res["solar_direct_pre"][s:s+24]
    disc    = res["discharge_pre"][s:s+24]
    chg     = res["charge_pre"][s:s+24]
    load_d  = load[s:s+24]

    solar_meter = solar_d * lf
    disc_meter  = disc * lf          # discharge_pre is already post-efficiency; applying eff twice was wrong
    shortfall   = np.maximum(load_d - solar_meter - disc_meter, 0.0)

    # ── Pre-compute RE-only shortfall for reservation planner ─────────────
    re_shortfall = np.empty(len(load))
    for _h in range(len(load)):
        _sv  = params["solar_capacity_mw"] * data["solar_cuf"][_h]
        _req = load[_h] / lf
        _sd  = min(_sv, _req)
        _direct = min(_sd, params["ppa_capacity_mw"])
        re_shortfall[_h] = max(load[_h] - _direct * lf, 0.0)

    container_size   = config.bess["bess"]["container"]["size_mwh"]
    charge_eff       = pe.charge_eff
    discharge_eff    = pe.discharge_eff
    aux_per_hour     = config.bess["bess"]["container"]["auxiliary_consumption_mwh_per_day"] / 24
    energy_cap       = params["bess_containers"] * container_size
    charge_pw_cap    = params["charge_c_rate"]    * energy_cap
    discharge_pw_cap = params["discharge_c_rate"] * energy_cap

    # Fast-forward SOC to start of day 250
    soc = 0.0
    for h in range(s):
        soc += res["charge_pre"][h] * charge_eff
        if soc > 0:
            ac  = min(params["bess_containers"], math.ceil(soc / container_size))
            soc = max(soc - ac * aux_per_hour, 0.0)
        soc -= res["discharge_pre"][h] / discharge_eff
        soc  = max(soc, 0.0)

    # Replay day 250 recording SOC + reservations at start of each hour
    soc_arr      = np.zeros(25)
    rsrv_eve_arr = np.zeros(24)
    rsrv_morn_arr= np.zeros(24)
    rsrv_fwd_arr = np.zeros(24)

    rsrv_evening      = 0.0
    rsrv_morning_next = 0.0
    rsrv_fwd_evening  = 0.0

    for i in range(24):
        h   = s + i
        hod = i
        day_start_h = h - hod

        soc_arr[i] = soc

        if hod == 11:
            rsrv_morning_next = 0.0
            eve_idx = [day_start_h + ep for ep in [18,19,20,21] if day_start_h+ep < len(load)]
            if eve_idx and energy_cap > 0:
                soc_est = soc
                for _cp in [11,12,13,14]:
                    _ci = day_start_h + _cp
                    if _ci >= len(load): break
                    _c = min(re_shortfall[_ci], charge_pw_cap,
                             max(energy_cap - soc_est, 0.0)) * charge_eff
                    soc_est = min(soc_est + _c, energy_cap)
                eve_need = sum(re_shortfall[fh]/(discharge_eff*lf) for fh in eve_idx)
                eve_need = min(eve_need, discharge_pw_cap * len(eve_idx))
                rsrv_fwd_evening = min(eve_need, soc_est)

        elif hod == 15:
            rsrv_fwd_evening = 0.0
            eve_idx  = [day_start_h + ep for ep in [18,19,20,21] if day_start_h+ep < len(load)]
            morn_idx = [day_start_h+24+mp for mp in [7,8,9,10]   if day_start_h+24+mp < len(load)]
            eve_need  = sum(re_shortfall[fh]/(discharge_eff*lf) for fh in eve_idx) if eve_idx else 0.0
            eve_need  = min(eve_need, discharge_pw_cap * max(len(eve_idx),1))
            rsrv_evening = min(eve_need, soc)
            morn_need = sum(re_shortfall[fh]/(discharge_eff*lf) for fh in morn_idx) if morn_idx else 0.0
            morn_need = min(morn_need, discharge_pw_cap * max(len(morn_idx),1))
            rsrv_morning_next = min(morn_need, max(soc - rsrv_evening, 0.0))

        rsrv_eve_arr[i]  = rsrv_evening
        rsrv_morn_arr[i] = rsrv_morning_next
        rsrv_fwd_arr[i]  = rsrv_fwd_evening

        soc += chg[i] * charge_eff
        if soc > 0:
            ac  = min(params["bess_containers"], math.ceil(soc / container_size))
            soc = max(soc - ac * aux_per_hour, 0.0)
        disc_raw = disc[i] / discharge_eff
        if hod in pe.morning_peak_hods:
            rsrv_morning_next = max(rsrv_morning_next - disc_raw, 0.0)
        elif hod in pe.evening_peak_hods:
            rsrv_evening = max(rsrv_evening - disc_raw, 0.0)
        soc -= disc_raw
        soc  = max(soc, 0.0)

    soc_arr[24] = soc

    def period_of(h):
        if h in pe.morning_peak_hods:  return "morning_peak"
        if h in pe.evening_peak_hods:  return "evening_peak"
        if h in pe.solar_offpeak_hods: return "solar_offpeak"
        return "normal"

    periods = [period_of(h) for h in range(24)]
    rates   = [RATES[p] for p in periods]

    def shade_periods(ax):
        for i, p in enumerate(periods):
            ax.axvspan(i - 0.5, i + 0.5,
                       color=PERIOD_COLORS[p], alpha=PERIOD_ALPHA, zorder=0)
        for trigger in [10.5, 14.5]:
            ax.axvline(trigger, color="#555555", lw=1.2, ls="--", alpha=0.6, zorder=1)

    x     = np.arange(24)
    x_soc = np.arange(25) - 0.5

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14),
        gridspec_kw={"height_ratios": [2.5, 2, 2.5, 1.2]},
        sharex=True,
    )
    fig.suptitle(
        f"Day {DAY} — ToD-Aware BESS Dispatch\n"
        f"Solar {params['solar_capacity_mw']:.0f} MW  |  "
        f"BESS {params['bess_containers']} × {container_size} MWh = {energy_cap:.0f} MWh  |  "
        f"Charge C-rate {params['charge_c_rate']}  |  Discharge C-rate {params['discharge_c_rate']}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Panel 1 — load coverage
    ax = axes[0]
    shade_periods(ax)
    ax.bar(x, shortfall,   color=C_SHORTFL, label="DISCOM draw",    alpha=0.85, zorder=3)
    ax.bar(x, disc_meter,  color=C_DISC,    label="BESS discharge",  alpha=0.85, zorder=3,
           bottom=shortfall)
    ax.bar(x, solar_meter, color=C_SOLAR,   label="Solar direct",    alpha=0.85, zorder=3,
           bottom=shortfall + disc_meter)
    ax.axhline(load_d[0], color="#333333", lw=1.5, ls=":",
               label=f"Load ({load_d[0]:.1f} MWh/h)", zorder=4)
    ax.set_ylabel("MWh / hour", fontsize=10)
    ax.set_title("Panel 1 — How Load Is Met (meter basis, post-loss)", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, load_d[0] * 1.35)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2 — charge / discharge
    ax = axes[1]
    shade_periods(ax)
    ax.bar(x,     chg,  color=C_CHG,  label="Charge (into battery)",   alpha=0.85, zorder=3)
    ax.bar(x,    -disc, color=C_DISC, label="Discharge (from battery)", alpha=0.85, zorder=3)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("MWh / hour", fontsize=10)
    ax.set_title("Panel 2 — BESS Charge / Discharge (busbar)", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _ylim = max(disc.max(), chg.max()) * 1.4
    ax.set_ylim(-_ylim, _ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        "Zero discharge\n(rsrv_fwd_evening\nblocking offpeak)",
        xy=(12.5, -5), xytext=(12.5, -chg.max() * 0.6),
        ha="center", va="top", fontsize=7.5, color="#B8860B",
        arrowprops=dict(arrowstyle="->", color="#B8860B", lw=1),
    )

    # Panel 3 — SOC + reservations
    ax = axes[2]
    shade_periods(ax)
    ax.step(x_soc, soc_arr, where="post", color=C_SOC, lw=2.2, label="SOC", zorder=4)
    ax.fill_between(x_soc, soc_arr, step="post", color=C_SOC, alpha=0.15)
    ax.fill_between(x - 0.5, rsrv_eve_arr,  step="post",
                    color=C_RSRV_E,   alpha=0.30, label="rsrv_evening")
    ax.fill_between(x - 0.5, rsrv_morn_arr, step="post",
                    color=C_RSRV_M,   alpha=0.30, label="rsrv_morning_next")
    ax.fill_between(x - 0.5, rsrv_fwd_arr,  step="post",
                    color=C_RSRV_FWD, alpha=0.30, label="rsrv_fwd_evening")
    ax.axhline(energy_cap, color="#999999", ls="--", lw=1,
               label=f"Capacity ({energy_cap:.0f} MWh)")
    ax.set_ylabel("MWh", fontsize=10)
    ax.set_title("Panel 3 — SOC Trajectory and Reservations", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, energy_cap * 1.15)
    ax.grid(axis="y", alpha=0.3)
    for tx, lbl in [(10.5, "Trigger 1\nhod=11"), (14.5, "Trigger 2\nhod=15")]:
        ax.annotate(
            lbl, xy=(tx, energy_cap * 0.55), fontsize=7.5, ha="center", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.8),
        )

    # Panel 4 — tariff
    ax = axes[3]
    shade_periods(ax)
    bars = ax.bar(x, rates,
                  color=[PERIOD_COLORS[p] for p in periods],
                  alpha=0.8, zorder=3, edgecolor="white", lw=0.5)
    ax.set_ylabel("₹ / kWh", fontsize=10)
    ax.set_title("Panel 4 — DISCOM Tariff Rate (LT)", fontsize=10, loc="left")
    ax.set_ylim(7.5, 9.8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    legend_patches = [
        mpatches.Patch(color=PERIOD_COLORS["morning_peak"],  alpha=0.5, label="Morning Peak  ₹9.182"),
        mpatches.Patch(color=PERIOD_COLORS["solar_offpeak"], alpha=0.5, label="Solar Offpeak  ₹8.027"),
        mpatches.Patch(color=PERIOD_COLORS["evening_peak"],  alpha=0.5, label="Evening Peak  ₹9.182"),
        mpatches.Patch(color=PERIOD_COLORS["normal"],        alpha=0.5, label="Normal  ₹8.687"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9,
               title="ToD Period", title_fontsize=8.5)

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Day-250 dispatch plot saved → {output_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config         = load_config()
    data           = load_timeseries_data(config)
    energy_engine  = Year1Engine(config, data)
    finance_engine = FinanceEngine(config, data)
    solver         = SolverEngine(config, data, energy_engine, finance_engine)

    # ── Phase 1: Solver (finds optimal plant without oversizing) ─────────────
    n_trials = config.solver["solver"].get("n_trials", 300)
    print(f"\nPhase 1 — Running solver ({n_trials} trials) …")
    result = solver.run(n_trials=n_trials, show_progress=True)

    params = result.best_params
    y1     = result.full_result["year1"]
    fi     = result.full_result["finance"]

    # ── Phase 2: Oversizing (add extra containers on top of solver-optimal) ──
    aug_cfg      = config.bess["bess"]["augmentation"]
    oversize_pct = float(aug_cfg.get("oversize_percent", 0.0))
    n_opt        = params["bess_containers"]
    n_extra      = max(math.ceil(n_opt * oversize_pct / 100), 0) if oversize_pct > 0 else 0

    if n_extra > 0:
        print(
            f"\nPhase 2 — Oversizing: adding {n_extra} container(s) "
            f"({oversize_pct:.1f}% × {n_opt} opt.) → "
            f"{n_opt + n_extra} total containers …"
        )
        params_oversize = {**params, "bess_containers": n_opt + n_extra}
        y1_oversize     = energy_engine.evaluate(**params_oversize)
        fi_oversize     = finance_engine.evaluate(
            y1_oversize,
            solar_capacity_mw = params_oversize["solar_capacity_mw"],
            wind_capacity_mw  = params_oversize["wind_capacity_mw"],
            ppa_capacity_mw   = params_oversize["ppa_capacity_mw"],
            fast_mode         = False,
        )
        params = params_oversize
        y1     = y1_oversize
        fi     = fi_oversize
    else:
        print(f"\nPhase 2 — Oversizing disabled (oversize_percent = {oversize_pct:.1f}%)")

    print_section1(params, y1, fi)
    print_section2(fi)
    print_section3(params, y1, fi, data, energy_engine)
    print_section4(fi)
    print_section5(fi)
    print_section6(fi)
    print_section7(fi)
    print_section8(fi, config, n_opt=n_opt, n_extra=n_extra)

    sep("SOLVER STATS")
    print(f"\n  {'Trials completed':<38} : {result.n_trials_completed}")
    print(f"  {'Feasible trials':<38} : {result.n_trials_feasible}")

    outputs_dir = find_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    aug_years = fi["augmentation"]["augmentation_years"] if fi["augmentation"]["enabled"] else []
    plot_dashboard(params, y1, fi, data, outputs_dir / "model_output.png",
                   augmentation_years=aug_years)
    plot_day250(params, config, data, outputs_dir / "day250_dispatch.png")
    if fi["augmentation"]["enabled"]:
        plot_augmentation(fi, config, outputs_dir / "augmentation_profile.png")