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

    charge_pre       = float(np.sum(y1["charge_pre"]))
    charge_loss      = float(np.sum(y1["charge_loss"]))
    discharge_loss   = float(np.sum(y1["discharge_loss"]))
    aux_loss         = float(np.sum(y1["aux_loss"]))
    end_soc          = float(y1.get("bess_end_soc_mwh", 0.0))

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

    # Plant CUF — transparent "naive" formula, the single source of truth.
    from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf as _plant_cuf
    plant_cuf_val = _plant_cuf(total_busbar, params["ppa_capacity_mw"])

    solar_cuf   = compute_cuf(raw_solar, params["solar_capacity_mw"])
    wind_cuf    = compute_cuf(raw_wind,  params["wind_capacity_mw"])
    loss_factor = float(y1["loss_factor"])
    ppa         = params["ppa_capacity_mw"]

    # PPA envelope and utilisation
    ppa_envelope_mwh = ppa * 8760

    print(f"\n  {'── GENERATION (Raw, Pre-Dispatch)'}")
    print(f"  {'Raw Solar Generation (MWh)':<38} : {round(raw_solar, 1)}")
    print(f"  {'Raw Wind Generation (MWh)':<38} : {round(raw_wind, 1)}")
    print(f"  {'Raw Total Generation (MWh)':<38} : {round(raw_solar + raw_wind, 1)}")
    print(f"\n  {'── BUSBAR (Pre-Loss)'}")
    print(f"  {'Solar Direct (MWh)':<38} : {round(solar_direct_pre, 1)}")
    print(f"  {'Wind Direct (MWh)':<38} : {round(wind_direct_pre, 1)}")
    print(f"  {'BESS Discharge (MWh)':<38} : {round(discharge_pre, 1)}")
    print(f"  {'Total Busbar Delivery (MWh)':<38} : {round(total_busbar, 1)}")
    print(f"  {'Excess Energy / Curtailment (MWh)':<38} : {round(curtailment, 1)}")
    print(f"\n  {'── BESS FLOWS (Pre-Loss)'}")
    print(f"  {'BESS Charging (MWh)':<38} : {round(charge_pre, 1)}")
    print(f"  {'Charge Loss (ηc=1-{:.4f}) (MWh)'.format(1-energy_engine.plant.charge_eff):<38} : {round(charge_loss, 1)}")
    print(f"  {'Discharge Loss (ηd=1-{:.4f}) (MWh)'.format(1-energy_engine.plant.discharge_eff):<38} : {round(discharge_loss, 1)}")
    print(f"  {'Aux Consumption (MWh)':<38} : {round(aux_loss, 1)}")
    print(f"  {'End-of-Year SOC (MWh)':<38} : {round(end_soc, 1)}")
    print(f"\n  {'── METER (Post-Loss)'}")
    print(f"  {'Grid Loss Factor':<38} : {round(loss_factor, 4)}")
    print(f"  {'Solar Direct at Meter (MWh)':<38} : {round(solar_direct_m, 1)}")
    print(f"  {'Wind Direct at Meter (MWh)':<38} : {round(wind_direct_m, 1)}")
    print(f"  {'BESS Discharge at Meter (MWh)':<38} : {round(discharge_m, 1)}")
    print(f"  {'Total RE at Meter (MWh)':<38} : {round(total_meter, 1)}")
    print(f"\n  {'── LOAD & DISCOM DEPENDENCY'}")
    print(f"  {'Annual Load (MWh)':<38} : {round(annual_load_mwh, 1)}")
    print(f"  {'DISCOM Draw (MWh)':<38} : {round(discom_draw_mwh, 1)}")
    print(f"  {'DISCOM Cost Year-1 (Rs Crore)':<38} : {cr(discom_cost_y1)}")
    print(f"  {'RE Penetration (%)':<38} : {round(re_penetration, 2)} %")
    print(f"\n  {'── CAPACITY UTILISATION (CUF)'}")
    print(f"  {'Solar CUF (%)':<38} : {round(solar_cuf, 2) if solar_cuf else 'N/A'}")
    print(f"  {'Wind CUF (%)':<38} : {round(wind_cuf, 2) if wind_cuf else 'N/A'}")
    print(f"  {'PPA Envelope (MWh = PPA×8760)':<38} : {round(ppa_envelope_mwh, 1)}")
    print(f"  {'Plant CUF (%) = busbar / PPA×8760':<38} : {round(plant_cuf_val, 2)}")


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

    # True capacity charge per meter kWh (now exposed explicitly by LandedTariffModel)
    cap_series    = lt.get("capacity_charge_per_kwh_series", None)
    markup_series = lt.get("lcoe_markup_per_kwh_series", None)

    if cap_series is not None:
        cap_y1, cap_y25       = cap_series[0], cap_series[-1]
    else:
        cap_y1 = cap_y25 = 0.0

    if markup_series is not None:
        markup_y1, markup_y25 = markup_series[0], markup_series[-1]
    else:
        markup_y1 = markup_y25 = 0.0

    print(f"\n  {'── LCOE BUILD-UP (NPV basis)'}")
    print(f"  {'NPV Total Cost (Rs Crore)':<38} : {cr(lcd['npv_total_cost'])}")
    print(f"    {'↳ Interest':<36} : {cr(lcd['npv_interest'])}")
    print(f"    {'↳ Principal':<36} : {cr(lcd['npv_principal'])}")
    print(f"    {'↳ ROE':<36} : {cr(lcd['npv_roe'])}")
    print(f"    {'↳ OPEX':<36} : {cr(lcd['npv_opex'])}")
    print(f"  {'NPV Busbar Energy (Bn kWh)':<38} : {round(lcd['npv_energy_kwh'] / 1e9, 4)}")
    print(f"  {'LCOE (Rs/kWh)':<38} : {round(lcoe, 4)}")

    print(f"\n  {'── LANDED TARIFF BUILD-UP (per meter kWh)'}")
    print(f"  {'Component':<38}    {'Year 1':>9}    {'Year 25':>9}")
    print(f"  {'LCOE (busbar basis)':<38} : {round(lcoe, 4):>9}    {round(lcoe, 4):>9}")
    print(f"  {'+ Busbar→Meter LCOE markup':<38} : {round(markup_y1, 4):>9}    {round(markup_y25, 4):>9}")
    print(f"  {'+ Wheeling charge':<38} : {round(wheeling, 4):>9}    {round(wheeling, 4):>9}")
    print(f"  {'+ Electricity tax':<38} : {round(elec_tax, 4):>9}    {round(elec_tax, 4):>9}")
    print(f"  {'+ Banking charge':<38} : {round(banking, 4):>9}    {round(banking, 4):>9}")
    print(f"  {'+ Capacity charge (CTU+STU+SLDC)':<38} : {round(cap_y1, 4):>9}    {round(cap_y25, 4):>9}")
    print(f"  {'= Landed Tariff (Rs/kWh)':<38} : {round(lts[0], 4):>9}    {round(lts[-1], 4):>9}")
    print(f"\n  {'DISCOM Tariff wt-avg (Rs/kWh)':<38} : {round(sv['discom_tariff'], 4)}")
    print(f"  {'Saving vs DISCOM Y1 (Rs/kWh)':<38} : {round(sv['discom_tariff'] - lts[0], 4)}")
    print(f"  {'Saving vs DISCOM Y25 (Rs/kWh)':<38} : {round(sv['discom_tariff'] - lts[-1], 4)}")


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
        ("Solar O&M",             "solar_om"),
        ("Wind O&M",              "wind_om"),
        ("BESS O&M",              "bess_om"),
        ("Solar Transmission O&M","solar_transmission_om"),
        ("Wind Transmission O&M", "wind_transmission_om"),
        ("Land Lease",            "land_lease"),
        ("Insurance",             "insurance"),
    ]
    print(f"\n  {'Component':<36}  {'Year 1':>10}  {'Year 25':>10}  {'Δ':>10}")
    sep()
    for label, key in cols:
        v1  = cr(ob_y1[key])
        v25 = cr(ob_y25[key])
        print(f"  {label:<36}  {v1:>10}  {v25:>10}  {round(v25 - v1, 4):>+10}")
    sep()
    print(f"  {'Total OPEX':<36}  {cr(ob_y1['total']):>10}  {cr(ob_y25['total']):>10}  {round(cr(ob_y25['total'])-cr(ob_y1['total']),4):>+10}")


def print_section7(fi, data, params, aug_data=None):
    sep("SECTION 7a — PER-YEAR ENERGY FLOWS (MWh, busbar)")
    ep  = fi["energy_projection"]

    # Per-year operating values (for the reference columns)
    from hybrid_plant.data_loader import operating_value, load_soh_curve
    from pathlib import Path
    root = find_project_root()
    import pandas as pd
    solar_eff_df = pd.read_csv(root / "data/solar_efficiency_curve.csv")
    wind_eff_df  = pd.read_csv(root / "data/wind_efficiency_curve.csv")
    solar_eff_curve = dict(zip(solar_eff_df["year"].astype(int), solar_eff_df["efficiency"]))
    wind_eff_curve  = dict(zip(wind_eff_df["year"].astype(int),  wind_eff_df["efficiency"]))
    # BESS blended SOH comes from cohorts; reconstruct via ratios if aug_data present
    soh_curve = load_soh_curve(None) if aug_data is not None else None

    # Reconstruct cohort containers per year
    container_counts = []
    init_c = params["bess_containers"]
    events = aug_data["event_log"] if aug_data else []
    cum = init_c
    for y in range(1, 26):
        for ev in events:
            if ev["year"] == y:
                cum += ev["k_containers"]
        container_counts.append(cum)

    print(f"\n  {'Yr':>3}  {'Cont':>5}  {'Solar Eff':>9}  {'Wind Eff':>9}  "
          f"{'Solar MWh':>11}  {'Wind MWh':>11}  {'BESS MWh':>11}  {'Busbar MWh':>11}  {'Meter MWh':>10}  {'Loss MWh':>10}")
    sep()
    for y in range(25):
        yr = y + 1
        s_eff = operating_value(solar_eff_curve, yr)
        w_eff = operating_value(wind_eff_curve,  yr)
        busbar = ep["delivered_pre_mwh"][y]
        meter  = ep["delivered_meter_mwh"][y]
        loss   = busbar - meter
        print(f"  {yr:>3}  {container_counts[y]:>5}  {s_eff:>9.4f}  {w_eff:>9.4f}  "
              f"{ep['solar_direct_mwh'][y]:>11.1f}  {ep['wind_direct_mwh'][y]:>11.1f}  "
              f"{ep['battery_mwh'][y]:>11.1f}  {busbar:>11.1f}  {meter:>10.1f}  {loss:>10.1f}")
    sep()

    # ─────────────────────────────────────────────────────────────────────
    sep("SECTION 7b — PER-YEAR LOAD, DISCOM, CUF")
    annual_load = float(np.sum(data["load_profile"]))
    ppa_env     = params["ppa_capacity_mw"] * 8760
    cuf_series  = aug_data.get("cuf_series", None) if aug_data else None

    raw_solar_annual = float(np.sum(params["solar_capacity_mw"] * data["solar_cuf"]))
    raw_wind_annual  = float(np.sum(params["wind_capacity_mw"]  * data["wind_cuf"]))

    print(f"\n  Annual Load (MWh)          : {annual_load:.1f}")
    print(f"  PPA Envelope (MWh)         : {ppa_env:.1f}")
    print(f"  Raw Solar @ nameplate Y1   : {raw_solar_annual:.1f}")
    print(f"  Raw Wind  @ nameplate Y1   : {raw_wind_annual:.1f}")

    print(f"\n  {'Yr':>3}  {'Meter MWh':>10}  {'DISCOM MWh':>10}  {'RE Pen%':>7}  "
          f"{'Solar CUF%':>10}  {'Plant CUF%':>10}  {'PPA Util%':>9}")
    sep()
    for y in range(25):
        yr = y + 1
        meter  = ep["delivered_meter_mwh"][y]
        busbar = ep["delivered_pre_mwh"][y]
        discom = max(annual_load - meter, 0)
        re_pen = meter / annual_load * 100 if annual_load else 0
        baseline_solar_cuf = raw_solar_annual / (params["solar_capacity_mw"] * 8760) * 100 if params["solar_capacity_mw"] else 0
        s_eff  = operating_value(solar_eff_curve, yr)
        solar_cuf_y = baseline_solar_cuf * s_eff
        plant_cuf   = cuf_series[y] if cuf_series else busbar / ppa_env * 100
        ppa_util    = busbar / ppa_env * 100
        print(f"  {yr:>3}  {meter:>10.1f}  {discom:>10.1f}  {re_pen:>6.2f}%  "
              f"{solar_cuf_y:>9.2f}%  {plant_cuf:>9.2f}%  {ppa_util:>8.2f}%")
    sep()

    # ─────────────────────────────────────────────────────────────────────
    sep("SECTION 7c — PER-YEAR ECONOMICS (Rs Crore unless noted)")
    sv   = fi["savings_breakdown"]
    lts  = fi["landed_tariff_series"]
    annual_savings = sv["annual_savings"]
    opex_proj      = fi["opex_projection"]
    baseline_annual = sv["baseline_annual_cost"]

    # Lump & recurring aug break-out (if available)
    aug_lump = aug_data["opex_augmentation_lump"] if aug_data else [0.0]*25
    aug_om   = aug_data["opex_augmentation_om"]   if aug_data else [0.0]*25

    print(f"\n  Baseline annual cost (flat, Rs Cr) : {baseline_annual/CRORE_TO_RS:.4f}")

    print(f"\n  {'Yr':>3}  {'Base OPEX':>9}  {'Aug Lump':>9}  {'Aug O&M':>8}  {'Tot OPEX':>9}  "
          f"{'Landed':>8}  {'Hybrid':>9}  {'Savings':>9}  {'Cum Sav':>10}")
    sep()
    cum = 0
    for y in range(25):
        yr = y + 1
        total_opex    = opex_proj[y]
        base_opex     = total_opex - aug_lump[y] - aug_om[y]
        hybrid        = sv["annual_hybrid_cost"][y]
        savings       = annual_savings[y]
        cum          += savings
        print(f"  {yr:>3}  {base_opex/CRORE_TO_RS:>9.4f}  {aug_lump[y]/CRORE_TO_RS:>9.4f}  "
              f"{aug_om[y]/CRORE_TO_RS:>8.4f}  {total_opex/CRORE_TO_RS:>9.4f}  "
              f"{lts[y]:>8.4f}  {hybrid/CRORE_TO_RS:>9.4f}  {savings/CRORE_TO_RS:>9.4f}  "
              f"{cum/CRORE_TO_RS:>10.4f}")
    sep()

def print_section6b(aug_data: dict, baseline_result, fi: dict) -> None:
    """Section 6b — Augmentation Timeline & Economics."""
    sep("SECTION 6b — AUGMENTATION")
    events          = aug_data["event_log"]
    init_cont       = aug_data["initial_containers"]
    container_size  = float(fi.get("_container_size_mwh", 5.015))
    solver_b_star   = baseline_result.best_params["bess_containers"]

    total_added = aug_data["total_containers_added"]
    n_events    = aug_data["n_events"]

    lump_total = aug_data["total_lump_cost_rs"]
    om_total   = aug_data["total_om_cost_rs"]

    # Oversize sweep metadata (present when oversizing ran)
    extra_containers = aug_data.get("extra_containers_oversized", init_cont - solver_b_star)
    sweep_log        = aug_data.get("oversize_sweep_log", [])

    print(f"\n  {'Trigger threshold CUF (Pass-1 baseline Y1)':<46} : {aug_data['trigger_threshold_cuf']:.4f} %")
    print()
    print(f"  {'Pass-1 solver containers (B*)':<46} : {solver_b_star}")
    print(f"  {'Extra containers added upfront (oversize)':<46} : {extra_containers}")
    print(f"  {'Initial BESS containers (B* + extra)':<46} : {init_cont}")
    print()

    if sweep_log:
        n_candidates = len(sweep_log)
        best_entry   = next((e for e in sweep_log if e["extra"] == extra_containers), None)
        best_npv_cr  = cr(best_entry["npv"]) if best_entry else float("nan")
        print(f"  OVERSIZE SWEEP SUMMARY")
        print(f"  {'Candidates evaluated':<46} : {n_candidates}")
        print(f"  {'Best extra':<46} : {extra_containers}")
        print(f"  {'Best oversized NPV':<46} : Rs {best_npv_cr:.2f} Cr")
        if sweep_log:
            print(f"\n  Sweep log (first 8 + last 3 entries):")
            shown = sweep_log[:8]
            if len(sweep_log) > 11:
                shown_tail = sweep_log[-3:]
            elif len(sweep_log) > 8:
                shown_tail = sweep_log[8:]
            else:
                shown_tail = []
            print(f"  {'Extra':>6}  {'InitCont':>8}  {'NPV (Cr)':>10}  {'Events':>7}  {'Reason'}")
            sep()
            for entry in shown:
                flag = " ◀ BEST" if entry["extra"] == extra_containers else ""
                print(f"  {entry['extra']:>6}  {entry['initial_containers']:>8}  "
                      f"{cr(entry['npv']):>10.2f}  {entry['n_events']:>7}  "
                      f"{entry.get('terminated_reason') or '':>10}{flag}")
            if shown_tail:
                print(f"  {'...':<6}")
                for entry in shown_tail:
                    flag = " ◀ BEST" if entry["extra"] == extra_containers else ""
                    print(f"  {entry['extra']:>6}  {entry['initial_containers']:>8}  "
                          f"{cr(entry['npv']):>10.2f}  {entry['n_events']:>7}  "
                          f"{entry.get('terminated_reason') or '':>10}{flag}")
            sep()
    print()
    print(f"  {'Augmentation events fired':<46} : {n_events}")
    print(f"  {'Total containers added (events only)':<46} : {total_added}")
    print(f"  {'Total augmentation lump-sum cost':<46} : Rs {cr(lump_total):.2f} Cr  (OPEX, undepreciated)")
    print(f"  {'Total augmentation O&M (lifetime)':<46} : Rs {cr(om_total):.2f} Cr")

    if events:
        print(f"\n  AUGMENTATION EVENT TIMELINE")
        print(f"  (Upfront oversize: {solver_b_star} → {init_cont} containers at Y1; "
              f"event-year additions shown below)\n")
        hdr = (
            f"  {'Yr':>4}  {'Pre-CUF':>9}  {'Cont':>5}  {'New MWh':>8}  "
            f"{'Lump (Cr)':>10}  {'Post-CUF':>9}  {'≥Thresh?':>9}  {'CumCont':>8}"
        )
        print(hdr)
        sep()
        cum_containers = init_cont
        for ev in events:
            k           = ev["k_containers"]
            cum_containers += k
            new_mwh     = k * container_size
            reached     = "Y" if ev.get("reached_hard", False) else "N"
            print(
                f"  {ev['year']:>4}  "
                f"{ev['trigger_cuf']:>8.4f}%  "
                f"{k:>5}  "
                f"{new_mwh:>8.2f}  "
                f"{cr(ev['lump_cost_rs']):>10.2f}  "
                f"{ev['post_event_cuf']:>8.4f}%  "
                f"{reached:>9}  "
                f"{cum_containers:>8}"
            )
        sep()
        print(f"  ≥Thresh? = post-event CUF ≥ hard threshold")
    else:
        print("\n  (No augmentation events triggered over 25-year life)")



# ─────────────────────────────────────────────────────────────────────────────
# Augmentation dashboard plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_augmentation_dashboard(
    aug_data:         dict,
    baseline_cuf_series: list[float] | None,
    output_path:      Path,
) -> None:
    """
    Generate a 3-panel augmentation dashboard PNG.

    Panel 1 — Plant CUF curve with trigger threshold line and event markers
    Panel 2 — Cohort effective capacity stacked area
    Panel 3 — Annual savings bars + lump-sum event costs + cumulative line
    """
    from hybrid_plant.constants import CRORE_TO_RS

    years      = np.arange(1, 26)
    cuf_series = np.array(aug_data["cuf_series"])
    events     = aug_data["event_log"]
    threshold  = aug_data["trigger_threshold_cuf"]
    timeline   = aug_data["cohort_capacity_timeline"]
    lump_series = np.array(aug_data["opex_augmentation_lump"]) / CRORE_TO_RS

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle("BESS Augmentation Dashboard — 25-Year Lifecycle", fontsize=13, fontweight="bold")

    COLORS = ["#4c9be8", "#66bb6a", "#f4a832", "#ab47bc", "#ef5350", "#26a69a"]

    # ── Panel 1: Plant CUF ───────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(years, cuf_series, color="#1565C0", lw=2.0, label="Scenario CUF (with aug)", zorder=3)

    if baseline_cuf_series is not None:
        ax1.plot(years, baseline_cuf_series, color="#BDBDBD", lw=1.5,
                 ls="--", label="Baseline (no aug)", zorder=2)

    ax1.axhline(threshold, color="#e53935", lw=1.5, ls="--",
                label=f"Hard threshold  {threshold:.2f}%", zorder=3)

    for ev in events:
        ax1.axvline(ev["year"], color="#FF8F00", lw=1.2, ls="--", alpha=0.8, zorder=2)
        ax1.annotate(
            f"+{ev['k_containers']}c",
            xy=(ev["year"], ev["post_event_cuf"]),
            xytext=(ev["year"] + 0.4, ev["post_event_cuf"] + 0.5),
            fontsize=8, color="#E65100",
            arrowprops=dict(arrowstyle="->", color="#E65100", lw=0.8),
        )

    ax1.set_ylabel("Plant CUF (%)", fontsize=10)
    ax1.set_title("Panel 1 — Plant CUF Curve", fontsize=10, loc="left")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(max(0, threshold - 5), max(cuf_series) + 2)

    # ── Panel 2: Cohort capacity stacked area ────────────────────────────────
    ax2 = axes[1]
    bottoms = np.zeros(25)
    for cohort_idx in sorted(timeline.keys()):
        cap = np.array(timeline[cohort_idx])
        color = COLORS[cohort_idx % len(COLORS)]
        label = f"Cohort {cohort_idx}" if cohort_idx == 0 else f"Aug Y{events[cohort_idx-1]['year']}"
        ax2.fill_between(years, bottoms, bottoms + cap,
                         color=color, alpha=0.70, label=label, step="mid")
        bottoms += cap

    for ev in events:
        ax2.axvline(ev["year"], color="#FF8F00", lw=1.2, ls="--", alpha=0.6, zorder=5)

    ax2.set_ylabel("Effective Capacity (MWh)", fontsize=10)
    ax2.set_title("Panel 2 — Cohort Effective Capacity Stack", fontsize=10, loc="left")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Annual savings + lump costs + cumulative ────────────────────
    ax3   = axes[2]
    ax3r  = ax3.twinx()

    # Annual savings from finance result (injected via caller)
    # Caller passes aug_data with annual_savings if available; otherwise skip
    if "annual_savings_cr" in aug_data:
        ann_sav = np.array(aug_data["annual_savings_cr"])
        ax3.bar(years, ann_sav, color="#26a69a", alpha=0.75, label="Annual savings", zorder=3)
        cumul   = np.cumsum(ann_sav)
        ax3r.plot(years, cumul, color="#7b1fa2", lw=2.0, label="Cumulative savings", zorder=4)
        ax3r.set_ylabel("Cumulative Savings (Rs Cr)", fontsize=9)
        ax3r.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Lump-sum costs at event years
    mask = lump_series > 0
    if mask.any():
        ax3.bar(years[mask], -lump_series[mask], color="#ef5350", alpha=0.8,
                label="Augmentation lump cost", zorder=4)

    ax3.axhline(0, color="#333333", lw=0.8)
    ax3.set_ylabel("Rs Crore", fontsize=10)
    ax3.set_xlabel("Project Year", fontsize=10)
    ax3.set_title("Panel 3 — Annual Cash Impact", fontsize=10, loc="left")
    ax3.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax3.set_xticks(years[::2])
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Augmentation dashboard saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(params, y1, fi, data, output_path: Path) -> None:
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

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Hybrid RE Plant — Model Output Dashboard\n"
        f"Solar {round(params['solar_capacity_mw'],1)} MW  |  "
        f"Wind {round(params['wind_capacity_mw'],1)} MW  |  "
        f"BESS {round(float(y1['energy_capacity_mwh']),1)} MWh  |  "
        f"PPA {round(params['ppa_capacity_mw'],1)} MW",
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

    # Panel 4: OPEX stack
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.stackplot(years, opex_solar, opex_wind, opex_bess, opex_trans, opex_land, opex_ins,
                  labels=["Solar O&M","Wind O&M","BESS O&M","Transmission O&M","Land Lease","Insurance"],
                  colors=["#fdd835","#42a5f5","#66bb6a","#ab47bc","#ff7043","#78909c"], alpha=0.85)
    ax4.set_title("OPEX Stack Over 25 Years", fontweight="bold")
    ax4.set_xlabel("Year"); ax4.set_ylabel("OPEX (Rs Crore)")
    ax4.legend(fontsize=8, loc="upper left"); ax4.grid(True, alpha=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard plot saved → {output_path}")
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

    aug_cfg     = config.bess["bess"]["augmentation"]
    aug_enabled = bool(aug_cfg.get("enabled", False))

    n_trials = config.solver["solver"].get("n_trials", 300)

    # ── PASS 1: Baseline solver ───────────────────────────────────────────────
    print(f"\n[Pass 1] Baseline solver ({n_trials} trials) …")
    baseline_result = solver.run(n_trials=n_trials, show_progress=True)

    aug_data         = None
    baseline_cuf_ser = None

    if aug_enabled:
        from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf as _cuf_fn, year1_busbar_mwh
        from hybrid_plant.data_loader import load_soh_curve, operating_value
        from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine
        from hybrid_plant.augmentation.oversize_optimizer import find_optimal_oversize

        # Guard: if solver chose bess_containers=0, augmentation has nothing to work with
        if baseline_result.best_params["bess_containers"] == 0:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Augmentation enabled but solver chose bess_containers=0. "
                "Skipping augmentation path."
            )
            aug_enabled = False
        else:
            soh_curve = load_soh_curve(config)

            # trigger_threshold_cuf = pre-oversize Y1 CUF (fixed hard floor)
            b_y1 = baseline_result.full_result.get("year1")
            if b_y1 is None:
                b_y1 = energy_engine.evaluate(**baseline_result.best_params)
            trigger_threshold_cuf = _cuf_fn(
                year1_busbar_mwh(b_y1),
                baseline_result.best_params["ppa_capacity_mw"],
            )
            print(f"  [Aug] Trigger threshold CUF (Pass-1 Y1): {trigger_threshold_cuf:.4f}%")

            aug_engine = AugmentationEngine(
                config, data, energy_engine, soh_curve,
                trigger_threshold_cuf = trigger_threshold_cuf,
            )

            # Compute NO-AUGMENTATION baseline CUF series for the dashboard plot
            def _load_curve_local(cfg_path: str, column: str) -> dict:
                import pandas as _pd
                root = find_project_root()
                df = _pd.read_csv(root / cfg_path)
                df.columns = df.columns.str.strip().str.lower()
                return dict(zip(df["year"].astype(int), df[column.lower()]))

            solar_eff_curve = _load_curve_local(
                config.project["generation"]["solar"]["degradation"]["file"], "efficiency"
            )
            wind_eff_curve = _load_curve_local(
                config.project["generation"]["wind"]["degradation"]["file"], "efficiency"
            )
            _base_params = baseline_result.best_params
            _ppa         = _base_params["ppa_capacity_mw"]
            baseline_cuf_ser = []
            for yr in range(1, 26):
                _s_eff = operating_value(solar_eff_curve, yr)
                _w_eff = operating_value(wind_eff_curve,  yr)
                _soh_y = operating_value(soh_curve,       yr)
                _sim = energy_engine.plant.simulate(
                    solar_capacity_mw  = _base_params["solar_capacity_mw"] * _s_eff,
                    wind_capacity_mw   = _base_params["wind_capacity_mw"]  * _w_eff,
                    bess_containers    = _base_params["bess_containers"],
                    charge_c_rate      = _base_params["charge_c_rate"],
                    discharge_c_rate   = _base_params["discharge_c_rate"],
                    ppa_capacity_mw    = _ppa,
                    dispatch_priority  = _base_params["dispatch_priority"],
                    bess_charge_source = _base_params["bess_charge_source"],
                    loss_factor        = energy_engine.grid.loss_factor,
                    bess_soh_factor    = _soh_y,
                )
                _busbar_yr = (float(np.sum(_sim["solar_direct_pre"]))
                              + float(np.sum(_sim["wind_direct_pre"]))
                              + float(np.sum(_sim["discharge_pre"])))
                baseline_cuf_ser.append(_cuf_fn(_busbar_yr, _ppa))

            # ── Oversize sweep ────────────────────────────────────────────────
            print("\n[Augmentation] Running oversize sweep (full mode) …")
            os_result = find_optimal_oversize(
                augmentation_engine  = aug_engine,
                base_params          = baseline_result.best_params,
                threshold_cuf        = trigger_threshold_cuf,
                max_extra_containers = int(aug_cfg.get("max_oversize_containers", 500)),
                patience             = int(aug_cfg.get("oversize_patience", 3)),
                tolerance            = float(aug_cfg.get("oversize_npv_tolerance_rs", 1e3)),
            )
            print(f"  [Aug] Oversize sweep done: best_extra={os_result.best_extra}  "
                  f"initial={os_result.best_initial_containers}  "
                  f"candidates={len(os_result.sweep_log)}")

            # Merge best-oversize finance into baseline_result for dashboard
            baseline_result.full_result["finance"] = os_result.best_result["finance"]
            baseline_result.full_result["year1"]   = os_result.best_result["year1"]
            aug_data = os_result.best_result["finance"]["augmentation"]
            # Attach sweep metadata for Section 6b
            aug_data["oversize_sweep_log"]            = os_result.sweep_log
            aug_data["extra_containers_oversized"]    = os_result.best_extra
            aug_data["initial_containers_oversized"]  = os_result.best_initial_containers

    # ── Resolve final result for rendering ───────────────────────────────────
    result = baseline_result

    # ── Render dashboard ──────────────────────────────────────────────────────
    params = result.best_params
    y1     = result.full_result["year1"]
    fi     = result.full_result["finance"]

    # Inject annual savings (Crore) into aug_data for plot panel 3
    if aug_data is not None:
        sv = fi["savings_breakdown"]
        aug_data["annual_savings_cr"] = [v / 1e7 for v in sv["annual_savings"]]

    print_section1(params, y1, fi)
    print_section2(fi)
    print_section3(params, y1, fi, data, energy_engine)
    print_section4(fi)
    print_section5(fi)
    print_section6(fi)
    if aug_enabled and aug_data is not None:
        print_section6b(aug_data, baseline_result, fi)
    print_section7(fi, data, params, aug_data)

    sep("SOLVER STATS")
    print(f"\n  {'Trials completed':<38} : {result.n_trials_completed}")
    print(f"  {'Feasible trials':<38} : {result.n_trials_feasible}")

    outputs_dir = find_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plot_dashboard(params, y1, fi, data, outputs_dir / "model_output.png")
    plot_day250(params, config, data, outputs_dir / "day250_dispatch.png")

    if aug_enabled and aug_data is not None:
        plot_augmentation_dashboard(
            aug_data,
            baseline_cuf_ser,
            outputs_dir / "augmentation_dashboard.png",
        )