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
    plant_cuf_val = compute_cuf(total_busbar, params["ppa_capacity_mw"])

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

    n_trials = config.solver["solver"].get("n_trials", 300)
    print(f"\nRunning solver ({n_trials} trials) …")
    result = solver.run(n_trials=n_trials, show_progress=True)

    params = result.best_params
    y1     = result.full_result["year1"]
    fi     = result.full_result["finance"]

    print_section1(params, y1, fi)
    print_section2(fi)
    print_section3(params, y1, fi, data, energy_engine)
    print_section4(fi)
    print_section5(fi)
    print_section6(fi)
    print_section7(fi)

    sep("SOLVER STATS")
    print(f"\n  {'Trials completed':<38} : {result.n_trials_completed}")
    print(f"  {'Feasible trials':<38} : {result.n_trials_feasible}")

    outputs_dir = find_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plot_dashboard(params, y1, fi, data, outputs_dir / "model_output.png")
    plot_day250(params, config, data, outputs_dir / "day250_dispatch.png")