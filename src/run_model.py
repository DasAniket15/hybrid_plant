"""
run_model.py
────────────
Master script: runs the full model pipeline and prints a complete dashboard.

Pipeline
────────
    1. Load config + data
    2. Run SolverEngine  (TPE, 300 trials)
    3. Extract best solution → Year1Engine + FinanceEngine outputs
    4. Print structured dashboard (Sections 1–7)
    5. Save model_output.png  (4-panel figure)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine
from solver.solver_engine import SolverEngine


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def cr(v):
    """Rs → Rs Crore, 4 dp."""
    return round(float(v) / 1e7, 4)

def pct(v):
    return round(float(v) * 100, 2)

def sep(title="", width=68):
    if title:
        pad = width - len(title) - 13
        print(f"\n{'─'*10} {title} {'─'*max(pad,2)}")
    else:
        print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Derived metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_payback_year(annual_savings):
    """First year where cumulative savings turns positive."""
    cumulative = 0.0
    for yr, s in enumerate(annual_savings, start=1):
        cumulative += s
        if cumulative > 0:
            return yr
    return None  # never pays back within project life


def compute_cuf(busbar_mwh_y1, capacity_mw, hours=8760):
    """CUF = annual busbar MWh / (capacity_mw × 8760)."""
    if capacity_mw <= 0:
        return None
    return busbar_mwh_y1 / (capacity_mw * hours) * 100  # percent


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Optimal Configuration
# ─────────────────────────────────────────────────────────────────────────────

def print_section1(params, y1, fi):
    sep("SECTION 1 — OPTIMAL CONFIGURATION")

    cap = fi["capex"]

    print(f"\n  {'Solar capacity (AC MW)':<38} : {round(params['solar_capacity_mw'], 2)}")
    print(f"  {'Solar capacity (DC MWp)':<38} : {round(cap['solar_dc_mwp'], 2)}")
    print(f"  {'AC/DC ratio':<38} : {round(cap['solar_dc_mwp'] / params['solar_capacity_mw'], 4) if params['solar_capacity_mw'] > 0 else 'N/A'}")
    print(f"  {'Wind capacity (MW)':<38} : {round(params['wind_capacity_mw'], 2)}")
    print(f"  {'BESS containers':<38} : {params['bess_containers']}")
    print(f"  {'BESS energy capacity (MWh)':<38} : {round(float(y1['energy_capacity_mwh']), 2)}")
    print(f"  {'BESS charge power (MW)':<38} : {round(float(y1['charge_power_mw']), 2)}")
    print(f"  {'BESS discharge power (MW)':<38} : {round(float(y1['discharge_power_mw']), 2)}")
    print(f"  {'PPA capacity (MW)':<38} : {round(params['ppa_capacity_mw'], 2)}")
    print(f"  {'Dispatch priority':<38} : {params['dispatch_priority']}")
    print(f"  {'BESS charge source':<38} : {params['bess_charge_source']}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Capital & Financing
# ─────────────────────────────────────────────────────────────────────────────

def print_section2(fi):
    sep("SECTION 2 — CAPITAL & FINANCING")

    cap  = fi["capex"]
    lcd  = fi["lcoe_breakdown"]

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Year-1 Energy Balance
# ─────────────────────────────────────────────────────────────────────────────

def print_section3(params, y1, fi, data, energy_engine):
    sep("SECTION 3 — YEAR-1 ENERGY BALANCE")

    solar_direct_pre  = float(np.sum(y1["solar_direct_pre"]))
    wind_direct_pre   = float(np.sum(y1["wind_direct_pre"]))
    discharge_pre     = float(np.sum(y1["discharge_pre"]))
    curtailment       = float(np.sum(y1["curtailment_pre"]))
    total_busbar      = solar_direct_pre + wind_direct_pre + discharge_pre

    solar_direct_m    = float(np.sum(y1["solar_direct_meter"]))
    wind_direct_m     = float(np.sum(y1["wind_direct_meter"]))
    discharge_m       = float(np.sum(y1["discharge_meter"]))
    total_meter       = solar_direct_m + wind_direct_m + discharge_m

    annual_load_mwh   = float(np.sum(data["load_profile"]))
    discom_draw_mwh   = max(annual_load_mwh - total_meter, 0)
    sv                = fi["savings_breakdown"]
    discom_cost_y1    = discom_draw_mwh * 1000 * sv["discom_tariff"]

    re_penetration    = total_meter / annual_load_mwh * 100

    # Raw solar and wind generation (for CUF calculation)
    raw_solar_gen     = float(np.sum(params["solar_capacity_mw"] * data["solar_cuf"]))
    raw_wind_gen      = float(np.sum(params["wind_capacity_mw"]  * data["wind_cuf"]))

    # Plant CUF — re-simulate with loss_factor=1.0 to match Excel convention:
    # dispatch is computed without grossing up load for grid losses, so
    # plant_export_pre reflects only what is needed to serve load at busbar.
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
    # Per-hour: min(ppa_mw, direct + discharge + curtailment)
    # This matches the Excel model which injects any remaining surplus
    # up to the PPA cap after direct dispatch and BESS discharge.
    plant_cuf_numerator = float(np.sum(
        np.minimum(
            y1_lf1["plant_export_pre"] + y1_lf1["curtailment_pre"],
            params["ppa_capacity_mw"]
        )
    ))

    solar_cuf = compute_cuf(raw_solar_gen,       params["solar_capacity_mw"])
    wind_cuf  = compute_cuf(raw_wind_gen,        params["wind_capacity_mw"])
    plant_cuf = compute_cuf(plant_cuf_numerator, params["ppa_capacity_mw"])

    loss_factor = float(y1["loss_factor"])

    print(f"\n  {'── BUSBAR (Pre-Loss)'}")
    print(f"  {'Solar Direct (MWh)':<38} : {round(solar_direct_pre, 1)}")
    print(f"  {'Wind Direct (MWh)':<38} : {round(wind_direct_pre, 1)}")
    print(f"  {'BESS Discharge (MWh)':<38} : {round(discharge_pre, 1)}")
    print(f"  {'Total Busbar Delivery (MWh)':<38} : {round(total_busbar, 1)}")
    print(f"  {'Curtailment (MWh)':<38} : {round(curtailment, 1)}")

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LCOE & Tariff Build-up
# ─────────────────────────────────────────────────────────────────────────────

def print_section4(fi):
    sep("SECTION 4 — LCOE & TARIFF BUILD-UP")

    lcd  = fi["lcoe_breakdown"]
    lt   = fi["landed_tariff_breakdown"]
    lts  = fi["landed_tariff_series"]
    sv   = fi["savings_breakdown"]
    lcoe = fi["lcoe_inr_per_kwh"]

    # Safely extract landed tariff components — fall back to deriving from
    # available values if an older landed_tariff_model is in use.
    wheeling   = lt.get("wheeling_per_kwh", 0.0)
    elec_tax   = lt.get("electricity_tax_per_kwh", 0.0)
    banking    = lt.get("banking_per_kwh", 0.0)

    cap_series = lt.get("capacity_charge_per_kwh_series", None)
    if cap_series is not None:
        cap_y1  = cap_series[0]
        cap_y25 = cap_series[-1]
    else:
        # Derive: capacity charge = landed - lcoe - energy_charges
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Client Savings
# ─────────────────────────────────────────────────────────────────────────────

def print_section5(fi):
    sep("SECTION 5 — CLIENT SAVINGS")

    sv  = fi["savings_breakdown"]
    lts = fi["landed_tariff_series"]

    annual_savings   = sv["annual_savings"]
    baseline         = sv["baseline_annual_cost"]
    savings_y1       = annual_savings[0]
    savings_pct      = savings_y1 / baseline * 100
    payback_yr       = compute_payback_year(annual_savings)
    cum_savings_25   = sum(annual_savings)

    print(f"\n  {'Baseline Annual Cost (Rs Crore)':<38} : {cr(baseline)}")
    print(f"  {'Hybrid Cost Year-1 (Rs Crore)':<38} : {cr(sv['annual_hybrid_cost'][0])}")
    print(f"  {'Savings Year-1 (Rs Crore)':<38} : {cr(savings_y1)}")
    print(f"  {'Savings as % of Baseline':<38} : {round(savings_pct, 2)} %")
    print(f"  {'Savings NPV (Rs Crore)':<38} : {cr(fi['savings_npv'])}")
    print(f"  {'Cumulative Savings 25 Yr (Rs Cr)':<38} : {cr(cum_savings_25)}")
    print(f"  {'Payback Year':<38} : {'Year ' + str(payback_yr) if payback_yr else 'Beyond project life'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OPEX Breakdown
# ─────────────────────────────────────────────────────────────────────────────

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
        d   = round(v25 - v1, 4)
        print(f"  {label:<36}  {v1:>10}  {v25:>10}  {d:>+10}")
    sep()
    v1t  = cr(ob_y1["total"])
    v25t = cr(ob_y25["total"])
    print(f"  {'Total OPEX':<36}  {v1t:>10}  {v25t:>10}  {round(v25t-v1t,4):>+10}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — 25-Year Projections Table
# ─────────────────────────────────────────────────────────────────────────────

def print_section7(fi):
    sep("SECTION 7 — 25-YEAR PROJECTIONS")

    ep   = fi["energy_projection"]
    sv   = fi["savings_breakdown"]
    lts  = fi["landed_tariff_series"]
    opex = fi["opex_projection"]

    annual_savings = sv["annual_savings"]
    cumulative     = 0.0

    hdr = (f"  {'Yr':>3}  {'Busbar MWh':>11}  {'Meter MWh':>10}  "
           f"{'OPEX (Cr)':>10}  {'Landed ₹/kWh':>13}  "
           f"{'Savings (Cr)':>13}  {'Cum Savings (Cr)':>17}")
    print(f"\n{hdr}")
    sep()

    for y in range(25):
        cumulative += annual_savings[y]
        print(
            f"  {y+1:>3}  "
            f"{ep['delivered_pre_mwh'][y]:>11.1f}  "
            f"{ep['delivered_meter_mwh'][y]:>10.1f}  "
            f"{opex[y]/1e7:>10.4f}  "
            f"{lts[y]:>13.4f}  "
            f"{annual_savings[y]/1e7:>13.4f}  "
            f"{cumulative/1e7:>17.4f}"
        )
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(params, y1, fi, data, output_path="model_output.png"):

    ep   = fi["energy_projection"]
    sv   = fi["savings_breakdown"]
    lts  = fi["landed_tariff_series"]
    opex = fi["opex_projection"]

    years = np.arange(1, 26)

    annual_savings = np.array(sv["annual_savings"])
    cumulative     = np.cumsum(annual_savings) / 1e7

    # Energy mix
    solar_mwh = ep["solar_direct_mwh"]
    wind_mwh  = ep["wind_direct_mwh"]
    bess_mwh  = ep["battery_mwh"]
    meter_mwh = ep["delivered_meter_mwh"]
    load_mwh  = float(np.sum(data["load_profile"]))
    discom_mwh = np.maximum(load_mwh - meter_mwh, 0)

    # OPEX stack (Crore)
    ob = fi["opex_breakdown"]
    opex_solar   = np.array([x["solar_om"]              for x in ob]) / 1e7
    opex_wind    = np.array([x["wind_om"]               for x in ob]) / 1e7
    opex_bess    = np.array([x["bess_om"]               for x in ob]) / 1e7
    opex_trans   = np.array([x["solar_transmission_om"] + x["wind_transmission_om"] for x in ob]) / 1e7
    opex_land    = np.array([x["land_lease"]             for x in ob]) / 1e7
    opex_ins     = np.array([x["insurance"]              for x in ob]) / 1e7

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Hybrid RE Plant — Model Output Dashboard\n"
        f"Solar {round(params['solar_capacity_mw'],1)} MW  |  "
        f"Wind {round(params['wind_capacity_mw'],1)} MW  |  "
        f"BESS {round(float(y1['energy_capacity_mwh']),1)} MWh  |  "
        f"PPA {round(params['ppa_capacity_mw'],1)} MW",
        fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    COLORS = {
        "solar":  "#f4a832",
        "wind":   "#4c9be8",
        "bess":   "#66bb6a",
        "discom": "#ef5350",
        "savings":"#26a69a",
        "cumul":  "#ab47bc",
    }

    # ── Panel 1: Annual savings bars + cumulative line ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bar_colors = [COLORS["savings"] if s >= 0 else COLORS["discom"]
                  for s in annual_savings]
    ax1.bar(years, annual_savings / 1e7, color=bar_colors, alpha=0.8,
            label="Annual Savings")
    ax1_r = ax1.twinx()
    ax1_r.plot(years, cumulative, color=COLORS["cumul"], linewidth=2,
               marker="o", markersize=3, label="Cumulative")
    ax1_r.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_title("Annual & Cumulative Savings", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Savings (Rs Crore)")
    ax1_r.set_ylabel("Cumulative Savings (Rs Crore)", color=COLORS["cumul"])
    ax1.tick_params(axis="y")
    ax1_r.tick_params(axis="y", labelcolor=COLORS["cumul"])
    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Landed tariff vs DISCOM tariff ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, lts, color="#1f77b4", linewidth=2, label="Landed Tariff")
    ax2.axhline(sv["discom_tariff"], color=COLORS["discom"], linewidth=1.5,
                linestyle="--", label=f"DISCOM Avg (Rs {round(sv['discom_tariff'],2)})")
    ax2.fill_between(years, lts, sv["discom_tariff"],
                     where=[l < sv["discom_tariff"] for l in lts],
                     alpha=0.15, color=COLORS["savings"], label="Savings band")
    ax2.set_title("Landed Tariff vs DISCOM Tariff", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Rs / kWh")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Energy mix stacked area ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.stackplot(
        years,
        solar_mwh / 1e3,
        wind_mwh  / 1e3,
        bess_mwh  / 1e3,
        discom_mwh / 1e3,
        labels=["Solar Direct", "Wind Direct", "BESS Discharge", "DISCOM Draw"],
        colors=[COLORS["solar"], COLORS["wind"], COLORS["bess"], COLORS["discom"]],
        alpha=0.85
    )
    ax3.set_title("Energy Mix Over 25 Years", fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Energy (GWh)")
    ax3.legend(fontsize=8, loc="lower left")
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: OPEX stack ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.stackplot(
        years,
        opex_solar,
        opex_wind,
        opex_bess,
        opex_trans,
        opex_land,
        opex_ins,
        labels=["Solar O&M", "Wind O&M", "BESS O&M",
                "Transmission O&M", "Land Lease", "Insurance"],
        colors=["#fdd835", "#42a5f5", "#66bb6a",
                "#ab47bc", "#ff7043", "#78909c"],
        alpha=0.85
    )
    ax4.set_title("OPEX Stack Over 25 Years", fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("OPEX (Rs Crore)")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, alpha=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard plot saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    config         = load_config()
    data           = load_timeseries_data(config)
    energy_engine  = Year1Engine(config, data)
    finance_engine = FinanceEngine(config, data)
    solver         = SolverEngine(config, data, energy_engine, finance_engine)

    n_trials = config.solver["solver"].get("n_trials", 300)

    # ── Solve ─────────────────────────────────────────────────────────────────
    print(f"\nRunning solver ({n_trials} trials) …")
    result = solver.run(n_trials=n_trials, show_progress=True)

    # ── Unpack best solution ──────────────────────────────────────────────────
    params = result.best_params
    y1     = result.full_result["year1"]
    fi     = result.full_result["finance"]

    # ── Print dashboard ───────────────────────────────────────────────────────
    print_section1(params, y1, fi)
    print_section2(fi)
    print_section3(params, y1, fi, data, energy_engine)
    print_section4(fi)
    print_section5(fi)
    print_section6(fi)
    print_section7(fi)

    # ── Solver stats ──────────────────────────────────────────────────────────
    sep("SOLVER STATS")
    print(f"\n  {'Trials completed':<38} : {result.n_trials_completed}")
    print(f"  {'Feasible trials':<38} : {result.n_trials_feasible}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_dashboard(params, y1, fi, data)