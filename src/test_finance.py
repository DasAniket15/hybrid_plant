import numpy as np

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine


# ─────────────────────────────────────────────────────────────────────────────
# Test Configuration  (Solar-only benchmark case)
# ─────────────────────────────────────────────────────────────────────────────

SOLAR_MW        = 195.415073395429
WIND_MW         = 0.0
BESS_CONTAINERS = 164
CHARGE_C        = 1.0
DISCHARGE_C     = 1.0
PPA_MW          = 67.5256615562851
DISPATCH        = "solar_first"
CHARGE_SOURCE   = "solar_only"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cr(value):
    """Convert Rs to Rs Crore, rounded to 4 dp."""
    return round(value / 1e7, 4)

def pct(value):
    return round(value * 100, 4)

def sep(title=""):
    if title:
        print(f"\n{'─' * 10} {title} {'─' * (50 - len(title))}")
    else:
        print("─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    config = load_config()
    data   = load_timeseries_data(config)

    energy_engine  = Year1Engine(config, data)
    finance_engine = FinanceEngine(config, data)

    # ── Year-1 energy simulation ──────────────────────────────────────────────
    year1 = energy_engine.evaluate(
        solar_capacity_mw  = SOLAR_MW,
        wind_capacity_mw   = WIND_MW,
        bess_containers    = BESS_CONTAINERS,
        charge_c_rate      = CHARGE_C,
        discharge_c_rate   = DISCHARGE_C,
        ppa_capacity_mw    = PPA_MW,
        dispatch_priority  = DISPATCH,
        bess_charge_source = CHARGE_SOURCE,
    )

    # ── Finance evaluation ────────────────────────────────────────────────────
    finance = finance_engine.evaluate(
        year1_results     = year1,
        solar_capacity_mw = SOLAR_MW,
        wind_capacity_mw  = WIND_MW,
        ppa_capacity_mw   = PPA_MW,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Print Results
    # ─────────────────────────────────────────────────────────────────────────

    sep("PLANT CONFIGURATION")
    print(f"  Solar capacity (AC MW)     : {round(SOLAR_MW, 3)}")
    print(f"  Solar capacity (DC MWp)    : {round(finance['capex']['solar_dc_mwp'], 3)}")
    print(f"  Wind capacity (MW)         : {round(WIND_MW, 3)}")
    print(f"  BESS containers            : {BESS_CONTAINERS}")
    print(f"  BESS energy capacity (MWh) : {round(year1['energy_capacity_mwh'], 2)}")
    print(f"  PPA capacity (MW)          : {round(PPA_MW, 3)}")

    # ── CAPEX ─────────────────────────────────────────────────────────────────
    cap = finance["capex"]
    sep("CAPEX  (Rs Crore)")
    print(f"  Solar                      : {cr(cap['solar_capex'])}")
    print(f"  Wind                       : {cr(cap['wind_capex'])}")
    print(f"  BESS                       : {cr(cap['bess_capex'])}")
    print(f"  Transmission               : {cr(cap['transmission_capex'])}")
    print(f"  ── Total CAPEX             : {cr(cap['total_capex'])}")

    # ── Financing ─────────────────────────────────────────────────────────────
    lcd = finance["lcoe_breakdown"]
    sep("FINANCING")
    print(f"  WACC                       : {pct(finance['wacc'])} %")
    print(f"  Debt amount (Rs Cr)        : {cr(lcd['debt_amount'])}")
    print(f"  Equity amount (Rs Cr)      : {cr(lcd['equity_amount'])}")
    print(f"  EMI (Rs Cr / year)         : {cr(lcd['emi'])}")
    print(f"  Annual ROE (Rs Cr)         : {cr(lcd['roe_schedule'][0])}")

    # ── OPEX Year-1 breakdown ─────────────────────────────────────────────────
    ob = finance["opex_breakdown"][0]
    sep("OPEX — YEAR 1  (Rs Crore)")
    print(f"  Solar O&M                  : {cr(ob['solar_om'])}")
    print(f"  Wind O&M                   : {cr(ob['wind_om'])}")
    print(f"  BESS O&M                   : {cr(ob['bess_om'])}")
    print(f"  Solar Transmission O&M     : {cr(ob['solar_transmission_om'])}")
    print(f"  Wind Transmission O&M      : {cr(ob['wind_transmission_om'])}")
    print(f"  Land Lease                 : {cr(ob['land_lease'])}")
    print(f"  Insurance                  : {cr(ob['insurance'])}")
    print(f"  ── Total OPEX Year 1       : {cr(ob['total'])}")
    print(f"  ── Total OPEX Year 25      : {cr(finance['opex_breakdown'][-1]['total'])}")

    # ── Energy projection ─────────────────────────────────────────────────────
    ep = finance["energy_projection"]
    sep("ENERGY PROJECTION  (MWh)")
    print(f"  Year 1 — busbar (pre-loss) : {round(ep['delivered_pre_mwh'][0], 2)}")
    print(f"  Year 1 — at meter          : {round(ep['delivered_meter_mwh'][0], 2)}")
    print(f"  Year 25 — busbar           : {round(ep['delivered_pre_mwh'][-1], 2)}")
    print(f"  Year 25 — at meter         : {round(ep['delivered_meter_mwh'][-1], 2)}")

    # ── LCOE ─────────────────────────────────────────────────────────────────
    sep("LCOE")
    print(f"  NPV of total costs (Rs Cr) : {cr(lcd['npv_total_cost'])}")
    print(f"    ↳ NPV interest  (Rs Cr)  : {cr(lcd['npv_interest'])}")
    print(f"    ↳ NPV principal (Rs Cr)  : {cr(lcd['npv_principal'])}")
    print(f"    ↳ NPV ROE       (Rs Cr)  : {cr(lcd['npv_roe'])}")
    print(f"    ↳ NPV OPEX      (Rs Cr)  : {cr(lcd['npv_opex'])}")
    print(f"  NPV of busbar energy (Bn kWh): {round(lcd['npv_energy_kwh'] / 1e9, 4)}")
    print(f"  ── LCOE                    : Rs {round(finance['lcoe_inr_per_kwh'], 4)} / kWh")

    # ── Landed tariff ─────────────────────────────────────────────────────────
    lt  = finance["landed_tariff_breakdown"]
    lts = finance["landed_tariff_series"]
    sep("LANDED TARIFF  (Rs / kWh)")
    print(f"  LCOE (busbar)              : {round(lt['lcoe_component'], 4)}")
    print(f"  Wheeling charge            : {round(lt['wheeling_per_kwh'], 4)}")
    print(f"  Electricity tax            : {round(lt['electricity_tax_per_kwh'], 4)}")
    print(f"  Banking charge             : {round(lt['banking_per_kwh'], 4)}")
    print(f"  Annual capacity charge (Rs Cr): {cr(lt['annual_capacity_charge_rs'])}")
    print(f"  Capacity charge/kWh Year 1 : {round(lt['capacity_charge_per_kwh_series'][0], 4)}")
    print(f"  Capacity charge/kWh Year 25: {round(lt['capacity_charge_per_kwh_series'][-1], 4)}")
    print(f"  ── Landed tariff Year 1    : Rs {round(lts[0], 4)} / kWh")
    print(f"  ── Landed tariff Year 25   : Rs {round(lts[-1], 4)} / kWh")

    # ── Client savings ────────────────────────────────────────────────────────
    sv = finance["savings_breakdown"]
    sep("CLIENT SAVINGS")
    print(f"  DISCOM tariff (wt. avg)    : Rs {round(sv['discom_tariff'], 4)} / kWh")
    print(f"  Annual load               : {round(sv['annual_load_kwh'] / 1e6, 4)} Bn kWh")
    print(f"  Baseline DISCOM cost (Cr)  : {cr(sv['baseline_annual_cost'])}")
    print(f"  Hybrid cost Year 1  (Cr)   : {cr(sv['annual_hybrid_cost'][0])}")
    print(f"  ── Savings Year 1   (Cr)   : {cr(finance['annual_savings_year1'])}")
    print(f"  ── Savings NPV      (Cr)   : {cr(finance['savings_npv'])}")

    # ── Full year-by-year table ───────────────────────────────────────────────
    sep("YEAR-BY-YEAR SUMMARY")
    print(f"  {'Year':>4}  {'Busbar MWh':>12}  {'Meter MWh':>12}  "
          f"{'OPEX (Cr)':>10}  {'Landed (Rs/kWh)':>16}  {'Savings (Cr)':>13}")
    sep()
    for y in range(25):
        print(
            f"  {y+1:>4}  "
            f"{ep['delivered_pre_mwh'][y]:>12.1f}  "
            f"{ep['delivered_meter_mwh'][y]:>12.1f}  "
            f"{finance['opex_projection'][y]/1e7:>10.4f}  "
            f"{lts[y]:>16.4f}  "
            f"{sv['annual_savings'][y]/1e7:>13.4f}"
        )
    sep()