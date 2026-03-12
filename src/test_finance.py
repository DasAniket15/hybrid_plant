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

def sep(title=""):
    width = 62
    if title:
        print(f"\n{'─' * 10} {title} {'─' * max(0, width - len(title) - 12)}")
    else:
        print("─" * width)


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
        # banked_energy_kwh_projection defaults to None → zeros (stub)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PLANT CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    sep("PLANT CONFIGURATION")
    print(f"  Solar capacity (AC MW)        : {round(SOLAR_MW, 3)}")
    print(f"  Solar capacity (DC MWp)       : {round(finance['capex']['solar_dc_mwp'], 3)}")
    print(f"  Wind capacity (MW)            : {round(WIND_MW, 3)}")
    print(f"  BESS containers               : {BESS_CONTAINERS}")
    print(f"  BESS energy capacity (MWh)    : {round(year1['energy_capacity_mwh'], 2)}")
    print(f"  PPA capacity (MW)             : {round(PPA_MW, 3)}")
    print(f"  Loss factor                   : {round(year1['loss_factor'], 6)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CAPEX
    # ─────────────────────────────────────────────────────────────────────────
    cap = finance["capex"]
    sep("CAPEX  (Rs Crore)")
    print(f"  Solar                         : {cr(cap['solar_capex'])}")
    print(f"  Wind                          : {cr(cap['wind_capex'])}")
    print(f"  BESS                          : {cr(cap['bess_capex'])}")
    print(f"  Transmission                  : {cr(cap['transmission_capex'])}")
    print(f"  ── Total CAPEX                : {cr(cap['total_capex'])}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FINANCING
    # ─────────────────────────────────────────────────────────────────────────
    lcd = finance["lcoe_breakdown"]
    sep("FINANCING")
    print(f"  WACC (computed)               : {round(finance['wacc'] * 100, 4)} %")
    print(f"  Debt amount (Rs Cr)           : {cr(lcd['debt_amount'])}")
    print(f"  Equity amount (Rs Cr)         : {cr(lcd['equity_amount'])}")
    print(f"  EMI (Rs Cr / year)            : {cr(lcd['emi'])}")
    print(f"  Annual ROE (Rs Cr)            : {cr(lcd['roe_schedule'][0])}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. OPEX — YEAR 1 BREAKDOWN
    # ─────────────────────────────────────────────────────────────────────────
    ob1  = finance["opex_breakdown"][0]
    ob25 = finance["opex_breakdown"][-1]
    sep("OPEX  (Rs Crore)")
    print(f"  {'Component':<30}  {'Year 1':>10}  {'Year 25':>10}")
    sep()
    print(f"  {'Solar O&M':<30}  {cr(ob1['solar_om']):>10}  {cr(ob25['solar_om']):>10}")
    print(f"  {'Wind O&M':<30}  {cr(ob1['wind_om']):>10}  {cr(ob25['wind_om']):>10}")
    print(f"  {'BESS O&M':<30}  {cr(ob1['bess_om']):>10}  {cr(ob25['bess_om']):>10}")
    print(f"  {'Solar Transmission O&M':<30}  {cr(ob1['solar_transmission_om']):>10}  {cr(ob25['solar_transmission_om']):>10}")
    print(f"  {'Wind Transmission O&M':<30}  {cr(ob1['wind_transmission_om']):>10}  {cr(ob25['wind_transmission_om']):>10}")
    print(f"  {'Land Lease':<30}  {cr(ob1['land_lease']):>10}  {cr(ob25['land_lease']):>10}")
    print(f"  {'Insurance':<30}  {cr(ob1['insurance']):>10}  {cr(ob25['insurance']):>10}")
    sep()
    print(f"  {'Total OPEX':<30}  {cr(ob1['total']):>10}  {cr(ob25['total']):>10}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ENERGY PROJECTION
    # ─────────────────────────────────────────────────────────────────────────
    ep = finance["energy_projection"]
    sep("ENERGY PROJECTION  (MWh)")
    print(f"  {'':30}  {'Year 1':>12}  {'Year 25':>12}")
    sep()
    print(f"  {'Solar direct (pre-loss)':<30}  {ep['solar_direct_mwh'][0]:>12.1f}  {ep['solar_direct_mwh'][-1]:>12.1f}")
    print(f"  {'Wind direct (pre-loss)':<30}  {ep['wind_direct_mwh'][0]:>12.1f}  {ep['wind_direct_mwh'][-1]:>12.1f}")
    print(f"  {'Battery discharge (pre-loss)':<30}  {ep['battery_mwh'][0]:>12.1f}  {ep['battery_mwh'][-1]:>12.1f}")
    print(f"  {'Delivered — busbar':<30}  {ep['delivered_pre_mwh'][0]:>12.1f}  {ep['delivered_pre_mwh'][-1]:>12.1f}")
    print(f"  {'Delivered — at meter':<30}  {ep['delivered_meter_mwh'][0]:>12.1f}  {ep['delivered_meter_mwh'][-1]:>12.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. LCOE
    # ─────────────────────────────────────────────────────────────────────────
    sep("LCOE")
    print(f"  NPV of interest    (Rs Cr)    : {cr(lcd['npv_interest'])}")
    print(f"  NPV of principal   (Rs Cr)    : {cr(lcd['npv_principal'])}")
    print(f"  NPV of ROE         (Rs Cr)    : {cr(lcd['npv_roe'])}")
    print(f"  NPV of OPEX        (Rs Cr)    : {cr(lcd['npv_opex'])}")
    sep()
    print(f"  NPV of total costs (Rs Cr)    : {cr(lcd['npv_total_cost'])}")
    print(f"  NPV of busbar energy (Bn kWh) : {round(lcd['npv_energy_kwh'] / 1e9, 6)}")
    sep()
    print(f"  ── LCOE                       : Rs {round(finance['lcoe_inr_per_kwh'], 4)} / kWh")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. LANDED TARIFF
    # ─────────────────────────────────────────────────────────────────────────
    lt  = finance["landed_tariff_breakdown"]
    lts = finance["landed_tariff_series"]
    sep("LANDED TARIFF — UNIT RATES")
    print(f"  CTU charge       (Rs/MW/month): {round(lt['ctu_per_mw_month'], 4)}")
    print(f"  STU charge       (Rs/MW/month): {round(lt['stu_per_mw_month'], 4)}")
    print(f"  SLDC charge      (Rs/MW/month): {round(lt['sldc_per_mw_month'], 4)}")
    print(f"  Wheeling charge  (Rs/kWh)     : {round(lt['wheeling_per_kwh'], 4)}")
    print(f"  Electricity tax  (Rs/kWh)     : {round(lt['electricity_tax_per_kwh'], 4)}")
    print(f"  Banking charge   (Rs/kWh)     : {round(lt['banking_per_kwh'], 4)}  [stub]")

    sep("LANDED TARIFF — ANNUAL COSTS  (Rs Crore)")
    print(f"  {'Component':<30}  {'Year 1':>10}  {'Year 25':>10}")
    sep()
    print(f"  {'RE payment (LCOE × busbar)':<30}  {cr(lt['annual_re_payment'][0]):>10}  {cr(lt['annual_re_payment'][-1]):>10}")
    print(f"  {'Capacity charges (CTU+STU+SLDC)':<30}  {cr(lt['annual_capacity_charge_rs']):>10}  {cr(lt['annual_capacity_charge_rs']):>10}")
    print(f"  {'Wheeling charges':<30}  {cr(lt['annual_wheeling'][0]):>10}  {cr(lt['annual_wheeling'][-1]):>10}")
    print(f"  {'Electricity tax':<30}  {cr(lt['annual_electricity_tax'][0]):>10}  {cr(lt['annual_electricity_tax'][-1]):>10}")
    print(f"  {'Banking charges':<30}  {cr(lt['annual_banking'][0]):>10}  {cr(lt['annual_banking'][-1]):>10}  [stub]")
    sep()
    print(f"  {'Total annual cost':<30}  {cr(lt['annual_total_cost'][0]):>10}  {cr(lt['annual_total_cost'][-1]):>10}")
    print(f"  {'Meter energy (MWh)':<30}  {ep['delivered_meter_mwh'][0]:>10.1f}  {ep['delivered_meter_mwh'][-1]:>10.1f}")
    sep()
    print(f"  ── Landed tariff Year 1       : Rs {round(lts[0], 4)} / kWh")
    print(f"  ── Landed tariff Year 25      : Rs {round(lts[-1], 4)} / kWh")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. CLIENT SAVINGS
    # ─────────────────────────────────────────────────────────────────────────
    sv = finance["savings_breakdown"]
    sep("CLIENT SAVINGS")
    print(f"  DISCOM tariff (wt. avg)       : Rs {round(sv['discom_tariff'], 4)} / kWh")
    print(f"  Annual load                   : {round(sv['annual_load_kwh'] / 1e6, 4)} Bn kWh")
    print(f"  Baseline DISCOM cost (Cr)     : {cr(sv['baseline_annual_cost'])}")
    print(f"  Hybrid cost Year 1   (Cr)     : {cr(sv['annual_hybrid_cost'][0])}")
    sep()
    print(f"  ── Savings Year 1    (Cr)     : {cr(finance['annual_savings_year1'])}")
    print(f"  ── Savings NPV       (Cr)     : {cr(finance['savings_npv'])}")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. FULL YEAR-BY-YEAR TABLE
    # ─────────────────────────────────────────────────────────────────────────
    sep("YEAR-BY-YEAR SUMMARY")
    print(
        f"  {'Yr':>3}  "
        f"{'Busbar MWh':>11}  "
        f"{'Meter MWh':>10}  "
        f"{'OPEX (Cr)':>10}  "
        f"{'Total Cost (Cr)':>15}  "
        f"{'Landed (Rs/kWh)':>15}  "
        f"{'Savings (Cr)':>12}"
    )
    sep()
    for y in range(25):
        print(
            f"  {y+1:>3}  "
            f"{ep['delivered_pre_mwh'][y]:>11.1f}  "
            f"{ep['delivered_meter_mwh'][y]:>10.1f}  "
            f"{finance['opex_projection'][y]/1e7:>10.4f}  "
            f"{lt['annual_total_cost'][y]/1e7:>15.4f}  "
            f"{lts[y]:>15.4f}  "
            f"{sv['annual_savings'][y]/1e7:>12.4f}"
        )
    sep()