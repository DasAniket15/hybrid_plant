"""
trace_pipeline.py
-----------------
Traces the complete data flow through every stage of the finance pipeline
for the augmented scenario, checking for consistency at each handoff.
"""
from __future__ import annotations
import math, warnings
warnings.filterwarnings("ignore")

import numpy as np
from hybrid_plant.config_loader import load_config
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.augmentation_engine import AugmentationEngine
from hybrid_plant.finance.finance_engine import FinanceEngine

config = load_config()
data   = load_timeseries_data(config)
ee     = Year1Engine(config, data)

n_oversize = 173  # 164 solver-optimal + 9 oversizing at 5%
BASE = {
    "solar_capacity_mw": 195.415073395429, "wind_capacity_mw": 0.0,
    "bess_containers": n_oversize, "charge_c_rate": 1.0, "discharge_c_rate": 1.0,
    "ppa_capacity_mw": 67.5256615562851, "dispatch_priority": "solar_first",
    "bess_charge_source": "solar_only",
}

# Run the pipeline
fe   = FinanceEngine(config, data)
y1   = ee.evaluate(**BASE)
fi   = fe.evaluate(y1, solar_capacity_mw=BASE["solar_capacity_mw"],
    wind_capacity_mw=BASE["wind_capacity_mw"], ppa_capacity_mw=BASE["ppa_capacity_mw"],
    fast_mode=False)

aug_res = fi["augmentation"]
ep      = fi["energy_projection"]
cap     = fi["capex"]
ob      = fi["opex_breakdown"]
op      = fi["opex_projection"]
lcd     = fi["lcoe_breakdown"]
ltb     = fi["landed_tariff_breakdown"]
lts     = fi["landed_tariff_series"]
sv      = fi["savings_breakdown"]

MWH_TO_KWH = 1000.0
CRORE = 1e7

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: CAPEX
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  CHECK 1 — CAPEX")
print("=" * 80)
bess_mwh_y1 = float(y1["energy_capacity_mwh"])  # nameplate, no SOH
bess_capex_expected = bess_mwh_y1 * float(config.finance["capex"]["bess"]["cost_per_mwh"])
print(f"  BESS containers          : {n_oversize}")
print(f"  Container size MWh       : {float(config.bess['bess']['container']['size_mwh'])}")
print(f"  energy_capacity_mwh (Y1) : {bess_mwh_y1:.4f}  (= {n_oversize} x {float(config.bess['bess']['container']['size_mwh'])} x SOH=1.0)")
print(f"  BESS CAPEX (computed)    : {cap['bess_capex']/CRORE:.4f} Cr")
print(f"  BESS CAPEX (expected)    : {bess_capex_expected/CRORE:.4f} Cr")
print(f"  Match? {abs(cap['bess_capex'] - bess_capex_expected) < 1.0}")
print(f"  Total CAPEX              : {cap['total_capex']/CRORE:.4f} Cr")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: Augmentation engine config and SOH
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 2 — AUGMENTATION ENGINE")
print("=" * 80)
aug_eng = AugmentationEngine(config)
print(f"  enabled                  : {aug_eng.enabled}")
print(f"  trigger_cuf_percent      : {aug_eng.trigger_cuf_percent}")
print(f"  restore_pct              : {aug_eng.restore_pct}")
print(f"  cost_per_mwh             : {aug_eng.cost_per_mwh/CRORE:.4f} Cr/MWh")
print(f"  container_size           : {aug_eng.container_size}")
print(f"  SOH curve[1..5]          : {[aug_eng.soh_curve.get(y) for y in range(1,6)]}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: Energy projection → augmentation schedule
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 3 — ENERGY PROJECTION + AUGMENTATION SCHEDULE")
print("=" * 80)
print(f"  Augmentation years       : {aug_res['augmentation_years']}")
print(f"  Cohorts                  : {aug_res['cohorts']}")
y1_cuf = float(ep["cuf_per_year"][0])
trig   = aug_eng.trigger_cuf_percent / 100.0 * y1_cuf
print(f"  Year-1 CUF               : {y1_cuf:.4f}%")
print(f"  Trigger CUF (abs)        : {trig:.4f}%")
print(f"  Year-1 eff MWh           : {aug_res['effective_containers_per_year'][1] * aug_eng.container_size * aug_res['effective_soh_per_year'][1]:.4f}")
print()
print(f"  {'Yr':>3}  {'Containers':>10}  {'Inst MWh':>10}  {'Eff SOH':>8}  {'Eff MWh':>9}  "
      f"{'CUF%':>7}  {'Busbar MWh':>11}  {'Meter MWh':>11}  {'AugPur Cr':>10}  {'Event':>5}")
print("  " + "-" * 105)

for yr in range(1, 26):
    nc   = aug_res["effective_containers_per_year"][yr]
    inst = aug_res["total_installed_mwh_per_year"][yr]
    soh  = aug_res["effective_soh_per_year"][yr]
    emwh = nc * aug_eng.container_size * soh
    cuf  = float(ep["cuf_per_year"][yr-1])
    bus  = float(ep["delivered_pre_mwh"][yr-1])
    met  = float(ep["delivered_meter_mwh"][yr-1])
    pur  = aug_res["augmentation_purchase_opex"].get(yr, 0.0)
    ev   = "AUG" if yr in aug_res["augmentation_years"] else ""
    print(f"  {yr:>3}  {nc:>10}  {inst:>10.2f}  {soh:>8.4f}  {emwh:>9.2f}  "
          f"{cuf:>7.3f}  {bus:>11.1f}  {met:>11.1f}  {pur/CRORE:>10.4f}  {ev:>5}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: OPEX consistency — does OPEX match the augmentation data?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 4 — OPEX CONSISTENCY")
print("=" * 80)
print()
issues = []
for yr in range(1, 26):
    idx = yr - 1
    bd  = ob[idx]

    # BESS O&M should be on total_installed_mwh
    expected_inst = aug_res["total_installed_mwh_per_year"].get(yr, bess_mwh_y1)
    actual_inst   = bd["bess_installed_mwh"]
    if abs(expected_inst - actual_inst) > 0.01:
        issues.append(f"  Year {yr}: bess_installed_mwh mismatch: expected {expected_inst:.2f}, got {actual_inst:.2f}")

    # Aug purchase should match
    expected_pur = aug_res["augmentation_purchase_opex"].get(yr, 0.0)
    actual_pur   = bd["augmentation_purchase"]
    if abs(expected_pur - actual_pur) > 1.0:
        issues.append(f"  Year {yr}: aug_purchase mismatch: expected {expected_pur:.0f}, got {actual_pur:.0f}")

    # total should equal sum of components
    comp_sum = (bd["solar_om"] + bd["wind_om"] + bd["bess_om"]
              + bd["solar_transmission_om"] + bd["wind_transmission_om"]
              + bd["land_lease"] + bd["insurance"] + bd["augmentation_purchase"])
    if abs(comp_sum - bd["total"]) > 1.0:
        issues.append(f"  Year {yr}: total mismatch: sum={comp_sum:.0f} vs total={bd['total']:.0f}")

    # opex_projection should match breakdown total
    if abs(op[idx] - bd["total"]) > 1.0:
        issues.append(f"  Year {yr}: opex_projection mismatch: {op[idx]:.0f} vs breakdown total {bd['total']:.0f}")

if issues:
    for iss in issues:
        print(iss)
else:
    print("  All OPEX checks PASSED (BESS O&M, aug purchase, totals)")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5: LCOE — verify NPV numerator/denominator
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 5 — LCOE VERIFICATION")
print("=" * 80)
wacc = fi["wacc"]
npv_energy = sum(float(ep["delivered_pre_mwh"][i]) * MWH_TO_KWH / (1+wacc)**(i+1) for i in range(25))
print(f"  WACC                     : {wacc*100:.4f}%")
print(f"  NPV Energy (B kWh)       : model={lcd['npv_energy_kwh']/1e9:.6f}  recomputed={npv_energy/1e9:.6f}  match={abs(npv_energy-lcd['npv_energy_kwh'])<1e3}")
print(f"  NPV OPEX (Cr)            : {lcd['npv_opex']/CRORE:.4f}")
print(f"  NPV Total Cost (Cr)      : {lcd['npv_total_cost']/CRORE:.4f}")
print(f"  LCOE = NPV_cost/NPV_energy: {lcd['npv_total_cost']/lcd['npv_energy_kwh']:.6f} Rs/kWh")
print(f"  LCOE (model)             : {fi['lcoe_inr_per_kwh']:.6f} Rs/kWh")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 6: LANDED TARIFF — trace Year 1 and an augmentation year
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 6 — LANDED TARIFF TRACE")
print("=" * 80)
lcoe = fi["lcoe_inr_per_kwh"]
ppa  = BASE["ppa_capacity_mw"]
cap_charges = ltb["annual_capacity_charge_rs"]
wheeling = ltb["wheeling_per_kwh"]
elec_tax = ltb["electricity_tax_per_kwh"]

for yr in [1, 8, 16, 25]:
    idx = yr - 1
    bus_kwh = float(ep["delivered_pre_mwh"][idx]) * MWH_TO_KWH
    met_kwh = float(ep["delivered_meter_mwh"][idx]) * MWH_TO_KWH
    re_pay  = lcoe * bus_kwh
    total   = re_pay + cap_charges + wheeling * met_kwh + elec_tax * met_kwh
    landed  = total / met_kwh
    print(f"\n  Year {yr}:")
    print(f"    busbar kWh          = {bus_kwh:,.0f}")
    print(f"    meter kWh           = {met_kwh:,.0f}")
    print(f"    RE payment          = LCOE({lcoe:.4f}) x busbar = {re_pay/CRORE:.4f} Cr")
    print(f"    Capacity charges    = {cap_charges/CRORE:.4f} Cr")
    print(f"    Wheeling            = {wheeling:.4f} x meter = {wheeling*met_kwh/CRORE:.4f} Cr")
    print(f"    Total cost          = {total/CRORE:.4f} Cr")
    print(f"    Landed = total/meter= {landed:.6f} Rs/kWh")
    print(f"    Model landed_tariff = {lts[idx]:.6f} Rs/kWh")
    print(f"    Match? {abs(landed - lts[idx]) < 1e-6}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 7: SAVINGS — trace Year 1 and an augmentation year
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 7 — SAVINGS TRACE")
print("=" * 80)
discom  = sv["discom_tariff"]
load_kwh = sv["annual_load_kwh"]
baseline = sv["baseline_annual_cost"]
print(f"  DISCOM tariff            : {discom:.4f} Rs/kWh")
print(f"  Annual load              : {load_kwh:,.0f} kWh")
print(f"  Baseline cost            : {baseline/CRORE:.4f} Cr")

for yr in [1, 8, 16, 25]:
    idx = yr - 1
    met_kwh = float(ep["delivered_meter_mwh"][idx]) * MWH_TO_KWH
    re_cost     = met_kwh * lts[idx]
    discom_draw = load_kwh - met_kwh
    discom_cost = discom_draw * discom
    hybrid_cost = re_cost + discom_cost
    savings     = baseline - hybrid_cost
    print(f"\n  Year {yr}:")
    print(f"    RE meter kWh          = {met_kwh:,.0f}")
    print(f"    RE cost               = RE_kwh x landed({lts[idx]:.4f}) = {re_cost/CRORE:.4f} Cr")
    print(f"    DISCOM draw kWh       = load - RE = {discom_draw:,.0f}")
    print(f"    DISCOM cost           = draw x tariff({discom:.4f}) = {discom_cost/CRORE:.4f} Cr")
    print(f"    Hybrid cost           = {hybrid_cost/CRORE:.4f} Cr")
    print(f"    Savings               = baseline - hybrid = {savings/CRORE:.4f} Cr")
    print(f"    Model savings         = {sv['annual_savings'][idx]/CRORE:.4f} Cr")
    print(f"    Match? {abs(savings - sv['annual_savings'][idx]) < 1.0}")

# Savings NPV
sav_npv = sum(sv['annual_savings'][i] / (1+wacc)**(i+1) for i in range(25))
print(f"\n  Savings NPV (recomputed)  : {sav_npv/CRORE:.4f} Cr")
print(f"  Savings NPV (model)       : {fi['savings_npv']/CRORE:.4f} Cr")
print(f"  Match? {abs(sav_npv - fi['savings_npv']) < 100}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 8: Year-by-year savings — does augmentation actually help?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 8 — SAVINGS TRAJECTORY  (is augmentation helping each year?)")
print("=" * 80)
print()
print(f"  {'Yr':>3}  {'RE kWh (M)':>11}  {'Landed':>8}  {'Margin':>8}  "
      f"{'Savings Cr':>11}  {'OPEX Cr':>9}  {'AugPur Cr':>10}  {'':>5}")
print("  " + "-" * 80)
for yr in range(1, 26):
    idx = yr - 1
    met_kwh = float(ep["delivered_meter_mwh"][idx]) * MWH_TO_KWH
    margin  = discom - lts[idx]
    sav     = sv["annual_savings"][idx]
    opx     = op[idx]
    pur     = aug_res["augmentation_purchase_opex"].get(yr, 0.0)
    ev      = "AUG" if yr in aug_res["augmentation_years"] else ""
    print(f"  {yr:>3}  {met_kwh/1e6:>11.4f}  {lts[idx]:>8.4f}  {margin:>8.4f}  "
          f"{sav/CRORE:>11.4f}  {opx/CRORE:>9.4f}  {pur/CRORE:>10.4f}  {ev:>5}")

print()
print("  NOTE: Margin = DISCOM tariff - Landed tariff")
print("  Savings = RE_kWh x Margin (approximately)")
tot_sav = sum(sv["annual_savings"]) / CRORE
print(f"\n  Total undiscounted savings : {tot_sav:.4f} Cr")
print(f"  Total discounted NPV       : {fi['savings_npv']/CRORE:.4f} Cr")
