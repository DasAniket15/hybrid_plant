"""
validate_lcoe.py
────────────────
First-principles LCOE validation.
Independently reconstructs LCOE from raw schedules and checks every
component of the numerator and denominator for both scenarios.
"""
from __future__ import annotations
import math, warnings
warnings.filterwarnings("ignore")

import numpy as np
from hybrid_plant.config_loader import load_config
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine

config = load_config()
data   = load_timeseries_data(config)
ee     = Year1Engine(config, data)

BASE = {
    "solar_capacity_mw":  195.415073395429,
    "wind_capacity_mw":   0.0,
    "bess_containers":    164,
    "charge_c_rate":      1.0,
    "discharge_c_rate":   1.0,
    "ppa_capacity_mw":    67.5256615562851,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_only",
}

aug_cfg      = config.bess["bess"]["augmentation"]
oversize_pct = float(aug_cfg.get("oversize_percent", 0.0))
n_opt        = 164
n_extra      = max(math.ceil(n_opt * oversize_pct / 100), 0) if oversize_pct > 0 else 0
n_oversize   = n_opt + n_extra   # 173

# ── Scenario A: no-aug, 164 containers ─────────────────────────────────────
aug_cfg["enabled"] = False
fe_a = FinanceEngine(config, data)
y1_a = ee.evaluate(**{**BASE, "bess_containers": n_opt})
fi_a = fe_a.evaluate(y1_a,
    solar_capacity_mw=BASE["solar_capacity_mw"],
    wind_capacity_mw=BASE["wind_capacity_mw"],
    ppa_capacity_mw=BASE["ppa_capacity_mw"],
    fast_mode=False)
aug_cfg["enabled"] = True

# ── Scenario B: aug + oversizing, 173 containers ────────────────────────────
fe_b = FinanceEngine(config, data)
y1_b = ee.evaluate(**{**BASE, "bess_containers": n_oversize})
fi_b = fe_b.evaluate(y1_b,
    solar_capacity_mw=BASE["solar_capacity_mw"],
    wind_capacity_mw=BASE["wind_capacity_mw"],
    ppa_capacity_mw=BASE["ppa_capacity_mw"],
    fast_mode=False)

WACC = fi_a["wacc"]

def npv(series, wacc=WACC):
    return sum(v / (1 + wacc) ** (t + 1) for t, v in enumerate(series))

# ── Manual LCOE reconstruction ───────────────────────────────────────────────
print("=" * 80)
print("  FIRST-PRINCIPLES LCOE RECONSTRUCTION")
print("=" * 80)

for label, fi, y1, n_cont in [
    ("Scen-A (164c, no-aug)", fi_a, y1_a, n_opt),
    ("Scen-B (173c, aug)",    fi_b, y1_b, n_oversize),
]:
    lcd = fi["lcoe_breakdown"]
    cap = fi["capex"]
    ep  = fi["energy_projection"]
    ob  = fi["opex_breakdown"]

    total_capex  = cap["total_capex"]
    debt_frac    = 0.70
    equity_frac  = 0.30
    interest_rate = 0.085
    roe_rate      = 0.20
    project_life  = 25
    debt_tenure   = 25

    debt   = total_capex * debt_frac
    equity = total_capex * equity_frac

    # Debt EMI schedule (fixed EMI)
    r, n = interest_rate, debt_tenure
    emi = debt * r * (1 + r)**n / ((1 + r)**n - 1)
    balance = debt
    interest_sched  = []
    principal_sched = []
    for yr in range(1, project_life + 1):
        if yr <= debt_tenure and balance > 1e-6:
            i = balance * r
            p = emi - i
            balance = max(balance - p, 0.0)
        else:
            i = p = 0.0
        interest_sched.append(i)
        principal_sched.append(p)

    roe_sched = [equity * roe_rate] * project_life

    # OPEX from model
    opex_sched = fi["opex_projection"]

    # Energy from model
    busbar_mwh = ep["delivered_pre_mwh"]
    busbar_kwh = [float(e) * 1e3 for e in busbar_mwh]   # MWh → kWh

    # NPVs (manual)
    npv_interest   = npv(interest_sched)
    npv_principal  = npv(principal_sched)
    npv_roe        = npv(roe_sched)
    npv_opex       = npv(opex_sched)
    npv_total      = npv_interest + npv_principal + npv_roe + npv_opex
    npv_energy_kwh = npv(busbar_kwh)

    lcoe_manual    = npv_total / npv_energy_kwh

    # Aug OPEX breakdown
    aug_res = fi.get("augmentation") or {}
    aug_pur = aug_res.get("augmentation_purchase_opex", {})
    aug_purchase_total  = sum(aug_pur.values())
    aug_purchase_npv    = npv([aug_pur.get(yr, 0.0) for yr in range(1, 26)])

    aug_yrs = sorted(aug_pur.keys())

    print(f"\n  ── {label} ──")
    print(f"  {'Initial CAPEX (Cr)':35} = {total_capex/1e7:.4f}")
    print(f"  {'  BESS CAPEX (Cr)':35} = {cap['bess_capex']/1e7:.4f}  ({n_cont} × {float(y1['energy_capacity_mwh'])/n_cont:.4f} MWh × 60L/MWh)")
    print(f"  {'Debt (Cr)':35} = {debt/1e7:.4f}  (70%)")
    print(f"  {'Equity (Cr)':35} = {equity/1e7:.4f}  (30%)")
    print(f"  {'Annual ROE (Cr/yr)':35} = {equity*roe_rate/1e7:.4f}  (20% × equity)")
    print()
    print(f"  {'NPV Interest (Cr)':35} = {npv_interest/1e7:.4f}  [model: {lcd['npv_interest']/1e7:.4f}]")
    print(f"  {'NPV Principal (Cr)':35} = {npv_principal/1e7:.4f}  [model: {lcd['npv_principal']/1e7:.4f}]")
    print(f"  {'NPV ROE (Cr)':35} = {npv_roe/1e7:.4f}  [model: {lcd['npv_roe']/1e7:.4f}]")
    print(f"  {'NPV OPEX (Cr)':35} = {npv_opex/1e7:.4f}  [model: {lcd['npv_opex']/1e7:.4f}]")
    print(f"  {'  of which: aug purchase (face)':35} = {aug_purchase_total/1e7:.4f} Cr  in years {aug_yrs}")
    print(f"  {'  of which: aug purchase (NPV)':35} = {aug_purchase_npv/1e7:.4f} Cr")
    print(f"  {'  OPEX ex-aug (NPV)':35} = {(npv_opex - aug_purchase_npv)/1e7:.4f} Cr")
    print()
    print(f"  {'NPV Total Costs (Cr)':35} = {npv_total/1e7:.4f}  [model: {lcd['npv_total_cost']/1e7:.4f}]")
    print(f"  {'NPV Energy (B kWh)':35} = {npv_energy_kwh/1e9:.6f}  [model: {lcd['npv_energy_kwh']/1e9:.6f}]")
    print()
    print(f"  {'LCOE manual (Rs/kWh)':35} = {lcoe_manual:.6f}")
    print(f"  {'LCOE model (Rs/kWh)':35} = {fi['lcoe_inr_per_kwh']:.6f}")
    print(f"  {'Match?':35} = {abs(lcoe_manual - fi['lcoe_inr_per_kwh']) < 1e-6}")

# ── Marginal LCOE of augmentation ────────────────────────────────────────────
lcd_a = fi_a["lcoe_breakdown"]
lcd_b = fi_b["lcoe_breakdown"]
ep_a  = fi_a["energy_projection"]
ep_b  = fi_b["energy_projection"]

delta_npv_cost   = lcd_b["npv_total_cost"]   - lcd_a["npv_total_cost"]
delta_npv_energy = lcd_b["npv_energy_kwh"]   - lcd_a["npv_energy_kwh"]
marginal_lcoe    = delta_npv_cost / delta_npv_energy

delta_busbar_25yr = sum(float(b) for b in ep_b["delivered_pre_mwh"]) \
                  - sum(float(b) for b in ep_a["delivered_pre_mwh"])

print()
print("=" * 80)
print("  MARGINAL LCOE OF AUGMENTATION  (why the blended delta is small)")
print("=" * 80)
print(f"\n  NPV Total Costs Scen-A (Cr):           {lcd_a['npv_total_cost']/1e7:.4f}")
print(f"  NPV Total Costs Scen-B (Cr):           {lcd_b['npv_total_cost']/1e7:.4f}")
print(f"  DELTA NPV Costs (Cr):                 +{delta_npv_cost/1e7:.4f}")
print()
print(f"  NPV Energy Scen-A (B kWh):             {lcd_a['npv_energy_kwh']/1e9:.6f}")
print(f"  NPV Energy Scen-B (B kWh):             {lcd_b['npv_energy_kwh']/1e9:.6f}")
print(f"  DELTA NPV Energy (M kWh):             +{delta_npv_energy/1e6:.2f}")
print()
print(f"  Marginal LCOE = dCost / dEnergy        {marginal_lcoe:.4f} Rs/kWh")
print(f"  Base LCOE Scen-A:                      {fi_a['lcoe_inr_per_kwh']:.4f} Rs/kWh")
print(f"  Blended LCOE Scen-B:                   {fi_b['lcoe_inr_per_kwh']:.4f} Rs/kWh")
print(f"  LCOE delta (blended):                 +{fi_b['lcoe_inr_per_kwh'] - fi_a['lcoe_inr_per_kwh']:.4f} Rs/kWh")
print()
print(f"  25-yr face busbar delta (MWh):        +{delta_busbar_25yr:,.0f}")
print(f"  Augmentation share of total energy:    {delta_busbar_25yr/sum(float(b) for b in ep_b['delivered_pre_mwh'])*100:.2f}%")
print()
print("  Why delta LCOE is small:")
print(f"  Scen-A base: {lcd_a['npv_total_cost']/1e7:.1f} Cr / {lcd_a['npv_energy_kwh']/1e9:.4f} B kWh = {fi_a['lcoe_inr_per_kwh']:.4f} Rs/kWh")
print(f"  Augm. marginal: {delta_npv_cost/1e7:.1f} Cr / {delta_npv_energy/1e6:.0f} M kWh = {marginal_lcoe:.4f} Rs/kWh")
print(f"  Blended (scen B): ({lcd_a['npv_total_cost']/1e7:.1f}+{delta_npv_cost/1e7:.1f}) / ({lcd_a['npv_energy_kwh']/1e9:.4f}B+{delta_npv_energy/1e6:.0f}M) = {fi_b['lcoe_inr_per_kwh']:.4f} Rs/kWh")

# ── Insurance bug check ───────────────────────────────────────────────────────
print()
print("=" * 80)
print("  INSURANCE BUG CHECK")
print("=" * 80)
print()
aug_res_b = fi_b.get("augmentation") or {}
aug_pur_b = aug_res_b.get("augmentation_purchase_opex", {})
cap_b     = fi_b["capex"]
ins_rate  = config.finance["opex"]["insurance"]["percent_of_total_capex"] / 100.0

print(f"  Insurance rate:  {ins_rate*100:.2f}% of asset replacement value per year")
print(f"  Initial CAPEX:   {cap_b['total_capex']/1e7:.4f} Cr")
print()
print(f"  Current model: insurance = {ins_rate*100:.2f}% × initial CAPEX only (CONSTANT)")
print(f"  Correct:        insurance = {ins_rate*100:.2f}% × (initial CAPEX + cumulative aug spend) per year")
print()

cumulative_aug = 0.0
insurance_model_total = 0.0
insurance_correct_total = 0.0
print(f"  {'Yr':>3}  {'CumAug Cr':>10}  {'Insured Cr':>12}  {'Ins_model Cr/yr':>16}  {'Ins_correct Cr/yr':>18}  {'Delta Cr/yr':>12}")
print("  " + "-" * 76)
for yr in range(1, 26):
    cumulative_aug += aug_pur_b.get(yr, 0.0)
    insured_base  = cap_b["total_capex"]
    insured_correct = insured_base + cumulative_aug
    ins_model_yr   = ins_rate * insured_base
    ins_correct_yr = ins_rate * insured_correct
    delta = ins_correct_yr - ins_model_yr
    insurance_model_total   += ins_model_yr
    insurance_correct_total += ins_correct_yr
    flag = " <-- aug" if yr in aug_pur_b else ""
    print(f"  {yr:>3}  {cumulative_aug/1e7:>10.4f}  {insured_correct/1e7:>12.4f}  "
          f"{ins_model_yr/1e7:>16.4f}  {ins_correct_yr/1e7:>18.4f}  {delta/1e7:>12.4f}{flag}")

print("  " + "-" * 76)
delta_total = insurance_correct_total - insurance_model_total
delta_npv_ins = npv([(ins_rate * (cap_b["total_capex"] + sum(aug_pur_b.get(y, 0.0) for y in range(1, yr+1)))
                      - ins_rate * cap_b["total_capex"])
                     for yr in range(1, 26)])
print(f"  {'TOTAL (face)':>50}  {insurance_model_total/1e7:>16.4f}  {insurance_correct_total/1e7:>18.4f}  {delta_total/1e7:>+12.4f} Cr")
print(f"  {'Insurance undercharge NPV (Cr)':>50}  {delta_npv_ins/1e7:>+12.4f}")
print()
print(f"  Impact on LCOE:  {delta_npv_ins/lcd_b['npv_energy_kwh']:>+.6f} Rs/kWh")

# ── ROE on augmentation equity ────────────────────────────────────────────────
print()
print("=" * 80)
print("  ROE ON AUGMENTATION EQUITY CHECK")
print("=" * 80)
print()
print("  In the current model, ROE = equity × 20% is fixed on INITIAL CAPEX only.")
print("  Augmentation purchases are funded as OPEX (from cashflows, not new equity).")
print("  => No ROE is charged on augmentation containers.")
print()
print("  Is this correct?")
print("  - If augmentation is funded from project operating cashflows: YES (OPEX treatment OK)")
print("  - If augmentation requires fresh equity injection: NO (should add ROE)")
print()
print("  Current design choice: OPEX (funded from cashflows).")
print("  Quantification IF treated as equity-funded:")

equity_frac = 0.30
roe_rate    = 0.20
extra_roe_series = [0.0] * 25
cumulative_aug = 0.0
for yr in range(1, 26):
    aug_this_yr = aug_pur_b.get(yr, 0.0)
    if aug_this_yr > 0:
        cumulative_aug += aug_this_yr * equity_frac
    extra_roe_series[yr - 1] = cumulative_aug * roe_rate

npv_extra_roe = npv(extra_roe_series)
extra_lcoe    = npv_extra_roe / lcd_b["npv_energy_kwh"]
print(f"  Cumulative aug equity (nominal) = {0.30 * sum(aug_pur_b.values())/1e7:.4f} Cr")
print(f"  NPV of extra ROE if equity-funded = {npv_extra_roe/1e7:.4f} Cr")
print(f"  LCOE impact if equity-funded      = {extra_lcoe:+.4f} Rs/kWh")
print(f"  Adjusted LCOE if equity-funded    = {fi_b['lcoe_inr_per_kwh'] + extra_lcoe:.4f} Rs/kWh")
print(f"  Adjusted LCOE delta vs Scen-A     = {fi_b['lcoe_inr_per_kwh'] + extra_lcoe - fi_a['lcoe_inr_per_kwh']:+.4f} Rs/kWh")
