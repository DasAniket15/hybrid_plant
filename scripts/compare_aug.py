"""
compare_aug.py
──────────────
Proper augmentation comparison:
  Scenario A: No augmentation, solver-optimal 164 containers
  Scenario B: With augmentation + 5% Phase-2 oversizing, 173 containers
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
n_opt        = BASE["bess_containers"]            # 164 (solver-optimal)
n_extra      = max(math.ceil(n_opt * oversize_pct / 100), 0) if oversize_pct > 0 else 0
n_oversize   = n_opt + n_extra                    # 173

print(f"  n_opt={n_opt}  oversize={oversize_pct}%  n_extra={n_extra}  n_oversize={n_oversize}")

# ── Scenario A: no-aug, 164 containers ─────────────────────────────────────
aug_cfg["enabled"] = False
fe_a = FinanceEngine(config, data)
y1_a = ee.evaluate(**{**BASE, "bess_containers": n_opt})
fi_a = fe_a.evaluate(
    y1_a,
    solar_capacity_mw=BASE["solar_capacity_mw"],
    wind_capacity_mw=BASE["wind_capacity_mw"],
    ppa_capacity_mw=BASE["ppa_capacity_mw"],
    fast_mode=False,
)
aug_cfg["enabled"] = True  # restore

# ── Scenario B: aug + oversizing, 173 containers ────────────────────────────
fe_b = FinanceEngine(config, data)
y1_b = ee.evaluate(**{**BASE, "bess_containers": n_oversize})
fi_b = fe_b.evaluate(
    y1_b,
    solar_capacity_mw=BASE["solar_capacity_mw"],
    wind_capacity_mw=BASE["wind_capacity_mw"],
    ppa_capacity_mw=BASE["ppa_capacity_mw"],
    fast_mode=False,
)

# ── helpers ──────────────────────────────────────────────────────────────────
ep_a  = fi_a["energy_projection"]
ep_b  = fi_b["energy_projection"]
bus_a = ep_a["delivered_pre_mwh"]
bus_b = ep_b["delivered_pre_mwh"]
met_a = ep_a["delivered_meter_mwh"]
met_b = ep_b["delivered_meter_mwh"]
cuf_a = ep_a.get("cuf_per_year", np.zeros(25))
cuf_b = ep_b.get("cuf_per_year", np.zeros(25))

aug_res  = fi_b.get("augmentation") or {}
aug_yrs  = set(aug_res.get("augmentation_years", []))
aug_pur  = aug_res.get("augmentation_purchase_opex", {})
eff_cont = aug_res.get("effective_containers_per_year", {})
cont_add = aug_res.get("containers_added_per_year", {})
cohorts  = aug_res.get("cohorts", [])

ob_a = fi_a["opex_breakdown"]
ob_b = fi_b["opex_breakdown"]

CONT_SIZE = float(config.bess["bess"]["container"]["size_mwh"])

# ── Year-by-year table ────────────────────────────────────────────────────────
print()
print("=" * 105)
print("  YEAR-BY-YEAR ENERGY  (Scenario A = no-aug 164c | Scenario B = aug+oversize 173c)")
print("=" * 105)
hdr = (
    f"  {'Yr':>3}  {'Bus_A MWh':>12}  {'Bus_B MWh':>12}"
    f"  {'dBus%':>7}  {'CUF_A%':>7}  {'CUF_B%':>7}"
    f"  {'AugPur Cr':>10}  {'N_cont':>6}  {'Added':>5}  {'':>5}"
)
print(hdr)
print("-" * 105)

tot_bus_a = tot_bus_b = tot_met_a = tot_met_b = tot_aug_pur = 0.0
for i in range(25):
    yr    = i + 1
    ba    = float(bus_a[i])
    bb    = float(bus_b[i])
    ca    = float(cuf_a[i])
    cb    = float(cuf_b[i])
    pur   = aug_pur.get(yr, 0.0)
    nc    = eff_cont.get(yr, n_oversize)
    added = cont_add.get(yr, 0)
    db    = (bb - ba) / ba * 100 if ba > 0 else 0.0
    ev    = "AUG <--" if yr in aug_yrs else ""
    tot_bus_a += ba; tot_bus_b += bb
    tot_met_a += float(met_a[i]); tot_met_b += float(met_b[i])
    tot_aug_pur += pur
    print(
        f"  {yr:>3}  {ba:>12.1f}  {bb:>12.1f}"
        f"  {db:>+7.2f}  {ca:>7.3f}  {cb:>7.3f}"
        f"  {pur/1e7:>10.4f}  {nc:>6}  {added:>5}  {ev}"
    )

print("-" * 105)
db_tot = (tot_bus_b - tot_bus_a) / tot_bus_a * 100
print(
    f"  {'TOT':>3}  {tot_bus_a:>12.1f}  {tot_bus_b:>12.1f}"
    f"  {db_tot:>+7.2f}  {'':>7}  {'':>7}"
    f"  {tot_aug_pur/1e7:>10.4f}"
)

# ── Financial summary ─────────────────────────────────────────────────────────
cap_a  = fi_a["capex"]
cap_b  = fi_b["capex"]
lcd_a  = fi_a["lcoe_breakdown"]
lcd_b  = fi_b["lcoe_breakdown"]

print()
print("=" * 72)
print("  FINANCIAL SUMMARY")
print("=" * 72)
rows = [
    ("CAPEX breakdown",              None, None, ""),
    ("  Solar CAPEX (Cr)",           cap_a["solar_capex"]/1e7,           cap_b["solar_capex"]/1e7,          ".4f"),
    ("  BESS CAPEX (Cr)",            cap_a["bess_capex"]/1e7,            cap_b["bess_capex"]/1e7,           ".4f"),
    ("  Transmission CAPEX (Cr)",    cap_a["transmission_capex"]/1e7,    cap_b["transmission_capex"]/1e7,   ".4f"),
    ("  Total CAPEX (Cr)",           cap_a["total_capex"]/1e7,           cap_b["total_capex"]/1e7,          ".4f"),
    ("",                             None, None, ""),
    ("Year-1 BESS nameplate MWh",    float(y1_a["energy_capacity_mwh"]), float(y1_b["energy_capacity_mwh"]),".4f"),
    ("",                             None, None, ""),
    ("NPV breakdown",                None, None, ""),
    ("  NPV Total Cost (Cr)",        lcd_a["npv_total_cost"]/1e7,        lcd_b["npv_total_cost"]/1e7,       ".4f"),
    ("  NPV OPEX (Cr)",              lcd_a["npv_opex"]/1e7,              lcd_b["npv_opex"]/1e7,             ".4f"),
    ("  NPV Interest (Cr)",          lcd_a["npv_interest"]/1e7,          lcd_b["npv_interest"]/1e7,         ".4f"),
    ("  NPV Principal (Cr)",         lcd_a["npv_principal"]/1e7,         lcd_b["npv_principal"]/1e7,        ".4f"),
    ("  NPV RoE (Cr)",               lcd_a["npv_roe"]/1e7,               lcd_b["npv_roe"]/1e7,              ".4f"),
    ("  NPV Energy (B kWh)",         lcd_a["npv_energy_kwh"]/1e9,        lcd_b["npv_energy_kwh"]/1e9,       ".6f"),
    ("",                             None, None, ""),
    ("Key outputs",                  None, None, ""),
    ("  LCOE (Rs/kWh)",              fi_a["lcoe_inr_per_kwh"],           fi_b["lcoe_inr_per_kwh"],          ".4f"),
    ("  Landed tariff Y1 (Rs/kWh)",  fi_a["landed_tariff_series"][0],    fi_b["landed_tariff_series"][0],   ".4f"),
    ("  Annual savings Y1 (Cr)",     fi_a["annual_savings_year1"]/1e7,   fi_b["annual_savings_year1"]/1e7,  ".4f"),
    ("  Savings NPV (Cr)",           fi_a["savings_npv"]/1e7,            fi_b["savings_npv"]/1e7,           ".4f"),
    ("",                             None, None, ""),
    ("25-yr energy totals",          None, None, ""),
    ("  Busbar MWh",                 tot_bus_a,                          tot_bus_b,                         ".0f"),
    ("  Meter MWh",                  tot_met_a,                          tot_met_b,                         ".0f"),
    ("  Aug purchase OPEX (Cr)",     0.0,                                tot_aug_pur/1e7,                   ".4f"),
]

print(f"  {'Metric':<34}  {'Scen-A':>12}  {'Scen-B':>12}  {'Delta':>12}")
print("-" * 72)
for label, va, vb, fmt in rows:
    if not fmt:
        print(f"  {label}")
        continue
    if va is None:
        print()
        continue
    delta = vb - va
    print(f"  {label:<34}  {va:>12{fmt}}  {vb:>12{fmt}}  {delta:>+12{fmt}}")

# ── OPEX by year ──────────────────────────────────────────────────────────────
SKIP = {"year", "total", "bess_installed_mwh"}

def show_opex(yr):
    idx = yr - 1
    ra  = ob_a[idx]  if idx < len(ob_a)  else {}
    rb  = ob_b[idx]  if idx < len(ob_b)  else {}
    keys = [k for k in sorted(set(ra) | set(rb)) if k not in SKIP]
    print(f"\n  Year {yr}:")
    print(f"  {'Component':<28}  {'Scen-A Cr':>10}  {'Scen-B Cr':>10}  {'Delta Cr':>10}")
    print("  " + "-" * 60)
    tot_a = tot_b = 0.0
    for k in keys:
        va = ra.get(k, 0); vb = rb.get(k, 0)
        tot_a += va; tot_b += vb
        print(f"  {k:<28}  {va/1e7:>10.4f}  {vb/1e7:>10.4f}  {(vb-va)/1e7:>+10.4f}")
    bim_a = ra.get("bess_installed_mwh", 0)
    bim_b = rb.get("bess_installed_mwh", 0)
    print(f"  {'[BESS installed MWh]':<28}  {bim_a:>10.2f}  {bim_b:>10.2f}  {bim_b-bim_a:>+10.2f}  (MWh)")
    print(f"  {'TOTAL':<28}  {tot_a/1e7:>10.4f}  {tot_b/1e7:>10.4f}  {(tot_b-tot_a)/1e7:>+10.4f}")

print()
print("=" * 72)
print("  OPEX BREAKDOWN (key years)")
print("=" * 72)
for yr in [1, 6, 13, 19, 25]:
    show_opex(yr)

# ── Augmentation cohorts ──────────────────────────────────────────────────────
print()
print("=" * 62)
print("  AUGMENTATION COHORTS — Scenario B")
print("=" * 62)
total_n_running = 0
for start, n in cohorts:
    added = cont_add.get(start, 0) if start > 1 else n_oversize
    pur   = aug_pur.get(start, 0.0) / 1e7
    total_mwh = n * CONT_SIZE
    print(
        f"  Year {start:>3}: {n:>4} total containers  added={added:>3}"
        f"  pur={pur:.4f} Cr  installed_MWh={total_mwh:.1f}"
    )
