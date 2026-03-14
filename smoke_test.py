"""
smoke_test.py
─────────────
Self-contained smoke test using only stdlib + numpy/pandas/pyyaml.
Validates every layer of the pipeline (config, data, energy, finance).
Solver layer is skipped (requires optuna — install separately).

Run:
    python smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# ── Make the src tree importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    icon = PASS if condition else FAIL
    print(f"  {icon}  {name:<55}  {detail}")


def section(title: str) -> None:
    print(f"\n{'─'*10} {title} {'─'*max(2, 68 - len(title) - 12)}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config loader
# ─────────────────────────────────────────────────────────────────────────────
section("1. CONFIG LOADER")
try:
    from hybrid_plant.config_loader import load_config, FullConfig
    config = load_config()
    check("load_config() returns FullConfig",       isinstance(config, FullConfig))
    check("project.yaml loaded",                   "project" in config.project)
    check("bess.yaml loaded",                      "bess"    in config.bess)
    check("finance.yaml loaded",                   "capex"   in config.finance)
    check("tariffs.yaml loaded",                   "discom"  in config.tariffs)
    check("regulatory.yaml loaded",                "regulatory" in config.regulatory)
    check("solver.yaml loaded",                    "solver"  in config.solver)
    check("project_life_years = 25",               config.project["project"]["project_life_years"] == 25)
    check("BESS container size > 0",               config.bess["bess"]["container"]["size_mwh"] > 0)
except Exception as e:
    check("Config loader — EXCEPTION", False, str(e))
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loader
# ─────────────────────────────────────────────────────────────────────────────
section("2. DATA LOADER")
try:
    from hybrid_plant.data_loader import load_timeseries_data
    data = load_timeseries_data(config)
    check("solar_cuf loaded",                      "solar_cuf"      in data)
    check("wind_cuf loaded",                       "wind_cuf"       in data)
    check("load_profile loaded",                   "load_profile"   in data)
    check("solar_cuf length = 8760",               len(data["solar_cuf"])    == 8760)
    check("wind_cuf length = 8760",                len(data["wind_cuf"])     == 8760)
    check("load_profile length = 8760",            len(data["load_profile"]) == 8760)
    check("solar_cuf in [0, 1]",                   data["solar_cuf"].max()    <= 1.0)
    check("wind_cuf in [0, 1]",                    data["wind_cuf"].max()     <= 1.0)
    check("load_profile > 0 (MWh)",                data["load_profile"].min() >  0.0)
    check("degradation curves loaded",             "solar_degradation_curve" in data)
except Exception as e:
    check("Data loader — EXCEPTION", False, str(e))
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constants
# ─────────────────────────────────────────────────────────────────────────────
section("3. CONSTANTS")
try:
    from hybrid_plant.constants import MWH_TO_KWH, LAKH_TO_RS, CRORE_TO_RS, HOURS_PER_YEAR
    check("MWH_TO_KWH = 1000",                     MWH_TO_KWH   == 1_000.0)
    check("LAKH_TO_RS = 1e5",                      LAKH_TO_RS   == 1e5)
    check("CRORE_TO_RS = 1e7",                     CRORE_TO_RS  == 1e7)
    check("HOURS_PER_YEAR = 8760",                 HOURS_PER_YEAR == 8_760)
except Exception as e:
    check("Constants — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. GridInterface
# ─────────────────────────────────────────────────────────────────────────────
section("4. GRID INTERFACE")
try:
    from hybrid_plant.energy.grid_interface import GridInterface
    gi = GridInterface(config)
    check("loss_factor computed",                  hasattr(gi, "loss_factor"))
    check("loss_factor in (0, 1]",                 0 < gi.loss_factor <= 1.0,
          f"= {round(gi.loss_factor, 4)}")
    arr = np.ones(8760) * 10.0
    res = gi.apply_losses(arr)
    check("apply_losses keys present",             {"meter_delivery","annual_meter_delivery","loss_factor"}.issubset(res))
    check("meter_delivery = arr × loss_factor",    np.allclose(res["meter_delivery"], arr * gi.loss_factor))
except Exception as e:
    check("GridInterface — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. MeterLayer
# ─────────────────────────────────────────────────────────────────────────────
section("5. METER LAYER")
try:
    from hybrid_plant.energy.meter_layer import MeterLayer
    ml  = MeterLayer(data)
    load = data["load_profile"]
    res  = ml.compute_shortfall(np.zeros(8760))    # zero delivery → full shortfall
    check("shortfall = load when delivery = 0",    np.allclose(res["shortfall"], load))
    res2 = ml.compute_shortfall(load + 99)         # surplus → zero shortfall
    check("shortfall = 0 when delivery > load",    np.all(res2["shortfall"] == 0))
except Exception as e:
    check("MeterLayer — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 6. PlantEngine — solar-only benchmark
# ─────────────────────────────────────────────────────────────────────────────
section("6. PLANT ENGINE  (solar-only benchmark)")

SOLAR_PARAMS = dict(
    solar_capacity_mw  = 195.415073395429,
    wind_capacity_mw   = 0.0,
    bess_containers    = 164,
    charge_c_rate      = 1.0,
    discharge_c_rate   = 1.0,
    ppa_capacity_mw    = 67.5256615562851,
    dispatch_priority  = "solar_first",
    bess_charge_source = "solar_only",
)

try:
    from hybrid_plant.energy.plant_engine import PlantEngine
    from hybrid_plant.energy.grid_interface import GridInterface
    pe  = PlantEngine(config, data)
    lf  = GridInterface(config).loss_factor
    res = pe.simulate(loss_factor=lf, **SOLAR_PARAMS)

    # Energy conservation
    raw_gen = float(np.sum(
        SOLAR_PARAMS["solar_capacity_mw"] * data["solar_cuf"]
        + SOLAR_PARAMS["wind_capacity_mw"]  * data["wind_cuf"]
    ))
    rhs = float(np.sum(res["solar_direct_pre"]) + np.sum(res["wind_direct_pre"])
                + np.sum(res["charge_pre"])       + np.sum(res["curtailment_pre"]))
    conservation_err = abs(raw_gen - rhs)

    check("Energy conservation (err < 0.001 MWh)",     conservation_err < 0.001,
          f"err = {conservation_err:.6f}")
    check("PPA cap not violated",
          bool(np.all(res["plant_export_pre"] <= SOLAR_PARAMS["ppa_capacity_mw"] + 1e-6)))
    check("SOC non-negative at year end",               res["bess_end_soc_mwh"] >= -1e-6,
          f"SOC = {round(res['bess_end_soc_mwh'], 3)} MWh")
    check("BESS was discharged",                        float(np.sum(res["discharge_pre"])) > 0,
          f"{round(float(np.sum(res['discharge_pre'])), 1)} MWh")
    check("No negative curtailment",                    bool(np.all(res["curtailment_pre"] >= -1e-9)))
    check("Output arrays are length 8760",              len(res["solar_direct_pre"]) == 8760)
    check("Wind direct = 0 (solar-only)",               float(np.sum(res["wind_direct_pre"])) == 0.0,
          f"{float(np.sum(res['wind_direct_pre'])):.1f} MWh")
except Exception as e:
    check("PlantEngine — EXCEPTION", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Year1Engine — end-to-end
# ─────────────────────────────────────────────────────────────────────────────
section("7. YEAR1 ENGINE  (end-to-end, solar-only)")
try:
    from hybrid_plant.energy.year1_engine import Year1Engine
    y1e = Year1Engine(config, data)
    y1  = y1e.evaluate(**SOLAR_PARAMS)

    annual_load = float(np.sum(data["load_profile"]))
    meter_y1    = float(y1["annual_meter_delivery"])

    check("loss_factor in result",                  "loss_factor" in y1)
    check("meter_delivery ≤ annual load",           meter_y1 <= annual_load + 1.0,
          f"{round(meter_y1,1)} ≤ {round(annual_load,1)} MWh")
    check("shortfall non-negative",                 bool(np.all(y1["shortfall"] >= 0)))
    check("shortfall + meter ≈ load",
          abs(float(np.sum(y1["shortfall"])) + meter_y1 - annual_load) < 1.0)
except Exception as e:
    check("Year1Engine — EXCEPTION", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CapexModel
# ─────────────────────────────────────────────────────────────────────────────
section("8. CAPEX MODEL")
try:
    from hybrid_plant.finance.capex_model import CapexModel
    cm  = CapexModel(config)
    cap = cm.compute(
        solar_capacity_mw        = SOLAR_PARAMS["solar_capacity_mw"],
        wind_capacity_mw         = SOLAR_PARAMS["wind_capacity_mw"],
        bess_energy_capacity_mwh = SOLAR_PARAMS["bess_containers"]
                                   * config.bess["bess"]["container"]["size_mwh"],
    )
    component_sum = (cap["solar_capex"] + cap["wind_capex"]
                     + cap["bess_capex"] + cap["transmission_capex"])
    check("total_capex = sum of components",        abs(cap["total_capex"] - component_sum) < 1.0,
          f"Rs {round(cap['total_capex']/1e7, 2)} Cr")
    check("wind_capex = 0 (no wind)",               cap["wind_capex"] == 0.0)
    check("solar_dc_mwp > solar_ac_mw",             cap["solar_dc_mwp"] > SOLAR_PARAMS["solar_capacity_mw"])
    check("total_capex > 0",                        cap["total_capex"] > 0)
except Exception as e:
    check("CapexModel — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 9. OpexModel
# ─────────────────────────────────────────────────────────────────────────────
section("9. OPEX MODEL")
try:
    from hybrid_plant.finance.opex_model import OpexModel
    om  = OpexModel(config)
    proj, breakdown = om.compute(
        solar_capacity_mw = SOLAR_PARAMS["solar_capacity_mw"],
        wind_capacity_mw  = SOLAR_PARAMS["wind_capacity_mw"],
        bess_energy_mwh   = SOLAR_PARAMS["bess_containers"] * config.bess["bess"]["container"]["size_mwh"],
        total_capex       = 1e10,
    )
    check("projection length = project_life",       len(proj) == config.project["project"]["project_life_years"],
          f"= {len(proj)}")
    check("all years positive",                     all(v > 0 for v in proj))
    check("year 25 ≥ year 1 (escalation)",          proj[-1] >= proj[0],
          f"Y1={round(proj[0]/1e7,3)} Cr  Y25={round(proj[-1]/1e7,3)} Cr")
    total_err = max(abs(p - b["total"]) for p, b in zip(proj, breakdown))
    check("breakdown totals match projection",      total_err < 1.0,
          f"max err = {total_err:.4f}")
except Exception as e:
    check("OpexModel — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 10. LCOEModel
# ─────────────────────────────────────────────────────────────────────────────
section("10. LCOE MODEL")
try:
    from hybrid_plant.finance.lcoe_model import LCOEModel
    from hybrid_plant.constants import PERCENT_TO_DECIMAL
    lm  = LCOEModel(config)
    fin = config.finance["financing"]
    d   = fin["debt_percent"]                          * PERCENT_TO_DECIMAL
    e   = fin["equity_percent"]                        * PERCENT_TO_DECIMAL
    rd  = fin["debt"]["interest_rate_percent"]         * PERCENT_TO_DECIMAL
    re  = fin["equity"]["return_on_equity_percent"]    * PERCENT_TO_DECIMAL
    tc  = fin["corporate_tax_rate_percent"]            * PERCENT_TO_DECIMAL
    expected_wacc = d * rd * (1 - tc) + e * re
    check("WACC formula correct",                   abs(lm.wacc - expected_wacc) < 1e-10,
          f"WACC = {round(lm.wacc*100, 4)} %")
    check("WACC > 0",                               lm.wacc > 0)
except Exception as e:
    check("LCOEModel — EXCEPTION", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 11. FinanceEngine — full pipeline
# ─────────────────────────────────────────────────────────────────────────────
section("11. FINANCE ENGINE  (full pipeline, solar-only)")
try:
    from hybrid_plant.finance.finance_engine import FinanceEngine
    fe  = FinanceEngine(config, data)
    fi  = fe.evaluate(
        year1_results     = y1,
        solar_capacity_mw = SOLAR_PARAMS["solar_capacity_mw"],
        wind_capacity_mw  = SOLAR_PARAMS["wind_capacity_mw"],
        ppa_capacity_mw   = SOLAR_PARAMS["ppa_capacity_mw"],
    )
    lcoe    = fi["lcoe_inr_per_kwh"]
    lts     = fi["landed_tariff_series"]
    sv      = fi["savings_breakdown"]
    discom  = sv["discom_tariff"]
    npv     = fi["savings_npv"]
    y1_sav  = fi["annual_savings_year1"]
    ep      = fi["energy_projection"]

    check("LCOE in plausible range (3–15 Rs/kWh)",  3.0 < lcoe < 15.0,
          f"LCOE = {round(lcoe, 4)} Rs/kWh")
    check("landed_tariff_series length = 25",        len(lts) == 25)
    check("Year-1 landed tariff < DISCOM tariff",    lts[0] < discom,
          f"{round(lts[0],4)} < {round(discom,4)} Rs/kWh")
    check("Year-1 savings positive",                 y1_sav > 0,
          f"Rs {round(y1_sav/1e7, 4)} Cr")
    check("Savings NPV positive",                    npv > 0,
          f"NPV = Rs {round(npv/1e7, 2)} Cr")
    check("Meter energy ≤ busbar energy",
          bool(np.all(ep["delivered_meter_mwh"] <= ep["delivered_pre_mwh"] + 1e-6)))
    check("Energy degrades over 25 years",
          ep["delivered_pre_mwh"][-1] < ep["delivered_pre_mwh"][0],
          f"Y1={round(ep['delivered_pre_mwh'][0],1)}  Y25={round(ep['delivered_pre_mwh'][-1],1)} MWh")
    check("DISCOM tariff in plausible range",         5.0 < discom < 15.0,
          f"{round(discom, 4)} Rs/kWh")
    check("opex_breakdown length = 25",               len(fi["opex_breakdown"]) == 25)
except Exception as e:
    check("FinanceEngine — EXCEPTION", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Solar + Wind case
# ─────────────────────────────────────────────────────────────────────────────
section("12. PLANT ENGINE  (solar + wind benchmark)")

WIND_PARAMS = dict(
    solar_capacity_mw  = 190.454972460807,
    wind_capacity_mw   = 116.130108575195,
    bess_containers    = 120,
    charge_c_rate      = 1.0,
    discharge_c_rate   = 1.0,
    ppa_capacity_mw    = 120.632227022855,
    dispatch_priority  = "solar_first",
    bess_charge_source = "solar_only",
)
try:
    y1e2  = Year1Engine(config, data)
    y1_sw = y1e2.evaluate(**WIND_PARAMS)
    raw_gen = float(np.sum(
        WIND_PARAMS["solar_capacity_mw"] * data["solar_cuf"]
        + WIND_PARAMS["wind_capacity_mw"]  * data["wind_cuf"]
    ))
    rhs = float(np.sum(y1_sw["solar_direct_pre"]) + np.sum(y1_sw["wind_direct_pre"])
                + np.sum(y1_sw["charge_pre"])       + np.sum(y1_sw["curtailment_pre"]))
    check("Energy conservation (solar+wind)",         abs(raw_gen - rhs) < 0.001,
          f"err = {abs(raw_gen - rhs):.6f}")
    check("Wind direct > 0",                          float(np.sum(y1_sw["wind_direct_pre"])) > 0,
          f"{round(float(np.sum(y1_sw['wind_direct_pre'])), 1)} MWh")
    check("Solar direct > 0",                         float(np.sum(y1_sw["solar_direct_pre"])) > 0)
except Exception as e:
    check("Solar+Wind — EXCEPTION", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"  Result: {passed}/{total} checks passed"
      + (f"  ({failed} FAILED)" if failed else "  — all green ✓"))
print(f"{'═'*70}\n")

if failed:
    print("Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL}  {name}  {detail}")
    sys.exit(1)
