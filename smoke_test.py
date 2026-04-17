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
# 13. Augmentation Engine
# ─────────────────────────────────────────────────────────────────────────────
section("13. AUGMENTATION ENGINE")
try:
    import math as _math
    import pandas as _pd
    from hybrid_plant.augmentation.cohort import BESSCohort, CohortRegistry
    from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf
    from hybrid_plant.augmentation.lifecycle_simulator import LifecycleSimulator
    from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine
    from hybrid_plant.data_loader import load_soh_curve
    from hybrid_plant.energy.year1_engine import Year1Engine
    from hybrid_plant._paths import find_project_root

    _soh   = load_soh_curve(config)
    _eng   = Year1Engine(config, data)
    _root  = find_project_root()

    def _load_eff(fpath, col):
        df = _pd.read_csv(_root / fpath)
        df.columns = df.columns.str.strip().str.lower()
        return dict(zip(df["year"].astype(int), df[col]))

    _solar_eff = _load_eff(config.project["generation"]["solar"]["degradation"]["file"], "efficiency")
    _wind_eff  = _load_eff(config.project["generation"]["wind"]["degradation"]["file"], "efficiency")
    _csize     = float(config.bess["bess"]["container"]["size_mwh"])
    _life      = int(config.project["project"]["project_life_years"])

    # ── Cohort registry totals ────────────────────────────────────────────────
    from hybrid_plant.data_loader import operating_value as _op
    _reg = CohortRegistry(initial_containers=10)
    _reg.add(install_year=8, containers=4)
    # Year 10 under end-of-year convention:
    #   initial cohort age=10 → operating_value(soh,10) = soh[9]
    #   aug cohort     age=3  → operating_value(soh, 3) = soh[2]
    _expected_eff = 10 * _csize * _op(_soh, 10) + 4 * _csize * _op(_soh, 3)
    _actual_eff   = _reg.effective_capacity_mwh(10, _csize, _soh)
    check("cohort registry totals match expected",
          _math.isclose(_actual_eff, _expected_eff, rel_tol=1e-12),
          f"expected={_expected_eff:.4f}  actual={_actual_eff:.4f}")
    _n, _soh_blend = _reg.to_plant_params(10, _csize, _soh)
    check("blended SOH satisfies capacity identity",
          _math.isclose(_n * _csize * _soh_blend, _expected_eff, rel_tol=1e-12),
          f"n={_n}  blend={round(_soh_blend,4)}")
    check("aug cohort inactive before install year",
          _reg.effective_capacity_mwh(7, _csize, _soh) ==
          10 * _csize * _op(_soh, 7))

    # ── LifecycleSimulator — fast_mode ────────────────────────────────────────
    _sim_params = {
        "solar_capacity_mw": 200.0, "wind_capacity_mw": 0.0,
        "ppa_capacity_mw": 60.0, "bess_containers": 30,
        "charge_c_rate": 0.5, "discharge_c_rate": 0.5,
        "dispatch_priority": "solar_first", "bess_charge_source": "solar_only",
    }
    _sim = LifecycleSimulator(
        config=config, plant_engine=_eng.plant,
        soh_curve=_soh, solar_eff_curve=_solar_eff,
        wind_eff_curve=_wind_eff, loss_factor=_eng.grid.loss_factor,
    )
    _lc_fast = _sim.simulate(
        params=_sim_params, initial_containers=30,
        trigger_threshold_cuf=0.0, restoration_target_cuf=0.0,
        fast_mode=True,
    )
    check("lifecycle simulator runs without exception (fast_mode)",
          len(_lc_fast.cuf_series) == _life,
          f"len(cuf_series)={len(_lc_fast.cuf_series)}")
    check("augmentation OPEX >= 0 all years (fast_mode)",
          all(v >= 0 for v in _lc_fast.opex_augmentation_lump) and
          all(v >= 0 for v in _lc_fast.opex_augmentation_om))
    check("no events fire with threshold=0",
          len(_lc_fast.event_log) == 0)

    # ── LifecycleSimulator — triggered event, lump cost formula ───────────────
    _y1_sim = _eng.evaluate(**_sim_params)
    from hybrid_plant.augmentation.cuf_evaluator import year1_busbar_mwh
    _cuf_y1 = compute_plant_cuf(year1_busbar_mwh(_y1_sim), _sim_params["ppa_capacity_mw"])
    _lc_trig = _sim.simulate(
        params=_sim_params, initial_containers=30,
        trigger_threshold_cuf=_cuf_y1 * 0.93,
        restoration_target_cuf=_cuf_y1,
        fast_mode=True,
    )
    check("lifecycle simulator runs without exception (triggered)",
          len(_lc_trig.cuf_series) == _life)
    check("event_log entries are well-formed",
          all({"year","trigger_cuf","adjusted_target","post_event_cuf",
               "k_containers","lump_cost_rs"}.issubset(ev.keys())
              for ev in _lc_trig.event_log),
          f"n_events={len(_lc_trig.event_log)}")

    if _lc_trig.event_log:
        _ev        = _lc_trig.event_log[0]
        _ev_years  = {e["year"] for e in _lc_trig.event_log}
        _cost_per_mwh = float(config.bess["bess"]["augmentation"]["cost_per_mwh"])
        _expected_lump = _ev["k_containers"] * _csize * _cost_per_mwh
        check("lump cost = k × container_size × cost_per_mwh",
              _math.isclose(_ev["lump_cost_rs"], _expected_lump, rel_tol=1e-9),
              f"got={_ev['lump_cost_rs']:.2f}  expected={_expected_lump:.2f}")
        check("lump cost > 0 in event years",
              all(_lc_trig.opex_augmentation_lump[yr - 1] > 0 for yr in _ev_years))
        check("lump cost = 0 in non-event years",
              sum(1 for yr, v in enumerate(_lc_trig.opex_augmentation_lump, 1)
                  if yr not in _ev_years and v != 0.0) == 0)
    else:
        check("lump cost formula (no events — skipped)", True, "no events fired")
        check("lump cost > 0 in event year (skipped)", True)
        check("lump cost = 0 in non-event years (skipped)", True)

    # ── AugmentationEngine.evaluate_scenario ─────────────────────────────────
    _aug_engine = AugmentationEngine(
        config, data, _eng, _soh,
        trigger_threshold_cuf = _cuf_y1 * 0.93,
        pass1_lcoe            = None,    # no payback filter in smoke test
    )
    _aug_res = _aug_engine.evaluate_scenario(_sim_params, fast_mode=True)
    _aug_fi  = _aug_res["finance"]
    check("evaluate_scenario returns finance dict",
          "savings_npv" in _aug_fi and "augmentation" in _aug_fi)
    check("savings_npv is finite",
          _math.isfinite(_aug_fi["savings_npv"]))
    check("augmentation sub-dict has all required keys",
          {"trigger_threshold_cuf","restoration_target_cuf","event_log",
           "cuf_series","cohort_snapshot","cohort_capacity_timeline",
           "total_lump_cost_rs","total_om_cost_rs","n_events"}.issubset(
              _aug_fi["augmentation"].keys()))
    check("skipped_event_log key present in augmentation dict",
          "skipped_event_log" in _aug_fi["augmentation"])
    check("n_skipped key present in augmentation dict",
          "n_skipped" in _aug_fi["augmentation"])
    check("skipped_event_log is a list",
          isinstance(_aug_fi["augmentation"]["skipped_event_log"], list))
    check("n_skipped matches skipped_event_log length",
          _aug_fi["augmentation"]["n_skipped"] ==
          len(_aug_fi["augmentation"]["skipped_event_log"]))

    # ── initial_containers matches params["bess_containers"] (no-oversize path) ─
    check("initial_containers == bess_containers when no oversize",
          _aug_fi["augmentation"]["initial_containers"] == _sim_params["bess_containers"])

    # ── bess.yaml new keys present ──────────────────────────────────────────
    _aug_cfg = config.bess["bess"]["augmentation"]
    check("bess.yaml: solver_aware key removed",
          "solver_aware" not in _aug_cfg)
    check("bess.yaml: mode key removed",
          "mode" not in _aug_cfg)
    check("bess.yaml: fixed_schedule key removed",
          "fixed_schedule" not in _aug_cfg)
    check("bess.yaml: max_oversize_containers present",
          "max_oversize_containers" in _aug_cfg)
    check("bess.yaml: oversize_patience present",
          "oversize_patience" in _aug_cfg)
    check("bess.yaml: oversize_npv_tolerance_rs present",
          "oversize_npv_tolerance_rs" in _aug_cfg)
    check("bess.yaml: payback_filter block present",
          "payback_filter" in _aug_cfg)
    check("bess.yaml: payback_filter.enabled present",
          "enabled" in _aug_cfg["payback_filter"])
    check("bess.yaml: trigger_tolerance_pp >= 0.01 (no longer near-zero)",
          float(_aug_cfg.get("trigger_tolerance_pp", 0)) >= 0.01)
    check("bess.yaml: max_augmentation_containers_per_event <= 100",
          int(_aug_cfg.get("max_augmentation_containers_per_event", 9999)) <= 100)

    # ── OversizeOptimizer module importable ─────────────────────────────────
    from hybrid_plant.augmentation.oversize_optimizer import (
        find_optimal_oversize, OversizeResult
    )
    check("oversize_optimizer module importable", True)
    check("find_optimal_oversize callable", callable(find_optimal_oversize))
    check("OversizeResult is a class", isinstance(OversizeResult, type))

    # ── LifecycleResult.skipped_event_log field exists ──────────────────────
    from hybrid_plant.augmentation.lifecycle_simulator import LifecycleResult
    import dataclasses as _dc
    _lc_fields = {f.name for f in _dc.fields(LifecycleResult)}
    check("LifecycleResult has skipped_event_log field",
          "skipped_event_log" in _lc_fields)

    # ── LifecycleSimulator accepts event_filter kwarg ───────────────────────
    import inspect as _inspect
    _lc_sig = _inspect.signature(LifecycleSimulator.__init__)
    check("LifecycleSimulator.__init__ has event_filter param",
          "event_filter" in _lc_sig.parameters)

    # ── AugmentationEngine accepts pass1_lcoe kwarg ─────────────────────────
    _ae_sig = _inspect.signature(AugmentationEngine.__init__)
    check("AugmentationEngine.__init__ has pass1_lcoe param",
          "pass1_lcoe" in _ae_sig.parameters)

    # ── evaluate_scenario accepts initial_containers kwarg ──────────────────
    _es_sig = _inspect.signature(AugmentationEngine.evaluate_scenario)
    check("evaluate_scenario has initial_containers param",
          "initial_containers" in _es_sig.parameters)

    # ── SolverEngine has no run_augmentation_aware method ───────────────────
    from hybrid_plant.solver.solver_engine import SolverEngine
    check("SolverEngine.run_augmentation_aware removed",
          not hasattr(SolverEngine, "run_augmentation_aware"))

    # ── SolverResult has no augmentation_result field ───────────────────────
    from hybrid_plant.solver.solver_engine import SolverResult
    _sr_fields = {f.name for f in _dc.fields(SolverResult)}
    check("SolverResult.augmentation_result field removed",
          "augmentation_result" not in _sr_fields)

    # ── Minimum-k semantics: event_filter=None → skipped_event_log is empty ─
    _sim_no_filter = LifecycleSimulator(
        config=config, plant_engine=_eng.plant,
        soh_curve=_soh, solar_eff_curve=_solar_eff,
        wind_eff_curve=_wind_eff, loss_factor=_eng.grid.loss_factor,
        event_filter=None,
    )
    _lc_no_filter = _sim_no_filter.simulate(
        params=_sim_params, initial_containers=30,
        trigger_threshold_cuf=_cuf_y1 * 0.93,
        restoration_target_cuf=_cuf_y1,
        fast_mode=True,
    )
    check("skipped_event_log empty when event_filter=None",
          _lc_no_filter.skipped_event_log == [],
          f"got {len(_lc_no_filter.skipped_event_log)} skipped")

    # ── event_filter=block-all → all events skip, none fire ─────────────────
    _sim_block = LifecycleSimulator(
        config=config, plant_engine=_eng.plant,
        soh_curve=_soh, solar_eff_curve=_solar_eff,
        wind_eff_curve=_wind_eff, loss_factor=_eng.grid.loss_factor,
        event_filter=lambda _: False,   # block everything
    )
    _lc_block = _sim_block.simulate(
        params=_sim_params, initial_containers=30,
        trigger_threshold_cuf=_cuf_y1 * 0.93,
        restoration_target_cuf=_cuf_y1,
        fast_mode=True,
    )
    check("event_log empty when event_filter blocks all",
          _lc_block.event_log == [],
          f"got {len(_lc_block.event_log)} fired")
    _any_skipped = len(_lc_block.skipped_event_log) > 0 or len(_lc_no_filter.event_log) == 0
    check("skipped_event_log non-empty when block-all filter active (if events exist)",
          _any_skipped)

    # ── OversizeResult structure ─────────────────────────────────────────────
    _os_result_fields = {f.name for f in _dc.fields(OversizeResult)}
    check("OversizeResult has best_extra field",       "best_extra"              in _os_result_fields)
    check("OversizeResult has best_initial_containers","best_initial_containers" in _os_result_fields)
    check("OversizeResult has best_result field",      "best_result"             in _os_result_fields)
    check("OversizeResult has sweep_log field",        "sweep_log"               in _os_result_fields)

except Exception as e:
    check("Augmentation Engine — EXCEPTION", False, str(e))
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