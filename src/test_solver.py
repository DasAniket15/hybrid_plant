"""
test_solver.py
──────────────
Runs the SolverEngine and prints a structured results report.
"""

import numpy as np

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine
from solver.solver_engine import SolverEngine


def cr(v):   return round(v / 1e7, 4)
def sep(t=""): print(f"\n{'─'*10} {t} {'─'*(50-len(t))}" if t else "─"*62)


if __name__ == "__main__":

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    config         = load_config()
    data           = load_timeseries_data(config)
    energy_engine  = Year1Engine(config, data)
    finance_engine = FinanceEngine(config, data)

    solver = SolverEngine(config, data, energy_engine, finance_engine)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("Running solver …")
    result = solver.run(n_trials=1500, show_progress=True)

    # ── Best solution ─────────────────────────────────────────────────────────
    sep("OPTIMAL CONFIGURATION")
    p = result.best_params
    y1 = result.full_result["year1"]
    fi = result.full_result["finance"]

    print(f"  Solar capacity (AC MW)     : {round(p['solar_capacity_mw'], 2)}")
    print(f"  Solar capacity (DC MWp)    : {round(fi['capex']['solar_dc_mwp'], 2)}")
    print(f"  Wind capacity (MW)         : {round(p['wind_capacity_mw'], 2)}")
    print(f"  BESS containers            : {p['bess_containers']}")
    print(f"  BESS energy capacity (MWh) : {round(float(y1['energy_capacity_mwh']), 2)}")
    print(f"  PPA capacity (MW)          : {round(p['ppa_capacity_mw'], 2)}")

    sep("CAPEX  (Rs Crore)")
    cap = fi["capex"]
    print(f"  Solar                      : {cr(cap['solar_capex'])}")
    print(f"  Wind                       : {cr(cap['wind_capex'])}")
    print(f"  BESS                       : {cr(cap['bess_capex'])}")
    print(f"  Transmission               : {cr(cap['transmission_capex'])}")
    print(f"  ── Total                   : {cr(cap['total_capex'])}")

    sep("LCOE & TARIFF")
    lts = fi["landed_tariff_series"]
    print(f"  WACC                       : {round(fi['wacc']*100, 4)} %")
    print(f"  LCOE                       : Rs {round(fi['lcoe_inr_per_kwh'], 4)} / kWh")
    print(f"  Landed tariff Year 1       : Rs {round(lts[0], 4)} / kWh")
    print(f"  Landed tariff Year 25      : Rs {round(lts[-1], 4)} / kWh")

    sep("CLIENT SAVINGS")
    sv = fi["savings_breakdown"]
    print(f"  DISCOM tariff (wt. avg)    : Rs {round(sv['discom_tariff'], 4)} / kWh")
    print(f"  Baseline annual cost (Cr)  : {cr(sv['baseline_annual_cost'])}")
    print(f"  Hybrid cost Year 1  (Cr)   : {cr(sv['annual_hybrid_cost'][0])}")
    print(f"  ── Savings Year 1   (Cr)   : {cr(result.best_year1_savings)}")
    print(f"  ── Savings NPV      (Cr)   : {cr(result.best_savings_npv)}")

    sep("SOLVER STATS")
    print(f"  Trials completed           : {result.n_trials_completed}")
    print(f"  Feasible trials            : {result.n_trials_feasible}")

    sep("TOP 10 TRIALS")
    cols = [
        "trial_number", "solar_capacity_mw", "wind_capacity_mw",
        "ppa_capacity_mw", "bess_containers",
        "lcoe", "landed_tariff_y1", "savings_npv_cr"
    ]
    available = [c for c in cols if c in result.all_trials.columns]
    print(result.all_trials[available].head(10).to_string(index=False))
    sep()