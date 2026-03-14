"""
validate_solver.py
──────────────────
Full 4-layer solver validation script.  Intended to be run directly, not
via pytest (it takes several minutes and produces plots).

    python tests/validate_solver.py

Layers
──────
  1. Benchmark   — solver NPV must beat the known hand-tuned case
  2. Convergence — best-so-far NPV plot vs trial number
  3. Sensitivity — 1-D sweeps around optimum for each decision variable
  4. Physical    — deterministic sanity checks on the best solution
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import load_config
from hybrid_plant.constants import CRORE_TO_RS
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine
from hybrid_plant.solver.solver_engine import SolverEngine, SolverResult


# ── Known benchmark (verified solar-only case) ────────────────────────────────

BENCHMARK = {
    "solar_capacity_mw":  202.0,
    "wind_capacity_mw":   0.0,
    "bess_containers":    162,
    "charge_c_rate":      1.0,
    "discharge_c_rate":   1.0,
    "ppa_capacity_mw":    60.5,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_and_wind",
}

SWEEP_POINTS = 40
N_TRIALS     = 1500


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cr(v: float) -> float:
    return round(v / CRORE_TO_RS, 4)

def sep(title: str = "") -> None:
    w = 62
    if title:
        print(f"\n{'─'*10} {title} {'─'*max(0, w - len(title) - 12)}")
    else:
        print("─" * w)

def run_engines(energy_engine, finance_engine, params):
    year1 = energy_engine.evaluate(
        solar_capacity_mw  = params["solar_capacity_mw"],
        wind_capacity_mw   = params["wind_capacity_mw"],
        bess_containers    = params["bess_containers"],
        charge_c_rate      = params["charge_c_rate"],
        discharge_c_rate   = params["discharge_c_rate"],
        ppa_capacity_mw    = params["ppa_capacity_mw"],
        dispatch_priority  = params["dispatch_priority"],
        bess_charge_source = params["bess_charge_source"],
    )
    finance = finance_engine.evaluate(
        year1_results     = year1,
        solar_capacity_mw = params["solar_capacity_mw"],
        wind_capacity_mw  = params["wind_capacity_mw"],
        ppa_capacity_mw   = params["ppa_capacity_mw"],
    )
    return year1, finance


# ─────────────────────────────────────────────────────────────────────────────
# 1. Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def validate_benchmark(energy_engine, finance_engine, result: SolverResult) -> bool:
    sep("VALIDATION 1 — BENCHMARK")
    _, bench_fi = run_engines(energy_engine, finance_engine, BENCHMARK)

    bench_npv    = bench_fi["savings_npv"]
    solver_npv   = result.best_savings_npv
    beat         = solver_npv >= bench_npv

    bench_lcoe   = bench_fi["lcoe_inr_per_kwh"]
    bench_landed = bench_fi["landed_tariff_series"][0]
    bench_y1     = bench_fi["annual_savings_year1"]

    print(f"\n  {'Metric':<30} {'Benchmark':>14} {'Solver Best':>14} {'Δ':>12}")
    sep()
    print(f"  {'Savings NPV (Rs Cr)':<30} {cr(bench_npv):>14} {cr(solver_npv):>14} {cr(solver_npv-bench_npv):>+12}")
    print(f"  {'Savings Y1 (Rs Cr)':<30} {cr(bench_y1):>14} {cr(result.best_year1_savings):>14} {cr(result.best_year1_savings-bench_y1):>+12}")
    print(f"  {'LCOE (Rs/kWh)':<30} {round(bench_lcoe,4):>14} {round(result.best_lcoe,4):>14} {round(result.best_lcoe-bench_lcoe,4):>+12}")
    print(f"  {'Landed Y1 (Rs/kWh)':<30} {round(bench_landed,4):>14} {round(result.best_landed_tariff_y1,4):>14} {round(result.best_landed_tariff_y1-bench_landed,4):>+12}")
    sep()
    print(f"\n  {'✓ PASS — solver beat the benchmark' if beat else '✗ FAIL — solver did NOT beat the benchmark'}")
    return beat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence
# ─────────────────────────────────────────────────────────────────────────────

def validate_convergence(solver: SolverEngine, result: SolverResult):
    sep("VALIDATION 2 — CONVERGENCE")
    log        = sorted(solver._trial_log, key=lambda x: x["trial_number"])
    best_npv   = -np.inf
    best_so_far: list[float] = []
    trial_nums: list[int]   = []

    for t in log:
        if t.get("feasible") and t.get("savings_npv_cr") is not None:
            best_npv = max(best_npv, t["savings_npv_cr"])
        best_so_far.append(best_npv if best_npv > -np.inf else float("nan"))
        trial_nums.append(t["trial_number"])

    tail     = int(len(best_so_far) * 0.8)
    tail_imp = best_so_far[-1] - best_so_far[tail]
    converged = tail_imp < 0.5

    print(f"\n  Best NPV at 10% trials  : {best_so_far[int(len(best_so_far)*0.1)]:.4f} Cr")
    print(f"  Best NPV at 50% trials  : {best_so_far[int(len(best_so_far)*0.5)]:.4f} Cr")
    print(f"  Best NPV at 80% trials  : {best_so_far[tail]:.4f} Cr")
    print(f"  Best NPV at 100% trials : {best_so_far[-1]:.4f} Cr")
    print(f"  Improvement in last 20% : {tail_imp:.4f} Cr")
    print(f"\n  {'✓ PASS — converged' if converged else '⚠ WARN — may need more trials'}")
    return trial_nums, best_so_far, converged


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def _sweep(energy_engine, finance_engine, base_params, var_name, values):
    results = []
    for v in values:
        params = {**base_params, var_name: v}
        try:
            _, fi = run_engines(energy_engine, finance_engine, params)
            results.append(fi["savings_npv"] / CRORE_TO_RS)
        except Exception:
            results.append(float("nan"))
    return results


def validate_sensitivity(energy_engine, finance_engine, result: SolverResult, config):
    sep("VALIDATION 3 — SENSITIVITY SWEEPS")
    p  = result.best_params
    dv = config.solver["solver"]["decision_variables"]

    sweep_vars = {
        "solar_capacity_mw": np.linspace(dv["solar_capacity_mw"]["min"] + 1, dv["solar_capacity_mw"]["max"], SWEEP_POINTS),
        "wind_capacity_mw":  np.linspace(dv["wind_capacity_mw"]["min"],      min(500, dv["wind_capacity_mw"]["max"]), SWEEP_POINTS),
        "ppa_capacity_mw":   np.linspace(dv["ppa_capacity_mw"]["min"] + 1,   min(500, dv["ppa_capacity_mw"]["max"]), SWEEP_POINTS),
        "bess_containers":   np.arange(dv["bess_containers"]["min"], min(400, dv["bess_containers"]["max"]) + 1, max(1, int(400 / SWEEP_POINTS))),
    }
    labels = {
        "solar_capacity_mw": "Solar Capacity (AC MW)",
        "wind_capacity_mw":  "Wind Capacity (MW)",
        "ppa_capacity_mw":   "PPA Capacity (MW)",
        "bess_containers":   "BESS Containers",
    }

    sweep_results = {}
    for var, values in sweep_vars.items():
        print(f"  Sweeping {var} ({len(values)} points) …")
        npvs = _sweep(energy_engine, finance_engine, p, var, values)
        sweep_results[var] = (values, npvs)
        clean = [(v, n) for v, n in zip(values, npvs) if not np.isnan(n)]
        if clean:
            best_v, best_n = max(clean, key=lambda x: x[1])
            print(f"  {labels[var]:<28}  peak = {round(best_v,1)} ({round(best_n,4)} Cr)  |  opt = {round(p[var],1)}")

    return sweep_results, labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. Physical audit
# ─────────────────────────────────────────────────────────────────────────────

def validate_physical(result: SolverResult, data) -> bool:
    sep("VALIDATION 4 — PHYSICAL AUDIT")
    p  = result.best_params
    y1 = result.full_result["year1"]
    fi = result.full_result["finance"]
    sv = fi["savings_breakdown"]

    annual_load_mwh = float(np.sum(data["load_profile"]))
    meter_mwh_y1    = float(fi["energy_projection"]["delivered_meter_mwh"][0])
    raw_gen_mwh     = float(np.sum(
        p["solar_capacity_mw"] * data["solar_cuf"]
        + p["wind_capacity_mw"]  * data["wind_cuf"]
    ))
    curtailment_pct = float(np.sum(y1["curtailment_pre"])) / raw_gen_mwh * 100 if raw_gen_mwh > 0 else 0

    checks = [
        ("PPA ≤ Solar + Wind capacity",     p["ppa_capacity_mw"] <= p["solar_capacity_mw"] + p["wind_capacity_mw"] + 1e-3, f"{round(p['ppa_capacity_mw'],2)} ≤ {round(p['solar_capacity_mw']+p['wind_capacity_mw'],2)} MW"),
        ("Meter delivery ≤ Annual load",    meter_mwh_y1 <= annual_load_mwh + 1e-3,                                       f"{round(meter_mwh_y1,1)} ≤ {round(annual_load_mwh,1)} MWh"),
        ("Savings Year-1 > 0",              fi["annual_savings_year1"] > 0,                                                f"Rs {cr(fi['annual_savings_year1'])} Cr"),
        ("Landed tariff < DISCOM tariff",   fi["landed_tariff_series"][0] < sv["discom_tariff"],                          f"Rs {round(fi['landed_tariff_series'][0],4)} < Rs {round(sv['discom_tariff'],4)} /kWh"),
        ("Curtailment < 40% raw gen",       curtailment_pct < 40.0,                                                        f"{round(curtailment_pct,2)} %"),
        ("BESS end SOC ≥ 0",                float(y1["bess_end_soc_mwh"]) >= 0,                                           f"{round(float(y1['bess_end_soc_mwh']),2)} MWh"),
        ("BESS was discharged",             float(np.sum(y1["discharge_pre"])) > 0,                                        f"{round(float(np.sum(y1['discharge_pre'])),1)} MWh"),
    ]

    all_pass = True
    print()
    for name, passed, detail in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {name:<40}  {detail}")
        all_pass = all_pass and passed

    sep()
    print(f"\n  {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(trial_nums, best_so_far, sweep_results, labels, result: SolverResult, output_path: Path):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Solver Validation Report", fontsize=14, fontweight="bold", y=0.98)

    gs      = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    opt_npv = result.best_savings_npv / CRORE_TO_RS
    p       = result.best_params

    # Convergence
    ax0 = fig.add_subplot(gs[0, :])
    valid = [(t, b) for t, b in zip(trial_nums, best_so_far) if not np.isnan(b)]
    if valid:
        tx, bx = zip(*valid)
        ax0.plot(tx, bx, color="#1f77b4", linewidth=1.5, label="Best-so-far NPV")
        ax0.axhline(opt_npv, color="red", linestyle="--", linewidth=1, label=f"Final best: {round(opt_npv,2)} Cr")
    ax0.set_title("Convergence: Best-so-far Savings NPV vs Trial Number")
    ax0.set_xlabel("Trial Number"); ax0.set_ylabel("Savings NPV (Rs Crore)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # Sensitivity sweeps
    var_order = ["solar_capacity_mw", "wind_capacity_mw", "ppa_capacity_mw", "bess_containers"]
    for var, gsp in zip(var_order, [gs[1,0], gs[1,1], gs[2,0], gs[2,1]]):
        ax   = fig.add_subplot(gsp)
        vals, npvs = sweep_results[var]
        cv   = [v for v, n in zip(vals, npvs) if not np.isnan(n)]
        cn   = [n for n in npvs if not np.isnan(n)]
        ax.plot(cv, cn, color="#2ca02c", linewidth=1.5)
        ax.axvline(p[var], color="red", linestyle="--", linewidth=1, label=f"Opt = {round(p[var],1)}")
        ax.axhline(opt_npv, color="orange", linestyle=":", linewidth=1, label=f"Best = {round(opt_npv,2)} Cr")
        ax.set_title(f"Sensitivity: {labels[var]}")
        ax.set_xlabel(labels[var]); ax.set_ylabel("Savings NPV (Rs Crore)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Validation plot saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config         = load_config()
    data           = load_timeseries_data(config)
    energy_engine  = Year1Engine(config, data)
    finance_engine = FinanceEngine(config, data)
    solver         = SolverEngine(config, data, energy_engine, finance_engine)

    print(f"\nRunning solver ({N_TRIALS} trials) …")
    result = solver.run(n_trials=N_TRIALS, show_progress=True)

    sep("OPTIMAL SOLUTION")
    p = result.best_params
    print(f"\n  Solar capacity (AC MW)  : {round(p['solar_capacity_mw'], 2)}")
    print(f"  Wind capacity (MW)      : {round(p['wind_capacity_mw'], 2)}")
    print(f"  BESS containers         : {p['bess_containers']}")
    print(f"  BESS energy (MWh)       : {round(float(result.full_result['year1']['energy_capacity_mwh']), 2)}")
    print(f"  PPA capacity (MW)       : {round(p['ppa_capacity_mw'], 2)}")
    print(f"  Savings NPV (Rs Cr)     : {cr(result.best_savings_npv)}")
    print(f"  LCOE (Rs/kWh)           : {round(result.best_lcoe, 4)}")

    b1 = validate_benchmark(energy_engine, finance_engine, result)
    trial_nums, best_so_far, converged = validate_convergence(solver, result)
    sweep_results, labels = validate_sensitivity(energy_engine, finance_engine, result, config)
    b4 = validate_physical(result, data)

    sep("VALIDATION SUMMARY")
    for name, passed in [("Benchmark", b1), ("Convergence", converged), ("Physical", b4)]:
        print(f"  {'✓' if passed else '✗'}  {name}")

    output_path = find_project_root() / "outputs" / "solver_validation.png"
    plot_all(trial_nums, best_so_far, sweep_results, labels, result, output_path)
