"""
validate_solver.py
──────────────────
Four-layer validation of the SolverEngine:

    1. Benchmark     — solver must beat the known hand-tuned case
    2. Convergence   — best-so-far NPV vs trial number (plot)
    3. Sensitivity   — 1-D sweeps around optimum for each variable (plots)
    4. Physical audit — deterministic sanity checks on the best solution
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine
from solver.solver_engine import SolverEngine


# ─────────────────────────────────────────────────────────────────────────────
# Known benchmark (from test_finance.py — verified solar-only case)
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = {
    "solar_capacity_mw": 195.415073395429,
    "wind_capacity_mw":  0.0,
    "bess_containers":   164,
    "charge_c_rate":     1.0,
    "discharge_c_rate":  1.0,
    "ppa_capacity_mw":   67.5256615562851,
    "dispatch_priority": "solar_first",
    "bess_charge_source":"solar_and_wind",
}

# Sensitivity sweep resolution (points per variable)
SWEEP_POINTS = 40

# Solver trials for main run
N_TRIALS = 1500


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cr(v):
    return round(v / 1e7, 4)

def sep(title=""):
    w = 62
    if title:
        pad = w - len(title) - 12
        print(f"\n{'─'*10} {title} {'─'*pad}")
    else:
        print("─" * w)

def run_engines(energy_engine, finance_engine, params):
    """Single full evaluation. Returns (year1, finance) or raises."""
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
# 1. BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def validate_benchmark(energy_engine, finance_engine, solver_result):
    sep("VALIDATION 1 — BENCHMARK")

    # Evaluate known case
    _, bench_finance = run_engines(energy_engine, finance_engine, BENCHMARK)
    bench_npv    = bench_finance["savings_npv"]
    bench_y1     = bench_finance["annual_savings_year1"]
    bench_lcoe   = bench_finance["lcoe_inr_per_kwh"]
    bench_landed = bench_finance["landed_tariff_series"][0]

    solver_npv    = solver_result.best_savings_npv
    solver_y1     = solver_result.best_year1_savings
    solver_lcoe   = solver_result.best_lcoe
    solver_landed = solver_result.best_landed_tariff_y1

    beat = solver_npv >= bench_npv

    print(f"\n  {'Metric':<30} {'Benchmark':>14} {'Solver Best':>14} {'Δ':>12}")
    sep()
    print(f"  {'Savings NPV (Rs Cr)':<30} {cr(bench_npv):>14} {cr(solver_npv):>14} {cr(solver_npv - bench_npv):>+12}")
    print(f"  {'Savings Year-1 (Rs Cr)':<30} {cr(bench_y1):>14} {cr(solver_y1):>14} {cr(solver_y1 - bench_y1):>+12}")
    print(f"  {'LCOE (Rs/kWh)':<30} {round(bench_lcoe,4):>14} {round(solver_lcoe,4):>14} {round(solver_lcoe-bench_lcoe,4):>+12}")
    print(f"  {'Landed Tariff Y1 (Rs/kWh)':<30} {round(bench_landed,4):>14} {round(solver_landed,4):>14} {round(solver_landed-bench_landed,4):>+12}")
    sep()
    status = "✓ PASS — solver beat the benchmark" if beat else "✗ FAIL — solver did NOT beat the benchmark"
    print(f"\n  {status}")

    return beat


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

def validate_convergence(solver, solver_result):
    sep("VALIDATION 2 — CONVERGENCE")

    log = solver._trial_log

    # Build best-so-far series from the trial log (all trials, ordered)
    all_sorted = sorted(log, key=lambda x: x["trial_number"])
    best_so_far = []
    current_best = -np.inf

    for t in all_sorted:
        npv = t.get("savings_npv_cr", None)
        if npv is not None and t.get("feasible", False):
            current_best = max(current_best, npv)
        best_so_far.append(current_best if current_best > -np.inf else np.nan)

    trial_nums = [t["trial_number"] for t in all_sorted]

    # Convergence metric: improvement in last 20% of trials
    tail_start = int(len(best_so_far) * 0.8)
    tail_improvement = best_so_far[-1] - best_so_far[tail_start]
    converged = tail_improvement < 0.5  # < 0.5 Cr improvement in last 20%

    print(f"\n  Best NPV at trial  10%  : {best_so_far[int(len(best_so_far)*0.1)]:.4f} Cr")
    print(f"  Best NPV at trial  50%  : {best_so_far[int(len(best_so_far)*0.5)]:.4f} Cr")
    print(f"  Best NPV at trial  80%  : {best_so_far[tail_start]:.4f} Cr")
    print(f"  Best NPV at trial 100%  : {best_so_far[-1]:.4f} Cr")
    print(f"  Improvement in last 20% : {tail_improvement:.4f} Cr")
    status = "✓ PASS — study converged" if converged else "⚠ WARN — study may need more trials"
    print(f"\n  {status}")

    return trial_nums, best_so_far, converged


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSITIVITY SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def _sweep(energy_engine, finance_engine, base_params, var_name, values):
    """Sweep one variable, return list of savings_npv (Cr), NaN on error."""
    results = []
    for v in values:
        params = {**base_params, var_name: v}
        try:
            _, fi = run_engines(energy_engine, finance_engine, params)
            results.append(fi["savings_npv"] / 1e7)
        except Exception:
            results.append(np.nan)
    return results


def validate_sensitivity(energy_engine, finance_engine, solver_result, config):
    sep("VALIDATION 3 — SENSITIVITY SWEEPS")

    p      = solver_result.best_params
    dv     = config.solver["solver"]["decision_variables"]
    y1_opt = solver_result.best_savings_npv / 1e7

    sweep_vars = {
        "solar_capacity_mw": np.linspace(dv["solar_capacity_mw"]["min"] + 1,
                                          dv["solar_capacity_mw"]["max"], SWEEP_POINTS),
        "wind_capacity_mw":  np.linspace(dv["wind_capacity_mw"]["min"],
                                          min(500, dv["wind_capacity_mw"]["max"]), SWEEP_POINTS),
        "ppa_capacity_mw":   np.linspace(dv["ppa_capacity_mw"]["min"] + 1,
                                          min(500, dv["ppa_capacity_mw"]["max"]), SWEEP_POINTS),
        "bess_containers":   np.arange(dv["bess_containers"]["min"],
                                        min(400, dv["bess_containers"]["max"]) + 1,
                                        max(1, int(400 / SWEEP_POINTS))),
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
        sweep_results[var] = (values, _sweep(energy_engine, finance_engine, p, var, values))

    # Check: peak is near optimum for at least solar and BESS
    checks = {}
    for var, (values, npvs) in sweep_results.items():
        clean = [(v, n) for v, n in zip(values, npvs) if not np.isnan(n)]
        if clean:
            best_v, best_n = max(clean, key=lambda x: x[1])
            checks[var] = (best_v, best_n)
            opt_v = p[var]
            print(f"  {labels[var]:<28}  sweep peak = {round(best_v,1)} ({round(best_n,4)} Cr)  |  solver opt = {round(opt_v,1)}")

    return sweep_results, labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. PHYSICAL AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def validate_physical(solver_result, data):
    sep("VALIDATION 4 — PHYSICAL AUDIT")

    p       = solver_result.best_params
    y1      = solver_result.full_result["year1"]
    fi      = solver_result.full_result["finance"]
    sv      = fi["savings_breakdown"]

    annual_load_mwh   = float(np.sum(data["load_profile"]))
    meter_mwh_y1      = float(fi["energy_projection"]["delivered_meter_mwh"][0])
    raw_gen_mwh       = float(np.sum(
        p["solar_capacity_mw"] * data["solar_cuf"]
        + p["wind_capacity_mw"] * data["wind_cuf"]
    ))
    curtailment_mwh   = float(np.sum(y1["curtailment_pre"]))
    curtailment_pct   = curtailment_mwh / raw_gen_mwh * 100 if raw_gen_mwh > 0 else 0

    discom_tariff     = sv["discom_tariff"]
    landed_y1         = fi["landed_tariff_series"][0]
    savings_y1        = fi["annual_savings_year1"]
    bess_end_soc      = float(y1["bess_end_soc_mwh"])
    bess_capacity     = float(y1["energy_capacity_mwh"])

    checks = [
        (
            "PPA ≤ Solar + Wind capacity",
            p["ppa_capacity_mw"] <= p["solar_capacity_mw"] + p["wind_capacity_mw"] + 1e-3,
            f"{round(p['ppa_capacity_mw'],2)} MW  ≤  {round(p['solar_capacity_mw'] + p['wind_capacity_mw'],2)} MW"
        ),
        (
            "Meter delivery ≤ Annual load",
            meter_mwh_y1 <= annual_load_mwh + 1e-3,
            f"{round(meter_mwh_y1,1)} MWh  ≤  {round(annual_load_mwh,1)} MWh"
        ),
        (
            "Savings Year-1 > 0",
            savings_y1 > 0,
            f"Rs {cr(savings_y1)} Cr"
        ),
        (
            "Landed tariff < DISCOM tariff",
            landed_y1 < discom_tariff,
            f"Rs {round(landed_y1,4)}  <  Rs {round(discom_tariff,4)} / kWh"
        ),
        (
            "Curtailment < 40% of raw generation",
            curtailment_pct < 40.0,
            f"{round(curtailment_pct,2)} %"
        ),
        (
            "BESS end SOC ≥ 0",
            bess_end_soc >= 0,
            f"{round(bess_end_soc,2)} MWh"
        ),
        (
            "BESS was used (discharge > 0)",
            float(np.sum(y1["discharge_pre"])) > 0,
            f"{round(float(np.sum(y1['discharge_pre'])),1)} MWh discharged"
        ),
    ]

    all_pass = True
    print()
    for name, passed, detail in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {name:<40}  {detail}")
        if not passed:
            all_pass = False

    sep()
    status = "✓ ALL PHYSICAL CHECKS PASSED" if all_pass else "✗ SOME PHYSICAL CHECKS FAILED"
    print(f"\n  {status}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(trial_nums, best_so_far, sweep_results, labels, solver_result):

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Solver Validation Report", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    opt_npv = solver_result.best_savings_npv / 1e7
    p       = solver_result.best_params

    # ── Plot 1: Convergence ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    valid = [(t, b) for t, b in zip(trial_nums, best_so_far) if not np.isnan(b)]
    if valid:
        tx, bx = zip(*valid)
        ax0.plot(tx, bx, color="#1f77b4", linewidth=1.5, label="Best-so-far NPV")
        ax0.axhline(opt_npv, color="red", linestyle="--", linewidth=1, label=f"Final best: {round(opt_npv,2)} Cr")
    ax0.set_title("Convergence: Best-so-far Savings NPV vs Trial Number")
    ax0.set_xlabel("Trial Number")
    ax0.set_ylabel("Savings NPV (Rs Crore)")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    # ── Plots 2–5: Sensitivity sweeps ────────────────────────────────────────
    var_order = ["solar_capacity_mw", "wind_capacity_mw", "ppa_capacity_mw", "bess_containers"]
    axes_pos  = [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]]

    for var, gsp in zip(var_order, axes_pos):
        ax = fig.add_subplot(gsp)
        values, npvs = sweep_results[var]
        clean_v = [v for v, n in zip(values, npvs) if not np.isnan(n)]
        clean_n = [n for n in npvs if not np.isnan(n)]

        ax.plot(clean_v, clean_n, color="#2ca02c", linewidth=1.5)
        ax.axvline(p[var], color="red", linestyle="--", linewidth=1,
                   label=f"Optimum = {round(p[var], 1)}")
        ax.axhline(opt_npv, color="orange", linestyle=":", linewidth=1,
                   label=f"Solver best = {round(opt_npv, 2)} Cr")

        ax.set_title(f"Sensitivity: {labels[var]}")
        ax.set_xlabel(labels[var])
        ax.set_ylabel("Savings NPV (Rs Crore)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.savefig("solver_validation.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved → solver_validation.png")
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

    solver = SolverEngine(config, data, energy_engine, finance_engine)

    # ── Run solver ────────────────────────────────────────────────────────────
    print(f"Running solver ({N_TRIALS} trials) …")
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

    # ── Run validations ───────────────────────────────────────────────────────
    b1 = validate_benchmark(energy_engine, finance_engine, result)
    trial_nums, best_so_far, converged = validate_convergence(solver, result)
    sweep_results, labels = validate_sensitivity(energy_engine, finance_engine, result, config)
    b4 = validate_physical(result, data)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep("VALIDATION SUMMARY")
    checks = [
        ("Benchmark",   b1),
        ("Convergence", converged),
        ("Physical",    b4),
    ]
    for name, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(trial_nums, best_so_far, sweep_results, labels, result)