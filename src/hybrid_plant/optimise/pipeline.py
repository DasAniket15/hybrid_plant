"""
optimise/pipeline.py
────────────────────
Cutover driver (Step 8): run the Pyomo optimizer and produce the SAME reporting
artifacts the Optuna SolverEngine produced, so run_model.py only needs a thin
engine switch.

LP-economics reporting
──────────────────────
The Pyomo LP is a *sizing + dispatch* oracle.  Its objective (1019 Cr for the
current config) is a ToD-valued proxy.  The reportable client savings uses the
established, Layer-2-validated FinanceEngine — evaluated on the LP's own
dispatch (``fast_mode=True`` scalar degradation, matching the single-mode LP's
own degradation model).  This is *higher* than the RTC heuristic oracle,
quantifying the value of optimal vs heuristic dispatch.

Three numbers surface, all reported for transparency:
  - ``lp_objective_npv``  : the LP's ToD-valued objective (optimization proxy)
  - finance ``savings_npv``: FinanceEngine on LP dispatch (the reported headline)
  - ``oracle_rtc_npv``    : FinanceEngine on the PlantEngine RTC heuristic
                            dispatch at the same sizing (the heuristic floor)

Definitional gaps between the LP objective and the FinanceEngine savings are
small (ToD-vs-flat valuation ~1%, aux netting ~0.5%) and are model-definition
differences, not errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.finance.finance_engine import FinanceEngine
from hybrid_plant.optimise.build import build_single_year_model
from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.params import OptParams, build_params
from hybrid_plant.optimise.report import compute_report
from hybrid_plant.optimise.sets import build_time_context
from hybrid_plant.optimise.solve import extract_dispatch, solve
from hybrid_plant.optimise.verify import VerifyReport, verify_solution

_HOURS = 8760
# Fixed future-scope dispatch params, matching SolverEngine._FUTURE_FIXED shape.
_DISPATCH_PRIORITY = "solar_first"


def _year1_from_lp(
    dispatch: dict[str, np.ndarray],
    sizing:   dict[str, float],
    params:   OptParams,
    re_pen_cost_inr: float,
) -> dict[str, Any]:
    """
    Build a Year1Engine-shaped result dict from the LP dispatch so the
    FinanceEngine (and dashboards) can consume it unchanged.

    All busbar quantities are pre-loss; meter quantities apply the loss factor.
    Single-mode degradation factors are 1.0, so cuf arrays give raw generation.
    """
    lf  = params.lf
    ed  = params.eta_d
    S   = sizing["S"]; W = sizing["W"]; nb = sizing["nb"]

    sd, wd = dispatch["sd"], dispatch["wd"]
    chg, dis, soc = dispatch["chg"], dispatch["dis"], dispatch["soc"]

    solar_direct_pre = sd
    wind_direct_pre  = wd
    discharge_pre    = ed * dis                      # busbar discharge (post-eff)
    plant_export_pre = solar_direct_pre + wind_direct_pre + discharge_pre

    gen_solar = S * params.cuf_s                     # deg = 1.0 (single mode)
    gen_wind  = W * params.cuf_w
    curtailment_pre = np.maximum(
        gen_solar + gen_wind - sd - wd - chg, 0.0
    )

    meter_total = plant_export_pre * lf
    e_cap = nb * params.cs

    return {
        "solar_direct_pre":    solar_direct_pre,
        "wind_direct_pre":     wind_direct_pre,
        "discharge_pre":       discharge_pre,
        "charge_pre":          chg,
        "curtailment_pre":     curtailment_pre,
        "plant_export_pre":    plant_export_pre,
        "solar_direct_meter":  solar_direct_pre * lf,
        "wind_direct_meter":   wind_direct_pre * lf,
        "discharge_meter":     discharge_pre * lf,
        "energy_capacity_mwh": e_cap,
        "charge_power_mw":     params.crc * e_cap,
        "discharge_power_mw":  params.crd * e_cap,
        "loss_factor":         lf,
        "bess_end_soc_mwh":    float(soc[-1]) if len(soc) else 0.0,
        "annual_meter_delivery": float(np.sum(meter_total)),
        "annual_discom":         float(np.sum(dispatch["ddraw"])),
        "annual_re_pen_cost_inr": float(re_pen_cost_inr),
        # EnergyProjection requires sim_params; fast_mode uses loss_factor here.
        "sim_params": {
            "solar_capacity_mw":  S,
            "wind_capacity_mw":   W,
            "bess_containers":    int(nb),
            "charge_c_rate":      params.crc,
            "discharge_c_rate":   params.crd,
            "ppa_capacity_mw":    sizing["P"],
            "dispatch_priority":  _DISPATCH_PRIORITY,
            "bess_charge_source": params.bess_charge_source,
            "loss_factor":        lf,
        },
    }


def _lp_re_pen_cost(dispatch: dict[str, np.ndarray], params: OptParams) -> float:
    """Annual hourly-RE-penetration penalty cost (INR) for the LP dispatch."""
    if not params.re_pen_penalty_enabled:
        return 0.0
    min_pct = params.re_pen_min_pct
    hrs = params.re_pen_penalty_hours
    hod = np.arange(_HOURS) % 24
    mask = np.ones(_HOURS, bool) if len(hrs) == 0 else np.isin(hod, np.asarray(hrs))
    shortfall = np.maximum(dispatch["ddraw"] - (1.0 - min_pct) * params.load, 0.0)
    shortfall = np.where(mask, shortfall, 0.0)
    return float(np.sum(shortfall * params.tod)) * 1000.0


def run_pyomo_optimization(
    config:         FullConfig,
    data:           dict[str, Any],
    opt_cfg:        OptModelConfig | None = None,
    compute_oracle: bool = True,
) -> dict[str, Any]:
    """
    Solve the single-year free MILP and assemble the reporting bundle.

    Returns
    -------
    dict with keys:
      engine, best_params, sizing, status, lp_objective_npv, verify (VerifyReport),
      year1 (LP-derived), finance (FinanceEngine on LP dispatch),
      lp_dispatch, and (if compute_oracle) oracle_rtc_npv.
    """
    opt_cfg = opt_cfg or OptModelConfig(horizon="single")
    params  = build_params(config, data)
    tc      = build_time_context(params, "single")

    model  = build_single_year_model(opt_cfg, params, objective="savings_npv")
    status = solve(model, opt_cfg)
    if "optimal" not in status["status"].lower():
        raise RuntimeError(f"Pyomo optimization did not reach optimal: {status['status']}")

    sizing = {
        "S":  float(pyo.value(model.S)),
        "W":  float(pyo.value(model.W)),
        "P":  float(pyo.value(model.P)),
        "nb": int(round(pyo.value(model.nb))),
    }
    dispatch = extract_dispatch(model, n_hours=tc.n_steps)
    verify:  VerifyReport = verify_solution(model, params, tc)

    best_params = {
        "solar_capacity_mw":  sizing["S"],
        "wind_capacity_mw":   sizing["W"],
        "ppa_capacity_mw":    sizing["P"],
        "bess_containers":    sizing["nb"],
        "charge_c_rate":      params.crc,
        "discharge_c_rate":   params.crd,
        "dispatch_priority":  _DISPATCH_PRIORITY,
        "bess_charge_source": params.bess_charge_source,
    }

    re_pen_cost = _lp_re_pen_cost(dispatch, params)
    year1 = _year1_from_lp(dispatch, sizing, params, re_pen_cost)
    finance = FinanceEngine(config, data).evaluate(
        year1_results     = year1,
        solar_capacity_mw = sizing["S"],
        wind_capacity_mw  = sizing["W"],
        ppa_capacity_mw   = sizing["P"],
        fast_mode         = True,   # scalar degradation — matches single-mode LP
    )

    result: dict[str, Any] = {
        "engine":            "pyomo",
        "best_params":       best_params,
        "sizing":            sizing,
        "status":            status,
        "lp_objective_npv":  status["obj_val"],
        "verify":            verify,
        "year1":             year1,
        "finance":           finance,
        "lp_dispatch":       dispatch,
    }
    if compute_oracle:
        oracle = compute_report(sizing, config, data, fast_mode=False)
        result["oracle_rtc_npv"] = oracle["savings_npv"]
    return result
