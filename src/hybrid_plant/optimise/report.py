"""
optimise/report.py
──────────────────
Post-solve reporting: LCOE, landed tariff, and the full dashboard.

Design principle (D2)
─────────────────────
LCOE and landed tariff are **outputs** of the optimisation, computed from the
optimal sizing via the existing ``LCOEModel`` / ``LandedTariffModel`` / ``FinanceEngine``.
They never enter the Pyomo model; this module is always called after solve.

Usage
─────
    sizing = {"S": ..., "W": ..., "P": ..., "nb": ...}
    report = compute_report(sizing, config, data, fast_mode=False)
    print(report["lcoe_inr_per_kwh"], report["savings_npv"])
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import PERCENT_TO_DECIMAL
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine


def compute_report(
    sizing:    dict[str, Any],
    config:    FullConfig,
    data:      dict[str, Any],
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Compute the full reporting dashboard from an optimal sizing.

    Parameters
    ----------
    sizing    : dict with keys ``"S"`` (MW), ``"W"`` (MW), ``"P"`` (MW),
                ``"nb"`` (int containers)
    config    : FullConfig
    data      : timeseries data dict from ``data_loader.load_timeseries_data``
    fast_mode : passed to FinanceEngine.evaluate().  False = full 25-year
                re-simulation (accurate dashboards); True = scalar scaling
                (fast, used in step-3 parity checks).

    Returns
    -------
    dict
        All keys from ``FinanceEngine.evaluate()`` — including
        ``"lcoe_inr_per_kwh"``, ``"landed_tariff_series"``,
        ``"savings_npv"``, ``"capex"``, ``"opex_projection"``, etc.
    """
    dv = config.solver["solver"]["decision_variables"]

    # C-rates: read fixed_value if present, else max (matching params.py convention)
    def _c_rate(key: str) -> float:
        cfg = dv.get(key, {})
        fv  = cfg.get("fixed_value")
        return float(fv) if fv is not None else float(cfg.get("max", 1.0))

    year1 = Year1Engine(config, data).evaluate(
        solar_capacity_mw  = float(sizing["S"]),
        wind_capacity_mw   = float(sizing["W"]),
        bess_containers    = int(sizing["nb"]),
        charge_c_rate      = _c_rate("bess_charge_c_rate"),
        discharge_c_rate   = _c_rate("bess_discharge_c_rate"),
        ppa_capacity_mw    = float(sizing["P"]),
        dispatch_priority  = "solar_first",
        bess_charge_source = "solar_only",
    )

    return FinanceEngine(config, data).evaluate(
        year1_results     = year1,
        solar_capacity_mw = float(sizing["S"]),
        wind_capacity_mw  = float(sizing["W"]),
        ppa_capacity_mw   = float(sizing["P"]),
        fast_mode         = fast_mode,
    )
