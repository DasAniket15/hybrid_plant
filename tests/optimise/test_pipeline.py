"""
tests/optimise/test_pipeline.py
────────────────────────────────
Step 8 cutover driver (optimise/pipeline.py): Pyomo optimizer -> reporting
bundle with LP-economics (FinanceEngine on the LP dispatch).

Fast: unit-test the LP->year1 adapter and the penalty-cost helper on synthetic
dispatch.  Slow: full end-to-end optimization pins the known optimum, the LP
objective, the FinanceEngine savings, the RTC oracle floor, and verify.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.optimise.params import OptParams, build_params
from hybrid_plant.optimise.pipeline import (
    _lp_re_pen_cost,
    _year1_from_lp,
    run_pyomo_optimization,
)


@pytest.fixture(scope="module")
def config(config) -> FullConfig:
    return config


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.fixture(scope="module")
def params(config: FullConfig, data: dict) -> OptParams:
    return build_params(config, data)


def _synthetic_dispatch(n: int = 8760) -> dict:
    rng = np.arange(n)
    return {
        "sd":    np.full(n, 3.0),
        "wd":    np.full(n, 2.0),
        "chg":   np.full(n, 1.0),
        "dis":   np.full(n, 0.5),
        "soc":   (rng % 10).astype(float),
        "ddraw": np.full(n, 4.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fast unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestYear1Adapter:

    def test_keys_and_values(self, params: OptParams) -> None:
        d = _synthetic_dispatch()
        sizing = {"S": 100.0, "W": 60.0, "P": 80.0, "nb": 20}
        y1 = _year1_from_lp(d, sizing, params, re_pen_cost_inr=0.0)

        # FinanceEngine-required keys present.
        for k in ("solar_direct_pre", "wind_direct_pre", "discharge_pre",
                  "energy_capacity_mwh", "sim_params"):
            assert k in y1
        assert "loss_factor" in y1["sim_params"]

        ed, lf = params.eta_d, params.lf
        assert np.allclose(y1["discharge_pre"], ed * d["dis"])
        assert np.allclose(y1["discharge_meter"], ed * d["dis"] * lf)
        assert y1["energy_capacity_mwh"] == 20 * params.cs
        # meter delivery = busbar * lf, summed
        busbar = d["sd"] + d["wd"] + ed * d["dis"]
        assert abs(y1["annual_meter_delivery"] - float(np.sum(busbar * lf))) < 1e-3

    def test_penalty_cost_disabled_is_zero(self, params: OptParams) -> None:
        assert _lp_re_pen_cost(_synthetic_dispatch(), params) == 0.0

    def test_penalty_cost_enabled_matches_manual(self, params: OptParams) -> None:
        p = dataclasses.replace(
            params, re_pen_penalty_enabled=True, re_pen_min_pct=0.90,
            re_pen_penalty_hours=(),
        )
        d = _synthetic_dispatch()
        d["ddraw"] = np.full(8760, 100.0)   # large draw -> guaranteed shortfall
        got = _lp_re_pen_cost(d, p)
        shortfall = np.maximum(d["ddraw"] - (1.0 - 0.90) * p.load, 0.0)
        want = float(np.sum(shortfall * p.tod)) * 1000.0
        assert abs(got - want) < 1e-3
        assert got > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Slow end-to-end
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPyomoPipelineEndToEnd:

    @pytest.fixture(scope="class")
    def result(self, config: FullConfig, data: dict) -> dict:
        return run_pyomo_optimization(config, data, compute_oracle=True)

    def test_optimal_and_known_sizing(self, result: dict) -> None:
        assert "optimal" in result["status"]["status"].lower()
        s = result["sizing"]
        assert abs(s["S"] - 71.76) < 2.0
        assert abs(s["W"] - 95.33) < 2.0
        assert abs(s["P"] - 56.39) < 2.0
        assert s["nb"] == 19

    def test_best_params_shape(self, result: dict) -> None:
        bp = result["best_params"]
        for k in ("solar_capacity_mw", "wind_capacity_mw", "ppa_capacity_mw",
                  "bess_containers", "charge_c_rate", "discharge_c_rate",
                  "dispatch_priority", "bess_charge_source"):
            assert k in bp

    def test_lp_objective_and_finance_savings(self, result: dict) -> None:
        lp = result["lp_objective_npv"] / 1e7
        fin = result["finance"]["savings_npv"] / 1e7
        assert 1000 < lp < 1040          # ToD proxy ~1019 Cr
        assert 1010 < fin < 1050         # FinanceEngine on LP dispatch ~1031 Cr

    def test_oracle_is_heuristic_floor(self, result: dict) -> None:
        # LP-optimal dispatch savings >= RTC heuristic dispatch savings.
        fin = result["finance"]["savings_npv"]
        oracle = result["oracle_rtc_npv"]
        assert oracle > 0
        assert fin >= oracle - 1e6        # LP dispatch beats/ties the heuristic

    def test_tod_annual_series_npvs_to_lp_objective(self, result: dict) -> None:
        # The ToD-aware annual savings series must reproduce the LP objective
        # exactly when discounted (it is the objective's per-year decomposition).
        tod = result["tod_savings_npv"]
        lp  = result["lp_objective_npv"]
        assert abs(tod - lp) / abs(lp) < 1e-9
        assert len(result["tod_annual_savings"]) == 25
        # Degradation makes the series non-increasing after year 1.
        s = result["tod_annual_savings"]
        assert s[0] >= s[-1]

    def test_verify_hard_invariants(self, result: dict) -> None:
        hard = {"nonneg", "soc_bounds", "charge_cap", "discharge_cap",
                "soc_dynamics", "solar_alloc", "wind_alloc", "ppa_cap", "load_balance"}
        rep = result["verify"]
        assert all(r.ok for r in rep.results if r.name in hard), str(rep)

    def test_developer_payback(self, result: dict) -> None:
        dp = result["developer_payback"]
        assert dp["unlevered"] is None or 1 <= dp["unlevered"] <= 25
        assert dp["levered"] is None or 1 <= dp["levered"] <= 25
        # Positive leverage (project return > cost of debt): equity returns
        # at least as fast as the whole asset.
        if dp["unlevered"] and dp["levered"]:
            assert dp["levered"] <= dp["unlevered"]
