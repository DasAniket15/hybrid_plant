"""
test_solver.py
──────────────
Smoke tests for the SolverEngine.

Runs a short optimisation (50 trials) to verify the solver completes,
returns the correct result structure, and finds a feasible solution that
beats the 100 % DISCOM baseline.

These tests are marked ``slow`` so they can be excluded from fast CI:
    pytest tests/ -m "not slow"
"""

from __future__ import annotations

import pytest

from hybrid_plant.solver.solver_engine import SolverEngine, SolverResult


@pytest.fixture(scope="module")
def solver_result(config, data, energy_engine, finance_engine):
    solver = SolverEngine(config, data, energy_engine, finance_engine)
    # 100 trials required: CUF-based augmentation changed the fast-mode objective
    # landscape so the TPE needs slightly more samples to find the feasible region.
    return solver.run(n_trials=100, show_progress=False)


@pytest.mark.slow
class TestSolverEngine:

    def test_returns_solver_result(self, solver_result):
        assert isinstance(solver_result, SolverResult)

    def test_best_savings_npv_positive(self, solver_result):
        assert solver_result.best_savings_npv > 0, \
            "Solver should find a solution with positive savings NPV"

    def test_best_lcoe_in_range(self, solver_result):
        assert 3.0 < solver_result.best_lcoe < 15.0

    def test_landed_tariff_below_discom(self, solver_result):
        fi     = solver_result.full_result["finance"]
        discom = fi["savings_breakdown"]["discom_tariff"]
        assert solver_result.best_landed_tariff_y1 < discom

    def test_best_params_keys(self, solver_result):
        required = {
            "solar_capacity_mw", "wind_capacity_mw", "ppa_capacity_mw",
            "bess_containers", "charge_c_rate", "discharge_c_rate",
            "dispatch_priority", "bess_charge_source",
        }
        assert required.issubset(solver_result.best_params.keys())

    def test_all_trials_dataframe_non_empty(self, solver_result):
        assert not solver_result.all_trials.empty

    def test_n_feasible_positive(self, solver_result):
        assert solver_result.n_trials_feasible > 0

    def test_solar_capacity_within_bounds(self, solver_result, config):
        dv  = config.solver["solver"]["decision_variables"]
        mw  = solver_result.best_params["solar_capacity_mw"]
        assert dv["solar_capacity_mw"]["min"] <= mw <= dv["solar_capacity_mw"]["max"]

    def test_bess_containers_within_bounds(self, solver_result, config):
        dv = config.solver["solver"]["decision_variables"]
        n  = solver_result.best_params["bess_containers"]
        assert dv["bess_containers"]["min"] <= n <= dv["bess_containers"]["max"]
