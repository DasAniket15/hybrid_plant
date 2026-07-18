"""
tests/optimise/test_verify.py
──────────────────────────────
Step 7 — post-solve invariant checks (verify.py) and the HiGHS↔CBC cross-check.

Fast: a valid solved mini model (single and full-degraded) passes every
invariant; synthetic corrupted dispatches trip exactly the intended check; the
report-only minimum_savings_npv gate fires when obj_val is below the floor.

The HiGHS↔CBC agreement test runs only when CBC is installed (it is the design's
cross-validation fallback); it skips otherwise.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyomo.environ as pyo
import pytest

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.constraints.allocation import add_allocation_constraints
from hybrid_plant.optimise.constraints.balance import add_balance_constraints
from hybrid_plant.optimise.constraints.ppa import add_ppa_constraint
from hybrid_plant.optimise.constraints.soc import add_soc_constraints
from hybrid_plant.optimise.objective import (
    add_savings_npv_objective,
    add_savings_npv_objective_full,
)
from hybrid_plant.optimise.params import OptParams, build_params
from hybrid_plant.optimise.sets import TimeContext, add_sets, build_time_context
from hybrid_plant.optimise.solve import extract_dispatch, solve
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars
from hybrid_plant.optimise.verify import check_invariants, verify_solution

N_HRS = 72
_FIXED = {"S": 150.0, "W": 80.0, "P": 100.0, "nb": 60}

_CBC_AVAILABLE = pyo.SolverFactory("cbc").available(exception_flag=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config(config) -> FullConfig:
    return config


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.fixture(scope="module")
def params(config: FullConfig, data: dict) -> OptParams:
    return build_params(config, data)


def _single_tc(n: int = N_HRS) -> TimeContext:
    return TimeContext(
        horizon="single", n_steps=n,
        hour_of=np.arange(n, dtype=int), year_of=np.ones(n, dtype=int),
        deg_s=np.ones(n), deg_w=np.ones(n), deg_b=np.ones(n),
        disc=np.full(n, np.nan),
    )


def _full_mini_tc(params: OptParams, n_hrs: int = 24, n_yrs: int = 2) -> TimeContext:
    H, Y = n_hrs, n_yrs
    d_s = np.array([1.0, 0.90]); d_w = np.array([1.0, 0.92]); d_b = np.array([1.0, 0.80])
    df  = np.array([1.0, 1.0 / (1.0 + params.wacc)])
    return TimeContext(
        horizon="full", n_steps=H * Y,
        hour_of=np.tile(np.arange(H), Y), year_of=np.repeat(np.arange(1, Y + 1), H),
        deg_s=np.repeat(d_s, H), deg_w=np.repeat(d_w, H), deg_b=np.repeat(d_b, H),
        disc=np.repeat(df, H),
    )


def _build_single(params: OptParams, tc: TimeContext, fixed: dict) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="verify_single")
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, params, fixed)
    add_dispatch_vars(m, params)
    add_allocation_constraints(m, params, tc)
    add_balance_constraints(m, params, tc)
    add_ppa_constraint(m, params, tc)
    add_soc_constraints(m, params, tc)
    add_savings_npv_objective(m, params)
    return m


def _no_gate(params: OptParams) -> OptParams:
    """
    Disable the economic minimum_savings_npv gate.

    The mini 72h/48h horizons pair a full-25yr cost side against a truncated
    revenue side, so their savings NPV is negative by construction — the gate
    would fire on an otherwise physically-valid solution.  Physical-invariant
    tests use this; the gate itself is exercised separately in TestMinSavingsGate.
    """
    return dataclasses.replace(
        params,
        opt_constraints=dataclasses.replace(
            params.opt_constraints, min_savings_npv_enabled=False
        ),
    )


def _build_full(params: OptParams, tc: TimeContext, fixed: dict) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="verify_full")
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, params, fixed)
    add_dispatch_vars(m, params)
    add_allocation_constraints(m, params, tc)
    add_balance_constraints(m, params, tc)
    add_ppa_constraint(m, params, tc)
    add_soc_constraints(m, params, tc)
    add_savings_npv_objective_full(m, params, tc)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 1. Valid solutions pass every invariant
# ─────────────────────────────────────────────────────────────────────────────

# Encoded-as-LP-rows invariants: these MUST hold on any optimal solution.
# no_simultaneous (D7) and min_savings_npv are NOT LP rows — see below.
_HARD = {
    "nonneg", "soc_bounds", "charge_cap", "discharge_cap", "soc_dynamics",
    "solar_alloc", "wind_alloc", "ppa_cap", "load_balance",
}


def _hard_ok(report) -> bool:
    return all(r.ok for r in report.results if r.name in _HARD)


class TestValidSolution:

    def test_single_mode_hard_invariants_ok(self, params: OptParams) -> None:
        tc = _single_tc()
        m = _build_single(params, tc, _FIXED)
        assert "optimal" in solve(m, OptModelConfig(horizon="single"))["status"].lower()
        report = verify_solution(m, _no_gate(params), tc)
        assert _hard_ok(report), str(report)
        # This single-mode optimum also happens to respect D7 exclusivity.
        assert next(r for r in report.results if r.name == "no_simultaneous").ok

    def test_full_degraded_hard_invariants_ok(self, params: OptParams) -> None:
        tc = _full_mini_tc(params)
        m = _build_full(params, tc, {"S": 100.0, "W": 60.0, "P": 80.0, "nb": 20})
        assert "optimal" in solve(m, OptModelConfig(horizon="full"))["status"].lower()
        report = verify_solution(m, _no_gate(params), tc)
        # Every LP-encoded invariant holds; D7 exclusivity is NOT an LP row and
        # a degenerate optimum may violate it — that is precisely what verify
        # surfaces (motivating the optional strict-D7 binary if fidelity needs).
        assert _hard_ok(report), str(report)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Corrupted dispatches trip the intended invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptionDetected:

    _SIZING = {"S": 100.0, "W": 50.0, "P": 80.0, "nb": 20}

    def _zeros(self) -> dict[str, np.ndarray]:
        z = lambda: np.zeros(N_HRS)  # noqa: E731
        return {k: z() for k in ("sd", "wd", "chg", "dis", "soc", "ddraw")}

    def _names(self, report) -> set[str]:
        return {r.name for r in report.failures()}

    def test_negative_var(self, params: OptParams) -> None:
        tc = _single_tc(); d = self._zeros(); d["ddraw"][3] = -5.0
        rep = check_invariants(d, self._SIZING, params, tc)
        assert not rep.ok and "nonneg" in self._names(rep)

    def test_soc_over_cap(self, params: OptParams) -> None:
        tc = _single_tc(); d = self._zeros()
        d["soc"][5] = self._SIZING["nb"] * params.cs + 10.0
        rep = check_invariants(d, self._SIZING, params, tc)
        assert not rep.ok and "soc_bounds" in self._names(rep)

    def test_ppa_exceeded(self, params: OptParams) -> None:
        tc = _single_tc(); d = self._zeros(); d["sd"][2] = 1000.0
        rep = check_invariants(d, self._SIZING, params, tc)
        assert not rep.ok and "ppa_cap" in self._names(rep)

    def test_simultaneous_charge_discharge(self, params: OptParams) -> None:
        tc = _single_tc(); d = self._zeros(); d["chg"][4] = 5.0; d["dis"][4] = 5.0
        rep = check_invariants(d, self._SIZING, params, tc)
        assert not rep.ok and "no_simultaneous" in self._names(rep)

    def test_assert_ok_raises(self, params: OptParams) -> None:
        tc = _single_tc(); d = self._zeros(); d["sd"][2] = 1000.0
        with pytest.raises(AssertionError):
            check_invariants(d, self._SIZING, params, tc).assert_ok()


# ─────────────────────────────────────────────────────────────────────────────
# 3. minimum_savings_npv report-only gate
# ─────────────────────────────────────────────────────────────────────────────

class TestMinSavingsGate:

    def test_gate_fires_below_floor(self, params: OptParams) -> None:
        tc = _single_tc()
        m = _build_single(params, tc, _FIXED)
        assert "optimal" in solve(m, OptModelConfig(horizon="single"))["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        sizing = {k: float(pyo.value(getattr(m, k))) for k in ("S", "W", "P", "nb")}
        obj_val = float(pyo.value(m.obj))   # single mode: unscaled INR

        # Floor above the achieved savings → gate must fail, others still pass.
        p2 = dataclasses.replace(
            params,
            opt_constraints=dataclasses.replace(
                params.opt_constraints,
                min_savings_npv_enabled=True,
                min_savings_npv_value=obj_val + 1e9,
            ),
        )
        rep = check_invariants(d, sizing, p2, tc, obj_val=obj_val)
        names = {r.name for r in rep.failures()}
        assert not rep.ok
        assert names == {"min_savings_npv"}

    def test_gate_passes_above_floor(self, params: OptParams) -> None:
        tc = _single_tc()
        m = _build_single(params, tc, _FIXED)
        solve(m, OptModelConfig(horizon="single"))
        d = extract_dispatch(m, n_hours=tc.n_steps)
        sizing = {k: float(pyo.value(getattr(m, k))) for k in ("S", "W", "P", "nb")}
        obj_val = float(pyo.value(m.obj))
        # Floor below the achieved savings → gate passes, all invariants ok.
        p2 = dataclasses.replace(
            params,
            opt_constraints=dataclasses.replace(
                params.opt_constraints,
                min_savings_npv_enabled=True,
                min_savings_npv_value=obj_val - 1e9,
            ),
        )
        rep = check_invariants(d, sizing, p2, tc, obj_val=obj_val)
        assert rep.ok, str(rep)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HiGHS ↔ CBC cross-check (runs only if CBC is installed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CBC_AVAILABLE, reason="CBC solver not installed")
def test_highs_cbc_objective_agreement(params: OptParams) -> None:
    tc = _single_tc()
    m_h = _build_single(params, tc, _FIXED)
    r_h = solve(m_h, OptModelConfig(horizon="single", solver_name="appsi_highs"))
    m_c = _build_single(params, tc, _FIXED)
    r_c = solve(m_c, OptModelConfig(horizon="single", solver_name="cbc"))
    assert "optimal" in r_h["status"].lower()
    assert "optimal" in r_c["status"].lower()
    rel = abs(r_h["obj_val"] - r_c["obj_val"]) / max(abs(r_h["obj_val"]), 1.0)
    assert rel < 1e-6, f"HiGHS {r_h['obj_val']:.4e} vs CBC {r_c['obj_val']:.4e}"
