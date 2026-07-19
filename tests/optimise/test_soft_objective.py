"""
tests/optimise/test_soft_objective.py
───────────────────────────────────────
Soft objective terms (design §6.6 / year1_engine parity).

Currently: the hourly RE-penetration penalty — a soft cost added to the hybrid
economics (reduces savings NPV) whenever meter RE delivery falls below a floor
in the penalty hours.  Unlike the hard re_penetration gate it rejects no
solution; it prices the shortfall so the optimizer sizes/dispatches to avoid it.

All on the fast mini 72h single model.
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
from hybrid_plant.optimise.objective import add_savings_npv_objective
from hybrid_plant.optimise.params import OptParams, build_params
from hybrid_plant.optimise.sets import TimeContext, add_sets
from hybrid_plant.optimise.solve import extract_dispatch, solve
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars

N_HRS = 72
_FIXED = {"S": 150.0, "W": 80.0, "P": 100.0, "nb": 60}


@pytest.fixture(scope="module")
def config(config) -> FullConfig:
    return config


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.fixture(scope="module")
def params(config: FullConfig, data: dict) -> OptParams:
    return build_params(config, data)


def _tc(n: int = N_HRS) -> TimeContext:
    return TimeContext(
        horizon="single", n_steps=n,
        hour_of=np.arange(n, dtype=int), year_of=np.ones(n, dtype=int),
        deg_s=np.ones(n), deg_w=np.ones(n), deg_b=np.ones(n),
        disc=np.full(n, np.nan),
    )


def _build_solve(params: OptParams) -> dict:
    tc = _tc()
    m = pyo.ConcreteModel(name="soft")
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, params, _FIXED)
    add_dispatch_vars(m, params)
    add_allocation_constraints(m, params, tc)
    add_balance_constraints(m, params, tc)
    add_ppa_constraint(m, params, tc)
    add_soc_constraints(m, params, tc)
    add_savings_npv_objective(m, params)
    st = solve(m, OptModelConfig(horizon="single"))
    return {"model": m, "status": st, "tc": tc}


def _with_penalty(params: OptParams, min_pct_decimal: float, hours: tuple = ()) -> OptParams:
    return dataclasses.replace(
        params,
        re_pen_penalty_enabled=True,
        re_pen_min_pct=min_pct_decimal,
        re_pen_penalty_hours=hours,
    )


class TestRePenetrationPenalty:

    def test_disabled_adds_no_var(self, params: OptParams) -> None:
        r = _build_solve(params)
        assert not hasattr(r["model"], "re_pen_short")

    def test_enabled_adds_var_and_stays_optimal(self, params: OptParams) -> None:
        r = _build_solve(_with_penalty(params, 0.90))
        assert "optimal" in r["status"]["status"].lower()
        assert hasattr(r["model"], "re_pen_short")
        assert hasattr(r["model"], "re_pen_short_con")

    def test_penalty_does_not_increase_savings(self, params: OptParams) -> None:
        base = _build_solve(params)["status"]["obj_val"]
        pen  = _build_solve(_with_penalty(params, 0.90))["status"]["obj_val"]
        # A soft cost can only lower (or leave) savings NPV.
        assert pen <= base + 1.0

    def test_shortfall_equals_breach(self, params: OptParams) -> None:
        p = _with_penalty(params, 0.90)          # floor 90% of load every hour
        r = _build_solve(p)
        m, tc = r["model"], r["tc"]
        d = extract_dispatch(m, n_hours=tc.n_steps)
        min_pct = p.re_pen_min_pct
        for t in range(tc.n_steps):
            breach = max(d["ddraw"][t] - (1.0 - min_pct) * float(p.load[t]), 0.0)
            sh = float(pyo.value(m.re_pen_short[t]))
            assert abs(sh - breach) < 1e-3, f"t={t}: sh={sh:.4f} breach={breach:.4f}"

    def test_penalty_hours_subset_only(self, params: OptParams) -> None:
        # Restrict penalty to evening hours (0-indexed 18..21); other hours have
        # no shortfall var.
        hours = (18, 19, 20, 21)
        r = _build_solve(_with_penalty(params, 0.90, hours))
        m, tc = r["model"], r["tc"]
        hod = np.arange(tc.n_steps) % 24
        in_window = {int(t) for t in np.nonzero(np.isin(hod, np.asarray(hours)))[0]}
        for t in range(tc.n_steps):
            if t in in_window:
                assert t in m.re_pen_short, f"t={t} in window missing shortfall var"
            else:
                assert t not in m.re_pen_short, f"t={t} outside window has a shortfall var"
