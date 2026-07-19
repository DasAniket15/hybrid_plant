"""
tests/optimise/test_charge_source.py
──────────────────────────────────────
D5 charge-source restriction (T8): bess_charge_source in
{solar_only, wind_only, solar_and_wind}.

solar_only (default) is byte-identical to the pre-D5 model (no chg_w var).
The split modes add a wind-charge portion chg_w; a zero-solar site makes the
gate decisive — with no solar to charge, solar_only cannot sustain any
discharge, while wind sources can.
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
from hybrid_plant.optimise.verify import check_invariants

N_HRS = 24
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


def _build(params: OptParams, source: str, *, zero_solar: bool = False,
           force_dis: float | None = None) -> tuple[pyo.ConcreteModel, OptParams, TimeContext]:
    p = dataclasses.replace(params, bess_charge_source=source)
    if zero_solar:
        p = dataclasses.replace(p, cuf_s=np.zeros_like(params.cuf_s))
    tc = _tc()
    m = pyo.ConcreteModel(name=f"csrc_{source}")
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, p, _FIXED)
    add_dispatch_vars(m, p)
    add_allocation_constraints(m, p, tc)
    add_balance_constraints(m, p, tc)
    add_ppa_constraint(m, p, tc)
    add_soc_constraints(m, p, tc)
    add_savings_npv_objective(m, p)
    if force_dis is not None:
        m.force_dis = pyo.Constraint(expr=pyo.quicksum(m.dis[t] for t in m.H) >= force_dis)
    return m, p, tc


def _sizing() -> dict:
    return {k: float(v) for k, v in _FIXED.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structural
# ─────────────────────────────────────────────────────────────────────────────

class TestStructural:

    def test_solar_only_has_no_chg_w(self, params: OptParams) -> None:
        m, _, _ = _build(params, "solar_only")
        assert not hasattr(m, "chg_w")
        assert not hasattr(m, "c_charge_source")

    @pytest.mark.parametrize("source", ["wind_only", "solar_and_wind"])
    def test_split_modes_have_chg_w(self, params: OptParams, source: str) -> None:
        m, _, _ = _build(params, source)
        assert hasattr(m, "chg_w")
        assert hasattr(m, "c_charge_source")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Valid solves + verify across all sources
# ─────────────────────────────────────────────────────────────────────────────

class TestValidAcrossSources:

    @pytest.mark.parametrize("source", ["solar_only", "wind_only", "solar_and_wind"])
    def test_optimal_and_invariants_hold(self, params: OptParams, source: str) -> None:
        m, p, tc = _build(params, source)
        assert "optimal" in solve(m, OptModelConfig(horizon="single"))["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        rep = check_invariants(
            d, _sizing(),
            dataclasses.replace(p, opt_constraints=dataclasses.replace(
                p.opt_constraints, min_savings_npv_enabled=False)),
            tc,
        )
        # Every LP-encoded invariant (incl. the source-aware alloc) holds.
        hard = {"nonneg", "soc_bounds", "charge_cap", "discharge_cap",
                "soc_dynamics", "solar_alloc", "wind_alloc", "ppa_cap", "load_balance"}
        assert all(r.ok for r in rep.results if r.name in hard), str(rep)

    def test_wind_only_charges_only_from_wind(self, params: OptParams) -> None:
        # Force some charge/discharge so chg > 0 somewhere, then check chg_w==chg.
        m, p, tc = _build(params, "wind_only", force_dis=5.0)
        assert "optimal" in solve(m, OptModelConfig(horizon="single"))["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        assert float(np.sum(d["chg"])) > 1e-6                       # some charging happened
        assert np.allclose(d["chg_w"], d["chg"], atol=1e-5)         # all of it from wind
        # Solar charge portion is zero.
        assert float(np.max(d["chg"] - d["chg_w"])) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decisive differential — zero-solar site
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroSolarGate:

    def test_solar_only_cannot_sustain_discharge(self, params: OptParams) -> None:
        # No solar anywhere → solar_only cannot charge → cannot discharge.
        m, _, _ = _build(params, "solar_only", zero_solar=True, force_dis=5.0)
        assert "optimal" not in solve(m, OptModelConfig(horizon="single"))["status"].lower()

    @pytest.mark.parametrize("source", ["wind_only", "solar_and_wind"])
    def test_wind_sources_can_sustain_discharge(self, params: OptParams, source: str) -> None:
        m, _, tc = _build(params, source, zero_solar=True, force_dis=5.0)
        assert "optimal" in solve(m, OptModelConfig(horizon="single"))["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        assert float(np.sum(d["dis"])) >= 5.0 - 1e-4                # discharge met via wind charge
