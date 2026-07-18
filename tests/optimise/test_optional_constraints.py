"""
tests/optimise/test_optional_constraints.py
────────────────────────────────────────────
Step 6 validation — toggleable PPA-contract constraints (design §3.6,
``constraints/optional.py``).

Every constraint is proven three ways, all on a fast mini single-horizon model
(72 h, real params) so the suite stays in the sub-second "not slow" band:

  1. Structural — the expected Pyomo component appears iff the toggle is on,
     and the all-off default adds nothing (base model unchanged).
  2. Binding    — a threshold tighter than the unconstrained optimum moves the
     solution to respect it (feasible direction).
  3. Infeasible — a physically impossible threshold flips the model from
     optimal to infeasible (proves the row is a real hard bound).

Fixed sizing is used for the dispatch-side constraints (plant_cuf,
re_penetration, minimum_bess_discharge) so the mini horizon's truncated
economics do not distort the test; minimum_bess_capacity uses free nb with the
rest fixed so the floor visibly forces the integer up.
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
from hybrid_plant.optimise.constraints.optional import add_optional_constraints
from hybrid_plant.optimise.constraints.ppa import add_ppa_constraint
from hybrid_plant.optimise.constraints.soc import add_soc_constraints
from hybrid_plant.optimise.objective import add_savings_npv_objective
from hybrid_plant.optimise.params import (
    OptionalConstraintsConfig,
    OptParams,
    build_params,
)
from hybrid_plant.optimise.sets import TimeContext, add_sets
from hybrid_plant.optimise.solve import extract_dispatch, solve
from hybrid_plant.optimise.variables import add_dispatch_vars, add_sizing_vars

N_HRS = 72
_FIXED = {"S": 150.0, "W": 80.0, "P": 100.0, "nb": 60}   # E_b = 60 * 5.015 = 300.9 MWh


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
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


def _mini_tc(n_hrs: int = N_HRS) -> TimeContext:
    H = n_hrs
    return TimeContext(
        horizon="single",
        n_steps=H,
        hour_of=np.arange(H, dtype=int),
        year_of=np.ones(H, dtype=int),
        deg_s=np.ones(H),
        deg_w=np.ones(H),
        deg_b=np.ones(H),
        disc=np.full(H, np.nan),
    )


def _build(
    params: OptParams,
    cfg: OptionalConstraintsConfig,
    fixed_sizing: dict | None,
) -> tuple[pyo.ConcreteModel, TimeContext]:
    """Assemble a mini single-horizon model with *cfg* optional constraints."""
    tc = _mini_tc()
    p  = dataclasses.replace(params, opt_constraints=cfg)
    m  = pyo.ConcreteModel(name="mini_optional")
    add_sets(m, tc.n_steps)
    add_sizing_vars(m, p, fixed_sizing)
    add_dispatch_vars(m, p)
    add_allocation_constraints(m, p, tc)
    add_balance_constraints(m, p, tc)
    add_ppa_constraint(m, p, tc)
    add_soc_constraints(m, p, tc)
    add_savings_npv_objective(m, p)
    add_optional_constraints(m, p, tc)
    return m, tc


def _solve(m: pyo.ConcreteModel) -> dict:
    return solve(m, OptModelConfig(horizon="single"))


def _busbar(d: dict, eta_d: float) -> float:
    return float(np.sum(d["sd"] + d["wd"] + eta_d * d["dis"]))


def _discharge(d: dict, eta_d: float) -> float:
    return float(np.sum(eta_d * d["dis"]))


def _meter(d: dict, params: OptParams, tc: TimeContext) -> float:
    total_load = float(np.sum(params.load[tc.hour_of]))
    return total_load - float(np.sum(d["ddraw"]))


@pytest.fixture(scope="module")
def baseline(params: OptParams) -> dict:
    """Unconstrained (all toggles off) fixed-sizing solve — reference values."""
    m, tc = _build(params, OptionalConstraintsConfig(), _FIXED)
    st = _solve(m)
    assert "optimal" in st["status"].lower()
    d = extract_dispatch(m, n_hours=tc.n_steps)
    total_load = float(np.sum(params.load[tc.hour_of]))
    return {
        "busbar":     _busbar(d, params.eta_d),
        "discharge":  _discharge(d, params.eta_d),
        "meter":      _meter(d, params, tc),
        "total_load": total_load,
        "n_steps":    tc.n_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structural — toggle on/off ⇒ component present/absent
# ─────────────────────────────────────────────────────────────────────────────

_OPT_COMPONENTS = [
    "opt_plant_cuf_min", "opt_plant_cuf_max",
    "opt_min_bess_capacity", "opt_min_bess_discharge",
    "opt_re_penetration_min", "opt_re_penetration_max",
]


class TestStructural:

    def test_all_off_adds_nothing(self, params: OptParams) -> None:
        m, _ = _build(params, OptionalConstraintsConfig(), _FIXED)
        for name in _OPT_COMPONENTS:
            assert not hasattr(m, name), f"{name} present with all toggles off"

    def test_plant_cuf_band_components(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(
            plant_cuf_enabled=True, plant_cuf_min_pct=5.0, plant_cuf_max_pct=40.0
        )
        m, _ = _build(params, cfg, _FIXED)
        assert hasattr(m, "opt_plant_cuf_min")
        assert hasattr(m, "opt_plant_cuf_max")

    def test_min_bess_capacity_component(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(
            min_bess_capacity_enabled=True, min_bess_capacity_mwh=250.0
        )
        m, _ = _build(params, cfg, _FIXED)
        assert hasattr(m, "opt_min_bess_capacity")

    def test_min_bess_discharge_component(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(
            min_bess_discharge_enabled=True, min_bess_discharge_annual_mwh=10.0
        )
        m, _ = _build(params, cfg, _FIXED)
        assert hasattr(m, "opt_min_bess_discharge")

    def test_re_penetration_band_components(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(
            re_penetration_enabled=True,
            re_penetration_min_pct=10.0, re_penetration_max_pct=60.0,
        )
        m, _ = _build(params, cfg, _FIXED)
        assert hasattr(m, "opt_re_penetration_min")
        assert hasattr(m, "opt_re_penetration_max")

    def test_min_savings_npv_is_report_only(self, params: OptParams) -> None:
        """minimum_savings_npv adds no LP row (report-only viability gate)."""
        cfg = OptionalConstraintsConfig(
            min_savings_npv_enabled=True, min_savings_npv_value=1e12
        )
        m, _ = _build(params, cfg, _FIXED)
        for name in _OPT_COMPONENTS:
            assert not hasattr(m, name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. plant_cuf
# ─────────────────────────────────────────────────────────────────────────────

class TestPlantCuf:

    def test_max_reduces_busbar(self, params: OptParams, baseline: dict) -> None:
        cuf0 = baseline["busbar"] / (_FIXED["P"] * baseline["n_steps"]) * 100.0
        cap_pct = 0.8 * cuf0
        cfg = OptionalConstraintsConfig(plant_cuf_enabled=True, plant_cuf_max_pct=cap_pct)
        m, tc = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" in st["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        busbar = _busbar(d, params.eta_d)
        cap = (cap_pct / 100.0) * _FIXED["P"] * tc.n_steps
        assert busbar <= cap + 1e-4
        assert busbar < baseline["busbar"] - 1e-3

    def test_min_infeasible_when_impossible(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(plant_cuf_enabled=True, plant_cuf_min_pct=99.0)
        m, _ = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" not in st["status"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. minimum_bess_capacity  (free nb, S/W/P fixed)
# ─────────────────────────────────────────────────────────────────────────────

class TestMinBessCapacity:

    _SWP = {"S": 150.0, "W": 80.0, "P": 100.0}   # nb left free

    def test_floor_forces_nb_up(self, params: OptParams) -> None:
        floor = 250.0
        # Off: objective drops BESS (full-scale capex vs tiny mini-horizon revenue).
        m_off, _ = _build(params, OptionalConstraintsConfig(), self._SWP)
        assert "optimal" in _solve(m_off)["status"].lower()
        e_off = float(pyo.value(m_off.E_b))

        m_on, _ = _build(
            params,
            OptionalConstraintsConfig(
                min_bess_capacity_enabled=True, min_bess_capacity_mwh=floor
            ),
            self._SWP,
        )
        assert "optimal" in _solve(m_on)["status"].lower()
        e_on = float(pyo.value(m_on.E_b))

        assert e_off < floor
        assert e_on >= floor - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 4. minimum_bess_discharge
# ─────────────────────────────────────────────────────────────────────────────

class TestMinBessDischarge:

    def test_low_floor_satisfied(self, params: OptParams, baseline: dict) -> None:
        floor = 0.5 * baseline["discharge"]
        cfg = OptionalConstraintsConfig(
            min_bess_discharge_enabled=True, min_bess_discharge_annual_mwh=floor
        )
        m, tc = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" in st["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        assert _discharge(d, params.eta_d) >= floor - 1e-4

    def test_impossible_floor_infeasible(self, params: OptParams, baseline: dict) -> None:
        # Far beyond what the fixed BESS (E_b=300.9, C-rate 1.0) can throughput.
        floor = 100.0 * max(baseline["discharge"], 1.0)
        cfg = OptionalConstraintsConfig(
            min_bess_discharge_enabled=True, min_bess_discharge_annual_mwh=floor
        )
        m, _ = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" not in st["status"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. re_penetration
# ─────────────────────────────────────────────────────────────────────────────

class TestRePenetration:

    def test_max_reduces_meter(self, params: OptParams, baseline: dict) -> None:
        ratio0 = baseline["meter"] / baseline["total_load"] * 100.0
        cap_pct = 0.8 * ratio0
        cfg = OptionalConstraintsConfig(re_penetration_enabled=True, re_penetration_max_pct=cap_pct)
        m, tc = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" in st["status"].lower()
        d = extract_dispatch(m, n_hours=tc.n_steps)
        meter = _meter(d, params, tc)
        cap = (cap_pct / 100.0) * baseline["total_load"]
        assert meter <= cap + 1e-3
        assert meter < baseline["meter"] - 1e-3

    def test_min_infeasible_when_impossible(self, params: OptParams) -> None:
        cfg = OptionalConstraintsConfig(re_penetration_enabled=True, re_penetration_min_pct=99.9)
        m, _ = _build(params, cfg, _FIXED)
        st = _solve(m)
        assert "optimal" not in st["status"].lower()
