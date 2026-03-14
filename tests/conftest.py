"""
conftest.py
───────────
Shared pytest fixtures for the hybrid_plant test suite.

The ``config`` and ``data`` fixtures are session-scoped so the heavy
YAML + CSV loading runs once per pytest session regardless of how many
test files import them.

The ``energy_engine`` and ``finance_engine`` fixtures are also session-scoped
because they hold no mutable state — each test evaluation call is stateless.
"""

from __future__ import annotations

import pytest

from hybrid_plant.config_loader import load_config, FullConfig
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config() -> FullConfig:
    """Fully loaded and validated FullConfig (loaded once per session)."""
    return load_config()


@pytest.fixture(scope="session")
def data(config: FullConfig) -> dict:
    """Time-series + degradation curve data dict (loaded once per session)."""
    return load_timeseries_data(config)


@pytest.fixture(scope="session")
def energy_engine(config: FullConfig, data: dict) -> Year1Engine:
    """Shared Year1Engine instance."""
    return Year1Engine(config, data)


@pytest.fixture(scope="session")
def finance_engine(config: FullConfig, data: dict) -> FinanceEngine:
    """Shared FinanceEngine instance."""
    return FinanceEngine(config, data)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical benchmark parameters
# ─────────────────────────────────────────────────────────────────────────────

SOLAR_ONLY_PARAMS: dict = {
    "solar_capacity_mw":  195.415073395429,
    "wind_capacity_mw":   0.0,
    "bess_containers":    164,
    "charge_c_rate":      1.0,
    "discharge_c_rate":   1.0,
    "ppa_capacity_mw":    67.5256615562851,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_only",
}

SOLAR_WIND_PARAMS: dict = {
    "solar_capacity_mw":  190.454972460807,
    "wind_capacity_mw":   116.130108575195,
    "bess_containers":    120,
    "charge_c_rate":      1.0,
    "discharge_c_rate":   1.0,
    "ppa_capacity_mw":    120.632227022855,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_only",
}


@pytest.fixture(scope="session")
def solar_only_params() -> dict:
    return SOLAR_ONLY_PARAMS


@pytest.fixture(scope="session")
def solar_wind_params() -> dict:
    return SOLAR_WIND_PARAMS
