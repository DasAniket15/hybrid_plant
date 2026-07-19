"""
tests/optimise/test_dashboard.py
─────────────────────────────────
HTML dashboard generation (optimise/dashboard.py).

Slow: runs the Pyomo pipeline once, renders both dashboards, and checks the
output is well-formed, self-contained, and carries the headline numbers.
"""

from __future__ import annotations

import pytest

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.optimise.dashboard import (
    compute_metrics,
    render_detailed_dashboard,
    render_executive_dashboard,
)
from hybrid_plant.optimise.pipeline import run_pyomo_optimization


@pytest.fixture(scope="module")
def config(config) -> FullConfig:
    return config


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.mark.slow
class TestDashboards:

    @pytest.fixture(scope="class")
    def result(self, config: FullConfig, data: dict) -> dict:
        return run_pyomo_optimization(config, data, compute_oracle=True)

    def test_metrics_sane(self, result: dict, config: FullConfig) -> None:
        m = compute_metrics(result, config)
        assert 900 < m["tod_npv_cr"] < 1100
        assert m["flat_npv_cr"] > 0 and m["rtc_npv_cr"] > 0
        assert 0 < m["re_penetration"] <= 100
        assert 0 < m["lcoe"] < 10
        assert m["verify_ok"]

    def test_executive_html(self, result: dict, config: FullConfig, data: dict) -> None:
        html = render_executive_dashboard(result, config, data)
        assert html.startswith("<!doctype html>")
        assert "Executive Summary" in html
        assert "data:image/png;base64," in html          # self-contained charts
        assert html.count("data:image/png") == 2          # two headline charts
        assert "http://" not in html and "https://" not in html   # no external assets
        assert "ToD-aware" in html

    def test_detailed_html(self, result: dict, config: FullConfig, data: dict) -> None:
        html = render_detailed_dashboard(result, config, data)
        assert html.startswith("<!doctype html>")
        assert html.count("data:image/png") == 5          # five charts
        assert "25-Year projection" in html
        assert "reconciliation" in html.lower()
        # 25 data rows in the projection table
        assert html.count("<tr>") >= 25
