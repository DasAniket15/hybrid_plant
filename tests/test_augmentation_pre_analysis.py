"""
test_augmentation_pre_analysis.py
──────────────────────────────────
Unit tests for AugmentationPreAnalysis.

Uses mock objects for baseline_ep and config so that no real YAML files
or simulation runs are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_plant.augmentation.augmentation_pre_analysis import (
    AugmentationPreAnalysis,
    PreAnalysisResult,
    SearchBounds,
    _contiguous_windows,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _MockAugConfig(dict):
    """dict subclass that also exposes .get() — behaves like the YAML-loaded dict."""
    pass


class _MockConfig:
    """Minimal mock for FullConfig, providing dict-style .augmentation and .project."""

    def __init__(
        self,
        project_life: int = 25,
        max_extra_containers: int = 10,
        max_augmentation_containers: int = 20,
        solar_oversize_enabled: bool = False,
        solar_oversize_max_mwp: float = 0.0,
    ) -> None:
        aug_optimizer = {
            "max_extra_containers": max_extra_containers,
            "max_augmentation_containers": max_augmentation_containers,
        }
        solar_oversize = {
            "enabled": solar_oversize_enabled,
            "max_extra_mwp": solar_oversize_max_mwp,
        }
        self.augmentation = {
            "augmentation_optimizer": aug_optimizer,
            "solar_oversize": solar_oversize,
        }
        self.project = {
            "project": {"project_life_years": project_life}
        }


def _make_baseline_ep(cuf_pct_series: np.ndarray, ppa_cap: float = 100.0) -> dict:
    """
    Build a baseline_ep dict whose delivered_meter_mwh is consistent with
    the given CUF % series.

    delivered_meter_mwh[t] = cuf_pct_series[t] / 100 * ppa_cap * 8760
    """
    delivered = cuf_pct_series / 100.0 * ppa_cap * 8760.0
    return {
        "delivered_meter_mwh": delivered,
        "ppa_capacity_mw": ppa_cap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _contiguous_windows
# ─────────────────────────────────────────────────────────────────────────────

class TestContiguousWindows:

    def test_no_shortfall(self):
        mask = np.array([False, False, False, False])
        assert _contiguous_windows(mask) == []

    def test_all_shortfall(self):
        mask = np.array([True, True, True])
        assert _contiguous_windows(mask) == [(1, 3)]

    def test_single_year_shortfall(self):
        mask = np.array([False, True, False, False])
        assert _contiguous_windows(mask) == [(2, 2)]

    def test_two_separate_windows(self):
        # Years 2–3 and years 5–6 (1-indexed)
        mask = np.array([False, True, True, False, True, True])
        assert _contiguous_windows(mask) == [(2, 3), (5, 6)]

    def test_window_at_end(self):
        # Years 4–5 shortfall, tenure = 5
        mask = np.array([False, False, False, True, True])
        assert _contiguous_windows(mask) == [(4, 5)]

    def test_window_at_start(self):
        # Year 1 shortfall (edge case — this should not normally happen
        # since Year 1 defines the floor, but the helper handles it correctly)
        mask = np.array([True, False, False])
        assert _contiguous_windows(mask) == [(1, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# AugmentationPreAnalysis
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentationPreAnalysis:

    def _run(self, cuf_series: np.ndarray, **config_kwargs) -> PreAnalysisResult:
        ep     = _make_baseline_ep(cuf_series)
        config = _MockConfig(**config_kwargs)
        return AugmentationPreAnalysis(ep, config).run()

    # ── CUF floor ────────────────────────────────────────────────────────────

    def test_cuf_floor_equals_year1(self):
        cuf = np.array([80.0, 78.0, 76.0, 74.0, 72.0])
        result = self._run(cuf, project_life=5)
        assert result.fixed_cuf_floor == pytest.approx(80.0)

    def test_cuf_floor_non_monotone(self):
        # Degradation can be non-monotone; floor is always Year-1 value
        cuf = np.array([75.0, 76.0, 74.0, 73.0])
        result = self._run(cuf, project_life=4)
        assert result.fixed_cuf_floor == pytest.approx(75.0)

    # ── Shortfall windows ────────────────────────────────────────────────────

    def test_no_shortfall_windows_when_flat(self):
        # All years equal to floor — no shortfall (strict <)
        cuf = np.array([80.0, 80.0, 80.0])
        result = self._run(cuf, project_life=3)
        assert result.shortfall_windows == []

    def test_single_contiguous_window(self):
        # Years 3–5 below floor
        cuf = np.array([80.0, 80.0, 79.0, 78.0, 77.0])
        result = self._run(cuf, project_life=5)
        assert result.shortfall_windows == [(3, 5)]

    def test_two_separate_windows(self):
        # Year 2 below floor, Year 4–5 below floor
        cuf = np.array([80.0, 79.0, 80.5, 78.0, 77.0])
        result = self._run(cuf, project_life=5)
        assert result.shortfall_windows == [(2, 2), (4, 5)]

    def test_three_windows(self):
        # Years 2, 4, 6 each form isolated shortfall windows
        cuf = np.array([80.0, 79.0, 80.0, 79.0, 80.0, 79.0])
        result = self._run(cuf, project_life=6)
        assert result.shortfall_windows == [(2, 2), (4, 4), (6, 6)]

    # ── N_max ────────────────────────────────────────────────────────────────

    def test_n_max_minimum_is_1_when_no_shortfall(self):
        cuf = np.array([80.0, 80.0, 80.0])
        result = self._run(cuf, project_life=3)
        assert result.N_max == 1

    def test_n_max_equals_window_count(self):
        # Two shortfall windows → N_max = 2
        cuf = np.array([80.0, 79.0, 80.5, 78.0, 77.0])
        result = self._run(cuf, project_life=5)
        assert result.N_max == 2

    def test_n_max_three_windows(self):
        cuf = np.array([80.0, 79.0, 80.0, 79.0, 80.0, 79.0])
        result = self._run(cuf, project_life=6)
        assert result.N_max == 3

    # ── baseline_cuf_series ──────────────────────────────────────────────────

    def test_baseline_cuf_series_shape(self):
        cuf = np.linspace(80.0, 70.0, 25)
        result = self._run(cuf, project_life=25)
        assert result.baseline_cuf_series.shape == (25,)

    def test_baseline_cuf_series_values(self):
        cuf = np.array([75.5, 74.0, 73.0])
        result = self._run(cuf, project_life=3)
        np.testing.assert_allclose(result.baseline_cuf_series, cuf, rtol=1e-9)

    # ── SearchBounds ─────────────────────────────────────────────────────────

    def test_search_bounds_year_min_always_2(self):
        cuf = np.array([80.0, 78.0])
        result = self._run(cuf, project_life=2)
        assert result.search_bounds.year_min == 2

    def test_search_bounds_year_max_equals_tenure(self):
        cuf = np.linspace(80.0, 70.0, 20)
        result = self._run(cuf, project_life=20)
        assert result.search_bounds.year_max == 20

    def test_search_bounds_x0_max(self):
        cuf = np.array([80.0, 78.0, 76.0])
        result = self._run(cuf, project_life=3, max_extra_containers=7)
        assert result.search_bounds.x0_max == 7

    def test_search_bounds_k_max(self):
        cuf = np.array([80.0, 78.0, 76.0])
        result = self._run(cuf, project_life=3, max_augmentation_containers=15)
        assert result.search_bounds.k_max == 15

    def test_s0_max_zero_when_solar_oversize_disabled(self):
        cuf = np.array([80.0, 78.0, 76.0])
        result = self._run(cuf, project_life=3, solar_oversize_enabled=False, solar_oversize_max_mwp=50.0)
        assert result.search_bounds.s0_max == 0.0

    def test_s0_max_from_config_when_enabled(self):
        cuf = np.array([80.0, 78.0, 76.0])
        result = self._run(cuf, project_life=3, solar_oversize_enabled=True, solar_oversize_max_mwp=50.0)
        assert result.search_bounds.s0_max == pytest.approx(50.0)

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_pre_analysis_result_instance(self):
        cuf = np.array([80.0, 78.0, 76.0])
        result = self._run(cuf, project_life=3)
        assert isinstance(result, PreAnalysisResult)
        assert isinstance(result.search_bounds, SearchBounds)

    # ── Full scenario: typical 25-year degradation ───────────────────────────

    def test_full_25_year_scenario(self):
        """
        Realistic scenario: Year-1 CUF = 75.05 %, then monotonically declining.
        Shortfall starts at Year 2 and runs to Year 25 — one contiguous window.
        """
        year1_cuf = 75.05
        cuf = np.concatenate([[year1_cuf], np.linspace(74.9, 65.0, 24)])
        result = self._run(cuf, project_life=25)

        assert result.fixed_cuf_floor == pytest.approx(year1_cuf)
        assert result.shortfall_windows == [(2, 25)]
        assert result.N_max == 1
        assert result.search_bounds.year_min == 2
        assert result.search_bounds.year_max == 25
