"""
augmentation_pre_analysis.py
─────────────────────────────
Pre-analysis step for the augmentation engine.

Derives the contractual CUF floor, shortfall windows, and search bounds
from the Phase-1 baseline energy projection — before any augmentation
optimisation is run.

Both BESS and solar augmentation are event-based: neither asset class is
oversized at Year 1.  Instead the optimizer chooses *when* to deploy new
capacity (any year ≥ 2) and *how much* to add at each event.

Each new solar cohort has its own age-based degradation curve, exactly as
each new BESS cohort does.

This module is intentionally independent of AugmentationEngine; it
produces structured inputs (PreAnalysisResult) that the engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchBounds:
    """
    Bounds on the augmentation decision-variable search space.

    Attributes
    ----------
    x0_max              : max extra BESS containers added at Year 1
    k_max               : max BESS containers added per future augmentation event
    solar_event_max_mwp : max solar DC MWp added per augmentation event
                          (0.0 if solar augmentation is disabled)
    year_min            : earliest augmentation year (always 2)
    year_max            : latest augmentation year (= project tenure)
    """
    x0_max:              int
    k_max:               int
    solar_event_max_mwp: float
    year_min:            int
    year_max:            int


@dataclass
class PreAnalysisResult:
    """
    Output of AugmentationPreAnalysis.run().

    Attributes
    ----------
    fixed_cuf_floor     : Year-1 CUF (%), used as the contractual floor.
    N_max               : Maximum number of BESS augmentation events (≥ 1).
    N_solar_max         : Maximum number of solar augmentation events (≥ 1).
                          Derived identically to N_max; same shortfall windows
                          can be addressed by either or both asset classes.
    shortfall_windows   : List of (start_year, end_year) 1-indexed tuples
                          identifying contiguous years where CUF < floor.
    baseline_cuf_series : CUF % per year, shape (tenure,).
    search_bounds       : SearchBounds instance.
    """
    fixed_cuf_floor:     float
    N_max:               int
    N_solar_max:         int
    shortfall_windows:   list[tuple[int, int]]
    baseline_cuf_series: np.ndarray
    search_bounds:       SearchBounds


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _contiguous_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_year, end_year) 1-indexed for contiguous True runs."""
    windows: list[tuple[int, int]] = []
    in_window = False
    start: int = 0  # assigned before use: only read when in_window is True
    for i, v in enumerate(mask):
        year = i + 1  # 1-indexed
        if v and not in_window:
            in_window = True
            start = year
        elif not v and in_window:
            windows.append((start, year - 1))
            in_window = False
    if in_window:
        windows.append((start, len(mask)))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationPreAnalysis:
    """
    Pre-analysis step that derives CUF floor, shortfall windows, and search
    bounds from the Phase-1 baseline energy projection.

    Parameters
    ----------
    baseline_ep : dict
        Dict returned by EnergyProjection.project(), or a superset thereof.
        Must contain:
          - ``delivered_pre_mwh``  : np.ndarray, shape (tenure,)  busbar basis
    config : object
        Config object with the following attribute paths:
          - config.augmentation["augmentation_optimizer"]["max_extra_containers"]
          - config.augmentation["augmentation_optimizer"]["max_augmentation_containers"]
          - config.augmentation.get("augmentation_optimizer", {})
              .get("solar_augmentation", {})
              -> {"enabled": bool, "max_extra_mwp_per_event": float}
          - config.project["project"]["project_life_years"]   (= tenure)
          - config.project["simulation"].get("hours_per_year", 8760)
    ppa_capacity_mw : float
        PPA contracted capacity in MW.  Passed explicitly so that this class
        does not depend on the layout of EnergyProjection.project() output.
    """

    def __init__(self, baseline_ep: dict, config, ppa_capacity_mw: float) -> None:
        self._baseline_ep = baseline_ep
        self._config = config
        self._ppa_capacity_mw = float(ppa_capacity_mw)

    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> PreAnalysisResult:
        """
        Run pre-analysis and return a PreAnalysisResult.

        Steps
        -----
        1. Extract the baseline CUF series from delivered_pre_mwh.
        2. Set fixed_cuf_floor = cuf_series[0]  (Year-1 value).
        3. Identify shortfall windows: contiguous years where CUF < floor.
        4. Derive N_max = max(1, len(shortfall_windows))  [BESS events].
        5. Derive N_solar_max = N_max  [solar events, same logic].
        6. Build SearchBounds from config.
        7. Return PreAnalysisResult.
        """
        ep     = self._baseline_ep
        config = self._config

        # ── 1. Baseline CUF series ────────────────────────────────────────────
        delivered = np.asarray(ep["delivered_pre_mwh"], dtype=float)
        if len(delivered) == 0:
            raise ValueError("baseline_ep['delivered_pre_mwh'] must not be empty.")
        ppa_cap        = self._ppa_capacity_mw
        hours_per_year = float(
            config.project["simulation"].get("hours_per_year", 8760)
        )

        cuf_series = delivered / (ppa_cap * hours_per_year) * 100.0

        # ── 2. CUF floor ─────────────────────────────────────────────────────
        fixed_cuf_floor = float(cuf_series[0])

        # ── 3. Shortfall windows ──────────────────────────────────────────────
        shortfall_mask    = cuf_series < fixed_cuf_floor
        shortfall_windows = _contiguous_windows(shortfall_mask)

        # ── 4–5. N_max / N_solar_max ──────────────────────────────────────────
        N_max       = max(1, len(shortfall_windows))
        N_solar_max = N_max

        # ── 6. SearchBounds ───────────────────────────────────────────────────
        aug_cfg = config.augmentation.get("augmentation_optimizer", {})

        solar_aug_cfg = aug_cfg.get("solar_augmentation", {})
        if solar_aug_cfg.get("enabled", False):
            solar_event_max_mwp = float(solar_aug_cfg.get("max_extra_mwp_per_event", 0.0))
        else:
            solar_event_max_mwp = 0.0

        x0_max   = int(aug_cfg.get("max_extra_containers", 10))
        k_max    = int(aug_cfg.get("max_augmentation_containers", 20))
        year_min = 2
        year_max = int(config.project["project"]["project_life_years"])

        search_bounds = SearchBounds(
            x0_max=x0_max,
            k_max=k_max,
            solar_event_max_mwp=solar_event_max_mwp,
            year_min=year_min,
            year_max=year_max,
        )

        # ── 7. Return ─────────────────────────────────────────────────────────
        return PreAnalysisResult(
            fixed_cuf_floor=fixed_cuf_floor,
            N_max=N_max,
            N_solar_max=N_solar_max,
            shortfall_windows=shortfall_windows,
            baseline_cuf_series=cuf_series,
            search_bounds=search_bounds,
        )
