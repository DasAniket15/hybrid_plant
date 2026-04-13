"""
augmentation_engine.py
──────────────────────
BESS augmentation config reader and cohort-math helper.

Role in the pipeline
────────────────────
This class no longer pre-computes a fixed schedule.  Instead it:

  1. Reads augmentation configuration from ``bess.yaml`` and exposes
     it as public properties (``enabled``, ``trigger_cuf_percent``, …).

  2. Provides two cohort-math helpers used by ``EnergyProjection._project_full``:
     • ``cohort_soh(project_year, cohort_start)`` — SOH for one cohort
     • ``compute_effective(year, cohorts)``  — blended effective MWh / SOH

  3. Provides ``passthrough_result(initial_containers)`` — returns the same
     dict structure as the augmentation result built by ``_project_full``,
     but with no augmentation events (plain SOH degradation only).  Used by
     the fast-mode energy projection during solver trials.

Trigger metric change
─────────────────────
The trigger is now **plant CUF %** (not BESS effective capacity %).
CUF requires a full 8760-hour simulation to measure, so the schedule
cannot be pre-computed here.  Instead, ``EnergyProjection._project_full``
simulates each year, checks whether CUF has fallen below
  ``trigger_cuf_percent / 100 × year1_cuf``
and, if so, adds a new cohort and re-simulates that year.

Cohorts method
──────────────
Each batch installed in a given year forms a cohort.  A cohort's SOH in
project year t is indexed from its own start year:

    cohort_soh(t) = soh_curve[ t − cohort.start_year + 1 ]

The blended (plant-level) effective SOH passed to PlantEngine is:

    effective_soh(t) = total_effective_mwh(t) / (total_containers × container_size)

Cost treatment
──────────────
Container purchase = OPEX in the augmentation year (not CAPEX).
BESS O&M in subsequent years is applied to total installed (nameplate) MWh.

Augmentation result dict structure (returned by _project_full / passthrough)
──────────────────────────────────────────────────────────────────────────────
  enabled                       : bool
  cohorts                       : list of (start_year, n_containers) tuples
  augmentation_years            : list[int]
  containers_added_per_year     : dict[int, int]
  effective_containers_per_year : dict[int, int]
  effective_soh_per_year        : dict[int, float]
  total_installed_mwh_per_year  : dict[int, float]
  augmentation_purchase_opex    : dict[int, float]   (Rs; 0 if no event)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.config_loader import FullConfig


class AugmentationEngine:
    """
    BESS augmentation config reader and cohort-math helper.

    Parameters
    ----------
    config : FullConfig
    """

    def __init__(self, config: FullConfig) -> None:
        aug_cfg = config.bess["bess"]["augmentation"]

        # ── Public config properties ──────────────────────────────────────────
        self.enabled              = bool(aug_cfg.get("enabled", False))
        self.trigger_cuf_percent  = float(aug_cfg.get("trigger_cuf_percent", 95.0))
        self.restore_pct          = float(aug_cfg.get("restore_to_percent_of_year1", 100.0))
        self.max_gap_years        = aug_cfg.get("max_gap_years")   # None or int
        if self.max_gap_years is not None:
            self.max_gap_years = int(self.max_gap_years)
        self.min_containers       = int(aug_cfg.get("minimum_augmentation_containers", 1))
        self.cost_per_mwh         = float(aug_cfg.get("cost_per_mwh", 0.60e7))
        self.container_size       = float(config.bess["bess"]["container"]["size_mwh"])
        self.project_life         = int(config.project["project"]["project_life_years"])

        # ── SOH curve {year: soh} ─────────────────────────────────────────────
        root = find_project_root()
        self.soh_curve: dict[int, float] = self._load_soh_curve(
            root / config.bess["bess"]["degradation"]["file"]
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_soh_curve(path: Path) -> dict[int, float]:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return dict(zip(df["year"].astype(int), df["soh"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Cohort-math helpers (public — called from EnergyProjection._project_full)
    # ─────────────────────────────────────────────────────────────────────────

    def cohort_soh(self, project_year: int, cohort_start: int) -> float:
        """SOH of a cohort at *project_year*, given it was installed at *cohort_start*."""
        age = project_year - cohort_start + 1          # 1 = brand new
        age = max(1, min(age, self.project_life))       # clamp to curve bounds
        return self.soh_curve.get(age, self.soh_curve[self.project_life])

    def compute_effective(
        self,
        year: int,
        cohorts: list[tuple[int, int]],
    ) -> tuple[float, int, float, float]:
        """
        Return ``(eff_mwh, total_n, effective_soh, total_installed_mwh)`` for *year*.

        Parameters
        ----------
        year    : project year (1-indexed)
        cohorts : list of (start_year, n_containers) tuples

        Returns
        -------
        eff_mwh            : total effective energy capacity across all cohorts (MWh)
        total_n            : total container count
        effective_soh      : blended SOH  = eff_mwh / (total_n × container_size)
        total_installed_mwh: nameplate sum (ignores SOH); used for O&M billing
        """
        eff_mwh   = 0.0
        total_n   = 0
        installed = 0.0
        for start_yr, n_i in cohorts:
            soh = self.cohort_soh(year, start_yr)
            eff_mwh   += n_i * self.container_size * soh
            total_n   += n_i
            installed += n_i * self.container_size

        eff_soh = eff_mwh / installed if installed > 0 else 0.0
        return eff_mwh, total_n, eff_soh, installed

    # ─────────────────────────────────────────────────────────────────────────
    # Fast-mode helper
    # ─────────────────────────────────────────────────────────────────────────

    def passthrough_result(self, initial_containers: int) -> dict[str, Any]:
        """
        Return an augmentation-result dict with *no* augmentation events.

        Effective SOH per year is taken directly from the plain SOH curve
        (single cohort starting at year 1).  Container count stays constant.

        Used by ``EnergyProjection._project_fast`` during solver trials so
        that all downstream consumers receive a valid result dict without
        needing conditional logic.

        Parameters
        ----------
        initial_containers : number of BESS containers installed in Year 1
        """
        cohorts  = [(1, initial_containers)]
        proj     = self.project_life
        cs       = self.container_size
        installed_yr = initial_containers * cs

        containers_added_per_year: dict[int, int]   = {}
        effective_containers_per_year: dict[int, int]   = {}
        effective_soh_per_year:        dict[int, float] = {}
        total_installed_mwh_per_year:  dict[int, float] = {}

        for yr in range(1, proj + 1):
            soh = self.soh_curve.get(yr, self.soh_curve[proj])
            containers_added_per_year[yr]     = 0
            effective_containers_per_year[yr] = initial_containers
            effective_soh_per_year[yr]        = soh
            total_installed_mwh_per_year[yr]  = installed_yr

        return {
            "enabled":                       self.enabled,
            "cohorts":                       cohorts,
            "augmentation_years":            [],
            "containers_added_per_year":     containers_added_per_year,
            "effective_containers_per_year": effective_containers_per_year,
            "effective_soh_per_year":        effective_soh_per_year,
            "total_installed_mwh_per_year":  total_installed_mwh_per_year,
            "augmentation_purchase_opex":    {},
        }
