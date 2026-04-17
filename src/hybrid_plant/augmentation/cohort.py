"""
cohort.py
─────────
BESS cohort data structures for the augmentation engine.

A "cohort" is a batch of BESS containers that share an install year and
therefore degrade on the same timeline.  The plant starts with one cohort
(the initial containers) and gains additional cohorts at each augmentation
event.

Cohort age convention
─────────────────────
For a cohort installed at ``install_year``, its age in calendar year
``year_t`` is:

    age = year_t - install_year + 1

This means:
  • In the year of installation (year_t == install_year), age = 1.
  • SOH for that year is **1.0** — the cohort is fresh and has not yet
    experienced any degradation.
  • For age ≥ 2, SOH is ``soh_curve[age - 1]`` (end-of-prior-year value).
    See ``data_loader.operating_value`` for the full convention.

The initial cohort uses ``install_year = 1`` (project Year 1), so its
age in Year 1 is 1 and it operates at SOH = 1.0.  This is consistent
with the end-of-year degradation convention used throughout.

Cohort aggregation
──────────────────
When multiple cohorts are active, ``CohortRegistry.to_plant_params``
translates the full cohort list into the ``(bess_containers, soh_factor)``
pair expected by ``PlantEngine.simulate``:

    total_containers = sum of all active cohort container counts
    total_eff_mwh    = sum of (containers × container_size × soh[age])
    blended_soh      = total_eff_mwh / (total_containers × container_size)

This produces a single "blended" SOH factor that, when multiplied by
the total container count and container size inside PlantEngine, yields
exactly the correct total effective energy capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Cohort descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BESSCohort:
    """
    Immutable descriptor for a single batch of BESS containers.

    Parameters
    ----------
    install_year : int
        Calendar project year in which this cohort was installed.
        Use 1 for the initial cohort; event cohorts use the event year.
    containers : int
        Number of physical BESS containers in this cohort.
    """

    install_year: int
    containers:   int

    def age(self, year_t: int) -> int:
        """
        Return this cohort's age in calendar year ``year_t``.

        Age is 1 in the install year, 2 the following year, etc.
        Returns 0 (inactive) for years before installation.
        """
        return year_t - self.install_year + 1

    def is_active(self, year_t: int) -> bool:
        """True if this cohort has been installed by ``year_t``."""
        return year_t >= self.install_year

    def effective_capacity_mwh(
        self,
        year_t:         int,
        container_size: float,
        soh_curve:      dict[int, float],
    ) -> float:
        """
        Effective energy capacity of this cohort in ``year_t`` (MWh).

        Applies the **operating** SOH for this cohort's age in year_t, using
        the end-of-year curve convention (see ``data_loader.operating_value``):
          • age == 1 → SOH = 1.0 (fresh cohort in its install year)
          • age >= 2 → SOH = soh_curve[age - 1]

        Returns 0.0 if the cohort is not yet active.
        """
        from hybrid_plant.data_loader import operating_value
        if not self.is_active(year_t):
            return 0.0
        age = self.age(year_t)
        soh = operating_value(soh_curve, age)
        return self.containers * container_size * soh


# ─────────────────────────────────────────────────────────────────────────────
# Cohort registry
# ─────────────────────────────────────────────────────────────────────────────

class CohortRegistry:
    """
    Mutable list-of-cohorts manager with helpers for per-year aggregation.

    The registry begins with a single initial cohort (install_year=1).
    Augmentation events add further cohorts via ``add()``.

    Parameters
    ----------
    initial_containers : int
        Number of BESS containers in the initial (project-start) cohort.
    """

    def __init__(self, initial_containers: int) -> None:
        self._cohorts: list[BESSCohort] = [
            BESSCohort(install_year=1, containers=initial_containers)
        ]

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, install_year: int, containers: int) -> None:
        """
        Register a new augmentation cohort.

        Parameters
        ----------
        install_year : int
            Calendar year in which the fresh containers are installed.
        containers   : int
            Number of new containers added in this event.
        """
        self._cohorts.append(BESSCohort(install_year=install_year, containers=containers))

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def total_containers(self, year_t: int) -> int:
        """Total physical container count active in ``year_t``."""
        return sum(c.containers for c in self._cohorts if c.is_active(year_t))

    def effective_capacity_mwh(
        self,
        year_t:         int,
        container_size: float,
        soh_curve:      dict[int, float],
    ) -> float:
        """
        Sum of effective energy capacity across all active cohorts in ``year_t``.

        Parameters
        ----------
        year_t         : calendar project year (1-indexed)
        container_size : MWh nameplate per container (from bess.yaml)
        soh_curve      : {age: soh_fraction} dict loaded from CSV

        Returns
        -------
        float — total effective MWh across all active cohorts
        """
        return sum(
            c.effective_capacity_mwh(year_t, container_size, soh_curve)
            for c in self._cohorts
            if c.is_active(year_t)
        )

    def to_plant_params(
        self,
        year_t:         int,
        container_size: float,
        soh_curve:      dict[int, float],
    ) -> tuple[int, float]:
        """
        Translate the full cohort list into ``(total_containers, blended_soh_factor)``
        for direct use with ``PlantEngine.simulate(bess_containers=..., bess_soh_factor=...)``.

        The blended SOH factor is defined so that:

            total_containers × container_size × blended_soh == total_effective_mwh

        Parameters
        ----------
        year_t         : calendar project year (1-indexed)
        container_size : MWh nameplate per container
        soh_curve      : {age: soh_fraction} dict

        Returns
        -------
        (total_containers, blended_soh_factor) — ready for PlantEngine.simulate
        """
        total_containers = self.total_containers(year_t)
        if total_containers == 0:
            return 0, 0.0
        total_eff_mwh = self.effective_capacity_mwh(year_t, container_size, soh_curve)
        blended_soh   = total_eff_mwh / (total_containers * container_size)
        return total_containers, blended_soh

    # ── Per-cohort capacity timeline ──────────────────────────────────────────

    def cohort_capacity_timeline(
        self,
        project_life:   int,
        container_size: float,
        soh_curve:      dict[int, float],
    ) -> dict[int, list[float]]:
        """
        Return per-cohort effective MWh for each project year.

        Returns
        -------
        dict mapping cohort index (0-based) to a list of length ``project_life``
        where each element is that cohort's effective MWh in year (index+1).
        Used for the stacked-area augmentation dashboard panel.
        """
        timeline: dict[int, list[float]] = {}
        for idx, cohort in enumerate(self._cohorts):
            timeline[idx] = [
                cohort.effective_capacity_mwh(year, container_size, soh_curve)
                for year in range(1, project_life + 1)
            ]
        return timeline

    # ── Introspection ─────────────────────────────────────────────────────────

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a list of dicts describing all cohorts (for dashboard logging)."""
        return [
            {"cohort_index": i, "install_year": c.install_year, "containers": c.containers}
            for i, c in enumerate(self._cohorts)
        ]

    @property
    def cohorts(self) -> list[BESSCohort]:
        """Read-only view of the cohort list."""
        return list(self._cohorts)

    def __len__(self) -> int:
        return len(self._cohorts)
