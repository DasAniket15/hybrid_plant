"""
cohort.py
─────────
Cohort-based BESS degradation model.

Each physical BESS installation (the initial build or any subsequent
augmentation) is represented by an independent ``BESSCohort``. Every cohort
ages from *its own* installation year, so its State-of-Health (SoH) in any
project year is read from the degradation curve at the cohort's operating
age — NOT the project age.

The plant's effective BESS energy at year Y is the sum of each active
cohort's degraded capacity:

    effective_mwh(Y) = Σ  cohort.containers × container_size × soh[Y − cohort.installation_year]
                      cohorts active in Y

Cohort lifecycle semantics
──────────────────────────
  installation_year = 0     initial cohort (installed at project start,
                            operational from Year 1)
  installation_year = K>0   augmentation installed during Year K,
                            becomes operational from Year K+1

  A cohort is "active" in project year Y iff Y > installation_year.
  Its age for SoH lookup is (Y − installation_year), which maps 1..25 onto
  the SoH CSV's year index — matching the existing single-battery model
  exactly when no augmentations are present.

Why this model
──────────────
  Collapsing cohorts into a single averaged SoH would under-credit a new
  cohort (dragged down by old cohorts) and over-degrade old cohorts
  (propped up by new ones). Keeping them independent preserves the physics
  and allows the augmentation engine to correctly size additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BESSCohort:
    """
    A single BESS installation event.

    Attributes
    ----------
    installation_year : int
        0 for the initial cohort (operational from Year 1); K > 0 for an
        augmentation installed in Year K (operational from Year K+1).
    containers : int
        Physical number of containers — always a non-negative integer.
        Fractional containers are not permitted.
    capacity_mwh : float
        Convenience field: containers × container_size_mwh (nameplate).
    """

    installation_year: int
    containers:        int
    capacity_mwh:      float

    def __post_init__(self) -> None:
        if self.installation_year < 0:
            raise ValueError(
                f"installation_year must be ≥ 0, got {self.installation_year}"
            )
        if self.containers < 0:
            raise ValueError(
                f"containers must be ≥ 0, got {self.containers}"
            )
        if not isinstance(self.containers, int):
            raise TypeError(
                f"containers must be an int (discrete sizing), "
                f"got {type(self.containers).__name__}"
            )

    def is_active(self, year: int) -> bool:
        """Cohort contributes in year Y iff Y > installation_year."""
        return year > self.installation_year

    def operating_age(self, year: int) -> int:
        """Age of the cohort in project year Y (1 = first operational year)."""
        return year - self.installation_year


class CohortManager:
    """
    Manages the list of active BESS cohorts and exposes aggregate
    quantities needed by the plant simulator.

    The SoH curve is the same CSV used by ``EnergyProjection`` (indexed by
    operating year 1..25). Age beyond the curve clamps to the last known
    value — augmentations installed late enough to escape the curve are
    effectively treated as still-degrading at the final-year rate.

    Parameters
    ----------
    container_size_mwh : float
        MWh per physical container (from ``bess.yaml``).
    soh_curve : dict[int, float]
        {operating_year: soh_factor} loaded from the SoH CSV. Must contain
        at least key 1 → soh at end of first operational year.
    """

    def __init__(
        self,
        container_size_mwh: float,
        soh_curve:          dict[int, float],
    ) -> None:
        if container_size_mwh <= 0:
            raise ValueError("container_size_mwh must be > 0")
        if not soh_curve:
            raise ValueError("soh_curve must not be empty")

        self.container_size: float = float(container_size_mwh)
        self.soh_curve:      dict[int, float] = dict(soh_curve)
        self._max_curve_age: int = max(self.soh_curve.keys())
        self.cohorts:        list[BESSCohort] = []

    # ─────────────────────────────────────────────────────────────────────────

    def add_initial(self, containers: int) -> BESSCohort:
        """Register the initial (Year-1) cohort. Only one is allowed."""
        if any(c.installation_year == 0 for c in self.cohorts):
            raise RuntimeError("Initial cohort already registered.")
        cohort = BESSCohort(
            installation_year=0,
            containers=int(containers),
            capacity_mwh=int(containers) * self.container_size,
        )
        self.cohorts.append(cohort)
        return cohort

    def add_augmentation(self, installation_year: int, containers: int) -> BESSCohort:
        """Append a new cohort installed in ``installation_year`` (must be ≥ 1)."""
        if installation_year < 1:
            raise ValueError(
                f"Augmentation installation_year must be ≥ 1, got {installation_year}"
            )
        cohort = BESSCohort(
            installation_year=int(installation_year),
            containers=int(containers),
            capacity_mwh=int(containers) * self.container_size,
        )
        self.cohorts.append(cohort)
        return cohort

    # ─────────────────────────────────────────────────────────────────────────

    def _soh_at_age(self, age: int) -> float:
        """
        SoH factor at a given cohort operating age.

        Age 0 (installed-this-year, not yet operational) → 1.0. Ages beyond
        the curve clamp to the last available value (no SoH curve provides
        data for ages > 25 in this model).
        """
        if age <= 0:
            return 1.0
        if age > self._max_curve_age:
            return self.soh_curve[self._max_curve_age]
        return self.soh_curve[age]

    def active_cohorts(self, year: int) -> list[BESSCohort]:
        return [c for c in self.cohorts if c.is_active(year)]

    def total_containers(self, year: int) -> int:
        return sum(c.containers for c in self.active_cohorts(year))

    def nameplate_mwh(self, year: int) -> float:
        """Sum of nameplate capacity across all active cohorts in year Y."""
        return sum(c.capacity_mwh for c in self.active_cohorts(year))

    def effective_mwh(self, year: int) -> float:
        """
        Degraded energy capacity in year Y:
            Σ cohort.capacity_mwh × soh[cohort.operating_age(Y)]
        """
        total = 0.0
        for c in self.active_cohorts(year):
            total += c.capacity_mwh * self._soh_at_age(c.operating_age(year))
        return total

    def aggregate_soh_factor(self, year: int) -> float:
        """
        Aggregate SoH ratio that, when passed to ``PlantEngine.simulate`` as
        ``bess_soh_factor`` together with ``bess_containers = total_containers``,
        reproduces the exact cohort-summed effective MWh.

        This mapping preserves PlantEngine's existing contract — effective
        energy = containers × container_size × soh_factor — so no plant-side
        code changes are required.

        Returns 0.0 if no cohort is active (pre-install years).
        """
        nameplate = self.nameplate_mwh(year)
        if nameplate <= 0:
            return 0.0
        return self.effective_mwh(year) / nameplate

    # ─────────────────────────────────────────────────────────────────────────

    def augmentation_events(self) -> list[tuple[int, int, float]]:
        """Return (installation_year, containers, capacity_mwh) for augmentations only."""
        return [
            (c.installation_year, c.containers, c.capacity_mwh)
            for c in self.cohorts
            if c.installation_year >= 1
        ]

    def snapshot(self) -> list[BESSCohort]:
        """Return a shallow copy of the cohort list (for logging/reporting)."""
        return list(self.cohorts)

    def __len__(self) -> int:
        return len(self.cohorts)

    def __iter__(self) -> Iterable[BESSCohort]:
        return iter(self.cohorts)
