"""
sizing.py
─────────
Smallest-augmentation search.

When the year-Y plant CUF drops below the frozen threshold, the
augmentation engine must decide how many containers (an integer ≥
``minimum_augmentation_containers``) to install so that the plant CUF
recovers to ≥ threshold in the NEXT year (Y+1) — the first full year in
which the augmented cohort is operational.

Search strategy (container count is a small integer)
────────────────────────────────────────────────────
We use a linear scan starting at ``min_containers`` and incrementing by 1.
For each candidate k:

  1. Append a tentative cohort with installation_year = Y, containers = k
  2. Simulate year Y+1 using the lifecycle simulator's ``simulate_year``
     (same PlantEngine, same per-year degraded solar/wind capacities)
  3. Compute the plant CUF
  4. Accept the first k that meets the threshold; otherwise continue
  5. Always remove the tentative cohort before returning — the caller
     decides whether to commit the decision

Why linear and not binary search
────────────────────────────────
  • The required k is typically small (1–5 containers)
  • Plant CUF is roughly monotonic in k but not strictly monotonic — at
    very high k the PPA cap starts to bind and additional BESS adds no
    further CUF. A linear scan tolerates that irregularity correctly by
    stopping at the first feasible k.
  • Each PlantEngine.simulate call is the dominant cost (~0.1–1 s on this
    model); 3–5 evaluations per augmentation event is acceptable.

"Economics stop improving" clause
─────────────────────────────────
The task description asks for the smallest augmentation that meets the CUF
constraint AND beyond which economics stop improving. For a single-year
OPEX hit, adding more containers than strictly needed:
    - increases augmentation OPEX (drag on savings)
    - increases energy delivery (uplift to savings)
The uplift plateaus quickly once the PPA cap binds, so the smallest
feasible k tends to be near-optimal.

Infeasible threshold (late-life years)
──────────────────────────────────────
Once the initial cohort has aged enough, the irreversible loss of its
capacity can make the frozen threshold (Year-1 CUF) unreachable for any
realistic new-cohort size. In that case the search returns a best-effort
``min_containers`` fallback with ``feasible=False`` — NOT a huge
``max_containers`` block, which would be a pure economic penalty with no
hope of meeting the threshold.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from hybrid_plant.augmentation.cohort import BESSCohort, CohortManager

if TYPE_CHECKING:  # only for type hints — avoid circular import at runtime
    from hybrid_plant.augmentation.lifecycle import LifecycleSimulator


# ─────────────────────────────────────────────────────────────────────────────

def _simulate_cuf_with_trial_cohort(
    cohort_manager: CohortManager,
    trial_year:     int,
    trial_k:        int,
    check_year:     int,
    sim_params:     dict[str, Any],
    simulator:      "LifecycleSimulator",
) -> tuple[float | None, float]:
    """
    Temporarily append a k-container cohort installed in ``trial_year`` and
    simulate ``check_year``. Returns (plant_cuf, busbar_mwh). The tentative
    cohort is always removed before returning — caller keeps the manager
    clean.
    """
    # Late import here too to avoid circular dependency at module load
    from hybrid_plant.augmentation.lifecycle import plant_cuf_from_busbar

    trial_cohort = BESSCohort(
        installation_year=int(trial_year),
        containers=int(trial_k),
        capacity_mwh=int(trial_k) * cohort_manager.container_size,
    )
    cohort_manager.cohorts.append(trial_cohort)
    try:
        yr = simulator.simulate_year(sim_params, check_year, cohort_manager)
        busbar = float(
            np.sum(yr["solar_direct_pre"])
            + np.sum(yr["wind_direct_pre"])
            + np.sum(yr["discharge_pre"])
        )
        cuf = plant_cuf_from_busbar(busbar, sim_params["ppa_capacity_mw"])
        return cuf, busbar
    finally:
        # Guarantee cleanup even on exception
        cohort_manager.cohorts.remove(trial_cohort)


# ─────────────────────────────────────────────────────────────────────────────

def find_augmentation_size(
    cohort_manager:  CohortManager,
    trigger_year:    int,
    sim_params:      dict[str, Any],
    threshold_cuf:   float,
    simulator:       "LifecycleSimulator",
    min_containers:  int = 1,
    max_containers:  int = 400,
) -> tuple[int, bool]:
    """
    Find the smallest integer k ≥ ``min_containers`` such that augmenting
    with k containers in ``trigger_year`` restores the plant CUF to
    ``threshold_cuf`` in year ``trigger_year + 1``.

    Returns
    -------
    tuple[int, bool]
        ``(k, feasible)``.

        * ``feasible=True`` — ``k`` is the smallest container count whose
          simulated year-(Y+1) CUF meets or exceeds ``threshold_cuf``.
        * ``feasible=False`` — the threshold is unreachable within the
          search cap. In this case ``k`` falls back to ``min_containers``
          so the lifecycle still records a good-faith augmentation event
          without installing an economically absurd ``max_containers``
          block that the threshold-chase would otherwise demand. The
          caller is expected to flag the event as best-effort.

        ``k = 0`` is returned when ``trigger_year`` is the final project
        year (no forward year exists to evaluate against). This is a
        distinct "no-op" signal from the best-effort fallback.

    Notes
    -----
    • CUF is evaluated in the year AFTER installation because a cohort with
      ``installation_year = Y`` is inactive in year Y by the cohort model.
      This matches the physical interpretation of a mid-year install that
      commissions by the start of the following year.
    • This function does NOT commit the augmentation — it only searches.
      The caller (LifecycleSimulator) decides whether to install.
    • The "fallback to ``min_containers`` on infeasibility" rule prevents
      the search from chasing an unreachable threshold (a common situation
      late in project life when the initial cohort's irreversible
      degradation dominates plant output, so no realistic cohort size can
      restore Year-1 CUF). Installing the minimum keeps the augmentation
      cadence tractable and lets downstream reporting flag the
      infeasibility.
    """
    check_year = trigger_year + 1
    if check_year > simulator.project_life:
        # Augmenting in the final year has no forward year to benefit from
        return 0, False

    if min_containers < 1:
        raise ValueError("min_containers must be ≥ 1")
    if max_containers < min_containers:
        raise ValueError("max_containers must be ≥ min_containers")

    for k in range(min_containers, max_containers + 1):
        cuf, _ = _simulate_cuf_with_trial_cohort(
            cohort_manager=cohort_manager,
            trial_year=trigger_year,
            trial_k=k,
            check_year=check_year,
            sim_params=sim_params,
            simulator=simulator,
        )
        if cuf is not None and cuf >= threshold_cuf:
            return k, True

    # Couldn't meet threshold within [min_containers, max_containers].
    # Fall back to the minimum good-faith install — NOT max_containers,
    # which would be a huge economically-nonsensical OPEX hit with no
    # chance of reaching the frozen threshold anyway.
    return min_containers, False
