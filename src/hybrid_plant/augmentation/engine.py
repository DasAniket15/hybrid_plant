"""
engine.py
─────────
Top-level BESS augmentation optimiser.

Consumes the base optimiser's solution (a ``params`` dict plus a
Year1Engine result) and returns an augmented lifecycle solution under one
of four modes:

  1. ``DISABLED`` — Return the base lifecycle with no augmentation
     (EnergyProjection-equivalent behaviour; provided for parity).
  2. ``TRIGGERED`` — CUF-based augmentation on top of the base initial
     BESS size; no oversizing.
  3. ``FIXED`` — User-specified augmentation years; each triggered install
     uses the configured minimum container count.
  4. ``OPTIMIZED`` (default) — Explore initial oversizing levels, for each
     run the CUF-triggered lifecycle, pick the candidate with maximum
     client savings NPV.

Threshold CUF
─────────────
Computed ONCE from the base optimiser's Year-1 plant CUF and held constant
across every candidate evaluated downstream. This is a strict rule of the
design — altering the threshold between candidates would let the engine
self-justify any sizing choice.

Oversizing search (Optimized mode)
──────────────────────────────────
We scan n = 0, 1, 2, … additional containers on top of the base solver's
choice. Stop when either:
  (a) Adding a container reduces client savings vs. the previous step
      (balanced-heuristic termination), OR
  (b) No augmentation is triggered for a candidate (CUF never drops
      below threshold — further oversizing is pure OPEX waste).

A small "patience" is applied to (a) to absorb a single non-improving
step caused by integer-discretisation noise.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

import numpy as np

from hybrid_plant.augmentation.lifecycle import (
    LifecycleSimulator,
    plant_cuf_from_busbar,
)
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.energy.year1_engine import Year1Engine


class AugmentationMode(str, Enum):
    """
    Engine dispatch modes.

    DISABLED  — skip augmentation entirely; report the base lifecycle.
    TRIGGERED — use the base initial size, augment by CUF trigger only.
    FIXED     — augment in the user-specified years (minimum containers).
    OPTIMIZED — full search over oversizing + CUF-triggered augmentation.
    """
    DISABLED  = "disabled"
    TRIGGERED = "triggered"
    FIXED     = "fixed"
    OPTIMIZED = "optimized"


# ─────────────────────────────────────────────────────────────────────────────

class AugmentationEngine:
    """
    Orchestrates the augmentation analysis on top of a base optimiser
    solution.

    Parameters
    ----------
    config       : FullConfig
    data         : dict
    year1_engine : Year1Engine

    The engine instantiates its own ``LifecycleSimulator`` so callers only
    need the three standard handles from the pipeline bootstrap.
    """

    def __init__(
        self,
        config:       FullConfig,
        data:         dict[str, Any],
        year1_engine: Year1Engine,
    ) -> None:
        self._config = config
        self._data   = data
        self._year1  = year1_engine
        self._sim    = LifecycleSimulator(config, data, year1_engine)

        aug_cfg = config.bess["bess"].get("augmentation", {}) or {}
        self._enabled: bool = bool(aug_cfg.get("enabled", True))
        self._min_aug_containers: int = int(aug_cfg.get("minimum_augmentation_containers", 1))

        # Oversizing search bounds — safe defaults; the CUF trigger usually
        # terminates long before this cap is reached.
        self._max_oversize_steps: int = int(aug_cfg.get("max_oversize_containers", 60))
        # Patience for the "savings decreased" stop rule. Set to 1 so we
        # allow one non-improving step to absorb integer noise before
        # declaring the search done.
        self._patience: int = int(aug_cfg.get("oversize_patience", 1))

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def simulator(self) -> LifecycleSimulator:
        return self._sim

    # ─────────────────────────────────────────────────────────────────────────

    def compute_threshold_cuf(
        self,
        sim_params: dict[str, Any],
        initial_containers: int,
    ) -> float:
        """
        Freeze the Year-1 plant CUF for the base initial size. This value
        is the constant threshold used across all modes and all oversizing
        candidates.

        The computation uses year 1 with no degradation (SoH = 1.0 via the
        container-count × container-size × SoH formula — the base-config
        Year-1 SoH is 0.9953 in the supplied CSV, matching the existing
        EnergyProjection semantics for Year 1). We intentionally run one
        full lifecycle year so this threshold is comparable with the yearly
        CUFs produced by the lifecycle simulator.
        """
        cm = self._sim.make_cohort_manager(int(initial_containers))
        yr = self._sim.simulate_year(sim_params, year=1, cm=cm)
        busbar = float(
            np.sum(yr["solar_direct_pre"])
            + np.sum(yr["wind_direct_pre"])
            + np.sum(yr["discharge_pre"])
        )
        cuf = plant_cuf_from_busbar(busbar, sim_params["ppa_capacity_mw"])
        if cuf is None:
            raise ValueError(
                "Cannot compute threshold CUF: PPA capacity is zero or negative."
            )
        return float(cuf)

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        sim_params:         dict[str, Any],
        base_initial_containers: int,
        mode:               AugmentationMode | str = AugmentationMode.OPTIMIZED,
        augmentation_years: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """
        Top-level entry point.

        Parameters
        ----------
        sim_params : dict
            Plant simulation parameters (from Year1Engine.evaluate()'s
            ``sim_params`` key). Must include all dispatch keys and
            ``loss_factor``.
        base_initial_containers : int
            The base optimiser's chosen BESS container count — the anchor
            for the threshold CUF and the starting point of the oversizing
            search.
        mode : AugmentationMode or str
            One of "disabled" / "triggered" / "fixed" / "optimized".
        augmentation_years : iterable[int] or None
            Required when mode == "fixed". Ignored otherwise.

        Returns
        -------
        dict
            mode                       : str
            enabled                    : bool
            threshold_cuf              : float | None
            best_initial_containers    : int
            best_initial_capacity_mwh  : float
            augmentation_events        : list[dict]
            total_augmentation_mwh     : float
            lifecycle_result           : dict  — full per-year diagnostics
            finance_result             : dict  — full pipeline output
            scenarios                  : list[dict]  — one per oversizing step (optimized mode) or one (other modes)
        """
        # Normalise mode
        if isinstance(mode, str):
            try:
                mode = AugmentationMode(mode.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown augmentation mode: {mode!r}. "
                    f"Valid: {[m.value for m in AugmentationMode]}"
                )

        # Respect the bess.yaml master switch — "enabled: false" forces DISABLED.
        effective_mode = mode if self._enabled else AugmentationMode.DISABLED

        if effective_mode == AugmentationMode.DISABLED:
            return self._run_single_scenario(
                sim_params=sim_params,
                initial_containers=int(base_initial_containers),
                threshold_cuf=None,
                augmentation_years=None,
                mode_label="disabled",
            )

        # Freeze threshold from the base (un-oversized) initial size
        threshold_cuf = self.compute_threshold_cuf(
            sim_params=sim_params,
            initial_containers=int(base_initial_containers),
        )

        if effective_mode == AugmentationMode.TRIGGERED:
            result = self._run_single_scenario(
                sim_params=sim_params,
                initial_containers=int(base_initial_containers),
                threshold_cuf=threshold_cuf,
                augmentation_years=None,
                mode_label="triggered",
            )
            return result

        if effective_mode == AugmentationMode.FIXED:
            if not augmentation_years:
                raise ValueError(
                    "AugmentationMode.FIXED requires a non-empty augmentation_years list."
                )
            # Fixed mode does NOT use the CUF trigger — purely user-driven schedule.
            result = self._run_single_scenario(
                sim_params=sim_params,
                initial_containers=int(base_initial_containers),
                threshold_cuf=None,
                augmentation_years=list(int(y) for y in augmentation_years),
                mode_label="fixed",
            )
            result["threshold_cuf"] = threshold_cuf  # still report for reference
            return result

        # OPTIMIZED — oversizing loop + CUF-triggered augmentation
        return self._run_optimized(
            sim_params=sim_params,
            base_initial=int(base_initial_containers),
            threshold_cuf=threshold_cuf,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _run_single_scenario(
        self,
        sim_params:         dict[str, Any],
        initial_containers: int,
        threshold_cuf:      float | None,
        augmentation_years: list[int] | None,
        mode_label:         str,
    ) -> dict[str, Any]:
        """Run one lifecycle + one finance evaluation and wrap the output."""
        lifecycle = self._sim.run_lifecycle(
            sim_params=sim_params,
            initial_containers=initial_containers,
            threshold_cuf=threshold_cuf,
            augmentation_years=augmentation_years,
        )
        finance = self._sim.run_finance(
            sim_params=sim_params,
            initial_containers=initial_containers,
            lifecycle_result=lifecycle,
        )

        total_aug_mwh = float(sum(e["mwh"] for e in lifecycle["augmentation_events"]))
        initial_mwh   = float(initial_containers) * self._sim.container_size

        return {
            "mode":                      mode_label,
            "enabled":                   self._enabled,
            "threshold_cuf":             threshold_cuf,
            "best_initial_containers":   initial_containers,
            "best_initial_capacity_mwh": initial_mwh,
            "augmentation_events":       lifecycle["augmentation_events"],
            "total_augmentation_mwh":    total_aug_mwh,
            "lifecycle_result":          lifecycle,
            "finance_result":            finance,
            "scenarios":                 [
                {
                    "initial_containers":  initial_containers,
                    "oversize_delta":      0,
                    "savings_npv":         finance["savings_npv"],
                    "augmentations":       len(lifecycle["augmentation_events"]),
                    "total_augmentation_mwh": total_aug_mwh,
                }
            ],
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _run_optimized(
        self,
        sim_params:     dict[str, Any],
        base_initial:   int,
        threshold_cuf:  float,
    ) -> dict[str, Any]:
        """
        Scan n = 0 … max_oversize_steps additional containers on top of
        ``base_initial``. For each candidate, run a full CUF-triggered
        lifecycle and record its savings NPV. Pick the best.

        Termination:
          (a) non-improving savings for ``patience`` consecutive steps, OR
          (b) no augmentation events were triggered (CUF stays above
              threshold for the whole horizon), OR
          (c) max_oversize_steps reached.
        """
        scenarios: list[dict[str, Any]] = []
        best:      dict[str, Any] | None = None
        best_savings: float = -float("inf")
        non_improving_streak: int = 0

        for n in range(0, self._max_oversize_steps + 1):
            initial = base_initial + n

            lifecycle = self._sim.run_lifecycle(
                sim_params=sim_params,
                initial_containers=initial,
                threshold_cuf=threshold_cuf,
            )
            finance = self._sim.run_finance(
                sim_params=sim_params,
                initial_containers=initial,
                lifecycle_result=lifecycle,
            )
            savings = float(finance["savings_npv"])
            n_aug   = len(lifecycle["augmentation_events"])
            total_aug_mwh = float(sum(e["mwh"] for e in lifecycle["augmentation_events"]))

            scenario = {
                "initial_containers":      initial,
                "oversize_delta":          n,
                "savings_npv":             savings,
                "augmentations":           n_aug,
                "total_augmentation_mwh":  total_aug_mwh,
            }
            scenarios.append(scenario)

            # Track the best
            improved = savings > best_savings
            if improved:
                best_savings = savings
                best = {
                    "initial":   initial,
                    "lifecycle": lifecycle,
                    "finance":   finance,
                }
                non_improving_streak = 0
            else:
                non_improving_streak += 1

            # Stop condition (b): no augmentation triggered — further
            # oversizing only adds waste OPEX via standard BESS O&M (more
            # containers) with no CUF motivation.
            if n_aug == 0:
                break

            # Stop condition (a): savings have been non-improving for
            # ``patience`` consecutive steps after at least one improvement.
            if non_improving_streak > self._patience:
                break

        assert best is not None, "Oversizing search did not evaluate any candidate."

        best_initial   = int(best["initial"])
        best_lifecycle = best["lifecycle"]
        best_finance   = best["finance"]
        best_total_aug = float(sum(e["mwh"] for e in best_lifecycle["augmentation_events"]))

        return {
            "mode":                      "optimized",
            "enabled":                   self._enabled,
            "threshold_cuf":             threshold_cuf,
            "best_initial_containers":   best_initial,
            "best_initial_capacity_mwh": best_initial * self._sim.container_size,
            "augmentation_events":       best_lifecycle["augmentation_events"],
            "total_augmentation_mwh":    best_total_aug,
            "lifecycle_result":          best_lifecycle,
            "finance_result":            best_finance,
            "scenarios":                 scenarios,
        }
