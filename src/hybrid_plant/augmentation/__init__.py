"""
augmentation
────────────
BESS Augmentation Engine sub-package.

Simulates BESS capacity degradation over the 25-year project life,
schedules augmentation events (adding fresh containers) when Plant CUF
falls below a threshold, and folds the result into the finance pipeline.

Public API
──────────
  BESSCohort          — immutable cohort descriptor
  CohortRegistry      — mutable cohort manager
  compute_plant_cuf   — canonical CUF formula
  LifecycleSimulator  — per-scenario 25-year simulation
  AugmentationEngine  — top-level facade used by solver and dashboard
"""

from __future__ import annotations

from hybrid_plant.augmentation.cohort import BESSCohort, CohortRegistry
from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf
from hybrid_plant.augmentation.lifecycle_simulator import LifecycleResult, LifecycleSimulator
from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine

__all__ = [
    "BESSCohort",
    "CohortRegistry",
    "compute_plant_cuf",
    "LifecycleResult",
    "LifecycleSimulator",
    "AugmentationEngine",
]
