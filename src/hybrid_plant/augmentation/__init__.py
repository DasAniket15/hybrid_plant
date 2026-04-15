"""
augmentation
────────────
BESS Augmentation Engine.

Simulates BESS degradation over the project lifetime using an independent
cohort model (each augmentation creates a new cohort that degrades from its
own installation age), triggers augmentation events when plant CUF drops
below the Year-1 threshold, and optimises the initial BESS oversizing and
augmentation sizing/timing to maximise 25-year client savings NPV.

Non-negotiable design rules
───────────────────────────
  • Cohort-based degradation — no single averaged battery
  • Container-discrete sizing — no fractional containers
  • Augmentation cost is OPEX in the year of installation — never CAPEX,
    never capitalised, never depreciated
  • CUF threshold is frozen from the base optimiser's Year-1 plant CUF
  • Initial oversizing is a top-level decision variable

Public API
──────────
  AugmentationEngine           Top-level entry point (4 modes)
  BESSCohort, CohortManager    Cohort data model
  LifecycleSimulator           25-year cohort-aware plant + finance runner
  find_augmentation_size       Smallest-k container search given a CUF target
"""

from hybrid_plant.augmentation.cohort import BESSCohort, CohortManager
from hybrid_plant.augmentation.engine import AugmentationEngine, AugmentationMode
from hybrid_plant.augmentation.lifecycle import LifecycleSimulator
from hybrid_plant.augmentation.sizing import find_augmentation_size

__all__ = [
    "AugmentationEngine",
    "AugmentationMode",
    "BESSCohort",
    "CohortManager",
    "LifecycleSimulator",
    "find_augmentation_size",
]
