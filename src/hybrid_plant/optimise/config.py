"""
optimise/config.py
──────────────────
Configuration dataclass for the Pyomo optimisation model.

Separate from FullConfig (which holds project/tariff/finance parameters):
OptModelConfig controls *how* the model is built and solved — horizon mode,
solver selection, unit scaling, and augmentation toggle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptModelConfig:
    """
    Configuration for the Pyomo MILP model build and solve.

    Parameters
    ----------
    horizon : str
        ``"single"`` — optimise over one representative year (8760 h), apply
        degradation only in the objective via D_s/D_w/D_b factors.
        ``"full"``   — optimise over all 25 × 8760 h simultaneously with
        per-year degraded capacity bounds and continuous SOC carryover.
    solver_name : str
        Pyomo solver interface name.  ``"appsi_highs"`` (default) uses the
        in-memory HiGHS API; ``"cbc"`` is the cross-validation fallback.
    scale_money : float
        Multiply INR values by this factor before passing to the solver.
        Default: 1e-7  (INR → Crore).  Inverted in ``report.py``.
    scale_energy : float
        Multiply MWh values by this factor before passing to the solver.
        Default: 1e-3  (MWh → GWh).  Inverted in ``report.py``.
    augmentation : bool
        When True, existing plant capacities are read from params and the
        model optimises *incremental* delta variables (Phase 2).
        When False (default, Phase 1 greenfield), existing = 0.
    """

    horizon:      str   = "single"
    solver_name:  str   = "appsi_highs"
    scale_money:  float = 1e-7
    scale_energy: float = 1e-3
    augmentation: bool  = False
