"""
cuf_evaluator.py
────────────────
Plant CUF — the single, simple formula used everywhere in the model.

    CUF (%) = annual busbar MWh  /  (PPA_MW × 8760)  × 100

This is the transparent "naive" formula that matches business intuition:
**"What fraction of my contracted PPA capacity did the plant actually use?"**

This metric naturally captures:
  • Solar & wind degradation   (affects busbar per year)
  • BESS degradation           (affects busbar per year)
  • Dispatch behaviour         (affects busbar per year)
  • PPA export cap             (already applied inside plant_engine)
  • Real grid loss factor      (busbar is computed with real lf)

Use this metric as:
  1. The year-over-year **Plant CUF** shown on the dashboard.
  2. The **augmentation trigger** metric — blended across all degradation
     sources (solar + wind + BESS), evaluated on actual dispatched busbar
     per year.  Trigger fires when ``cuf_t < threshold_cuf - tolerance``.

Callers are responsible for computing the year's busbar themselves (via
``Year1Engine.evaluate`` for Y1, ``EnergyProjection.project`` for a no-
augmentation projection, or ``LifecycleSimulator.simulate`` for a cohort-
aware projection).  This file just holds the formula.
"""

from __future__ import annotations

from hybrid_plant.constants import HOURS_PER_YEAR


def compute_plant_cuf(
    busbar_mwh:      float,
    ppa_capacity_mw: float,
    hours:           int = HOURS_PER_YEAR,
) -> float:
    """
    Return Plant CUF as a percentage (e.g. 84.48 for 84.48 %).

    Parameters
    ----------
    busbar_mwh       : Total annual busbar delivery (MWh) for the year.
                       Typically ``sum(solar_direct_pre) + sum(wind_direct_pre)
                       + sum(discharge_pre)`` from a PlantEngine simulation,
                       or the pre-computed ``delivered_pre_mwh[year-1]`` entry
                       from an EnergyProjection / LifecycleSimulator output.
    ppa_capacity_mw  : Contracted PPA export capacity (MW).
    hours            : Hours in the year (default 8760).

    Returns
    -------
    float — CUF as percent. Returns 0.0 if ``ppa_capacity_mw`` ≤ 0.
    """
    if ppa_capacity_mw <= 0:
        return 0.0
    return busbar_mwh / (ppa_capacity_mw * hours) * 100.0


def year1_busbar_mwh(year1_result: dict) -> float:
    """
    Convenience helper: extract Year-1 annual busbar MWh from a
    ``Year1Engine.evaluate()`` result dict.

    Parameters
    ----------
    year1_result : dict with keys ``solar_direct_pre``, ``wind_direct_pre``,
                   ``discharge_pre`` (each an np.ndarray of shape (8760,)).

    Returns
    -------
    float — total annual busbar MWh.
    """
    import numpy as np
    return float(
        np.sum(year1_result["solar_direct_pre"])
        + np.sum(year1_result["wind_direct_pre"])
        + np.sum(year1_result["discharge_pre"])
    )


def busbar_from_sim(sim: dict) -> float:
    """
    Extract busbar MWh from any PlantEngine.simulate() / Year1Engine.evaluate()
    output dict. Alias for convenience in augmentation/lifecycle code.
    """
    return year1_busbar_mwh(sim)
