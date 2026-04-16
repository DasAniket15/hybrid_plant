"""
cuf_evaluator.py
────────────────
Single source of truth for the Plant CUF formula.

Plant CUF definition (from print_section3 in run_model.py)
──────────────────────────────────────────────────────────
CUF is computed by re-running dispatch with loss_factor=1.0 (eliminates
grid losses so the numerator is the pure plant output, not net meter delivery).
The numerator sums hourly min(plant_export_pre + curtailment_pre, ppa_mw)
— this captures both what the plant actually exported and what was curtailed
due to the PPA cap, giving a fair picture of plant generation capability.

Why loss_factor=1.0?
    The PPA contract specifies a capacity in MW; CUF measures how much of
    that contracted capacity was actually utilised by the plant.  Including
    grid losses in the denominator would conflate plant performance with
    transmission infrastructure quality.  Rerunning with loss_factor=1.0
    removes the loss artefact from both numerator (via re-dispatch) and
    implicitly from the denominator (PPA MW × 8760 is grid-loss-independent).

This function is used in:
  • run_model.print_section3  — dashboard CUF display (via refactored call)
  • AugmentationEngine        — trigger threshold from baseline Year-1
  • LifecycleSimulator        — per-year CUF check + restoration target
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.constants import HOURS_PER_YEAR


def compute_plant_cuf(
    plant_engine: Any,
    params:       dict[str, Any],
    bess_containers:  int,
    bess_soh_factor:  float,
) -> float:
    """
    Compute Plant CUF using the canonical formula.

    Re-runs the full 8760-hour dispatch with ``loss_factor=1.0`` and
    the supplied ``bess_soh_factor``, then computes:

        numerator = Σ min(plant_export_pre[h] + curtailment_pre[h], ppa_mw)
        CUF (%)   = numerator / (ppa_mw × 8760) × 100

    Parameters
    ----------
    plant_engine     : PlantEngine — the shared plant simulation instance
    params           : dict — must contain ``solar_capacity_mw``,
                       ``wind_capacity_mw``, ``ppa_capacity_mw``,
                       ``charge_c_rate``, ``discharge_c_rate``,
                       ``dispatch_priority``, ``bess_charge_source``
    bess_containers  : int — container count to pass to the simulation
                       (may differ from params["bess_containers"] when
                       evaluating post-augmentation states)
    bess_soh_factor  : float — blended SOH factor for the container set

    Returns
    -------
    float — Plant CUF as a percentage (e.g. 38.45 for 38.45 %)

    Notes
    -----
    The ``loss_factor=1.0`` re-run uses the same containers and C-rates
    as the actual Year-1 dispatch but strips out grid losses.  This ensures
    CUF reflects plant capability, not grid infrastructure quality.
    """
    ppa_mw = float(params["ppa_capacity_mw"])

    sim = plant_engine.simulate(
        solar_capacity_mw  = params["solar_capacity_mw"],
        wind_capacity_mw   = params["wind_capacity_mw"],
        bess_containers    = bess_containers,
        charge_c_rate      = params["charge_c_rate"],
        discharge_c_rate   = params["discharge_c_rate"],
        ppa_capacity_mw    = ppa_mw,
        dispatch_priority  = params["dispatch_priority"],
        bess_charge_source = params["bess_charge_source"],
        loss_factor        = 1.0,
        bess_soh_factor    = bess_soh_factor,
    )

    numerator = float(np.sum(
        np.minimum(
            sim["plant_export_pre"] + sim["curtailment_pre"],
            ppa_mw,
        )
    ))

    return numerator / (ppa_mw * HOURS_PER_YEAR) * 100.0
