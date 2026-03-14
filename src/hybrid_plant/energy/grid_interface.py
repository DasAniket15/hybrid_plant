"""
grid_interface.py
─────────────────
Computes the blended HT/LT grid loss factor and applies it to plant
export to derive meter delivery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import PERCENT_TO_DECIMAL


class GridInterface:
    """
    Translates plant-busbar export (pre-loss) to client-meter delivery
    (post-loss) using a blended HT/LT loss factor.

    The loss factor is computed once at construction from ``regulatory.yaml``
    and reused for every simulation call, avoiding repeated YAML lookups.

    Attributes
    ----------
    loss_factor : float
        Fraction of busbar energy that arrives at the client meter [0–1].
    """

    def __init__(self, config: FullConfig) -> None:
        reg   = config.regulatory["regulatory"]
        split = reg["connection"]["ht_lt_split_percent"] * PERCENT_TO_DECIMAL

        ht = reg["losses"]["ht_side"]
        lt = reg["losses"]["lt_side"]

        self.loss_factor: float = (
            split       * (1 - ht["ctu_percent"]      * PERCENT_TO_DECIMAL)
                        * (1 - ht["stu_percent"]      * PERCENT_TO_DECIMAL)
                        * (1 - ht["wheeling_percent"] * PERCENT_TO_DECIMAL)
            + (1 - split) * (1 - lt["ctu_percent"]      * PERCENT_TO_DECIMAL)
                          * (1 - lt["stu_percent"]      * PERCENT_TO_DECIMAL)
                          * (1 - lt["wheeling_percent"] * PERCENT_TO_DECIMAL)
        )

    def apply_losses(self, plant_export_pre: np.ndarray) -> dict[str, Any]:
        """
        Scale busbar export by the loss factor to produce meter delivery.

        Parameters
        ----------
        plant_export_pre : np.ndarray
            Hourly plant export at the busbar (MWh), shape (8760,).

        Returns
        -------
        dict
            meter_delivery        : np.ndarray  (8760,) MWh at client meter
            annual_meter_delivery : float       sum of meter_delivery
            loss_factor           : float       the blended loss factor used
        """
        meter = plant_export_pre * self.loss_factor
        return {
            "meter_delivery":        meter,
            "annual_meter_delivery": float(np.sum(meter)),
            "loss_factor":           self.loss_factor,
        }
