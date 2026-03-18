"""
meter_layer.py
──────────────
Computes hourly DISCOM shortfall — the portion of client load not covered
by RE delivery at the meter.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MeterLayer:
    """
    Derives the DISCOM draw (shortfall) from the difference between client
    load and RE meter delivery.

    Parameters
    ----------
    data : dict
        Must contain ``load_profile`` as an ``np.ndarray`` of shape (8760,)
        in MWh.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.load: np.ndarray = data["load_profile"]

    def compute_shortfall(self, meter_delivery: np.ndarray) -> dict[str, Any]:
        """
        Compute hourly DISCOM shortfall.

        Parameters
        ----------
        meter_delivery : np.ndarray
            Hourly RE energy delivered at the client meter (MWh), shape (8760,).

        Returns
        -------
        dict
            shortfall      : np.ndarray  max(load − delivery, 0) per hour
            annual_discom  : float       total annual DISCOM draw (MWh)
        """
        shortfall = np.maximum(self.load - meter_delivery, 0)
        return {
            "shortfall":     shortfall,
            "annual_discom": float(np.sum(shortfall)),
        }