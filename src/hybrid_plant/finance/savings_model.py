"""
savings_model.py
────────────────
Computes client electricity cost savings versus a 100 % DISCOM baseline.

Baseline cost
─────────────
    baseline = annual_load_kWh × weighted-average DISCOM ToD tariff

    The weighted-average tariff is a blended figure:
        blended = (lt_fraction × lt_avg) + (ht_fraction × ht_avg)
    where lt_fraction and ht_fraction come from ``ht_lt_split_percent`` in
    ``regulatory.yaml`` and lt_avg / ht_avg are hour-count-weighted averages
    over each voltage level's tod_periods in ``tariffs.yaml``.

Hybrid cost (per year t)
────────────────────────
    hybrid_t = (RE_meter_kWh_t × landed_tariff_t)
             + (DISCOM_draw_kWh_t × discom_tariff)

    where  DISCOM_draw_kWh_t = annual_load_kWh − RE_meter_kWh_t

Savings
───────
    savings_t = baseline − hybrid_t

Savings NPV
───────────
    NPV discounted at WACC using Excel-style convention (t = 1 … project_life).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import MWH_TO_KWH


class SavingsModel:
    """
    Calculates annual savings and their NPV against the 100 % DISCOM baseline.

    Parameters
    ----------
    config : FullConfig
    data   : dict  — must contain ``load_profile`` (np.ndarray, MWh)
    """

    def __init__(self, config: FullConfig, data: dict[str, Any]) -> None:
        self._project_life = config.project["project"]["project_life_years"]
        self._discom_tariff = self._weighted_discom_tariff(config)
        self._annual_load_kwh = float(np.sum(data["load_profile"])) * MWH_TO_KWH
        self._baseline_cost   = self._annual_load_kwh * self._discom_tariff

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _period_weighted_avg(periods: dict) -> float:
        """
        Hour-count-weighted average rate across a single voltage level's
        tod_periods dict.

        Returns
        -------
        float  Rs / kWh
        """
        total_weighted = sum(p["rate_inr_per_kwh"] * len(p["hours"]) for p in periods.values())
        total_hours    = sum(len(p["hours"]) for p in periods.values())
        return total_weighted / total_hours

    @staticmethod
    def _weighted_discom_tariff(config: FullConfig) -> float:
        """
        Compute the blended load-weighted average DISCOM ToD tariff.

        The blend is driven by ``ht_lt_split_percent`` in ``regulatory.yaml``:
            blended = (lt_fraction × lt_avg) + (ht_fraction × ht_avg)

        With ``ht_lt_split_percent = 0`` (100 % LT), the result equals the
        pure LT weighted average.

        Returns
        -------
        float  Rs / kWh
        """
        ht_fraction = (
            config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"] / 100.0
        )
        lt_fraction = 1.0 - ht_fraction

        lt_avg = SavingsModel._period_weighted_avg(
            config.tariffs["discom"]["lt"]["tod_periods"]
        )
        ht_avg = SavingsModel._period_weighted_avg(
            config.tariffs["discom"]["ht"]["tod_periods"]
        )
        return lt_fraction * lt_avg + ht_fraction * ht_avg

    def _npv(self, series: list[float], wacc: float) -> float:
        """Excel-style NPV: series[0] → Year 1, discounted at t = 1."""
        return sum(v / (1 + wacc) ** (t + 1) for t, v in enumerate(series))

    # ─────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        landed_tariff_series:        list[float],
        meter_energy_mwh_projection: Any,
        wacc:                        float,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        landed_tariff_series        : list[float]  annual landed tariff (Rs/kWh)
        meter_energy_mwh_projection : array-like   annual RE meter energy (MWh)
        wacc                        : float         discount rate (decimal)

        Returns
        -------
        dict
            annual_savings_year1 : float
            savings_npv          : float
            annual_savings       : list[float]  full 25-year series
            + supporting cost series and scalars for reporting
        """
        annual_savings:     list[float] = []
        annual_hybrid_cost: list[float] = []
        annual_re_cost:     list[float] = []
        annual_discom_cost: list[float] = []

        for landed_t, meter_mwh_t in zip(landed_tariff_series, meter_energy_mwh_projection):
            re_kwh_t     = float(meter_mwh_t) * MWH_TO_KWH
            discom_kwh_t = self._annual_load_kwh - re_kwh_t

            re_cost_t     = re_kwh_t     * landed_t
            discom_cost_t = discom_kwh_t * self._discom_tariff
            hybrid_t      = re_cost_t + discom_cost_t
            savings_t     = self._baseline_cost - hybrid_t

            annual_savings.append(savings_t)
            annual_hybrid_cost.append(hybrid_t)
            annual_re_cost.append(re_cost_t)
            annual_discom_cost.append(discom_cost_t)

        savings_npv = self._npv(annual_savings, wacc)

        return {
            "annual_savings_year1":  annual_savings[0],
            "savings_npv":           savings_npv,
            "annual_savings":        annual_savings,
            "baseline_annual_cost":  self._baseline_cost,
            "annual_hybrid_cost":    annual_hybrid_cost,
            "annual_re_cost":        annual_re_cost,
            "annual_discom_cost":    annual_discom_cost,
            "discom_tariff":         self._discom_tariff,
            "annual_load_kwh":       self._annual_load_kwh,
        }