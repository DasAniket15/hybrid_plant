"""
landed_tariff_model.py
──────────────────────
Computes the annual landed tariff series (Rs/kWh) — the all-in RE cost
delivered to the client meter.

    landed_tariff_t = total_annual_cost_t / meter_kwh_t

where total_annual_cost_t =
      RE payment         (LCOE × busbar kWh)
    + capacity charges   (CTU + STU + SLDC on PPA MW × 12 months)
    + wheeling charges   (Rs/kWh × meter kWh)
    + electricity tax    (Rs/kWh × meter kWh)
    + banking charges    (Rs/kWh × banked kWh)   ← stub = 0

Charge bases
────────────
  Capacity-based (Rs/MW/month) applied on contracted PPA capacity:
      CTU, STU, SLDC
  Energy-based (Rs/kWh) applied on RE delivered at client meter:
      Wheeling, Electricity tax
  Banking-based (Rs/kWh) applied on banked energy:
      Banking charge  (stub = 0)

All rates are weighted by HT/LT split from ``regulatory.yaml`` and sourced
from ``finance.yaml`` — nothing is hardcoded.
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import MONTHS_PER_YEAR, MWH_TO_KWH, PERCENT_TO_DECIMAL


class LandedTariffModel:
    """
    Converts absolute annual cost components into a per-kWh landed tariff
    series across the project lifetime.

    Parameters
    ----------
    config : FullConfig
    """

    def __init__(self, config: FullConfig) -> None:
        rc  = config.finance["regulatory_charges"]
        split = (
            config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]
            * PERCENT_TO_DECIMAL
        )
        ht_frac = split
        lt_frac = 1.0 - split

        ht, lt = rc["ht"], rc["lt"]

        # ── Capacity-based rates (Rs/MW/month) ───────────────────────────────
        self._ctu_per_mw_month  = ht_frac * ht["ctu_charge_inr_per_mw_per_month"]  + lt_frac * lt["ctu_charge_inr_per_mw_per_month"]
        self._stu_per_mw_month  = ht_frac * ht["stu_charge_inr_per_mw_per_month"]  + lt_frac * lt["stu_charge_inr_per_mw_per_month"]
        self._sldc_per_mw_month = ht_frac * ht["sldc_charge_inr_per_mw_per_month"] + lt_frac * lt["sldc_charge_inr_per_mw_per_month"]

        # ── Energy-based rates (Rs/kWh) ───────────────────────────────────────
        self._wheeling_per_kwh = ht_frac * ht["wheeling_charge_inr_per_kwh"] + lt_frac * lt["wheeling_charge_inr_per_kwh"]
        self._elec_tax_per_kwh = ht_frac * ht["electricity_tax_inr_per_kwh"] + lt_frac * lt["electricity_tax_inr_per_kwh"]
        self._banking_per_kwh  = ht_frac * ht["banking_charge_inr_per_kwh"]  + lt_frac * lt["banking_charge_inr_per_kwh"]

    # ─────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        lcoe_inr_per_kwh:             float,
        ppa_capacity_mw:              float,
        busbar_energy_mwh_projection: Any,
        meter_energy_mwh_projection:  Any,
        banked_energy_kwh_projection: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        lcoe_inr_per_kwh             : float  — from LCOEModel
        ppa_capacity_mw              : float  — basis for capacity charges
        busbar_energy_mwh_projection : array-like, length = project_life
        meter_energy_mwh_projection  : array-like, length = project_life
        banked_energy_kwh_projection : list[float] or None — defaults to zeros

        Returns
        -------
        dict
            landed_tariff_series : list[float]   primary output (Rs/kWh)
            + per-component annual cost series and unit rates
        """
        n = len(meter_energy_mwh_projection)
        if banked_energy_kwh_projection is None:
            banked_energy_kwh_projection = [0.0] * n

        annual_capacity_rs = (
            (self._ctu_per_mw_month + self._stu_per_mw_month + self._sldc_per_mw_month)
            * ppa_capacity_mw
            * MONTHS_PER_YEAR
        )

        landed_series:     list[float] = []
        re_payment_series: list[float] = []
        wheeling_series:   list[float] = []
        elec_tax_series:   list[float] = []
        banking_series:    list[float] = []
        total_cost_series: list[float] = []
        capacity_per_kwh_series: list[float] = []
        lcoe_markup_per_kwh_series: list[float] = []

        for busbar_mwh, meter_mwh, banked_kwh in zip(
            busbar_energy_mwh_projection,
            meter_energy_mwh_projection,
            banked_energy_kwh_projection,
        ):
            busbar_kwh = float(busbar_mwh) * MWH_TO_KWH
            meter_kwh  = float(meter_mwh)  * MWH_TO_KWH

            re_payment = lcoe_inr_per_kwh   * busbar_kwh
            wheeling   = self._wheeling_per_kwh * meter_kwh
            elec_tax   = self._elec_tax_per_kwh * meter_kwh
            banking    = self._banking_per_kwh  * float(banked_kwh)

            total = re_payment + annual_capacity_rs + wheeling + elec_tax + banking
            landed = total / meter_kwh if meter_kwh > 0 else 0.0

            # Decompose the landed tariff build-up:
            #   capacity_per_kwh = the TRUE capacity charge expressed per meter kWh
            #   lcoe_markup      = LCOE × (busbar/meter - 1) — the extra paid per meter
            #                      kWh because LCOE applies to busbar but landed
            #                      is measured on meter (grid-loss markup)
            # Together: landed = LCOE + wheeling + elec_tax + banking_per_kwh
            #                  + capacity_per_kwh + lcoe_markup
            cap_per_kwh    = annual_capacity_rs / meter_kwh if meter_kwh > 0 else 0.0
            lcoe_markup    = (
                lcoe_inr_per_kwh * (busbar_kwh / meter_kwh - 1)
                if meter_kwh > 0 else 0.0
            )

            landed_series.append(landed)
            re_payment_series.append(re_payment)
            wheeling_series.append(wheeling)
            elec_tax_series.append(elec_tax)
            banking_series.append(banking)
            total_cost_series.append(total)
            capacity_per_kwh_series.append(cap_per_kwh)
            lcoe_markup_per_kwh_series.append(lcoe_markup)

        return {
            # Primary
            "landed_tariff_series":      landed_series,
            # Annual cost components (Rs)
            "annual_re_payment":         re_payment_series,
            "annual_capacity_charge_rs": annual_capacity_rs,
            "annual_wheeling":           wheeling_series,
            "annual_electricity_tax":    elec_tax_series,
            "annual_banking":            banking_series,
            "annual_total_cost":         total_cost_series,
            # Per-kWh decomposition of the landed tariff (useful for dashboards)
            "capacity_charge_per_kwh_series":  capacity_per_kwh_series,
            "lcoe_markup_per_kwh_series":      lcoe_markup_per_kwh_series,
            # Unit rates (for audit)
            "ctu_per_mw_month":          self._ctu_per_mw_month,
            "stu_per_mw_month":          self._stu_per_mw_month,
            "sldc_per_mw_month":         self._sldc_per_mw_month,
            "wheeling_per_kwh":          self._wheeling_per_kwh,
            "electricity_tax_per_kwh":   self._elec_tax_per_kwh,
            "banking_per_kwh":           self._banking_per_kwh,
        }