"""
opex_model.py
─────────────
Projects annual OPEX across the full project lifetime.

Components and escalation
─────────────────────────
  Solar O&M         Rs Lakh / DC MWp      escalates at solar_esc % YoY
  Wind O&M          Rs Lakh / MW          escalates at wind_esc % YoY
  Land lease        Rs Crore / month      escalates at land_esc % YoY
  BESS O&M          Rs Lakh / MWh         no escalation
  Solar trans. O&M  Rs Lakh / DC MWp      no escalation
  Wind trans. O&M   Rs Lakh / MW          no escalation
  Insurance         % of total CAPEX      no escalation

All rates are sourced from ``finance.yaml`` — nothing is hardcoded.
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import CRORE_TO_RS, LAKH_TO_RS, MONTHS_PER_YEAR, PERCENT_TO_DECIMAL


class OpexModel:
    """
    Computes a 25-year annual OPEX projection with per-component detail.

    Parameters
    ----------
    config : FullConfig
    """

    def __init__(self, config: FullConfig) -> None:
        self._cfg          = config.finance["opex"]
        self._project_life = config.project["project"]["project_life_years"]
        self._ac_dc_ratio  = config.finance["capex"]["solar"]["ac_dc_ratio"]

    def compute(
        self,
        solar_capacity_mw:    float,
        wind_capacity_mw:     float,
        bess_energy_mwh:      float,
        total_capex:          float,
        augmentation_result:  dict | None = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """
        Parameters
        ----------
        solar_capacity_mw   : AC solar installed capacity (MW)
        wind_capacity_mw    : Wind installed capacity (MW)
        bess_energy_mwh     : Initial Year-1 BESS nameplate capacity (MWh)
        total_capex         : Total project CAPEX (Rs) — basis for insurance
        augmentation_result : Optional dict from AugmentationEngine.compute().
                              When provided:
                                • BESS O&M is computed on the total installed
                                  (nameplate) BESS capacity per year, which
                                  grows as cohorts are added.
                                • The one-time container purchase cost is
                                  added as OPEX in the augmentation year.

        Returns
        -------
        opex_projection : list[float]
            Total annual OPEX per year (Rs), length = project_life.
        opex_breakdown : list[dict]
            Per-component annual breakdown, length = project_life.
        """
        cfg      = self._cfg
        solar_dc = solar_capacity_mw * self._ac_dc_ratio   # DC MWp

        # ── Base (Year-1) values for escalating components ────────────────────
        solar_om_base   = cfg["solar"]["rate_lakh_per_mwp"]            * LAKH_TO_RS  * solar_dc
        wind_om_base    = cfg["wind"]["rate_lakh_per_mw"]              * LAKH_TO_RS  * wind_capacity_mw
        land_lease_base = cfg["land_lease"]["base_monthly_cost_crore"] * CRORE_TO_RS * MONTHS_PER_YEAR

        # ── Fixed (non-escalating) components ────────────────────────────────────
        solar_trans_om   = cfg["solar_transmission"]["rate_lakh_per_mwp"] * LAKH_TO_RS * solar_dc
        wind_trans_om    = cfg["wind_transmission"]["rate_lakh_per_mw"]   * LAKH_TO_RS * wind_capacity_mw
        insurance_rate   = cfg["insurance"]["percent_of_total_capex"] * PERCENT_TO_DECIMAL
        # Insurance base (initial CAPEX): fixed for all years; augmentation
        # purchases accumulate as additional insured asset value each year
        # (see insurance_yr computation inside the annual loop below).

        # ── BESS O&M rate (Rs per installed MWh per year, no escalation) ──────
        bess_om_rate_per_mwh = cfg["bess"]["rate_lakh_per_mwh"] * LAKH_TO_RS

        # ── Escalation rates ──────────────────────────────────────────────────
        solar_esc = cfg["solar"]["escalation_percent"]      * PERCENT_TO_DECIMAL
        wind_esc  = cfg["wind"]["escalation_percent"]       * PERCENT_TO_DECIMAL
        land_esc  = cfg["land_lease"]["escalation_percent"] * PERCENT_TO_DECIMAL

        # ── Augmentation data (optional) ──────────────────────────────────────
        aug_installed_mwh: dict[int, float] = {}
        aug_purchase_opex: dict[int, float] = {}
        if augmentation_result is not None:
            aug_installed_mwh = augmentation_result.get("total_installed_mwh_per_year", {})
            aug_purchase_opex = augmentation_result.get("augmentation_purchase_opex", {})

        # ── Annual projection ─────────────────────────────────────────────────
        opex_projection: list[float]          = []
        opex_breakdown:  list[dict[str, Any]] = []

        cumulative_aug_spend: float = 0.0   # running sum of aug purchase costs (Rs)

        for year in range(1, self._project_life + 1):
            solar_om_yr   = solar_om_base   * (1 + solar_esc) ** (year - 1)
            wind_om_yr    = wind_om_base    * (1 + wind_esc)  ** (year - 1)
            land_lease_yr = land_lease_base * (1 + land_esc)  ** (year - 1)

            # BESS O&M applied to total installed (nameplate) MWh this year.
            # With augmentation, this grows each time a new cohort is added.
            bess_installed_yr = aug_installed_mwh.get(year, bess_energy_mwh)
            bess_om_yr        = bess_om_rate_per_mwh * bess_installed_yr

            # One-time augmentation container purchase cost (OPEX in that year).
            aug_purchase_yr    = aug_purchase_opex.get(year, 0.0)
            # Augmentation purchase in this year increases the insured asset value
            # from the FOLLOWING year — containers are installed at end of year.
            # Insurance in the current year is based on asset value before purchase.
            insurance_yr = insurance_rate * (total_capex + cumulative_aug_spend)
            cumulative_aug_spend += aug_purchase_yr   # update AFTER computing this year's insurance

            total_yr = (
                solar_om_yr + wind_om_yr + bess_om_yr
                + solar_trans_om + wind_trans_om
                + land_lease_yr + insurance_yr
                + aug_purchase_yr
            )

            opex_projection.append(total_yr)
            opex_breakdown.append({
                "year":                  year,
                "solar_om":              solar_om_yr,
                "wind_om":               wind_om_yr,
                "bess_om":               bess_om_yr,
                "bess_installed_mwh":    bess_installed_yr,
                "augmentation_purchase": aug_purchase_yr,
                "solar_transmission_om": solar_trans_om,
                "wind_transmission_om":  wind_trans_om,
                "land_lease":            land_lease_yr,
                "insurance":             insurance_yr,
                "total":                 total_yr,
            })

        return opex_projection, opex_breakdown