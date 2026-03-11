class OpexModel:
    """
    Computes annual OPEX projection across the project lifetime.

    Components:
        - Solar O&M         : Rs Lakh/DC MWp,  escalates 2% YoY
        - Wind O&M          : Rs Lakh/MW,       escalates 2% YoY
        - BESS O&M          : Rs Lakh/MWh,      no escalation
        - Solar Trans. O&M  : Rs Lakh/DC MWp,   no escalation
        - Wind Trans. O&M   : Rs Lakh/MW,        no escalation
        - Land Lease        : Rs Crore/month,    escalates 3% YoY
        - Insurance         : % of total CAPEX,  no escalation

    All rates are sourced from finance.yaml opex section.
    Nothing is hardcoded.

    Returns:
        opex_projection  : list[float], total annual OPEX per year (Rs), length = project_life
        opex_breakdown   : list[dict],  per-component annual breakdown
    """

    def __init__(self, config):
        self.opex_cfg     = config.finance["opex"]
        self.project_life = config.project["project"]["project_life_years"]
        self.ac_dc_ratio  = config.finance["capex"]["solar"]["ac_dc_ratio"]

    # ------------------------------------------------------------------

    def compute(
        self,
        solar_capacity_mw,
        wind_capacity_mw,
        bess_energy_mwh,
        total_capex,
    ):
        ac_dc    = self.ac_dc_ratio
        solar_dc = solar_capacity_mw * ac_dc          # DC MWp

        cfg = self.opex_cfg

        # -------------------------------------------------------
        # Conversion helpers
        # -------------------------------------------------------
        # 1 Lakh  = 1e5 Rs
        # 1 Crore = 1e7 Rs

        LAKH  = 1e5
        CRORE = 1e7

        # -------------------------------------------------------
        # Base (Year-1) values for escalating components
        # -------------------------------------------------------
        solar_om_base = (
            cfg["solar"]["rate_lakh_per_mwp"] * LAKH * solar_dc
        )
        wind_om_base = (
            cfg["wind"]["rate_lakh_per_mw"] * LAKH * wind_capacity_mw
        )
        land_lease_base = (
            cfg["land_lease"]["base_monthly_cost_crore"] * CRORE * 12   # annual
        )

        # -------------------------------------------------------
        # Fixed (non-escalating) components
        # -------------------------------------------------------
        bess_om = (
            cfg["bess"]["rate_lakh_per_mwh"] * LAKH * bess_energy_mwh
        )
        solar_trans_om = (
            cfg["solar_transmission"]["rate_lakh_per_mwp"] * LAKH * solar_dc
        )
        wind_trans_om = (
            cfg["wind_transmission"]["rate_lakh_per_mw"] * LAKH * wind_capacity_mw
        )
        insurance = (
            cfg["insurance"]["percent_of_total_capex"] / 100 * total_capex
        )

        # -------------------------------------------------------
        # Escalation rates
        # -------------------------------------------------------
        solar_esc      = cfg["solar"]["escalation_percent"]      / 100
        wind_esc       = cfg["wind"]["escalation_percent"]       / 100
        land_lease_esc = cfg["land_lease"]["escalation_percent"] / 100

        # -------------------------------------------------------
        # Build annual projection
        # -------------------------------------------------------
        opex_projection = []
        opex_breakdown  = []

        for year in range(1, self.project_life + 1):

            factor_solar = (1 + solar_esc)      ** (year - 1)
            factor_wind  = (1 + wind_esc)       ** (year - 1)
            factor_land  = (1 + land_lease_esc) ** (year - 1)

            solar_om_yr      = solar_om_base  * factor_solar
            wind_om_yr       = wind_om_base   * factor_wind
            land_lease_yr    = land_lease_base * factor_land

            total_yr = (
                solar_om_yr
                + wind_om_yr
                + bess_om
                + solar_trans_om
                + wind_trans_om
                + land_lease_yr
                + insurance
            )

            opex_projection.append(total_yr)

            opex_breakdown.append({
                "year":                 year,
                "solar_om":             solar_om_yr,
                "wind_om":              wind_om_yr,
                "bess_om":              bess_om,
                "solar_transmission_om": solar_trans_om,
                "wind_transmission_om": wind_trans_om,
                "land_lease":           land_lease_yr,
                "insurance":            insurance,
                "total":                total_yr,
            })

        return opex_projection, opex_breakdown