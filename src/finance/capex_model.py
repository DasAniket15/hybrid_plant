class CapexModel:
    """
    Computes project CAPEX broken down by component.

    Solar CAPEX is applied on DC MWp basis (AC capacity × AC/DC ratio).
    All costs sourced from finance.yaml capex section.
    """

    def __init__(self, config):
        self.capex_cfg = config.finance["capex"]

    def compute(
        self,
        solar_capacity_mw,
        wind_capacity_mw,
        bess_energy_capacity_mwh,
    ):
        solar_dc_mwp = solar_capacity_mw * self.capex_cfg["solar"]["ac_dc_ratio"]

        solar_capex       = solar_dc_mwp          * float(self.capex_cfg["solar"]["cost_per_mwp"])
        wind_capex        = wind_capacity_mw       * float(self.capex_cfg["wind"]["cost_per_mw"])
        bess_capex        = bess_energy_capacity_mwh * float(self.capex_cfg["bess"]["cost_per_mwh"])
        transmission_capex = (
            float(self.capex_cfg["transmission"]["length_km"])
            * float(self.capex_cfg["transmission"]["cost_per_km"])
        )

        total_capex = solar_capex + wind_capex + bess_capex + transmission_capex

        return {
            "solar_dc_mwp":         solar_dc_mwp,
            "solar_capex":          solar_capex,
            "wind_capex":           wind_capex,
            "bess_capex":           bess_capex,
            "transmission_capex":   transmission_capex,
            "total_capex":          total_capex,
        }