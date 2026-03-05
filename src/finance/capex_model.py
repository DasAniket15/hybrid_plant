class CapexModel:

    def __init__(self, config):
        self.config = config
        self.capex_cfg = config.finance["capex"]

    def compute(
        self,
        solar_capacity_mw,
        wind_capacity_mw,
        bess_energy_capacity_mwh,
    ):

        solar_cost = float(self.capex_cfg["solar"]["cost_per_mwp"])
        wind_cost = float(self.capex_cfg["wind"]["cost_per_mw"])
        bess_cost = float(self.capex_cfg["bess"]["cost_per_mwh"])
        trans_cost = float(self.capex_cfg["transmission"]["cost_per_km"])
        trans_len = float(self.capex_cfg["transmission"]["length_km"])
        
        solar_capex = solar_capacity_mw * solar_cost

        wind_capex = wind_capacity_mw * wind_cost

        bess_capex = bess_energy_capacity_mwh * bess_cost

        transmission_capex = trans_len * trans_cost

        total_capex = (
            solar_capex
            + wind_capex
            + bess_capex
            + transmission_capex
        )

        return {
            "solar_capex": solar_capex,
            "wind_capex": wind_capex,
            "bess_capex": bess_capex,
            "transmission_capex": transmission_capex,
            "total_capex": total_capex,
        }