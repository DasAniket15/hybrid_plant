class OpexModel:

    def __init__(self, config):
        self.config = config
        self.opex_cfg = config.finance["opex"]
        self.project_life = config.project["project"]["project_life_years"]

    def compute(self, total_capex):

        solar_opex_percent = float(self.opex_cfg["solar_percent_of_capex"])
        wind_opex_percent = float(self.opex_cfg["wind_percent_of_capex"])
        bess_opex_percent = float(self.opex_cfg["bess_percent_of_capex"])
        land_lease_base_cost = float(self.opex_cfg["land_lease"]["base_annual_cost"])
        land_lease_escalation = float(self.opex_cfg["land_lease"]["escalation_percent"])
        insurance_percent = float(self.opex_cfg["insurance_percent_of_capex"])

        opex_projection = []

        for year in range(1, self.project_life + 1):

            solar_opex = total_capex * solar_opex_percent / 100

            wind_opex = total_capex * wind_opex_percent / 100

            bess_opex = total_capex * bess_opex_percent / 100

            land_cost = land_lease_base_cost * ((1 + land_lease_escalation / 100) ** (year - 1))

            insurance = total_capex * insurance_percent / 100

            total_opex = (
                solar_opex
                + wind_opex
                + bess_opex
                + land_cost
                + insurance
            )

            opex_projection.append(total_opex)

        return opex_projection