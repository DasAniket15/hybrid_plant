from .capex_model          import CapexModel
from .opex_model           import OpexModel
from .energy_projection    import EnergyProjection
from .lcoe_model           import LCOEModel
from .landed_tariff_model  import LandedTariffModel
from .savings_model        import SavingsModel


class FinanceEngine:
    """
    Top-level orchestrator for the LCOE-based finance engine.

    ── PIPELINE ─────────────────────────────────────────────────────────

        1. CAPEX           → total project capital cost breakdown
        2. OPEX            → 25-year annual OPEX projection
        3. Energy          → 25-year busbar + meter energy (degraded)
        4. LCOE            → NPV(costs) / NPV(busbar energy)
        5. Landed Tariff   → absolute annual costs → divide by meter kWh
        6. Savings         → client savings vs. 100% DISCOM baseline

    ── INPUTS ───────────────────────────────────────────────────────────
        year1_results      : dict from Year1Engine.evaluate()
        solar_capacity_mw  : float (AC MW)
        wind_capacity_mw   : float (AC MW)
        ppa_capacity_mw    : float (contracted MW)

    ── PRIMARY OUTPUTS ──────────────────────────────────────────────────
        lcoe_inr_per_kwh
        landed_tariff_series     (annual series, Rs/kWh)
        annual_savings_year1
        savings_npv
        + full breakdown dicts for all intermediate results
    """

    def __init__(self, config, data):
        self.config = config
        self.data   = data

        self.capex_model         = CapexModel(config)
        self.opex_model          = OpexModel(config)
        self.lcoe_model          = LCOEModel(config)
        self.landed_tariff_model = LandedTariffModel(config)
        self.savings_model       = SavingsModel(config, data)

    # ------------------------------------------------------------------

    def evaluate(
        self,
        year1_results,
        solar_capacity_mw,
        wind_capacity_mw,
        ppa_capacity_mw,
        banked_energy_kwh_projection=None,
    ):
        """
        Run the full finance pipeline for a given plant configuration.

        Parameters
        ----------
        year1_results               : dict
            Output of Year1Engine.evaluate().

        solar_capacity_mw           : float
            AC solar capacity (MW).

        wind_capacity_mw            : float
            Wind capacity (MW).

        ppa_capacity_mw             : float
            Contracted PPA capacity (MW).

        banked_energy_kwh_projection : list[float] or None
            Annual banked energy (kWh) across project life.
            Defaults to zero (stub until banking module is implemented).

        Returns
        -------
        dict — full finance results
        """

        bess_mwh = float(year1_results["energy_capacity_mwh"])

        # ── 1. CAPEX ─────────────────────────────────────────────────
        capex = self.capex_model.compute(
            solar_capacity_mw        = solar_capacity_mw,
            wind_capacity_mw         = wind_capacity_mw,
            bess_energy_capacity_mwh = bess_mwh,
        )
        total_capex = capex["total_capex"]

        # ── 2. OPEX projection ────────────────────────────────────────
        opex_projection, opex_breakdown = self.opex_model.compute(
            solar_capacity_mw = solar_capacity_mw,
            wind_capacity_mw  = wind_capacity_mw,
            bess_energy_mwh   = bess_mwh,
            total_capex       = total_capex,
        )

        # ── 3. Energy projection ──────────────────────────────────────
        # loss_factor comes directly from year1_results (set by
        # GridInterface) to guarantee consistency with energy engine.
        loss_factor = float(year1_results["loss_factor"])

        projection = EnergyProjection(
            config            = self.config,
            data              = self.data,
            year1_results     = year1_results,
            solar_capacity_mw = solar_capacity_mw,
            wind_capacity_mw  = wind_capacity_mw,
            loss_factor       = loss_factor,
        ).project()

        busbar_mwh = projection["delivered_pre_mwh"]    # LCOE denominator
        meter_mwh  = projection["delivered_meter_mwh"]  # savings + landed tariff

        # ── 4. LCOE ──────────────────────────────────────────────────
        lcoe_result = self.lcoe_model.compute(
            total_capex                  = total_capex,
            opex_projection              = opex_projection,
            busbar_energy_mwh_projection = busbar_mwh,
        )
        lcoe = lcoe_result["lcoe_inr_per_kwh"]
        wacc = lcoe_result["wacc"]

        # ── 5. Landed tariff (annual series) ─────────────────────────
        landed_result = self.landed_tariff_model.compute(
            lcoe_inr_per_kwh             = lcoe,
            ppa_capacity_mw              = ppa_capacity_mw,
            busbar_energy_mwh_projection = busbar_mwh,
            meter_energy_mwh_projection  = meter_mwh,
            banked_energy_kwh_projection = banked_energy_kwh_projection,
        )

        # ── 6. Client savings ─────────────────────────────────────────
        savings_result = self.savings_model.compute(
            landed_tariff_series        = landed_result["landed_tariff_series"],
            meter_energy_mwh_projection = meter_mwh,
            wacc                        = wacc,
        )

        # ── Return ────────────────────────────────────────────────────
        return {
            # ── Primary outputs ───────────────────────────────────────
            "lcoe_inr_per_kwh":        lcoe,
            "landed_tariff_series":    landed_result["landed_tariff_series"],
            "annual_savings_year1":    savings_result["annual_savings_year1"],
            "savings_npv":             savings_result["savings_npv"],

            # ── Supporting outputs ────────────────────────────────────
            "wacc":                    wacc,
            "capex":                   capex,
            "opex_projection":         opex_projection,
            "opex_breakdown":          opex_breakdown,
            "energy_projection":       projection,
            "lcoe_breakdown":          lcoe_result,
            "landed_tariff_breakdown": landed_result,
            "savings_breakdown":       savings_result,
        }