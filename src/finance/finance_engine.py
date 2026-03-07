import numpy as np
import numpy_financial as npf

from .capex_model import CapexModel
from .opex_model import OpexModel
from .cashflow_model import CashflowModel
from .energy_projection import EnergyProjection


class FinanceEngine:

    def __init__(self, config, data):

        self.config = config
        self.data = data

        self.capex_model = CapexModel(config)
        self.opex_model = OpexModel(config)
        self.cashflow_model = CashflowModel(config)

        self.project_life = config.project["project"]["project_life_years"]
        self.wacc = config.finance["discounting"]["wacc_percent"] / 100

        self.discom_tariff = self._get_flat_discom_tariff()

    # -------------------------------------------------
    # DISCOM Tariff (Weighted Avg ToD)
    # -------------------------------------------------

    def _get_flat_discom_tariff(self):

        tod_periods = self.config.tariffs["discom"]["tod_periods"]

        total_weighted = 0
        total_hours = 0

        for period in tod_periods.values():
            rate = period["rate_inr_per_kwh"]
            hours = len(period["hours"])

            total_weighted += rate * hours
            total_hours += hours

        return total_weighted / total_hours

    # -------------------------------------------------
    # Main Evaluation
    # -------------------------------------------------

    def evaluate(
        self,
        year1_results,
        solar_capacity_mw,
        wind_capacity_mw,
        ppa_capacity_mw,
        target_irr_percent,
    ):

        # -------------------------
        # 1️⃣ Energy Projection
        # -------------------------

        projection = EnergyProjection(
            self.config,
            self.data,
            year1_results,
            solar_capacity_mw,
            wind_capacity_mw,
        ).project()

        # -------------------------
        # 2️⃣ CAPEX
        # -------------------------
 
        capex = self.capex_model.compute(
            solar_capacity_mw,
            wind_capacity_mw,
            year1_results["energy_capacity_mwh"],
        )

        total_capex = capex["total_capex"]

        # -------------------------
        # 3️⃣ OPEX
        # -------------------------

        opex_projection = self.opex_model.compute(total_capex)

        # -------------------------
        # 4️⃣ Solve Required Tariff (IRR Constraint)
        # -------------------------

        required_tariff = self.cashflow_model.solve_tariff_for_target_irr(
            total_capex=total_capex,
            delivered_meter_projection=projection["delivered_meter_mwh"],
            opex_projection=opex_projection,
            target_irr_percent=target_irr_percent,
        )

        if required_tariff is None:
            return {"invalid_solution": True}

        achieved_irr_percent = self.cashflow_model.equity_irr(
            total_capex=total_capex,
            delivered_meter_projection=projection["delivered_meter_mwh"],
            opex_projection=opex_projection,
            ppa_tariff=required_tariff,
        ) * 100

        equity_cashflows = self.cashflow_model.compute_equity_cashflows(
            total_capex,
            projection["delivered_meter_mwh"],
            opex_projection,
            required_tariff,
        )

        # -------------------------
        # 5️⃣ Client Economics (With Regulatory Charges)
        # -------------------------

        annual_load_mwh = np.sum(self.data["load_profile"])
        baseline_cost = annual_load_mwh * 1000 * self.discom_tariff

        reg_cfg = self.config.finance["regulatory_charges"]
        ht_percent = self.config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]
        lt_percent = 100 - ht_percent

        # Capacity-based charges (₹/MW/month)
        ctu_charge = (
            ht_percent / 100 * reg_cfg["ht"]["ctu_charge_inr_per_mw_per_month"]
            + lt_percent / 100 * reg_cfg["lt"]["ctu_charge_inr_per_mw_per_month"]
        )

        stu_charge = (
            ht_percent / 100 * reg_cfg["ht"]["stu_charge_inr_per_mw_per_month"]
            + lt_percent / 100 * reg_cfg["lt"]["stu_charge_inr_per_mw_per_month"]
        )

        sldc_charge = (
            ht_percent / 100 * reg_cfg["ht"]["sldc_charge_inr_per_mw_per_month"]
            + lt_percent / 100 * reg_cfg["lt"]["sldc_charge_inr_per_mw_per_month"]
        )

        annual_capacity_charge = (
            (ctu_charge + stu_charge + sldc_charge)
            * ppa_capacity_mw
            * 12
        )

        # Energy-based charges (₹/kWh)
        wheeling_charge = (
            ht_percent / 100 * reg_cfg["ht"]["wheeling_charge_inr_per_kwh"]
            + lt_percent / 100 * reg_cfg["lt"]["wheeling_charge_inr_per_kwh"]
        )

        electricity_tax = (
            ht_percent / 100 * reg_cfg["ht"]["electricity_tax_inr_per_kwh"]
            + lt_percent / 100 * reg_cfg["lt"]["electricity_tax_inr_per_kwh"]
        )

        # Banking charge placeholder (future)
        banking_charge = (
            ht_percent / 100 * reg_cfg["ht"]["banking_charge_inr_per_kwh"]
            + lt_percent / 100 * reg_cfg["lt"]["banking_charge_inr_per_kwh"]
        )

        savings_projection = []
        hybrid_cost_projection = []

        for year in range(self.project_life):

            re_energy = projection["delivered_meter_mwh"][year]
            residual_discom = annual_load_mwh - re_energy

            energy_based_charges = (
                re_energy * 1000 * (wheeling_charge + electricity_tax)
            )

            # Banking not implemented yet → 0 placeholder
            banking_cost = 0

            hybrid_cost = (
                re_energy * 1000 * required_tariff
                + energy_based_charges
                + annual_capacity_charge
                + banking_cost
                + residual_discom * 1000 * self.discom_tariff
            )

            savings = baseline_cost - hybrid_cost

            savings_projection.append(savings)
            hybrid_cost_projection.append(hybrid_cost)

        savings_npv = npf.npv(self.wacc, savings_projection)

        # -------------------------
        # Return Structured Output
        # -------------------------

        return {
            "invalid_solution": False,
            "required_ppa_tariff": required_tariff,
            "achieved_equity_irr": achieved_irr_percent,

            "projection": projection,
            "capex": capex,
            "opex_projection": opex_projection,
            "equity_cashflows": equity_cashflows,
            
            "annual_savings": savings_projection,
            "savings_npv": savings_npv,
            "objective_value": savings_npv,

            "discom_tariff": self.discom_tariff
        }