import numpy as np
import numpy_financial as npf


class CashflowModel:

    def __init__(self, config):
        self.config = config
        self.project_life = config.project["project"]["project_life_years"]
        self.financing = config.finance["financing"]
        self.wacc = config.finance["discounting"]["wacc_percent"] / 100

    # -------------------------------------------------

    def _compute_emi(self, debt_amount):

        r = self.financing["debt"]["interest_rate_percent"] / 100
        n = self.financing["debt"]["tenure_years"]

        emi = debt_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

        return emi

    # -------------------------------------------------

    def _equity_irr(
        self,
        total_capex,
        delivered_meter_projection,
        opex_projection,
        ppa_tariff,
    ):

        debt_percent = self.financing["debt_percent"] / 100
        equity_percent = self.financing["equity_percent"] / 100

        debt_amount = total_capex * debt_percent
        equity_amount = total_capex * equity_percent

        emi = self._compute_emi(debt_amount)

        equity_cashflows = [-equity_amount]

        for year in range(self.project_life):

            revenue = delivered_meter_projection[year] * 1000 * ppa_tariff
            opex = opex_projection[year]

            cash_after_opex = revenue - opex
            cash_after_debt = cash_after_opex - emi

            equity_cashflows.append(cash_after_debt)

        irr = npf.irr(equity_cashflows)

        return irr

    # -------------------------------------------------

    def solve_tariff_for_target_irr(
        self,
        total_capex,
        delivered_meter_projection,
        opex_projection,
        target_irr_percent,
    ):

        target_irr = target_irr_percent / 100

        # Bounds for tariff search
        low = 0.01
        high = 20.0

        print("IRR at tariff 2:", self._equity_irr(total_capex, delivered_meter_projection, opex_projection, 2))
        print("IRR at tariff 5:", self._equity_irr(total_capex, delivered_meter_projection, opex_projection, 5))
        print("IRR at tariff 10:", self._equity_irr(total_capex, delivered_meter_projection, opex_projection, 10))
        print("IRR at tariff 20:", self._equity_irr(total_capex, delivered_meter_projection, opex_projection, 20))

        for _ in range(60):  # 60 iterations = extremely precise

            mid = (low + high) / 2

            irr = self._equity_irr(
                total_capex,
                delivered_meter_projection,
                opex_projection,
                mid,
            )

            if irr is None or np.isnan(irr):
                low = mid  # If IRR can't be computed, we need a higher tariff
                continue

            if irr >= target_irr:
                high = mid
            else:
                low = mid
        
        return mid