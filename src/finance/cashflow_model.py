import numpy as np
import numpy_financial as npf


class CashflowModel:

    def __init__(self, config):

        self.config = config
        self.project_life = config.project["project"]["project_life_years"]

        self.financing = config.finance["financing"]

        self.debt_percent = self.financing["debt_percent"] / 100
        self.equity_percent = self.financing["equity_percent"] / 100

        self.interest_rate = (
            self.financing["debt"]["interest_rate_percent"] / 100
        )

        self.debt_tenure = self.financing["debt"]["tenure_years"]

    # -------------------------------------------------
    # Debt EMI
    # -------------------------------------------------

    def _compute_emi(self, debt_amount):

        r = self.interest_rate
        n = self.debt_tenure

        emi = debt_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

        return emi

    # -------------------------------------------------
    # Equity Cashflow Series
    # -------------------------------------------------

    def compute_equity_cashflows(
        self,
        total_capex,
        delivered_meter_projection,
        opex_projection,
        ppa_tariff,
    ):

        debt_amount = total_capex * self.debt_percent
        equity_amount = total_capex * self.equity_percent

        emi = self._compute_emi(debt_amount)

        equity_cashflows = [-equity_amount]

        for year in range(self.project_life):

            revenue = delivered_meter_projection[year] * 1000 * ppa_tariff
            opex = opex_projection[year]

            cash_after_opex = revenue - opex
            cash_after_debt = cash_after_opex - emi

            equity_cashflows.append(cash_after_debt)

        return np.array(equity_cashflows)

    # -------------------------------------------------
    # IRR Calculation
    # -------------------------------------------------

    def equity_irr(
        self,
        total_capex,
        delivered_meter_projection,
        opex_projection,
        ppa_tariff,
    ):

        equity_cashflows = self.compute_equity_cashflows(
            total_capex,
            delivered_meter_projection,
            opex_projection,
            ppa_tariff,
        )

        irr = npf.irr(equity_cashflows)

        return irr

    # -------------------------------------------------
    # Tariff Solver (Binary Search)
    # -------------------------------------------------

    def solve_tariff_for_target_irr(
        self,
        total_capex,
        delivered_meter_projection,
        opex_projection,
        target_irr_percent,
    ):

        target_irr = target_irr_percent / 100

        low = 0.01
        high = 20.0

        for _ in range(60):

            mid = (low + high) / 2

            irr = self.equity_irr(
                total_capex,
                delivered_meter_projection,
                opex_projection,
                mid,
            )

            if irr is None or np.isnan(irr):
                low = mid
                continue

            if irr >= target_irr:
                high = mid
            else:
                low = mid

        return mid