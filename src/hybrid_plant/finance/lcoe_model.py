"""
lcoe_model.py
─────────────
NPV-based Levelised Cost of Energy (LCOE).

    LCOE (Rs/kWh) = NPV(total costs) / NPV(busbar energy in kWh)

Numerator components (all discounted at WACC)
─────────────────────────────────────────────
  1. Debt interest payments      (Years 1 – debt_tenure)
  2. Debt principal repayments   (Years 1 – debt_tenure)
  3. Equity ROE                  (Years 1 – project_life, fixed annual)
  4. OPEX                        (Years 1 – project_life)

WACC
────
  WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)

NPV convention
──────────────
  Matches Excel =NPV():  CF_t / (1 + r)^t  for t = 1 … n.
  CAPEX at Year 0 is recovered implicitly through debt service + ROE.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import MWH_TO_KWH, PERCENT_TO_DECIMAL


class LCOEModel:
    """
    Computes the project LCOE and supporting financial schedules.

    WACC is computed once at construction and reused across all ``compute``
    calls so callers can read ``model.wacc`` without re-evaluating.

    Parameters
    ----------
    config : FullConfig
    """

    def __init__(self, config: FullConfig) -> None:
        fin = config.finance["financing"]

        self._project_life   = config.project["project"]["project_life_years"]
        self._debt_frac      = fin["debt_percent"]                          * PERCENT_TO_DECIMAL
        self._equity_frac    = fin["equity_percent"]                        * PERCENT_TO_DECIMAL
        self._interest_rate  = fin["debt"]["interest_rate_percent"]         * PERCENT_TO_DECIMAL
        self._debt_tenure    = fin["debt"]["tenure_years"]
        self._roe            = fin["equity"]["return_on_equity_percent"]    * PERCENT_TO_DECIMAL
        self._tax_rate       = fin["corporate_tax_rate_percent"]            * PERCENT_TO_DECIMAL

        self.wacc: float = self._compute_wacc()

    # ── WACC ─────────────────────────────────────────────────────────────────

    def _compute_wacc(self) -> float:
        """WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)"""
        return (
            self._debt_frac   * self._interest_rate * (1 - self._tax_rate)
            + self._equity_frac * self._roe
        )

    # ── NPV helper ────────────────────────────────────────────────────────────

    def _npv(self, series: list[float]) -> float:
        """
        Excel-style NPV: series[0] is Year 1, discounted at t = 1.

            NPV = Σ series[t] / (1 + wacc)^(t+1)   for t = 0 … n-1
        """
        return sum(v / (1 + self.wacc) ** (t + 1) for t, v in enumerate(series))

    # ── Debt amortisation ─────────────────────────────────────────────────────

    def _debt_schedule(self, debt_amount: float) -> tuple[list[float], list[float]]:
        """
        Fixed-EMI amortising loan schedule.

        Returns two lists of length ``project_life``:
            interest_schedule, principal_schedule
        Both are padded with zeros after ``debt_tenure``.
        """
        r, n = self._interest_rate, self._debt_tenure
        emi  = (debt_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)) if r > 0 else debt_amount / n

        balance: float          = debt_amount
        interest_schedule:  list[float] = []
        principal_schedule: list[float] = []

        for year in range(1, self._project_life + 1):
            if year <= self._debt_tenure and balance > 1e-6:
                interest  = balance * r
                principal = emi - interest
                balance   = max(balance - principal, 0.0)
            else:
                interest = principal = 0.0

            interest_schedule.append(interest)
            principal_schedule.append(principal)

        return interest_schedule, principal_schedule

    # ── Main computation ──────────────────────────────────────────────────────

    def compute(
        self,
        total_capex:                  float,
        opex_projection:              list[float],
        busbar_energy_mwh_projection: np.ndarray,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        total_capex                   : Total project CAPEX (Rs)
        opex_projection               : Annual OPEX (Rs), length = project_life
        busbar_energy_mwh_projection  : Annual busbar energy (MWh), length = project_life

        Returns
        -------
        dict
            lcoe_inr_per_kwh : float  — primary output
            wacc             : float
            + full breakdown (npv components, schedules, capital split)
        """
        debt_amount   = total_capex * self._debt_frac
        equity_amount = total_capex * self._equity_frac

        interest_schedule, principal_schedule = self._debt_schedule(debt_amount)

        roe_annual   = equity_amount * self._roe
        roe_schedule = [roe_annual] * self._project_life

        npv_interest  = self._npv(interest_schedule)
        npv_principal = self._npv(principal_schedule)
        npv_roe       = self._npv(roe_schedule)
        npv_opex      = self._npv(opex_projection)
        npv_total     = npv_interest + npv_principal + npv_roe + npv_opex

        busbar_kwh    = [float(e) * MWH_TO_KWH for e in busbar_energy_mwh_projection]
        npv_energy    = self._npv(busbar_kwh)

        if npv_energy <= 0:
            raise ValueError("NPV of busbar energy is zero or negative — check energy projection.")

        lcoe = npv_total / npv_energy   # Rs / kWh

        emi = (
            interest_schedule[0] + principal_schedule[0]
            if self._debt_tenure > 0 else 0.0
        )

        return {
            "lcoe_inr_per_kwh":   lcoe,
            "wacc":               self.wacc,
            "npv_total_cost":     npv_total,
            "npv_interest":       npv_interest,
            "npv_principal":      npv_principal,
            "npv_roe":            npv_roe,
            "npv_opex":           npv_opex,
            "npv_energy_kwh":     npv_energy,
            "interest_schedule":  interest_schedule,
            "principal_schedule": principal_schedule,
            "roe_schedule":       roe_schedule,
            "emi":                emi,
            "debt_amount":        debt_amount,
            "equity_amount":      equity_amount,
        }