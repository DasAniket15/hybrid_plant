import numpy as np


class LCOEModel:
    """
    Computes Levelised Cost of Energy (LCOE) using an NPV-based approach.

    LCOE (Rs/kWh) = NPV of total costs / NPV of busbar energy (kWh)

    ── NUMERATOR ────────────────────────────────────────────────────────
    NPV of total costs comprises four components, each discounted at WACC:

        1. NPV of debt principal repayments  (Years 1 – debt_tenure)
        2. NPV of debt interest payments     (Years 1 – debt_tenure)
        3. NPV of ROE payments               (Years 1 – project_life)
                ROE = fixed_annual = roe_percent × equity_amount  (flat)
        4. NPV of total OPEX                 (Years 1 – project_life)

    ── DENOMINATOR ──────────────────────────────────────────────────────
    NPV of annual busbar energy delivered (pre-loss, in kWh), discounted
    at the same WACC.

    ── WACC ─────────────────────────────────────────────────────────────
    Computed dynamically:
        WACC = (debt_% × interest_% × (1 − tax_%)) + (equity_% × roe_%)

    ── NPV CONVENTION ───────────────────────────────────────────────────
    Matches Excel NPV():  cashflow at Year t is discounted as CF/(1+r)^t
    where t runs from 1 to n.  CAPEX at Year 0 is not separately added
    here — it is implicitly recovered through debt service + ROE.
    """

    def __init__(self, config):
        fin = config.finance["financing"]

        self.project_life  = config.project["project"]["project_life_years"]
        self.debt_percent  = fin["debt_percent"]  / 100
        self.equity_percent = fin["equity_percent"] / 100
        self.interest_rate  = fin["debt"]["interest_rate_percent"] / 100
        self.debt_tenure    = fin["debt"]["tenure_years"]
        self.roe_percent    = fin["equity"]["return_on_equity_percent"] / 100
        self.tax_rate       = fin["corporate_tax_rate_percent"] / 100

        self.wacc = self._compute_wacc()

    # ------------------------------------------------------------------
    # WACC
    # ------------------------------------------------------------------

    def _compute_wacc(self):
        """
        WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)
        """
        return (
            self.debt_percent  * self.interest_rate * (1 - self.tax_rate)
            + self.equity_percent * self.roe_percent
        )

    # ------------------------------------------------------------------
    # NPV helper  (Excel-style, t = 1 … n)
    # ------------------------------------------------------------------

    def _npv(self, series):
        """
        NPV = Σ  series[t-1] / (1 + wacc)^t   for t = 1 … len(series)
        Matches Excel =NPV(wacc, series).
        """
        return sum(
            v / (1 + self.wacc) ** (t + 1)
            for t, v in enumerate(series)
        )

    # ------------------------------------------------------------------
    # Debt amortisation schedule
    # ------------------------------------------------------------------

    def _debt_schedule(self, debt_amount):
        """
        Fixed EMI amortising loan.
        Returns two lists of length project_life:
            interest_schedule, principal_schedule
        Padded with zeros after debt_tenure if tenure < project_life.
        """
        r = self.interest_rate
        n = self.debt_tenure

        if r == 0:
            emi = debt_amount / n
        else:
            emi = debt_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

        balance           = debt_amount
        interest_schedule  = []
        principal_schedule = []

        for year in range(1, self.project_life + 1):
            if year <= self.debt_tenure and balance > 1e-6:
                interest  = balance * r
                principal = emi - interest
                balance   = max(balance - principal, 0.0)
            else:
                interest  = 0.0
                principal = 0.0

            interest_schedule.append(interest)
            principal_schedule.append(principal)

        return interest_schedule, principal_schedule

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute(
        self,
        total_capex,
        opex_projection,
        busbar_energy_mwh_projection,
    ):
        """
        Parameters
        ----------
        total_capex                   : float     — total project CAPEX (Rs)
        opex_projection               : list[float] — annual OPEX (Rs), length = project_life
        busbar_energy_mwh_projection  : array-like — annual busbar energy (MWh), length = project_life

        Returns
        -------
        dict with lcoe_inr_per_kwh and full breakdown
        """
        debt_amount   = total_capex * self.debt_percent
        equity_amount = total_capex * self.equity_percent

        # ── Debt schedule ────────────────────────────────────────────
        interest_schedule, principal_schedule = self._debt_schedule(debt_amount)

        # ── ROE  (flat annual payment = roe_percent × equity_amount) ─
        roe_annual    = equity_amount * self.roe_percent
        roe_schedule  = [roe_annual] * self.project_life

        # ── NPV of each cost component ───────────────────────────────
        npv_interest   = self._npv(interest_schedule)
        npv_principal  = self._npv(principal_schedule)
        npv_roe        = self._npv(roe_schedule)
        npv_opex       = self._npv(opex_projection)

        npv_total_cost = npv_interest + npv_principal + npv_roe + npv_opex

        # ── NPV of busbar energy (kWh) ────────────────────────────────
        busbar_kwh    = [float(e) * 1000.0 for e in busbar_energy_mwh_projection]
        npv_energy_kwh = self._npv(busbar_kwh)

        if npv_energy_kwh <= 0:
            raise ValueError("NPV of busbar energy is zero or negative — check energy projection.")

        lcoe = npv_total_cost / npv_energy_kwh   # Rs / kWh

        return {
            # Primary output
            "lcoe_inr_per_kwh":    lcoe,
            "wacc":                self.wacc,

            # Numerator breakdown
            "npv_total_cost":      npv_total_cost,
            "npv_interest":        npv_interest,
            "npv_principal":       npv_principal,
            "npv_roe":             npv_roe,
            "npv_opex":            npv_opex,

            # Denominator
            "npv_energy_kwh":      npv_energy_kwh,

            # Raw schedules (for audit / export)
            "interest_schedule":   interest_schedule,
            "principal_schedule":  principal_schedule,
            "roe_schedule":        roe_schedule,
            "emi":                 interest_schedule[0] + principal_schedule[0] if self.debt_tenure > 0 else 0,

            # Capital split
            "debt_amount":         debt_amount,
            "equity_amount":       equity_amount,
        }