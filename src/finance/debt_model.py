class DebtModel:
    """
    Amortizing loan model.
    """

    def __init__(self, principal, interest_rate_percent, tenure_years):
        self.principal = principal
        self.rate = interest_rate_percent / 100
        self.tenure = tenure_years

        self.emi = self._calculate_emi()

    # -------------------------------------------------

    def _calculate_emi(self):
        r = self.rate
        n = self.tenure
        D = self.principal

        if r == 0:
            return D / n

        return (r * D) / (1 - (1 + r) ** (-n))

    # -------------------------------------------------

    def amortization_schedule(self):
        """
        Returns list of dicts:
        [
            {
                "year": t,
                "interest": ...,
                "principal": ...,
                "total_payment": ...,
                "outstanding": ...
            }
        ]
        """

        balance = self.principal
        schedule = []

        for year in range(1, self.tenure + 1):

            interest = balance * self.rate
            principal_payment = self.emi - interest
            balance -= principal_payment

            schedule.append({
                "year": year,
                "interest": interest,
                "principal": principal_payment,
                "total_payment": self.emi,
                "outstanding": max(balance, 0)
            })

        return schedule