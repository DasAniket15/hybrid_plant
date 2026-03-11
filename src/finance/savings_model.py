import numpy as np


class SavingsModel:
    """
    Computes the client's annual electricity cost savings vs. a 100% DISCOM baseline.

    ── BASELINE ─────────────────────────────────────────────────────────
    Baseline cost = total annual load (kWh) × weighted-average DISCOM ToD tariff

    ── HYBRID COST (per year) ───────────────────────────────────────────
    hybrid_cost_t = (RE_meter_kwh_t × landed_tariff_t)
                  + (DISCOM_load_kwh_t × discom_tariff)

    where:
        DISCOM_load_kwh_t = total_load_kwh − RE_meter_kwh_t
        landed_tariff_t   = annual landed tariff for year t (Rs/kWh)

    ── SAVINGS ──────────────────────────────────────────────────────────
    savings_t = baseline_cost − hybrid_cost_t

    ── SAVINGS NPV ──────────────────────────────────────────────────────
    Discounted at WACC using Excel-style convention (t = 1 … project_life).

    Note: DISCOM tariff is the weighted-average ToD rate computed from
    tariffs.yaml.  Total annual load is constant across years (Year-1
    load profile is used as a fixed annual reference).
    """

    def __init__(self, config, data):

        self.project_life    = config.project["project"]["project_life_years"]
        self.discom_tariff   = self._get_discom_tariff(config)
        self.annual_load_kwh = float(np.sum(data["load_profile"])) * 1000.0  # MWh → kWh
        self.baseline_cost   = self.annual_load_kwh * self.discom_tariff      # Rs

    # ------------------------------------------------------------------

    @staticmethod
    def _get_discom_tariff(config):
        """Weighted-average ToD tariff (Rs/kWh) across all 24 hours."""
        tod = config.tariffs["discom"]["tod_periods"]
        total_weighted = sum(p["rate_inr_per_kwh"] * len(p["hours"]) for p in tod.values())
        total_hours    = sum(len(p["hours"]) for p in tod.values())
        return total_weighted / total_hours

    # ------------------------------------------------------------------

    def _npv(self, series, wacc):
        """Excel-style NPV: series[0] is Year 1, discounted at t = 1."""
        return sum(v / (1 + wacc) ** (t + 1) for t, v in enumerate(series))

    # ------------------------------------------------------------------

    def compute(self, landed_tariff_series, meter_energy_mwh_projection, wacc):
        """
        Parameters
        ----------
        landed_tariff_series        : list[float]  — annual landed tariff (Rs/kWh)
        meter_energy_mwh_projection : array-like   — annual meter energy (MWh)
        wacc                        : float        — discount rate (decimal)

        Returns
        -------
        dict with annual_savings, savings_npv, annual_savings_year1, and
        full cost series for reporting.
        """
        annual_savings       = []
        annual_hybrid_cost   = []
        annual_re_cost       = []
        annual_discom_cost   = []

        for landed_t, meter_mwh_t in zip(landed_tariff_series, meter_energy_mwh_projection):

            re_kwh_t      = float(meter_mwh_t) * 1000.0
            discom_kwh_t  = self.annual_load_kwh - re_kwh_t

            re_cost_t     = re_kwh_t     * landed_t
            discom_cost_t = discom_kwh_t * self.discom_tariff
            hybrid_cost_t = re_cost_t + discom_cost_t

            savings_t = self.baseline_cost - hybrid_cost_t

            annual_savings.append(savings_t)
            annual_hybrid_cost.append(hybrid_cost_t)
            annual_re_cost.append(re_cost_t)
            annual_discom_cost.append(discom_cost_t)

        savings_npv = self._npv(annual_savings, wacc)

        return {
            # Primary outputs
            "annual_savings_year1":  annual_savings[0],
            "savings_npv":           savings_npv,
            "annual_savings":        annual_savings,      # full 25-year series

            # Cost series (for reporting / diagnostics)
            "baseline_annual_cost":  self.baseline_cost,
            "annual_hybrid_cost":    annual_hybrid_cost,
            "annual_re_cost":        annual_re_cost,
            "annual_discom_cost":    annual_discom_cost,
            "discom_tariff":         self.discom_tariff,
            "annual_load_kwh":       self.annual_load_kwh,
        }