class LandedTariffModel:
    """
    Computes the annual landed tariff series (Rs/kWh) for the client.

    Landed tariff = LCOE (busbar) + regulatory charges (converted to Rs/kWh)

    ── REGULATORY CHARGE TYPES ──────────────────────────────────────────

    Energy-based (Rs/kWh) — constant across years:
        • Wheeling charge
        • Electricity tax
        • Banking charge   (stub = 0 for now)

    Capacity-based (Rs/MW/month) — converted to Rs/kWh using the meter
    energy of THAT SPECIFIC YEAR (so this component changes year on year
    as delivered energy degrades):
        • CTU charge
        • STU charge
        • SLDC charge

    Conversion:
        annual_capacity_charge (Rs) = (CTU + STU + SLDC) × PPA_MW × 12
        capacity_charge_per_kwh_t   = annual_capacity_charge / meter_kwh_t

    The landed tariff is therefore an annual series, not a single fixed number.

    All rates are sourced from finance.yaml (regulatory_charges) and
    regulatory.yaml (ht_lt_split_percent).
    """

    def __init__(self, config):

        reg_charges = config.finance["regulatory_charges"]
        ht_lt_split = config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]

        ht_frac = ht_lt_split / 100.0
        lt_frac = 1.0 - ht_frac

        ht = reg_charges["ht"]
        lt = reg_charges["lt"]

        # ── Energy-based charges (Rs/kWh) ────────────────────────────
        self.wheeling_per_kwh = (
            ht_frac * ht["wheeling_charge_inr_per_kwh"]
            + lt_frac * lt["wheeling_charge_inr_per_kwh"]
        )
        self.elec_tax_per_kwh = (
            ht_frac * ht["electricity_tax_inr_per_kwh"]
            + lt_frac * lt["electricity_tax_inr_per_kwh"]
        )
        self.banking_per_kwh = (
            ht_frac * ht["banking_charge_inr_per_kwh"]
            + lt_frac * lt["banking_charge_inr_per_kwh"]
        )

        # Total fixed energy charge (Rs/kWh)
        self.energy_charge_per_kwh = (
            self.wheeling_per_kwh
            + self.elec_tax_per_kwh
            + self.banking_per_kwh
        )

        # ── Capacity-based charges (Rs/MW/month) ─────────────────────
        self.ctu_per_mw_month  = (
            ht_frac * ht["ctu_charge_inr_per_mw_per_month"]
            + lt_frac * lt["ctu_charge_inr_per_mw_per_month"]
        )
        self.stu_per_mw_month  = (
            ht_frac * ht["stu_charge_inr_per_mw_per_month"]
            + lt_frac * lt["stu_charge_inr_per_mw_per_month"]
        )
        self.sldc_per_mw_month = (
            ht_frac * ht["sldc_charge_inr_per_mw_per_month"]
            + lt_frac * lt["sldc_charge_inr_per_mw_per_month"]
        )

    # ------------------------------------------------------------------

    def compute(self, lcoe_inr_per_kwh, ppa_capacity_mw, meter_energy_mwh_projection):
        """
        Parameters
        ----------
        lcoe_inr_per_kwh            : float       — LCOE from LCOEModel (Rs/kWh)
        ppa_capacity_mw             : float       — contracted PPA capacity (MW)
        meter_energy_mwh_projection : array-like  — annual meter energy (MWh), length = project_life

        Returns
        -------
        dict with landed_tariff_series (list[float], Rs/kWh) and component breakdown
        """

        # Annual capacity charge in Rs (fixed across years, same PPA capacity)
        annual_capacity_charge_rs = (
            (self.ctu_per_mw_month + self.stu_per_mw_month + self.sldc_per_mw_month)
            * ppa_capacity_mw
            * 12
        )

        landed_tariff_series        = []
        capacity_charge_per_kwh_series = []

        for meter_mwh in meter_energy_mwh_projection:

            meter_kwh = float(meter_mwh) * 1000.0

            if meter_kwh > 0:
                capacity_charge_per_kwh = annual_capacity_charge_rs / meter_kwh
            else:
                capacity_charge_per_kwh = 0.0

            landed = lcoe_inr_per_kwh + self.energy_charge_per_kwh + capacity_charge_per_kwh

            capacity_charge_per_kwh_series.append(capacity_charge_per_kwh)
            landed_tariff_series.append(landed)

        return {
            "landed_tariff_series":             landed_tariff_series,        # Rs/kWh, per year

            # Components (for audit / export)
            "lcoe_component":                   lcoe_inr_per_kwh,
            "wheeling_per_kwh":                 self.wheeling_per_kwh,
            "electricity_tax_per_kwh":          self.elec_tax_per_kwh,
            "banking_per_kwh":                  self.banking_per_kwh,
            "energy_charge_per_kwh":            self.energy_charge_per_kwh,  # Rs/kWh, fixed
            "annual_capacity_charge_rs":        annual_capacity_charge_rs,   # Rs/year, fixed
            "capacity_charge_per_kwh_series":   capacity_charge_per_kwh_series,  # Rs/kWh, per year
        }