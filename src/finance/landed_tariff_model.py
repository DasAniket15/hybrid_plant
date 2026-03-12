class LandedTariffModel:
    """
    Computes the annual landed tariff series (Rs/kWh) for the client.

    ── APPROACH ─────────────────────────────────────────────────────────
    All charges are first computed in absolute Rs per year, then the
    total annual cost is divided by meter energy to derive the landed
    tariff for that year.

        landed_tariff_t = total_annual_cost_t / meter_kwh_t

    where:

        total_annual_cost_t =
              annual_RE_payment_t          (LCOE × busbar_kwh_t)
            + capacity_charges_t           (CTU + STU + SLDC on PPA MW)
            + wheeling_charges_t           (on RE meter kWh)
            + electricity_tax_t            (on RE meter kWh)
            + banking_charges_t            (on banked kWh, stub = 0)

    ── CHARGE BASES ─────────────────────────────────────────────────────
    Capacity-based  (Rs/MW/month) — applied on contracted PPA capacity:
        CTU, STU, SLDC

    Energy-based (Rs/kWh) — applied on RE delivered at client meter:
        Wheeling charge, Electricity tax

    Banking-based (Rs/kWh) — applied on total banked energy:
        Banking charge  (stub = 0 until banking module is implemented)

    All charges are weighted by HT/LT split from regulatory.yaml.
    Nothing is hardcoded.
    """

    def __init__(self, config):

        reg_charges = config.finance["regulatory_charges"]
        ht_lt_split = config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]

        self.ht_frac = ht_lt_split / 100.0
        self.lt_frac = 1.0 - self.ht_frac

        ht = reg_charges["ht"]
        lt = reg_charges["lt"]

        # ── Capacity-based rates (Rs/MW/month) ───────────────────────
        self.ctu_per_mw_month = (
            self.ht_frac * ht["ctu_charge_inr_per_mw_per_month"]
            + self.lt_frac * lt["ctu_charge_inr_per_mw_per_month"]
        )
        self.stu_per_mw_month = (
            self.ht_frac * ht["stu_charge_inr_per_mw_per_month"]
            + self.lt_frac * lt["stu_charge_inr_per_mw_per_month"]
        )
        self.sldc_per_mw_month = (
            self.ht_frac * ht["sldc_charge_inr_per_mw_per_month"]
            + self.lt_frac * lt["sldc_charge_inr_per_mw_per_month"]
        )

        # ── Energy-based rates (Rs/kWh) ──────────────────────────────
        self.wheeling_per_kwh = (
            self.ht_frac * ht["wheeling_charge_inr_per_kwh"]
            + self.lt_frac * lt["wheeling_charge_inr_per_kwh"]
        )
        self.elec_tax_per_kwh = (
            self.ht_frac * ht["electricity_tax_inr_per_kwh"]
            + self.lt_frac * lt["electricity_tax_inr_per_kwh"]
        )

        # ── Banking rate (Rs/kWh) — stub ─────────────────────────────
        self.banking_per_kwh = (
            self.ht_frac * ht["banking_charge_inr_per_kwh"]
            + self.lt_frac * lt["banking_charge_inr_per_kwh"]
        )

    # ------------------------------------------------------------------

    def compute(
        self,
        lcoe_inr_per_kwh,
        ppa_capacity_mw,
        busbar_energy_mwh_projection,
        meter_energy_mwh_projection,
        banked_energy_kwh_projection=None,
    ):
        """
        Parameters
        ----------
        lcoe_inr_per_kwh                : float
            LCOE from LCOEModel (Rs/kWh, busbar basis).

        ppa_capacity_mw                 : float
            Contracted PPA capacity (MW) — basis for capacity charges.

        busbar_energy_mwh_projection    : array-like
            Annual busbar energy (MWh), length = project_life.
            Used to compute annual RE payment = LCOE × busbar_kWh.

        meter_energy_mwh_projection     : array-like
            Annual meter energy (MWh), length = project_life.
            Used as denominator for landed tariff and basis for
            wheeling / electricity tax.

        banked_energy_kwh_projection    : list[float] or None
            Annual banked energy (kWh), length = project_life.
            Defaults to zero for all years (stub until banking module
            is implemented).

        Returns
        -------
        dict
        """
        n = len(meter_energy_mwh_projection)

        if banked_energy_kwh_projection is None:
            banked_energy_kwh_projection = [0.0] * n

        # ── Annual capacity charges (Rs) — fixed across years ────────
        annual_capacity_charge_rs = (
            (self.ctu_per_mw_month + self.stu_per_mw_month + self.sldc_per_mw_month)
            * ppa_capacity_mw
            * 12
        )

        # ── Per-year computation ──────────────────────────────────────
        landed_tariff_series         = []
        annual_re_payment_series     = []
        annual_wheeling_series       = []
        annual_elec_tax_series       = []
        annual_banking_series        = []
        annual_total_cost_series     = []

        for busbar_mwh, meter_mwh, banked_kwh in zip(
            busbar_energy_mwh_projection,
            meter_energy_mwh_projection,
            banked_energy_kwh_projection,
        ):
            busbar_kwh = float(busbar_mwh) * 1000.0
            meter_kwh  = float(meter_mwh)  * 1000.0

            # RE payment: client pays LCOE on busbar energy
            re_payment = lcoe_inr_per_kwh * busbar_kwh

            # Energy-based charges on RE delivered at meter
            wheeling   = self.wheeling_per_kwh * meter_kwh
            elec_tax   = self.elec_tax_per_kwh * meter_kwh

            # Banking charge on banked energy
            banking    = self.banking_per_kwh * float(banked_kwh)

            total_annual_cost = (
                re_payment
                + annual_capacity_charge_rs
                + wheeling
                + elec_tax
                + banking
            )

            landed_t = total_annual_cost / meter_kwh if meter_kwh > 0 else 0.0

            landed_tariff_series.append(landed_t)
            annual_re_payment_series.append(re_payment)
            annual_wheeling_series.append(wheeling)
            annual_elec_tax_series.append(elec_tax)
            annual_banking_series.append(banking)
            annual_total_cost_series.append(total_annual_cost)

        return {
            # Primary output
            "landed_tariff_series":      landed_tariff_series,

            # Absolute annual cost components (Rs)
            "annual_re_payment":         annual_re_payment_series,
            "annual_capacity_charge_rs": annual_capacity_charge_rs,  # fixed Rs/year
            "annual_wheeling":           annual_wheeling_series,
            "annual_electricity_tax":    annual_elec_tax_series,
            "annual_banking":            annual_banking_series,
            "annual_total_cost":         annual_total_cost_series,

            # Unit rates (for audit)
            "ctu_per_mw_month":          self.ctu_per_mw_month,
            "stu_per_mw_month":          self.stu_per_mw_month,
            "sldc_per_mw_month":         self.sldc_per_mw_month,
            "wheeling_per_kwh":          self.wheeling_per_kwh,
            "electricity_tax_per_kwh":   self.elec_tax_per_kwh,
            "banking_per_kwh":           self.banking_per_kwh,
        }