"""
tests/optimise/test_params.py
─────────────────────────────
Unit tests for ``optimise.params.build_params``.

Every parameter in ``OptParams`` is reconciled 1-to-1 against the source
``FullConfig`` and CSVs, plus cross-checks of precomputed constants against
the existing finance pipeline (LCOEModel, OpexModel).

Tolerances
──────────
- Exact integer/string fields: ``==``
- Float fields derived directly from YAML: abs diff < 1e-12
- Precomputed constants (floating-point chain): rel diff < 1e-10
- Cross-check against LCOEModel phi: rel diff < 1e-10
- Cross-check against OpexModel G_esc sums: rel diff < 1e-10
- D_s/D_w/D_b cross-check: rel diff < 1e-10
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_plant.config_loader import FullConfig, load_config
from hybrid_plant.constants import (
    CRORE_TO_RS,
    HOURS_PER_DAY,
    LAKH_TO_RS,
    MONTHS_PER_YEAR,
    PERCENT_TO_DECIMAL,
)
from hybrid_plant.data_loader import load_timeseries_data, operating_value
from hybrid_plant.energy.grid_interface import GridInterface
from hybrid_plant.energy.year1_engine import _build_hourly_discom_tariff
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.opex_model import OpexModel
from hybrid_plant.optimise.params import OptParams, build_params


# ─────────────────────────────────────────────────────────────────────────────
# Session fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config() -> FullConfig:
    return load_config()


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.fixture(scope="module")
def params(config: FullConfig, data: dict) -> OptParams:
    return build_params(config, data)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rel(a: float, b: float) -> float:
    """Relative difference |a - b| / max(|b|, 1e-30)."""
    return abs(a - b) / max(abs(b), 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hourly time-series
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeSeries:
    def test_cuf_s_matches_data(self, data: dict, params: OptParams) -> None:
        assert np.allclose(params.cuf_s, data["solar_cuf"])

    def test_cuf_w_matches_data(self, data: dict, params: OptParams) -> None:
        assert np.allclose(params.cuf_w, data["wind_cuf"])

    def test_load_matches_data(self, data: dict, params: OptParams) -> None:
        assert np.allclose(params.load, data["load_profile"])

    def test_tod_matches_builder(self, config: FullConfig, data: dict, params: OptParams) -> None:
        expected = _build_hourly_discom_tariff(config, n_hours=len(data["load_profile"]))
        assert np.allclose(params.tod, expected)

    def test_array_lengths_are_8760(self, params: OptParams) -> None:
        assert len(params.cuf_s)  == 8760
        assert len(params.cuf_w)  == 8760
        assert len(params.load)   == 8760
        assert len(params.tod)    == 8760

    def test_cufs_in_unit_interval(self, params: OptParams) -> None:
        assert float(params.cuf_s.min()) >= 0.0
        assert float(params.cuf_s.max()) <= 1.0
        assert float(params.cuf_w.min()) >= 0.0
        assert float(params.cuf_w.max()) <= 1.0

    def test_tod_positive(self, params: OptParams) -> None:
        assert float(params.tod.min()) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Physics scalars
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysics:
    def test_lf_matches_grid_interface(self, config: FullConfig, params: OptParams) -> None:
        expected = GridInterface(config).loss_factor
        assert abs(params.lf - expected) < 1e-12

    def test_eta_c(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.bess["bess"]["efficiency"]["charge_efficiency"])
        assert abs(params.eta_c - expected) < 1e-12

    def test_eta_d(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.bess["bess"]["efficiency"]["discharge_efficiency"])
        assert abs(params.eta_d - expected) < 1e-12

    def test_cs(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.bess["bess"]["container"]["size_mwh"])
        assert abs(params.cs - expected) < 1e-12

    def test_aux_pc(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.bess["bess"]["container"]["auxiliary_consumption_mwh_per_day"])
            / HOURS_PER_DAY
        )
        assert abs(params.aux_pc - expected) < 1e-12

    def test_crc_positive_and_bounded(self, params: OptParams) -> None:
        assert 0.0 < params.crc <= 1.0

    def test_crd_positive_and_bounded(self, params: OptParams) -> None:
        assert 0.0 < params.crd <= 1.0

    def test_crc_from_solver_yaml(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]
        c_cfg = dv.get("bess_charge_c_rate", {})
        expected_fv = c_cfg.get("fixed_value")
        expected_max = float(c_cfg.get("max", 1.0))
        expected = float(expected_fv) if expected_fv is not None else expected_max
        assert abs(params.crc - expected) < 1e-12

    def test_crd_from_solver_yaml(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]
        d_cfg = dv.get("bess_discharge_c_rate", {})
        expected_fv = d_cfg.get("fixed_value")
        expected_max = float(d_cfg.get("max", 1.0))
        expected = float(expected_fv) if expected_fv is not None else expected_max
        assert abs(params.crd - expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grid charges
# ─────────────────────────────────────────────────────────────────────────────

class TestGridCharges:
    def _ht_frac(self, config: FullConfig) -> float:
        return (
            config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]
            * PERCENT_TO_DECIMAL
        )

    def test_wheel(self, config: FullConfig, params: OptParams) -> None:
        rc = config.finance["regulatory_charges"]
        ht = self._ht_frac(config)
        lt = 1.0 - ht
        expected = (
            ht * float(rc["ht"]["wheeling_charge_inr_per_kwh"])
            + lt * float(rc["lt"]["wheeling_charge_inr_per_kwh"])
        )
        assert abs(params.wheel - expected) < 1e-12

    def test_tax(self, config: FullConfig, params: OptParams) -> None:
        rc = config.finance["regulatory_charges"]
        ht = self._ht_frac(config)
        lt = 1.0 - ht
        expected = (
            ht * float(rc["ht"]["electricity_tax_inr_per_kwh"])
            + lt * float(rc["lt"]["electricity_tax_inr_per_kwh"])
        )
        assert abs(params.tax - expected) < 1e-12

    def test_cap_rate(self, config: FullConfig, params: OptParams) -> None:
        rc = config.finance["regulatory_charges"]
        ht = self._ht_frac(config)
        lt = 1.0 - ht
        expected = (
            ht * (
                float(rc["ht"]["ctu_charge_inr_per_mw_per_month"])
                + float(rc["ht"]["stu_charge_inr_per_mw_per_month"])
                + float(rc["ht"]["sldc_charge_inr_per_mw_per_month"])
            )
            + lt * (
                float(rc["lt"]["ctu_charge_inr_per_mw_per_month"])
                + float(rc["lt"]["stu_charge_inr_per_mw_per_month"])
                + float(rc["lt"]["sldc_charge_inr_per_mw_per_month"])
            )
        )
        assert abs(params.cap_rate - expected) < 1e-12

    def test_cap_rate_non_negative(self, params: OptParams) -> None:
        assert params.cap_rate >= 0.0

    def test_wheel_non_negative(self, params: OptParams) -> None:
        assert params.wheel >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. CAPEX rates
# ─────────────────────────────────────────────────────────────────────────────

class TestCapex:
    def test_solar_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["capex"]["solar"]["cost_per_mwp"])
        assert abs(params.solar_rate - expected) < 1e-6

    def test_ac_dc(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["capex"]["solar"]["ac_dc_ratio"])
        assert abs(params.ac_dc - expected) < 1e-12

    def test_wind_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["capex"]["wind"]["cost_per_mw"])
        assert abs(params.wind_rate - expected) < 1e-6

    def test_bess_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["capex"]["bess"]["cost_per_mwh"])
        assert abs(params.bess_rate - expected) < 1e-6

    def test_trans_fixed(self, config: FullConfig, params: OptParams) -> None:
        cfg = config.finance["capex"]["transmission"]
        expected = float(cfg["length_km"]) * float(cfg["cost_per_km"])
        assert abs(params.trans_fixed - expected) < 1e-4

    def test_trans_fixed_matches_capex_model(self, config: FullConfig, params: OptParams) -> None:
        """Cross-check: trans_fixed must equal CapexModel(solar=0, wind=0, bess=0).total_capex."""
        from hybrid_plant.finance.capex_model import CapexModel
        result = CapexModel(config).compute(0.0, 0.0, 0.0)
        assert abs(params.trans_fixed - result["transmission_capex"]) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPEX base rates
# ─────────────────────────────────────────────────────────────────────────────

class TestOpex:
    def test_solar_om_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["solar"]["rate_lakh_per_mwp"]) * LAKH_TO_RS
        assert abs(params.solar_om_rate - expected) < 1.0  # INR, so 1 Rs tolerance

    def test_solar_om_esc(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["solar"]["escalation_percent"]) * PERCENT_TO_DECIMAL
        assert abs(params.solar_om_esc - expected) < 1e-12

    def test_wind_om_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["wind"]["rate_lakh_per_mw"]) * LAKH_TO_RS
        assert abs(params.wind_om_rate - expected) < 1.0

    def test_wind_om_esc(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["wind"]["escalation_percent"]) * PERCENT_TO_DECIMAL
        assert abs(params.wind_om_esc - expected) < 1e-12

    def test_land_lease_monthly(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["opex"]["land_lease"]["base_monthly_cost_crore"]) * CRORE_TO_RS
        )
        assert abs(params.land_lease_monthly - expected) < 1.0

    def test_land_esc(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["land_lease"]["escalation_percent"]) * PERCENT_TO_DECIMAL
        assert abs(params.land_esc - expected) < 1e-12

    def test_bess_om_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["opex"]["bess"]["rate_lakh_per_mwh"]) * LAKH_TO_RS
        assert abs(params.bess_om_rate - expected) < 1.0

    def test_solar_trans_om_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["opex"]["solar_transmission"]["rate_lakh_per_mwp"]) * LAKH_TO_RS
        )
        assert abs(params.solar_trans_om_rate - expected) < 1.0

    def test_wind_trans_om_rate(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["opex"]["wind_transmission"]["rate_lakh_per_mw"]) * LAKH_TO_RS
        )
        assert abs(params.wind_trans_om_rate - expected) < 1.0

    def test_insurance_pct(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["opex"]["insurance"]["percent_of_total_capex"]) * PERCENT_TO_DECIMAL
        )
        assert abs(params.insurance_pct - expected) < 1e-12

    def test_opex_cross_check_year1(self, config: FullConfig, params: OptParams) -> None:
        """
        Cross-check: for a reference sizing, OptParams OPEX formula (Year-1 only)
        must reproduce OpexModel.compute Year-1 total within 1 Rs.
        """
        solar_mw = 100.0
        wind_mw  = 50.0
        bess_mwh = 200.0
        capex_for_insurance = (
            params.solar_rate * solar_mw * params.ac_dc
            + params.wind_rate * wind_mw
            + params.bess_rate * bess_mwh
            + params.trans_fixed
        )

        # Year-1 OPEX from OptParams formula (t=1 → no escalation factor)
        solar_dc = solar_mw * params.ac_dc
        opex_y1 = (
            solar_dc * params.solar_om_rate
            + wind_mw * params.wind_om_rate
            + params.land_lease_monthly * MONTHS_PER_YEAR
            + bess_mwh * params.bess_om_rate
            + solar_dc * params.solar_trans_om_rate
            + wind_mw * params.wind_trans_om_rate
            + capex_for_insurance * params.insurance_pct
        )

        # Year-1 OPEX from OpexModel
        opex_model = OpexModel(config)
        projection, _ = opex_model.compute(solar_mw, wind_mw, bess_mwh, capex_for_insurance)
        assert abs(opex_y1 - projection[0]) < 1.0, (
            f"Year-1 OPEX mismatch: params={opex_y1:.2f}, OpexModel={projection[0]:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Financing scalars
# ─────────────────────────────────────────────────────────────────────────────

class TestFinancing:
    def test_debt_frac(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["financing"]["debt_percent"]) * PERCENT_TO_DECIMAL
        assert abs(params.debt_frac - expected) < 1e-12

    def test_eq_frac(self, config: FullConfig, params: OptParams) -> None:
        expected = float(config.finance["financing"]["equity_percent"]) * PERCENT_TO_DECIMAL
        assert abs(params.eq_frac - expected) < 1e-12

    def test_debt_eq_sum_to_one(self, params: OptParams) -> None:
        assert abs(params.debt_frac + params.eq_frac - 1.0) < 1e-12

    def test_r(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["financing"]["debt"]["interest_rate_percent"]) * PERCENT_TO_DECIMAL
        )
        assert abs(params.r - expected) < 1e-12

    def test_tenure(self, config: FullConfig, params: OptParams) -> None:
        expected = int(config.finance["financing"]["debt"]["tenure_years"])
        assert params.tenure == expected

    def test_roe(self, config: FullConfig, params: OptParams) -> None:
        expected = (
            float(config.finance["financing"]["equity"]["return_on_equity_percent"])
            * PERCENT_TO_DECIMAL
        )
        assert abs(params.roe - expected) < 1e-12

    def test_project_life(self, config: FullConfig, params: OptParams) -> None:
        expected = int(config.project["project"]["project_life_years"])
        assert params.project_life == expected

    def test_wacc_formula(self, config: FullConfig, params: OptParams) -> None:
        """WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)"""
        fin      = config.finance["financing"]
        tax_rate = float(fin["corporate_tax_rate_percent"]) * PERCENT_TO_DECIMAL
        expected = (
            params.debt_frac * params.r * (1.0 - tax_rate)
            + params.eq_frac * params.roe
        )
        assert abs(params.wacc - expected) < 1e-12

    def test_wacc_matches_lcoe_model(self, config: FullConfig, params: OptParams) -> None:
        """WACC in OptParams must equal LCOEModel.wacc."""
        assert abs(params.wacc - LCOEModel(config).wacc) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 7. Precomputed constants
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecomputed:
    def test_df_shape(self, params: OptParams) -> None:
        assert len(params.df) == params.project_life

    def test_df_values(self, params: OptParams) -> None:
        for i in range(params.project_life):
            expected = (1.0 + params.wacc) ** (-(i + 1))
            assert abs(params.df[i] - expected) < 1e-12

    def test_A_N(self, params: OptParams) -> None:
        expected = float(np.sum(params.df))
        assert abs(params.A_N - expected) < 1e-12

    def test_A_n_leq_A_N(self, params: OptParams) -> None:
        assert params.A_n <= params.A_N + 1e-12

    def test_A_n(self, params: OptParams) -> None:
        expected = float(np.sum(params.df[:params.tenure]))
        assert abs(params.A_n - expected) < 1e-12

    def test_emi_factor_formula(self, params: OptParams) -> None:
        r, n = params.r, params.tenure
        if r > 0:
            expected = r * (1 + r) ** n / ((1 + r) ** n - 1)
        else:
            expected = 1.0 / n
        assert abs(params.emi_factor - expected) < 1e-12

    def test_phi_formula(self, params: OptParams) -> None:
        expected = params.debt_frac * params.emi_factor * params.A_n + params.eq_frac * params.roe * params.A_N
        assert abs(params.phi - expected) < 1e-12

    def test_phi_cross_check_lcoemodel(self, config: FullConfig, params: OptParams) -> None:
        """
        Cross-check Φ: for any total_capex, NPV(debt service + ROE) from
        LCOEModel must equal total_capex × phi within 1e-10 relative error.
        """
        total_capex = 5e9  # arbitrary
        lcoe = LCOEModel(config)

        opex_zeros = [0.0] * params.project_life
        busbar_large = [1e7] * params.project_life   # large enough to avoid division error

        result = lcoe.compute(total_capex, opex_zeros, np.array(busbar_large))
        npv_financing = (
            result["npv_interest"] + result["npv_principal"] + result["npv_roe"]
        )
        assert _rel(npv_financing, total_capex * params.phi) < 1e-10, (
            f"phi mismatch: LCOEModel={npv_financing:.6f}, capex*phi={total_capex * params.phi:.6f}"
        )

    def test_G_solar_om_formula(self, params: OptParams) -> None:
        n = params.project_life
        expected = float(np.dot(params.df, (1.0 + params.solar_om_esc) ** np.arange(n)))
        assert _rel(params.G_solar_om, expected) < 1e-12

    def test_G_wind_om_formula(self, params: OptParams) -> None:
        n = params.project_life
        expected = float(np.dot(params.df, (1.0 + params.wind_om_esc) ** np.arange(n)))
        assert _rel(params.G_wind_om, expected) < 1e-12

    def test_G_land_formula(self, params: OptParams) -> None:
        n = params.project_life
        expected = float(np.dot(params.df, (1.0 + params.land_esc) ** np.arange(n)))
        assert _rel(params.G_land, expected) < 1e-12

    def test_non_escalating_G_equal_A_N(self, params: OptParams) -> None:
        for name, val in [
            ("G_bess_om",     params.G_bess_om),
            ("G_solar_trans", params.G_solar_trans),
            ("G_wind_trans",  params.G_wind_trans),
            ("G_insurance",   params.G_insurance),
        ]:
            assert abs(val - params.A_N) < 1e-12, f"{name} should equal A_N"

    def test_G_esc_cross_check_opex_npv(self, config: FullConfig, params: OptParams) -> None:
        """
        Cross-check G_* sums: NPV(solar_om) from OptParams formula must
        match NPV(solar_om column) from OpexModel across 25 years.
        """
        solar_mw = 100.0

        opex_model = OpexModel(config)
        # Zero wind, bess, and capex to isolate solar O&M NPV
        proj, breakdown = opex_model.compute(solar_mw, 0.0, 0.0, 0.0)

        solar_dc = solar_mw * params.ac_dc
        solar_om_series = [row["solar_om"] for row in breakdown]
        npv_solar_om_ref  = sum(
            solar_om_series[i] * float(params.df[i]) for i in range(params.project_life)
        )
        npv_solar_om_params = solar_dc * params.solar_om_rate * params.G_solar_om
        assert _rel(npv_solar_om_params, npv_solar_om_ref) < 1e-10, (
            f"G_solar_om NPV mismatch: params={npv_solar_om_params:.2f}, OpexModel={npv_solar_om_ref:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Degradation arrays
# ─────────────────────────────────────────────────────────────────────────────

class TestDegradation:
    def _load_curve(self, path_str: str, col: str) -> dict:
        import pandas as pd
        from hybrid_plant._paths import find_project_root
        df = pd.read_csv(find_project_root() / path_str)
        df.columns = df.columns.str.strip().str.lower()
        return dict(zip(df["year"].astype(int), df[col.lower()]))

    def test_d_s_shape(self, config: FullConfig, params: OptParams) -> None:
        assert len(params.d_s) == params.project_life

    def test_d_w_shape(self, config: FullConfig, params: OptParams) -> None:
        assert len(params.d_w) == params.project_life

    def test_d_b_shape(self, config: FullConfig, params: OptParams) -> None:
        assert len(params.d_b) == params.project_life

    def test_d_s_year1_is_one(self, params: OptParams) -> None:
        assert abs(params.d_s[0] - 1.0) < 1e-12, "Year-1 solar degradation must be 1.0 (fresh)"

    def test_d_w_year1_is_one(self, params: OptParams) -> None:
        assert abs(params.d_w[0] - 1.0) < 1e-12

    def test_d_b_year1_is_one(self, params: OptParams) -> None:
        assert abs(params.d_b[0] - 1.0) < 1e-12

    def test_d_s_non_increasing(self, params: OptParams) -> None:
        diffs = np.diff(params.d_s)
        assert float(diffs.max()) <= 1e-10, "Solar degradation should be non-increasing"

    def test_d_b_non_increasing(self, params: OptParams) -> None:
        diffs = np.diff(params.d_b)
        assert float(diffs.max()) <= 1e-10, "BESS SOH should be non-increasing"

    def test_d_s_matches_operating_value(self, config: FullConfig, params: OptParams) -> None:
        curve = self._load_curve(
            config.project["generation"]["solar"]["degradation"]["file"], "efficiency"
        )
        for i in range(params.project_life):
            year = i + 1
            expected = operating_value(curve, year)
            assert abs(params.d_s[i] - expected) < 1e-12, (
                f"d_s mismatch at year {year}: got {params.d_s[i]}, expected {expected}"
            )

    def test_d_w_matches_operating_value(self, config: FullConfig, params: OptParams) -> None:
        curve = self._load_curve(
            config.project["generation"]["wind"]["degradation"]["file"], "efficiency"
        )
        for i in range(params.project_life):
            year = i + 1
            expected = operating_value(curve, year)
            assert abs(params.d_w[i] - expected) < 1e-12

    def test_d_b_matches_operating_value(self, config: FullConfig, params: OptParams) -> None:
        curve = self._load_curve(
            config.bess["bess"]["degradation"]["file"], "soh"
        )
        for i in range(params.project_life):
            year = i + 1
            expected = operating_value(curve, year)
            assert abs(params.d_b[i] - expected) < 1e-12

    def test_D_s_formula(self, params: OptParams) -> None:
        expected = float(np.dot(params.df, params.d_s))
        assert _rel(params.D_s, expected) < 1e-12

    def test_D_w_formula(self, params: OptParams) -> None:
        expected = float(np.dot(params.df, params.d_w))
        assert _rel(params.D_w, expected) < 1e-12

    def test_D_b_formula(self, params: OptParams) -> None:
        expected = float(np.dot(params.df, params.d_b))
        assert _rel(params.D_b, expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 9. Decision-variable bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestBounds:
    def test_solar_bounds(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]["solar_capacity_mw"]
        assert params.s_min == float(dv["min"])
        assert params.s_max == float(dv["max"])

    def test_wind_bounds(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]["wind_capacity_mw"]
        assert params.w_min == float(dv["min"])
        assert params.w_max == float(dv["max"])

    def test_ppa_bounds(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]["ppa_capacity_mw"]
        assert params.p_min == float(dv["min"])
        assert params.p_max == float(dv["max"])

    def test_bess_bounds(self, config: FullConfig, params: OptParams) -> None:
        dv = config.solver["solver"]["decision_variables"]["bess_containers"]
        assert params.nb_min == int(dv["min"])
        assert params.nb_max == int(dv["max"])

    def test_all_mins_non_negative(self, params: OptParams) -> None:
        assert params.s_min  >= 0.0
        assert params.w_min  >= 0.0
        assert params.p_min  >= 0.0
        assert params.nb_min >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Optional constraint parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionalConstraintParams:
    def test_re_pen_enabled_matches_yaml(self, config: FullConfig, params: OptParams) -> None:
        hrep = (
            config.solver["solver"]
            .get("constraints", {})
            .get("hourly_re_penetration_penalty", {})
        )
        expected = bool(hrep.get("enabled", False))
        assert params.re_pen_penalty_enabled == expected

    def test_re_pen_min_pct(self, config: FullConfig, params: OptParams) -> None:
        hrep = (
            config.solver["solver"]
            .get("constraints", {})
            .get("hourly_re_penetration_penalty", {})
        )
        expected = float(hrep.get("min_percent", 0.0)) * PERCENT_TO_DECIMAL
        assert abs(params.re_pen_min_pct - expected) < 1e-12

    def test_re_pen_hours_are_zero_indexed(self, config: FullConfig, params: OptParams) -> None:
        hrep = (
            config.solver["solver"]
            .get("constraints", {})
            .get("hourly_re_penetration_penalty", {})
        )
        yaml_hours_1idx = hrep.get("penalty_hours", [])
        expected_0idx = tuple(int(h) - 1 for h in yaml_hours_1idx)
        assert params.re_pen_penalty_hours == expected_0idx

    def test_re_pen_hours_in_valid_range(self, params: OptParams) -> None:
        for h in params.re_pen_penalty_hours:
            assert 0 <= h <= 23, f"Penalty hour {h} out of 0–23 range"
