"""
test_finance.py
───────────────
Unit and integration tests for the finance pipeline.

Covers
──────
- CapexModel component sums
- OpexModel escalation direction and component positivity
- LCOEModel: WACC formula, NPV positivity, LCOE in plausible range
- LandedTariffModel: tariff monotonicity, unit rate positivity
- SavingsModel: baseline, savings sign, NPV sign
- FinanceEngine end-to-end integration (solar-only benchmark)
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_plant.finance.capex_model import CapexModel
from hybrid_plant.finance.opex_model import OpexModel
from hybrid_plant.finance.lcoe_model import LCOEModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared Year-1 result (session-scoped via fixture dependency)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def year1(energy_engine, solar_only_params):
    return energy_engine.evaluate(**solar_only_params)


@pytest.fixture(scope="module")
def finance(finance_engine, year1, solar_only_params):
    p = solar_only_params
    return finance_engine.evaluate(
        year1_results     = year1,
        solar_capacity_mw = p["solar_capacity_mw"],
        wind_capacity_mw  = p["wind_capacity_mw"],
        ppa_capacity_mw   = p["ppa_capacity_mw"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CapexModel
# ─────────────────────────────────────────────────────────────────────────────

class TestCapexModel:

    def test_total_equals_sum_of_components(self, config, solar_only_params):
        model  = CapexModel(config)
        p      = solar_only_params
        result = model.compute(
            solar_capacity_mw        = p["solar_capacity_mw"],
            wind_capacity_mw         = p["wind_capacity_mw"],
            bess_energy_capacity_mwh = p["bess_containers"] * config.bess["bess"]["container"]["size_mwh"],
        )
        components = (result["solar_capex"] + result["wind_capex"]
                      + result["bess_capex"] + result["transmission_capex"])
        assert abs(result["total_capex"] - components) < 1.0

    def test_all_components_non_negative(self, config, solar_only_params):
        model  = CapexModel(config)
        p      = solar_only_params
        result = model.compute(
            p["solar_capacity_mw"], p["wind_capacity_mw"],
            p["bess_containers"] * config.bess["bess"]["container"]["size_mwh"],
        )
        for key in ("solar_capex","wind_capex","bess_capex","transmission_capex","total_capex"):
            assert result[key] >= 0, f"{key} is negative"

    def test_zero_wind_zero_wind_capex(self, config, solar_only_params):
        model  = CapexModel(config)
        p      = solar_only_params
        result = model.compute(p["solar_capacity_mw"], 0.0, 100.0)
        assert result["wind_capex"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OpexModel
# ─────────────────────────────────────────────────────────────────────────────

class TestOpexModel:

    @pytest.fixture(scope="class")
    def opex_result(self, config, solar_only_params):
        model  = OpexModel(config)
        p      = solar_only_params
        bess_mwh = p["bess_containers"] * config.bess["bess"]["container"]["size_mwh"]
        capex_dummy = 1e9
        return model.compute(p["solar_capacity_mw"], p["wind_capacity_mw"], bess_mwh, capex_dummy)

    def test_projection_length(self, opex_result, config):
        projection, _ = opex_result
        assert len(projection) == config.project["project"]["project_life_years"]

    def test_all_years_positive(self, opex_result):
        projection, _ = opex_result
        assert all(v > 0 for v in projection)

    def test_year25_ge_year1(self, opex_result):
        projection, _ = opex_result
        assert projection[-1] >= projection[0], "Total OPEX should not decrease"

    def test_breakdown_totals_match_projection(self, opex_result):
        projection, breakdown = opex_result
        for i, (total, row) in enumerate(zip(projection, breakdown)):
            assert abs(total - row["total"]) < 1.0, f"Year {i+1} total mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# LCOEModel
# ─────────────────────────────────────────────────────────────────────────────

class TestLCOEModel:

    def test_wacc_positive(self, config):
        model = LCOEModel(config)
        assert model.wacc > 0

    def test_wacc_formula(self, config):
        from hybrid_plant.constants import PERCENT_TO_DECIMAL
        fin     = config.finance["financing"]
        d       = fin["debt_percent"]    * PERCENT_TO_DECIMAL
        e       = fin["equity_percent"]  * PERCENT_TO_DECIMAL
        rd      = fin["debt"]["interest_rate_percent"] * PERCENT_TO_DECIMAL
        re      = fin["equity"]["return_on_equity_percent"] * PERCENT_TO_DECIMAL
        tc      = fin["corporate_tax_rate_percent"] * PERCENT_TO_DECIMAL
        expected = d * rd * (1 - tc) + e * re
        model    = LCOEModel(config)
        assert abs(model.wacc - expected) < 1e-10

    def test_lcoe_in_plausible_range(self, finance):
        lcoe = finance["lcoe_inr_per_kwh"]
        assert 3.0 < lcoe < 15.0, f"LCOE {lcoe} outside plausible range"


# ─────────────────────────────────────────────────────────────────────────────
# FinanceEngine — end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestFinanceEngineIntegration:

    def test_required_keys_present(self, finance):
        required = {
            "lcoe_inr_per_kwh", "landed_tariff_series", "annual_savings_year1",
            "savings_npv", "wacc", "capex", "opex_projection", "opex_breakdown",
            "energy_projection", "lcoe_breakdown", "landed_tariff_breakdown",
            "savings_breakdown",
        }
        assert required.issubset(finance.keys())

    def test_landed_tariff_series_length(self, finance, config):
        lts = finance["landed_tariff_series"]
        assert len(lts) == config.project["project"]["project_life_years"]

    def test_landed_tariff_below_discom(self, finance):
        lts    = finance["landed_tariff_series"]
        discom = finance["savings_breakdown"]["discom_tariff"]
        assert lts[0] < discom, "Year-1 landed tariff should be below DISCOM tariff"

    def test_savings_year1_positive(self, finance):
        assert finance["annual_savings_year1"] > 0

    def test_savings_npv_positive(self, finance):
        assert finance["savings_npv"] > 0

    def test_energy_projection_shape(self, finance, config):
        n = config.project["project"]["project_life_years"]
        ep = finance["energy_projection"]
        for key in ("delivered_pre_mwh", "delivered_meter_mwh", "solar_direct_mwh"):
            assert len(ep[key]) == n, f"{key} wrong length"

    def test_meter_le_busbar(self, finance):
        ep = finance["energy_projection"]
        assert np.all(ep["delivered_meter_mwh"] <= ep["delivered_pre_mwh"] + 1e-6)

    def test_energy_degrades_over_time(self, finance):
        ep = finance["energy_projection"]
        assert ep["delivered_pre_mwh"][-1] < ep["delivered_pre_mwh"][0], \
            "Busbar energy should degrade over project life"

    def test_discom_tariff_reasonable(self, finance):
        discom = finance["savings_breakdown"]["discom_tariff"]
        assert 5.0 < discom < 15.0, f"DISCOM tariff {discom} outside plausible range"
