"""
test_energy.py
──────────────
Unit and integration tests for the energy simulation layer.

Covers
──────
- PlantEngine physical constraints (energy conservation, PPA cap, SOC bounds)
- GridInterface loss factor computation
- MeterLayer shortfall calculation
- Year1Engine end-to-end integration (solar-only and solar+wind cases)
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_plant.constants import HOURS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# GridInterface
# ─────────────────────────────────────────────────────────────────────────────

class TestGridInterface:

    def test_loss_factor_in_range(self, energy_engine):
        lf = energy_engine.grid.loss_factor
        assert 0 < lf <= 1.0, f"Loss factor {lf} out of range (0, 1]"

    def test_apply_losses_scales_correctly(self, energy_engine):
        lf   = energy_engine.grid.loss_factor
        arr  = np.ones(HOURS_PER_YEAR) * 10.0        # 10 MW flat
        result = energy_engine.grid.apply_losses(arr)
        np.testing.assert_allclose(result["meter_delivery"], arr * lf)

    def test_annual_meter_delivery_equals_sum(self, energy_engine):
        arr    = np.random.uniform(0, 50, HOURS_PER_YEAR)
        result = energy_engine.grid.apply_losses(arr)
        assert abs(result["annual_meter_delivery"] - np.sum(result["meter_delivery"])) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# MeterLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestMeterLayer:

    def test_no_shortfall_when_delivery_exceeds_load(self, energy_engine):
        load     = energy_engine.meter.load
        delivery = load + 1.0          # always surplus
        result   = energy_engine.meter.compute_shortfall(delivery)
        assert np.all(result["shortfall"] == 0)

    def test_shortfall_equals_deficit(self, energy_engine):
        load     = energy_engine.meter.load
        delivery = np.zeros_like(load)  # no RE delivery
        result   = energy_engine.meter.compute_shortfall(delivery)
        np.testing.assert_allclose(result["shortfall"], load)
        np.testing.assert_allclose(result["annual_discom"], np.sum(load))


# ─────────────────────────────────────────────────────────────────────────────
# PlantEngine — solar-only benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestPlantEngineSolarOnly:

    @pytest.fixture(scope="class")
    def result(self, energy_engine, solar_only_params):
        p = solar_only_params
        return energy_engine.plant.simulate(
            solar_capacity_mw  = p["solar_capacity_mw"],
            wind_capacity_mw   = p["wind_capacity_mw"],
            bess_containers    = p["bess_containers"],
            charge_c_rate      = p["charge_c_rate"],
            discharge_c_rate   = p["discharge_c_rate"],
            ppa_capacity_mw    = p["ppa_capacity_mw"],
            dispatch_priority  = p["dispatch_priority"],
            bess_charge_source = p["bess_charge_source"],
            loss_factor        = energy_engine.grid.loss_factor,
        )

    def test_output_keys_present(self, result):
        required = {
            "solar_direct_pre", "wind_direct_pre", "charge_pre", "discharge_pre",
            "curtailment_pre", "plant_export_pre", "energy_capacity_mwh",
            "bess_end_soc_mwh",
        }
        assert required.issubset(result.keys())

    def test_array_lengths(self, result):
        for key in ("solar_direct_pre", "wind_direct_pre", "charge_pre",
                    "discharge_pre", "curtailment_pre", "plant_export_pre"):
            assert len(result[key]) == HOURS_PER_YEAR, f"{key} wrong length"

    def test_energy_conservation(self, result, data, solar_only_params):
        # Invariant: all generation must be accounted for as direct delivery,
        # BESS charging, or curtailment.  solar_direct_pre and wind_direct_pre
        # store post-PPA-cap actual delivery values (not pre-cap allocations),
        # so this balance always holds exactly.
        p = solar_only_params
        raw_gen = np.sum(p["solar_capacity_mw"] * data["solar_cuf"]
                         + p["wind_capacity_mw"]  * data["wind_cuf"])
        rhs = (np.sum(result["solar_direct_pre"])
               + np.sum(result["wind_direct_pre"])
               + np.sum(result["charge_pre"])
               + np.sum(result["curtailment_pre"]))
        assert abs(raw_gen - rhs) < 1e-3, f"Energy not conserved: {raw_gen:.2f} vs {rhs:.2f}"

    def test_discharge_loss_correct(self, result, energy_engine):
        # discharge_loss should equal discharge_raw × (1 − discharge_eff),
        # where discharge_raw = discharge_pre / discharge_eff.
        eff = energy_engine.plant.discharge_eff
        discharge_pre  = result["discharge_pre"]
        discharge_loss = result["discharge_loss"]
        discharge_raw  = np.where(eff > 0, discharge_pre / eff, 0.0)
        expected_loss  = discharge_raw * (1 - eff)
        np.testing.assert_allclose(
            discharge_loss, expected_loss, rtol=1e-6,
            err_msg="discharge_loss does not equal discharge_raw × (1 − discharge_eff)",
        )

    def test_direct_delivery_matches_export(self, result):
        # plant_export_pre = solar_direct_pre + wind_direct_pre + discharge_pre
        export    = result["plant_export_pre"]
        recon     = result["solar_direct_pre"] + result["wind_direct_pre"] + result["discharge_pre"]
        np.testing.assert_allclose(
            export, recon, atol=1e-9,
            err_msg="plant_export_pre != solar_direct + wind_direct + discharge",
        )

    def test_ppa_cap_not_violated(self, result, solar_only_params):
        ppa = solar_only_params["ppa_capacity_mw"]
        assert np.all(result["plant_export_pre"] <= ppa + 1e-6), "PPA cap violated"

    def test_soc_non_negative(self, result):
        assert result["bess_end_soc_mwh"] >= -1e-6, "End SOC negative"

    def test_no_negative_arrays(self, result):
        for key in ("solar_direct_pre", "wind_direct_pre", "discharge_pre",
                    "curtailment_pre", "charge_pre"):
            assert np.all(result[key] >= -1e-9), f"{key} has negative values"

    def test_bess_was_used(self, result):
        assert np.sum(result["discharge_pre"]) > 0, "BESS never discharged"


# ─────────────────────────────────────────────────────────────────────────────
# Year1Engine — solar-only integration
# ─────────────────────────────────────────────────────────────────────────────

class TestYear1EngineSolarOnly:

    @pytest.fixture(scope="class")
    def result(self, energy_engine, solar_only_params):
        return energy_engine.evaluate(**solar_only_params)

    def test_meter_delivery_le_load(self, result, data):
        load = np.sum(data["load_profile"])
        meter = result["annual_meter_delivery"]
        assert meter <= load + 1.0, "Meter delivery exceeds annual load"

    def test_loss_factor_in_result(self, result):
        assert "loss_factor" in result
        assert 0 < result["loss_factor"] <= 1.0

    def test_shortfall_non_negative(self, result):
        assert np.all(result["shortfall"] >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# Year1Engine — solar + wind integration
# ─────────────────────────────────────────────────────────────────────────────

class TestYear1EngineSolarWind:

    @pytest.fixture(scope="class")
    def result(self, energy_engine, solar_wind_params):
        return energy_engine.evaluate(**solar_wind_params)

    def test_wind_direct_non_zero(self, result):
        assert np.sum(result["wind_direct_pre"]) > 0

    def test_energy_conservation(self, result, data, solar_wind_params):
        p = solar_wind_params
        raw_gen = np.sum(p["solar_capacity_mw"] * data["solar_cuf"]
                         + p["wind_capacity_mw"]  * data["wind_cuf"])
        rhs = (np.sum(result["solar_direct_pre"])
               + np.sum(result["wind_direct_pre"])
               + np.sum(result["charge_pre"])
               + np.sum(result["curtailment_pre"]))
        assert abs(raw_gen - rhs) < 1e-3
