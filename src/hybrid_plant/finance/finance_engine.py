"""
finance_engine.py
─────────────────
Top-level finance pipeline orchestrator.

Pipeline
────────
  1. CAPEX          → total project capital cost
  2. OPEX           → 25-year annual OPEX projection
  3. Energy         → 25-year busbar + meter energy (degraded)
  4. LCOE           → NPV(costs) / NPV(busbar energy)
  5. Landed tariff  → absolute annual costs → divide by meter kWh
  6. Client savings → hybrid vs. 100 % DISCOM baseline
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.finance.capex_model import CapexModel
from hybrid_plant.finance.energy_projection import EnergyProjection
from hybrid_plant.finance.landed_tariff_model import LandedTariffModel
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.opex_model import OpexModel
from hybrid_plant.finance.savings_model import SavingsModel


class FinanceEngine:
    """
    Orchestrates the full LCOE-based finance pipeline for a given
    plant configuration.

    Parameters
    ----------
    config : FullConfig
    data   : dict  — loaded by ``data_loader.load_timeseries_data``
    """

    def __init__(self, config: FullConfig, data: dict[str, Any]) -> None:
        self._config = config
        self._data   = data

        self._capex   = CapexModel(config)
        self._opex    = OpexModel(config)
        self._lcoe    = LCOEModel(config)
        self._landed  = LandedTariffModel(config)
        self._savings = SavingsModel(config, data)

    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        year1_results:               dict[str, Any],
        solar_capacity_mw:           float,
        wind_capacity_mw:            float,
        ppa_capacity_mw:             float,
        banked_energy_kwh_projection: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full finance pipeline for a given plant configuration.

        Parameters
        ----------
        year1_results                : dict from Year1Engine.evaluate()
        solar_capacity_mw            : AC solar capacity (MW)
        wind_capacity_mw             : Wind capacity (MW)
        ppa_capacity_mw              : Contracted PPA capacity (MW)
        banked_energy_kwh_projection : annual banked energy (kWh), defaults to zeros

        Returns
        -------
        dict — full finance results including primary outputs and all breakdowns
        """
        bess_mwh    = float(year1_results["energy_capacity_mwh"])
        loss_factor = float(year1_results["loss_factor"])

        # ── 1. CAPEX ──────────────────────────────────────────────────────────
        capex     = self._capex.compute(solar_capacity_mw, wind_capacity_mw, bess_mwh)
        total_capex = capex["total_capex"]

        # ── 2. OPEX ───────────────────────────────────────────────────────────
        opex_projection, opex_breakdown = self._opex.compute(
            solar_capacity_mw, wind_capacity_mw, bess_mwh, total_capex
        )

        # ── 3. Energy projection ──────────────────────────────────────────────
        projection = EnergyProjection(
            config            = self._config,
            data              = self._data,
            year1_results     = year1_results,
            solar_capacity_mw = solar_capacity_mw,
            wind_capacity_mw  = wind_capacity_mw,
            loss_factor       = loss_factor,
        ).project()

        busbar_mwh = projection["delivered_pre_mwh"]
        meter_mwh  = projection["delivered_meter_mwh"]

        # ── 4. LCOE ───────────────────────────────────────────────────────────
        lcoe_result = self._lcoe.compute(total_capex, opex_projection, busbar_mwh)
        lcoe = lcoe_result["lcoe_inr_per_kwh"]
        wacc = lcoe_result["wacc"]

        # ── 5. Landed tariff ──────────────────────────────────────────────────
        landed_result = self._landed.compute(
            lcoe_inr_per_kwh             = lcoe,
            ppa_capacity_mw              = ppa_capacity_mw,
            busbar_energy_mwh_projection = busbar_mwh,
            meter_energy_mwh_projection  = meter_mwh,
            banked_energy_kwh_projection = banked_energy_kwh_projection,
        )

        # ── 6. Client savings ─────────────────────────────────────────────────
        savings_result = self._savings.compute(
            landed_tariff_series        = landed_result["landed_tariff_series"],
            meter_energy_mwh_projection = meter_mwh,
            wacc                        = wacc,
        )

        return {
            # Primary outputs
            "lcoe_inr_per_kwh":        lcoe,
            "landed_tariff_series":    landed_result["landed_tariff_series"],
            "annual_savings_year1":    savings_result["annual_savings_year1"],
            "savings_npv":             savings_result["savings_npv"],
            # Supporting
            "wacc":                    wacc,
            "capex":                   capex,
            "opex_projection":         opex_projection,
            "opex_breakdown":          opex_breakdown,
            "energy_projection":       projection,
            "lcoe_breakdown":          lcoe_result,
            "landed_tariff_breakdown": landed_result,
            "savings_breakdown":       savings_result,
        }
