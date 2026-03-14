"""
capex_model.py
──────────────
Computes project CAPEX broken down by component.

Solar CAPEX is on a DC MWp basis (AC capacity × AC/DC ratio).
Transmission CAPEX is fixed (length × cost per km).
All rates are sourced from ``finance.yaml`` — nothing is hardcoded.
"""

from __future__ import annotations

from typing import Any

from hybrid_plant.config_loader import FullConfig


class CapexModel:
    """
    Calculates total project CAPEX and a per-component breakdown.

    Parameters
    ----------
    config : FullConfig
    """

    def __init__(self, config: FullConfig) -> None:
        self._cfg = config.finance["capex"]

    def compute(
        self,
        solar_capacity_mw:        float,
        wind_capacity_mw:         float,
        bess_energy_capacity_mwh: float,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        solar_capacity_mw        : AC solar installed capacity (MW)
        wind_capacity_mw         : Wind installed capacity (MW)
        bess_energy_capacity_mwh : Total BESS energy capacity (MWh)

        Returns
        -------
        dict
            solar_dc_mwp       : float  DC solar capacity (MWp)
            solar_capex        : float  Rs
            wind_capex         : float  Rs
            bess_capex         : float  Rs
            transmission_capex : float  Rs
            total_capex        : float  Rs
        """
        cfg = self._cfg
        solar_dc_mwp = solar_capacity_mw * cfg["solar"]["ac_dc_ratio"]

        solar_capex        = solar_dc_mwp            * float(cfg["solar"]["cost_per_mwp"])
        wind_capex         = wind_capacity_mw         * float(cfg["wind"]["cost_per_mw"])
        bess_capex         = bess_energy_capacity_mwh * float(cfg["bess"]["cost_per_mwh"])
        transmission_capex = (
            float(cfg["transmission"]["length_km"]) * float(cfg["transmission"]["cost_per_km"])
        )
        total_capex = solar_capex + wind_capex + bess_capex + transmission_capex

        return {
            "solar_dc_mwp":         solar_dc_mwp,
            "solar_capex":          solar_capex,
            "wind_capex":           wind_capex,
            "bess_capex":           bess_capex,
            "transmission_capex":   transmission_capex,
            "total_capex":          total_capex,
        }
