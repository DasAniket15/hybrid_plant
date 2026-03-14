"""
plant_engine.py
───────────────
Pure plant-layer physics — all calculations are PRE-LOSS (busbar basis).

Dispatch sequence each hour
────────────────────────────
  1. Direct dispatch  (solar and/or wind → load, up to PPA cap)
  2. BESS charging    (surplus RE → battery, respecting charge power cap)
  3. Curtailment      (any remaining surplus after charging)
  4. Aux consumption  (parasitic draw on active BESS containers)
  5. BESS discharge   (meet residual shortfall, up to PPA headroom)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import HOURS_PER_DAY


class PlantEngine:
    """
    Simulates one full year (8760 h) of hybrid plant dispatch.

    All power and energy values are at the busbar (pre-grid-loss).
    The ``loss_factor`` parameter is passed in by ``Year1Engine`` from
    ``GridInterface`` so there is a single source of truth.

    Parameters
    ----------
    config : FullConfig
    data   : dict   — must contain ``solar_cuf``, ``wind_cuf``, ``load_profile``
    """

    def __init__(self, config: FullConfig, data: dict[str, Any]) -> None:
        self.config = config
        self.data   = data

        bess_cfg = config.bess["bess"]

        self.container_size: float = bess_cfg["container"]["size_mwh"]
        self.aux_per_hour:   float = (
            bess_cfg["container"]["auxiliary_consumption_mwh_per_day"] / HOURS_PER_DAY
        )
        self.charge_eff:    float = bess_cfg["efficiency"]["charge_efficiency"]
        self.discharge_eff: float = bess_cfg["efficiency"]["discharge_efficiency"]
        self.debug:         bool  = config.solver["solver"].get("debug_mode", False)

    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        solar_capacity_mw:  float,
        wind_capacity_mw:   float,
        bess_containers:    int,
        charge_c_rate:      float,
        discharge_c_rate:   float,
        ppa_capacity_mw:    float,
        dispatch_priority:  str,
        bess_charge_source: str,
        loss_factor:        float,
    ) -> dict[str, Any]:
        """
        Run the full 8760-hour dispatch simulation.

        Parameters
        ----------
        solar_capacity_mw  : AC solar installed capacity (MW)
        wind_capacity_mw   : Wind installed capacity (MW)
        bess_containers    : Number of BESS containers
        charge_c_rate      : BESS charge C-rate (fraction of energy capacity per hour)
        discharge_c_rate   : BESS discharge C-rate
        ppa_capacity_mw    : Contracted PPA export cap (MW)
        dispatch_priority  : "solar_first" | "wind_first" | "proportional"
        bess_charge_source : "solar_only" | "wind_only" | "solar_and_wind"
        loss_factor        : Grid loss factor from GridInterface

        Returns
        -------
        dict
            Full set of hourly arrays and scalar summaries — see return block.
        """
        solar_profile = self.data["solar_cuf"]
        wind_profile  = self.data["wind_cuf"]
        load          = self.data["load_profile"]

        hours = len(load)

        energy_capacity    = bess_containers * self.container_size
        charge_power_cap   = charge_c_rate   * energy_capacity
        discharge_power_cap = discharge_c_rate * energy_capacity

        soc: float = 0.0

        # Pre-allocate output arrays
        solar_direct       = np.zeros(hours)
        wind_direct        = np.zeros(hours)
        solar_direct_meter = np.zeros(hours)
        wind_direct_meter  = np.zeros(hours)
        solar_charge       = np.zeros(hours)
        wind_charge        = np.zeros(hours)
        charge             = np.zeros(hours)
        charge_loss        = np.zeros(hours)
        discharge          = np.zeros(hours)
        discharge_loss     = np.zeros(hours)
        aux_loss           = np.zeros(hours)
        discharge_meter    = np.zeros(hours)
        curtailment        = np.zeros(hours)
        export             = np.zeros(hours)

        for h in range(hours):

            # ── Raw generation (pre-loss) ─────────────────────────────────
            solar_pre  = solar_capacity_mw * solar_profile[h]
            wind_pre   = wind_capacity_mw  * wind_profile[h]
            total_pre  = solar_pre + wind_pre
            required_pre = load[h] / loss_factor

            # ── 1. Direct dispatch ────────────────────────────────────────
            if dispatch_priority == "solar_first":
                solar_d = min(solar_pre, required_pre)
                wind_d  = min(wind_pre, required_pre - solar_d)

            elif dispatch_priority == "wind_first":
                wind_d  = min(wind_pre, required_pre)
                solar_d = min(solar_pre, required_pre - wind_d)

            else:  # proportional
                if total_pre > 0:
                    ratio_s = solar_pre / total_pre
                    solar_d = min(solar_pre, required_pre * ratio_s)
                    wind_d  = min(wind_pre,  required_pre * (1 - ratio_s))
                else:
                    solar_d = wind_d = 0.0

            direct_pre = min(solar_d + wind_d, ppa_capacity_mw)  # apply PPA cap

            solar_d_meter  = solar_d    * loss_factor
            wind_d_meter   = wind_d     * loss_factor
            direct_meter   = direct_pre * loss_factor
            shortfall      = max(load[h] - direct_meter, 0.0)

            # ── 2. BESS charging (from surplus) ───────────────────────────
            if bess_charge_source == "solar_only":
                solar_surplus = solar_pre - solar_d
                wind_surplus  = 0.0
            elif bess_charge_source == "wind_only":
                solar_surplus = 0.0
                wind_surplus  = wind_pre - wind_d
            else:  # solar_and_wind
                solar_surplus = solar_pre - solar_d
                wind_surplus  = wind_pre  - wind_d

            total_surplus = solar_surplus + wind_surplus

            charge_pre = min(total_surplus, charge_power_cap, energy_capacity - soc)

            if total_surplus > 0:
                solar_charge_pre = charge_pre * (solar_surplus / total_surplus)
                wind_charge_pre  = charge_pre * (wind_surplus  / total_surplus)
            else:
                solar_charge_pre = wind_charge_pre = 0.0

            solar_charge[h]  = solar_charge_pre
            wind_charge[h]   = wind_charge_pre
            charge_loss[h]   = charge_pre * (1 - self.charge_eff)
            soc             += charge_pre * self.charge_eff
            charge[h]        = charge_pre

            # ── 3. Curtailment ────────────────────────────────────────────
            used = solar_d + wind_d + solar_charge_pre + wind_charge_pre
            curtailment[h] = max(total_pre - used, 0.0)

            # ── 4. Aux consumption (only when BESS has charge) ────────────
            if soc > 0:
                active_containers = min(bess_containers, math.ceil(soc / self.container_size))
                aux_energy = active_containers * self.aux_per_hour
                aux_loss[h] = aux_energy
                soc = max(soc - aux_energy, 0.0)

            # ── 5. BESS discharge ─────────────────────────────────────────
            required_discharge_pre = (
                shortfall / (self.discharge_eff * loss_factor) if shortfall > 0 else 0.0
            )
            remaining_headroom = ppa_capacity_mw - direct_pre

            discharge_pre = min(
                required_discharge_pre,
                soc,
                discharge_power_cap,
                remaining_headroom,
            )
            soc -= discharge_pre

            discharge_pre       = discharge_pre * self.discharge_eff   # post-efficiency
            discharge_post      = discharge_pre * loss_factor           # post-losses at meter

            discharge[h]        = discharge_pre
            discharge_loss[h]   = discharge_pre * (1 - self.discharge_eff)
            export[h]           = direct_pre + discharge_pre

            solar_direct[h]       = solar_d
            wind_direct[h]        = wind_d
            solar_direct_meter[h] = solar_d_meter
            wind_direct_meter[h]  = wind_d_meter
            discharge_meter[h]    = discharge_post

        # ── Debug assertions ──────────────────────────────────────────────────
        if self.debug:
            generation_total = float(np.sum(
                solar_capacity_mw * solar_profile + wind_capacity_mw * wind_profile
            ))
            rhs = float(np.sum(solar_direct) + np.sum(wind_direct)
                        + np.sum(charge) + np.sum(curtailment))
            if abs(generation_total - rhs) > 1e-6:
                raise ValueError("Energy conservation violated.")
            if np.any(export > ppa_capacity_mw + 1e-6):
                raise ValueError("PPA cap violated.")
            if soc < -1e-6:
                raise ValueError("SOC became negative.")
            if soc > energy_capacity + 1e-6:
                raise ValueError("SOC exceeded capacity.")
            if np.any(discharge > discharge_power_cap + 1e-6):
                raise ValueError("Discharge power limit exceeded.")
            if np.any(charge > charge_power_cap + 1e-6):
                raise ValueError("Charge power limit exceeded.")

        return {
            # Direct dispatch (pre-loss)
            "solar_direct_pre":    solar_direct,
            "wind_direct_pre":     wind_direct,
            # BESS charge split
            "solar_charge_pre":    solar_charge,
            "wind_charge_pre":     wind_charge,
            # BESS flows
            "charge_pre":          charge,
            "charge_loss":         charge_loss,
            "discharge_pre":       discharge,
            "discharge_loss":      discharge_loss,
            "aux_loss":            aux_loss,
            # Meter-side quantities
            "solar_direct_meter":  solar_direct_meter,
            "wind_direct_meter":   wind_direct_meter,
            "discharge_meter":     discharge_meter,
            # Plant totals
            "plant_export_pre":    export,
            "curtailment_pre":     curtailment,
            # BESS sizing (scalars)
            "energy_capacity_mwh": energy_capacity,
            "charge_power_mw":     charge_power_cap,
            "discharge_power_mw":  discharge_power_cap,
            "bess_end_soc_mwh":    soc,
        }
