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
  5. BESS discharge   (ToD-aware, priority-ordered — see below)

ToD-aware BESS discharge
─────────────────────────
Discharge priority (highest value first):
  1. Evening peak shortfall today   (hod 18–21, rate ₹9.182)
  2. Morning peak shortfall tomorrow (hod  7–10, rate ₹9.182)
  3. Normal slots today/tonight     (hod  0–6, 15–17, 22–23, rate ₹8.687)
  4. Solar-offpeak slots            (hod 11–14, rate ₹8.027) — lowest priority

Two SOC reservations are maintained and updated at two fixed trigger points per day:

  hod = 11  (solar window opens)
      • Clear rsrv_morning_next (morning peak just ended)
      • Compute rsrv_fwd_evening: forward estimate of evening peak need,
        bounded by projected SOC at end of charging window.
        Prevents cheap discharge during solar hours from stealing SOC
        needed for tonight's evening peak.

  hod = 15  (solar window closes, actual SOC known)
      • Zero rsrv_fwd_evening (replaced by definitive value)
      • Set rsrv_evening:      definitive reservation for hod 18–21 today
      • Set rsrv_morning_next: reservation for hod  7–10 tomorrow,
        funded from SOC remaining after rsrv_evening

Available SOC per period:
  morning_peak  → soc                               (use the reservation)
  solar_offpeak → max(soc − rsrv_fwd_evening, 0)
  normal        → max(soc − rsrv_evening − rsrv_morning_next, 0)
  evening_peak  → max(soc − rsrv_morning_next, 0)   (protect tomorrow's morning)
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

        # ── ToD period sets (0-indexed hod, i.e. h % 24) ─────────────────────
        # LT periods are the canonical reference for hour definitions
        # (HT has identical hours; only rates differ).
        # Tariff hours are 1-indexed (1–24); convert to 0-indexed (0–23).
        tod = config.tariffs["discom"]["lt"]["tod_periods"]

        self.morning_peak_hods:  set[int] = {(h - 1) % 24 for h in tod["morning_peak"]["hours"]}
        self.evening_peak_hods:  set[int] = {(h - 1) % 24 for h in tod["evening_peak"]["hours"]}
        self.solar_offpeak_hods: set[int] = {(h - 1) % 24 for h in tod["solar_offpeak"]["hours"]}

        # Convenience union (for debug / external reference)
        self.peak_hours: set[int] = self.morning_peak_hods | self.evening_peak_hods

        # ── Dispatch mask (optional client override) ──────────────────────────
        # Loaded once here; applied at two points in simulate():
        #   1. Pre-compute pass  — blocked hours zeroed in re_shortfall so the
        #      ToD reservation planner never ring-fences SOC for them (full block).
        #   2. Discharge step    — discharge_raw forced to 0 for blocked hours.
        # Charging is never affected.
        _mask_cfg = config.bess["bess"].get("dispatch_mask", {})
        _blocked: set[int] = (
            {int(h) for h in _mask_cfg.get("blocked_hours", [])}
            if _mask_cfg.get("enabled", False)
            else set()
        )
        self.discharge_allowed_hods: set[int] = set(range(24)) - _blocked

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
        bess_soh_factor:    float = 1.0,
    ) -> dict[str, Any]:
        """
        Run the full 8760-hour dispatch simulation.

        Parameters
        ----------
        solar_capacity_mw  : AC solar installed capacity (MW)
        wind_capacity_mw   : Wind installed capacity (MW)
        bess_containers    : Number of BESS containers (physical, unchanged by degradation)
        charge_c_rate      : BESS charge C-rate (fraction of energy capacity per hour)
        discharge_c_rate   : BESS discharge C-rate
        ppa_capacity_mw    : Contracted PPA export cap (MW)
        dispatch_priority  : "solar_first" | "wind_first" | "proportional"
        bess_charge_source : "solar_only" | "wind_only" | "solar_and_wind"
        loss_factor        : Grid loss factor from GridInterface
        bess_soh_factor    : State-of-health multiplier [0–1], default 1.0 (Year 1).
                             Scales effective energy capacity and therefore power caps.
                             Used by EnergyProjection for year-wise re-simulation.

        Returns
        -------
        dict
            Full set of hourly arrays and scalar summaries — see return block.
        """
        solar_profile = self.data["solar_cuf"]
        wind_profile  = self.data["wind_cuf"]
        load          = self.data["load_profile"]

        hours = len(load)

        # Effective energy capacity shrinks with SOH; power caps scale proportionally
        # since they are expressed as C-rate × energy capacity.
        energy_capacity     = bess_containers * self.container_size * bess_soh_factor
        charge_power_cap    = charge_c_rate    * energy_capacity
        discharge_power_cap = discharge_c_rate * energy_capacity

        # ── Pre-compute RE-only shortfall and charging surplus (BESS-independent) ──
        # Used by the ToD reservation planner; computed once, O(8760).
        re_shortfall = np.empty(hours)
        re_surplus   = np.empty(hours)
        for _h in range(hours):
            _s   = solar_capacity_mw * solar_profile[_h]
            _w   = wind_capacity_mw  * wind_profile[_h]
            _req = load[_h] / loss_factor

            if dispatch_priority == "solar_first":
                _sd = min(_s, _req)
                _wd = min(_w, _req - _sd)
            elif dispatch_priority == "wind_first":
                _wd = min(_w, _req)
                _sd = min(_s, _req - _wd)
            else:  # proportional
                _tot = _s + _w
                if _tot > 0:
                    _sd = min(_s, _req * _s / _tot)
                    _wd = min(_w, _req * _w / _tot)
                else:
                    _sd = _wd = 0.0

            _direct = min(_sd + _wd, ppa_capacity_mw)
            re_shortfall[_h] = max(load[_h] - _direct * loss_factor, 0.0)

            # Full block: reservation planner treats blocked hours as zero-shortfall
            # so no SOC is ring-fenced for hours that will never discharge.
            if (_h % 24) not in self.discharge_allowed_hods:
                re_shortfall[_h] = 0.0

            if bess_charge_source == "solar_only":
                re_surplus[_h] = max(_s - _sd, 0.0)
            elif bess_charge_source == "wind_only":
                re_surplus[_h] = max(_w - _wd, 0.0)
            else:  # solar_and_wind
                re_surplus[_h] = max(_s - _sd, 0.0) + max(_w - _wd, 0.0)

        # ── State ────────────────────────────────────────────────────────────
        soc: float = 0.0

        # Three reservation variables — see module docstring for lifecycle.
        rsrv_evening:      float = 0.0  # ring-fenced for tonight's evening peak
        rsrv_morning_next: float = 0.0  # ring-fenced for tomorrow's morning peak
        rsrv_fwd_evening:  float = 0.0  # forward estimate during solar window

        # ── Pre-allocate output arrays ───────────────────────────────────────
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

            hod          = h % 24
            day_start_h  = h - hod   # index of midnight for the current day

            # ── ToD reservation updates ──────────────────────────────────────
            #
            # Trigger 1 — hod 11 (first solar-offpeak hour, morning peak just ended)
            #   • Zero out rsrv_morning_next (consumed / expired).
            #   • Estimate rsrv_fwd_evening: walk the charging window (hod 11–14)
            #     to project SOC at end of solar charging, then cap evening need
            #     to that projected SOC.  Bounding by projected (not current) SOC
            #     prevents the estimate from being zero on days with low carry-in.
            #
            if hod == 11:
                rsrv_morning_next = 0.0

                evening_indices = [
                    day_start_h + ep for ep in [18, 19, 20, 21]
                    if day_start_h + ep < hours
                ]
                if evening_indices and energy_capacity > 0:
                    # Walk the charging window to estimate post-charging SOC
                    soc_est = soc
                    for _cp in [11, 12, 13, 14]:
                        _ci = day_start_h + _cp
                        if _ci >= hours:
                            break
                        _c = (
                            min(re_surplus[_ci], charge_power_cap,
                                max(energy_capacity - soc_est, 0.0))
                            * self.charge_eff
                        )
                        soc_est = min(soc_est + _c, energy_capacity)

                    evening_need = sum(
                        re_shortfall[fh] / (self.discharge_eff * loss_factor)
                        for fh in evening_indices
                    )
                    evening_need = min(evening_need,
                                       discharge_power_cap * len(evening_indices))
                    rsrv_fwd_evening = min(evening_need, soc_est)
                else:
                    rsrv_fwd_evening = 0.0

            # Trigger 2 — hod 15 (first normal-day hour, solar window just closed)
            #   • Replace forward estimate with definitive rsrv_evening based on
            #     actual post-charging SOC.
            #   • Compute rsrv_morning_next for hod 7–10 of tomorrow, funded from
            #     SOC remaining after tonight's evening reservation.
            #
            elif hod == 15:
                rsrv_fwd_evening = 0.0

                evening_indices = [
                    day_start_h + ep for ep in [18, 19, 20, 21]
                    if day_start_h + ep < hours
                ]
                morning_indices = [
                    day_start_h + 24 + mp for mp in [7, 8, 9, 10]
                    if day_start_h + 24 + mp < hours
                ]

                evening_need = (
                    sum(re_shortfall[fh] / (self.discharge_eff * loss_factor)
                        for fh in evening_indices)
                    if evening_indices else 0.0
                )
                evening_need  = min(evening_need,
                                    discharge_power_cap * max(len(evening_indices), 1))
                rsrv_evening  = min(evening_need, soc)

                morning_need = (
                    sum(re_shortfall[fh] / (self.discharge_eff * loss_factor)
                        for fh in morning_indices)
                    if morning_indices else 0.0
                )
                morning_need      = min(morning_need,
                                        discharge_power_cap * max(len(morning_indices), 1))
                rsrv_morning_next = min(morning_need,
                                        max(soc - rsrv_evening, 0.0))

            # ── Raw generation (pre-loss) ────────────────────────────────────
            solar_pre    = solar_capacity_mw * solar_profile[h]
            wind_pre     = wind_capacity_mw  * wind_profile[h]
            total_pre    = solar_pre + wind_pre
            required_pre = load[h] / loss_factor

            # ── 1. Direct dispatch ───────────────────────────────────────────
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

            direct_pre = min(solar_d + wind_d, ppa_capacity_mw)   # apply PPA cap

            solar_d_meter = solar_d    * loss_factor
            wind_d_meter  = wind_d     * loss_factor
            shortfall     = max(load[h] - direct_pre * loss_factor, 0.0)

            # ── 2. BESS charging (from surplus) ──────────────────────────────
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

            solar_charge[h] = solar_charge_pre
            wind_charge[h]  = wind_charge_pre
            charge_loss[h]  = charge_pre * (1 - self.charge_eff)
            soc            += charge_pre * self.charge_eff
            charge[h]       = charge_pre

            # ── 3. Curtailment ───────────────────────────────────────────────
            used           = solar_d + wind_d + solar_charge_pre + wind_charge_pre
            curtailment[h] = max(total_pre - used, 0.0)

            # ── 4. Aux consumption (only when BESS has charge) ───────────────
            if soc > 0:
                # Each container's effective capacity degrades with SOH, so
                # ceil(soc / effective_size) may be higher than at nameplate.
                effective_container_size = self.container_size * bess_soh_factor
                active_containers = min(
                    bess_containers,
                    math.ceil(soc / effective_container_size) if effective_container_size > 0 else bess_containers,
                )
                aux_energy  = active_containers * self.aux_per_hour
                aux_loss[h] = aux_energy
                soc         = max(soc - aux_energy, 0.0)

            # ── 5. BESS discharge (ToD-aware priority dispatch) ──────────────
            #
            # Available SOC depends on which period we're in:
            #
            #   morning_peak  → soc  (ring-fenced SOC is FOR this window)
            #   solar_offpeak → soc − rsrv_fwd_evening
            #   normal        → soc − rsrv_evening − rsrv_morning_next
            #   evening_peak  → soc − rsrv_morning_next  (protect tomorrow's morning)
            #
            required_discharge_pre = (
                shortfall / (self.discharge_eff * loss_factor) if shortfall > 0 else 0.0
            )
            remaining_headroom = ppa_capacity_mw - direct_pre

            is_morning_peak  = hod in self.morning_peak_hods
            is_evening_peak  = hod in self.evening_peak_hods
            is_solar_offpeak = hod in self.solar_offpeak_hods

            if is_morning_peak:
                available_soc = soc
            elif is_evening_peak:
                available_soc = max(soc - rsrv_morning_next, 0.0)
            elif is_solar_offpeak:
                available_soc = max(soc - rsrv_fwd_evening, 0.0)
            else:  # normal (day or night)
                available_soc = max(soc - rsrv_evening - rsrv_morning_next, 0.0)

            discharge_raw = min(
                required_discharge_pre,
                available_soc,
                discharge_power_cap,
                remaining_headroom / self.discharge_eff,  # headroom is on busbar export (direct_pre + discharge_raw×eff ≤ ppa_cap)
            ) if hod in self.discharge_allowed_hods else 0.0
            soc -= discharge_raw

            # Consume reservations as they are used
            if is_morning_peak:
                rsrv_morning_next = max(rsrv_morning_next - discharge_raw, 0.0)
            elif is_evening_peak:
                rsrv_evening = max(rsrv_evening - discharge_raw, 0.0)

            discharge_pre  = discharge_raw * self.discharge_eff   # post-efficiency
            discharge_post = discharge_pre  * loss_factor          # post-losses at meter

            discharge[h]      = discharge_pre
            # True round-trip loss = discharge_raw - discharge_pre = discharge_raw × (1 - η_d).
            # Previously used `discharge_pre × (1 - η_d)` which under-reported the loss by
            # a factor of η_d (a cosmetic bug — not used downstream in any economic calc,
            # but the returned diagnostic array was wrong).
            discharge_loss[h] = discharge_raw * (1 - self.discharge_eff)
            export[h]         = direct_pre + discharge_pre

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