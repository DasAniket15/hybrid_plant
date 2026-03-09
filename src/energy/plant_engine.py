import numpy as np
import math


class PlantEngine:
    """
    Pure plant-layer physics.
    All calculations are PRE-LOSS.
    """

    def __init__(self, config, data):

        self.config = config
        self.data = data

        bess_cfg = config.bess["bess"]

        self.container_size = bess_cfg["container"]["size_mwh"]
        self.aux_per_day = bess_cfg["container"]["auxiliary_consumption_mwh_per_day"]

        self.charge_eff = bess_cfg["efficiency"]["charge_efficiency"]
        self.discharge_eff = bess_cfg["efficiency"]["discharge_efficiency"]

        self.aux_per_hour = self.aux_per_day / 24

        self.debug = config.solver["solver"].get("debug_mode", False)

    # ------------------------------------------------------------------

    def simulate(
        self,
        solar_capacity_mw,
        wind_capacity_mw,
        bess_containers,
        charge_c_rate,
        discharge_c_rate,
        ppa_capacity_mw,
        dispatch_priority,
        bess_charge_source,
        loss_factor
    ):

        solar_profile = self.data["solar_cuf"]
        wind_profile = self.data["wind_cuf"]
        load = self.data["load_profile"]

        hours = len(load)

        energy_capacity = bess_containers * self.container_size
        charge_power_cap = charge_c_rate * energy_capacity
        discharge_power_cap = discharge_c_rate * energy_capacity

        soc = 0.0

        solar_direct = np.zeros(hours)
        wind_direct = np.zeros(hours)

        solar_direct_meter = np.zeros(hours)
        wind_direct_meter = np.zeros(hours)
        
        charge = np.zeros(hours)
        solar_charge = np.zeros(hours)
        wind_charge = np.zeros(hours)
        charge_loss = np.zeros(hours)

        discharge = np.zeros(hours)
        discharge_loss = np.zeros(hours)
        aux_loss = np.zeros(hours)
        discharge_meter = np.zeros(hours)
        
        curtailment = np.zeros(hours)
        export = np.zeros(hours)

        for h in range(hours):

            # ---------------------------------
            # Raw generation (pre-loss)
            # ---------------------------------

            solar_pre = solar_capacity_mw * solar_profile[h]
            wind_pre = wind_capacity_mw * wind_profile[h]
            total_pre = solar_pre + wind_pre

            required_pre = load[h] / loss_factor

            # ---------------------------------
            # Direct dispatch
            # ---------------------------------

            if dispatch_priority == "solar_first":

                solar_d = min(solar_pre, required_pre)
                wind_d = min(wind_pre, required_pre - solar_d)

            elif dispatch_priority == "wind_first":

                wind_d = min(wind_pre, required_pre)
                solar_d = min(solar_pre, required_pre - wind_d)

            else:  # proportional

                if total_pre > 0:

                    ratio_s = solar_pre / total_pre
                    solar_d = min(solar_pre, required_pre * ratio_s)
                    wind_d = min(wind_pre, required_pre * (1 - ratio_s))

                else:

                    solar_d = 0
                    wind_d = 0

            direct_pre = solar_d + wind_d

            # Apply PPA export cap
            direct_pre = min(direct_pre, ppa_capacity_mw)

            solar_d_meter = solar_d * loss_factor
            wind_d_meter = wind_d * loss_factor
            direct_meter = direct_pre * loss_factor
            shortfall = max(load[h] - direct_meter, 0)

            # =====================================================
            # 1️⃣ BESS CHARGING (from surplus)
            # =====================================================

            if bess_charge_source == "solar_only":
                solar_surplus = solar_pre - solar_d
                wind_surplus = 0

            elif bess_charge_source == "wind_only":
                solar_surplus = 0
                wind_surplus = wind_pre - wind_d

            else:
                solar_surplus = solar_pre - solar_d
                wind_surplus = wind_pre - wind_d

            total_surplus = solar_surplus + wind_surplus

            charge_pre = min(
                total_surplus,
                charge_power_cap,
                energy_capacity - soc
            )

            # split charge sources
            if total_surplus > 0:
                solar_charge_pre = charge_pre * (solar_surplus / total_surplus)
                wind_charge_pre = charge_pre * (wind_surplus / total_surplus)
            
            else:
                solar_charge_pre = 0
                wind_charge_pre = 0

            solar_charge[h] = solar_charge_pre
            wind_charge[h] = wind_charge_pre

            # charging loss
            charge_loss[h] = charge_pre * (1 - self.charge_eff)

            # SOC increase
            soc += charge_pre * self.charge_eff

            charge[h] = charge_pre

            # =====================================================
            # 2️⃣ CURTAILMENT
            # =====================================================

            used = solar_d + wind_d + solar_charge_pre + wind_charge_pre
            curtailment[h] = max(total_pre - used, 0)

            # =====================================================
            # 3️⃣ AUX CONSUMPTION (only if SOC > 0)
            # =====================================================

            if soc > 0:

                active_containers = min(
                    bess_containers,
                    math.ceil(soc / self.container_size)
                )

                aux_energy = active_containers * self.aux_per_hour

                aux_loss[h] = aux_energy

                soc = max(soc - aux_energy, 0)

            # =====================================================
            # 4️⃣ BESS DISCHARGE
            # =====================================================

            required_discharge_pre = (
                shortfall / (self.discharge_eff * loss_factor)
                if shortfall > 0 else 0
            )

            remaining_headroom = ppa_capacity_mw - direct_pre

            discharge_pre = min(
                required_discharge_pre,
                soc,
                discharge_power_cap,
                remaining_headroom
            )

            soc -= discharge_pre          
                        
            discharge_pre = discharge_pre * self.discharge_eff  # Post discharge efficiency
            discharge_post = discharge_pre * loss_factor  # Post losses at meter

            discharge[h] = discharge_pre

            discharge_loss[h] = discharge_pre * (1 - self.discharge_eff)

            # =====================================================
            # Export
            # =====================================================

            export[h] = direct_pre + discharge_pre

            solar_direct[h] = solar_d
            wind_direct[h] = wind_d
            solar_direct_meter[h] = solar_d_meter
            wind_direct_meter[h] = wind_d_meter
            discharge_meter[h] = discharge_post

        # ======================================================
        # DEBUG ASSERTIONS
        # ======================================================

        if self.debug:

            generation_total = np.sum(
                solar_capacity_mw * solar_profile
                + wind_capacity_mw * wind_profile
            )

            rhs = (
                np.sum(solar_direct)
                + np.sum(wind_direct)
                + np.sum(charge)
                + np.sum(curtailment)
            )

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

        # ======================================================
        # RETURN RESULTS
        # ======================================================

        return {

            "solar_direct_pre": solar_direct,
            "wind_direct_pre": wind_direct,
            
            "solar_charge_pre": solar_charge,
            "wind_charge_pre": wind_charge,

            "charge_pre": charge,
            "charge_loss": charge_loss,
            "discharge_pre": discharge,
            "discharge_loss": discharge_loss,
            "aux_loss": aux_loss,
            
            "solar_direct_meter": solar_direct_meter,
            "wind_direct_meter": wind_direct_meter,
            "discharge_meter": discharge_meter,

            "plant_export_pre": export,
            "curtailment_pre": curtailment,

            "energy_capacity_mwh": energy_capacity,
            "charge_power_mw": charge_power_cap,
            "discharge_power_mw": discharge_power_cap,

            "bess_end_soc_mwh": soc
        }