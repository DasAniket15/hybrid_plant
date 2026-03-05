import numpy as np

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine


# -------------------------------------------------
# Helper Function
# -------------------------------------------------

def run_test_case(
    name,
    solar_capacity_mw,
    wind_capacity_mw,
    bess_containers,
    charge_c_rate,
    discharge_c_rate,
    ppa_capacity_mw,
    dispatch_priority,
    bess_charge_source,
):

    print(f"\n================ {name} ================\n")

    plant_results = energy_engine.evaluate(
        solar_capacity_mw=solar_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        bess_containers=bess_containers,
        charge_c_rate=charge_c_rate,
        discharge_c_rate=discharge_c_rate,
        ppa_capacity_mw=ppa_capacity_mw,
        dispatch_priority=dispatch_priority,
        bess_charge_source=bess_charge_source,
    )

    # ----------------------------
    # Extract Arrays
    # ----------------------------

    solar_direct = plant_results["solar_direct_pre"]
    wind_direct = plant_results["wind_direct_pre"]
    discharge = plant_results["discharge_pre"]
    charge = plant_results["charge_pre"]
    curtailment = plant_results["curtailment_pre"]
    plant_export = plant_results["plant_export_pre"]
    meter_delivery = plant_results["meter_delivery"]

    # ----------------------------
    # Annual Aggregates
    # ----------------------------

    raw_generation = np.sum(
        solar_capacity_mw * data["solar_cuf"]
        + wind_capacity_mw * data["wind_cuf"]
    )

    delivered_pre_loss = np.sum(plant_export)
    delivered_meter = np.sum(meter_delivery)
    total_loss = delivered_pre_loss - delivered_meter

    print("Raw Generation (MWh):", round(raw_generation, 2))
    print("Solar Direct (MWh):", round(np.sum(solar_direct), 2))
    print("Wind Direct (MWh):", round(np.sum(wind_direct), 2))
    print("BESS Charge (MWh):", round(np.sum(charge), 2))
    print("BESS Discharge (MWh):", round(np.sum(discharge), 2))
    print("Delivered Pre-Loss (MWh):", round(delivered_pre_loss, 2))
    print("Delivered At Meter (MWh):", round(delivered_meter, 2))
    print("Total Grid Loss (MWh):", round(total_loss, 2))
    print("Curtailment (MWh):", round(np.sum(curtailment), 2))

    print("BESS Energy Capacity (MWh):", round(plant_results["energy_capacity_mwh"], 2))
    print("Charge Power Cap (MW):", round(plant_results["charge_power_mw"], 2))
    print("Discharge Power Cap (MW):", round(plant_results["discharge_power_mw"], 2))
    print("End of year BESS SOC (MWh):", round(plant_results["bess_end_soc_mwh"], 2))

    if delivered_pre_loss > 0:
        print("Loss Ratio (Post/Pre):",
              round(delivered_meter / delivered_pre_loss, 4))

    print("\n=========================================\n")

# -------------------------------------------------
# Main Execution
# -------------------------------------------------

if __name__ == "__main__":

    config = load_config()
    data = load_timeseries_data(config)

    energy_engine = Year1Engine(config, data)
    """
    # ======================================================
    # Solar + Wind + BESS Excel Model Comparison Case
    # ======================================================

    run_test_case(
        name="Excel Model Comparison",
        solar_capacity_mw=190.454972460807,
        wind_capacity_mw=116.130108575195,
        bess_containers=120,
        charge_c_rate=1.0,
        discharge_c_rate=1.0,
        ppa_capacity_mw=120.632227022855,
        dispatch_priority="solar_first",
        bess_charge_source="solar_only",
    )
    """
    # ======================================================
    # Solar + BESS Excel Model Comparison Case
    # ======================================================

    run_test_case(
        name="Excel Model Comparison",
        solar_capacity_mw=389.273054572568,
        wind_capacity_mw=0.0,
        bess_containers=327,
        charge_c_rate=1.0,
        discharge_c_rate=1.0,
        ppa_capacity_mw=134.51327003244,
        dispatch_priority="solar_first",
        bess_charge_source="solar_only",
    )