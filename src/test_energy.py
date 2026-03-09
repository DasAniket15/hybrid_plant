import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine


# -------------------------------------------------
# Helper Function
# -------------------------------------------------

def plot_stacked_dispatch(
    data,
    plant_results,
    solar_capacity_mw,
    wind_capacity_mw
):

    solar_gen = data["solar_cuf"] * solar_capacity_mw
    wind_gen = data["wind_cuf"] * wind_capacity_mw
    load = data["load_profile"]

    solar_direct = plant_results["solar_direct_pre"]
    wind_direct = plant_results["wind_direct_pre"]
    discharge = plant_results["discharge_pre"]
    charge = plant_results["charge_pre"]

    hours = np.arange(len(load))

    plt.figure(figsize=(14,6))

    # Stacked supply
    plt.stackplot(
        hours,
        solar_direct,
        wind_direct,
        discharge,
        labels=[
            "Solar Direct",
            "Wind Direct",
            "BESS Discharge"
        ],
        alpha=0.8
    )

    # Load line
    plt.plot(hours, load, color="black", linewidth=2, label="Load")

    # Charging below zero
    plt.plot(hours, -charge, label="BESS Charging", linestyle="--")

    plt.title("Stacked Dispatch Plot (8760 Hours)")
    plt.xlabel("Hour of Year")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.show()

def plot_representative_week(
    data,
    plant_results,
    solar_capacity_mw,
    wind_capacity_mw,
    week_index=20
):

    start = week_index * 168
    end = start + 168

    hours = np.arange(168)

    solar_gen = data["solar_cuf"][start:end] * solar_capacity_mw
    wind_gen = data["wind_cuf"][start:end] * wind_capacity_mw
    load = data["load_profile"][start:end]

    solar_direct = plant_results["solar_direct_pre"][start:end]
    wind_direct = plant_results["wind_direct_pre"][start:end]
    discharge = plant_results["discharge_pre"][start:end]
    charge = plant_results["charge_pre"][start:end]
    curtailment = plant_results["curtailment_pre"][start:end]

    plt.figure(figsize=(14,6))

    plt.stackplot(
        hours,
        solar_direct,
        wind_direct,
        discharge,
        labels=["Solar Direct", "Wind Direct", "BESS Discharge"],
        alpha=0.8
    )

    plt.plot(hours, load, color="black", linewidth=2, label="Load")

    # Charging shown below zero
    plt.plot(hours, -charge, label="BESS Charge", linestyle="--")

    # Optional: show curtailment
    plt.plot(hours, curtailment, label="Curtailment", linestyle=":")

    plt.title(f"Representative Week Dispatch (Week {week_index})")
    plt.xlabel("Hour of Week")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.show()

def plot_energy_sankey(plant_results, data, solar_capacity_mw, wind_capacity_mw):

    # ============================
    # GENERATION
    # ============================

    solar_gen = np.sum(solar_capacity_mw * data["solar_cuf"])
    wind_gen = np.sum(wind_capacity_mw * data["wind_cuf"])

    solar_direct = np.sum(plant_results["solar_direct_pre"])
    wind_direct = np.sum(plant_results["wind_direct_pre"])

    solar_charge = np.sum(plant_results["solar_charge_pre"])
    wind_charge = np.sum(plant_results["wind_charge_pre"])

    curtailment = np.sum(plant_results["curtailment_pre"])

    # ============================
    # BESS FLOWS
    # ============================

    charge = np.sum(plant_results["charge_pre"])
    discharge = np.sum(plant_results["discharge_pre"])

    charge_loss = np.sum(plant_results["charge_loss"])
    discharge_loss = np.sum(plant_results["discharge_loss"])
    aux_loss = np.sum(plant_results["aux_loss"])

    soc_end = plant_results["bess_end_soc_mwh"]

    # ============================
    # GRID DELIVERY
    # ============================

    solar_meter = np.sum(plant_results["solar_direct_meter"])
    wind_meter = np.sum(plant_results["wind_direct_meter"])
    bess_meter = np.sum(plant_results["discharge_meter"])

    pre_loss_delivery = solar_direct + wind_direct + discharge
    meter_delivery = solar_meter + wind_meter + bess_meter

    grid_losses = pre_loss_delivery - meter_delivery

    # ============================
    # SANKEY NODES
    # ============================

    labels = [

        "Solar Generation",
        "Wind Generation",

        "Solar Direct",
        "Wind Direct",

        "Solar → BESS",
        "Wind → BESS",

        "Curtailment",

        "BESS Charge",

        "BESS Charge Loss",
        "Stored Energy",

        "BESS Discharge",

        "BESS Discharge Loss",
        "BESS Aux Loss",

        "Grid Loss",

        "Solar at Meter",
        "Wind at Meter",
        "BESS at Meter"
    ]

    # ============================
    # SANKEY FLOWS
    # ============================

    source = [

        # Solar generation
        0, 0, 0,

        # Wind generation
        1, 1, 1,

        # Charging inputs
        4, 5,

        # Charge → stored / loss
        7, 7,

        # Stored → discharge / aux
        9, 9,

        # Discharge → loss / grid
        10, 10,

        # Grid → meter
        2, 3, 10
    ]

    target = [

        2, 4, 6,
        3, 5, 6,

        7, 7,

        8, 9,

        10, 12,

        11, 13,

        14, 15, 16
    ]

    values = [

        solar_direct,
        solar_charge,
        curtailment * solar_gen/(solar_gen + wind_gen + 1e-9),

        wind_direct,
        wind_charge,
        curtailment * wind_gen/(solar_gen + wind_gen + 1e-9),

        solar_charge,
        wind_charge,

        charge_loss,
        charge - charge_loss,

        discharge,
        aux_loss,

        discharge_loss,
        discharge - discharge_loss,

        solar_meter,
        wind_meter,
        bess_meter
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        )
    ))

    fig.update_layout(
        title="Hybrid Plant Energy Balance",
        font_size=12
    )

    fig.show()

def plot_residual_load_curve(
    data, 
    plant_results, 
    solar_capacity_mw, 
    wind_capacity_mw
):

    solar_gen = data["solar_cuf"] * solar_capacity_mw
    wind_gen = data["wind_cuf"] * wind_capacity_mw
    load = data["load_profile"]

    discharge = plant_results["discharge_pre"]
    meter_delivery = plant_results["meter_delivery"]

    # Residual before BESS
    residual_raw = load - (solar_gen + wind_gen)

    # Residual after BESS discharge
    residual_after_bess = load - (solar_gen + wind_gen + discharge)

    # Residual after hybrid plant (DISCOM supply)
    residual_grid = load - meter_delivery

    plt.figure(figsize=(12,6))

    plt.plot(sorted(residual_raw, reverse=True), label="Residual Load (After RE)")
    plt.plot(sorted(residual_after_bess, reverse=True), label="Residual Load (After BESS)")
    plt.plot(sorted(residual_grid, reverse=True), label="DISCOM Supply Requirement")

    plt.axhline(0, linestyle="--")

    plt.title("Residual Load Duration Curve")
    plt.xlabel("Hour Rank")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True)

    plt.show()

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

    print(f"\n================ {name} ================")

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
    solar_direct_meter = plant_results["solar_direct_meter"]
    wind_direct_meter = plant_results["wind_direct_meter"]
    charge = plant_results["charge_pre"]
    discharge = plant_results["discharge_pre"]
    discharge_meter = plant_results["discharge_meter"]
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
    print(f"================ Plant Details ================\n")

    print("Solar Capacity (MW):", round(solar_capacity_mw, 2))
    print("Wind Capacity (MW):", round(wind_capacity_mw, 2))
    print("BESS Containers:", bess_containers)
    print("BESS Energy Capacity (MWh):", round(plant_results["energy_capacity_mwh"], 2))
    print("Charge Power Cap (MW):", round(plant_results["charge_power_mw"], 2))
    print("Discharge Power Cap (MW):", round(plant_results["discharge_power_mw"], 2))

    print(f"\n================ Pre-Losses (Bus-Bar) ================\n")

    print("Raw Generation (MWh):", round(raw_generation, 2))
    print("Solar Direct (MWh):", round(np.sum(solar_direct), 2))
    print("Wind Direct (MWh):", round(np.sum(wind_direct), 2))
    print("BESS Charge (MWh):", round(np.sum(charge), 2))
    print("BESS Discharge (MWh):", round(np.sum(discharge), 2))
    print("Delivered Pre-Loss (MWh):", round(delivered_pre_loss, 2))

    print(f"\n================ Post-Losses (At Meter) ================\n")

    print("Solar Direct At Meter (MWh):", round(np.sum(solar_direct_meter), 2))
    print("Wind Direct At Meter (MWh):", round(np.sum(wind_direct_meter), 2))
    print("BESS Discharge At Meter (MWh):", round(np.sum(discharge_meter), 2))
    print("Delivered At Meter (MWh):", round(delivered_meter, 2))
    print("Total Grid Loss (MWh):", round(total_loss, 2))
    print("Curtailment (MWh):", round(np.sum(curtailment), 2))
    print("End of year BESS SOC (MWh):", round(plant_results["bess_end_soc_mwh"], 2))

    if delivered_pre_loss > 0:
        print("Loss Ratio (Post/Pre):",
              round(delivered_meter / delivered_pre_loss, 4))

    print("\n=========================================\n")

    """
    plot_stacked_dispatch(
        data, 
        plant_results,
        solar_capacity_mw,
        wind_capacity_mw
    )

    plot_representative_week(
        data, 
        plant_results, 
        solar_capacity_mw, 
        wind_capacity_mw, 
        week_index=26
    )
    """
    plot_energy_sankey(
        plant_results, 
        data, 
        solar_capacity_mw, 
        wind_capacity_mw
    )
    """
    plot_residual_load_curve(
        data,
        plant_results,
        solar_capacity_mw,
        wind_capacity_mw
    )
    """
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
        solar_capacity_mw=195.415073395429,
        wind_capacity_mw=0.0,
        bess_containers=164,
        charge_c_rate=1.0,
        discharge_c_rate=1.0,
        ppa_capacity_mw=67.5256615562851,
        dispatch_priority="solar_first",
        bess_charge_source="solar_only",
    )