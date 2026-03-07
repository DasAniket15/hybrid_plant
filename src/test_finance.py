import numpy as np
import matplotlib.pyplot as plt

from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine

config = load_config()
data = load_timeseries_data(config)

energy_engine = Year1Engine(config, data)
finance_engine = FinanceEngine(config, data)

def plot_tariff_vs_irr(
    finance_engine, 
    finance
):

    energy = finance["projection"]["delivered_meter_mwh"]
    opex = finance["opex_projection"]
    capex = finance["capex"]["total_capex"]

    tariffs = np.linspace(1, 10, 40)
    irr_values = []

    for t in tariffs:

        irr = finance_engine.cashflow_model.equity_irr(
            total_capex=capex,
            delivered_meter_projection=energy,
            opex_projection=opex,
            ppa_tariff=t
        )

        irr_values.append(irr * 100 if irr else np.nan)

    plt.figure(figsize=(8,5))

    plt.plot(tariffs, irr_values)

    plt.xlabel("PPA Tariff (₹/kWh)")
    plt.ylabel("Equity IRR (%)")

    plt.title("Tariff vs IRR")

    plt.grid(True)

    plt.show()

def plot_project_cashflows(finance):

    projection = finance["projection"]
    opex = np.array(finance["opex_projection"])

    energy = np.array(projection["delivered_meter_mwh"])
    tariff = finance["required_ppa_tariff"]

    revenue = energy * tariff

    cashflow = revenue - opex

    years = np.arange(1, len(energy)+1)

    plt.figure(figsize=(10,5))

    plt.plot(years, revenue/1e7, label="Revenue (₹Cr)")
    plt.plot(years, opex/1e7, label="OPEX (₹Cr)")
    plt.plot(years, cashflow/1e7, label="Net Cashflow (₹Cr)")

    plt.xlabel("Year")
    plt.ylabel("₹ Crore")

    plt.title("Project Cashflows")

    plt.legend()
    plt.grid(True)

    plt.show()

def plot_client_savings(finance, data):

    savings = np.array(finance["annual_savings"])

    annual_load = np.sum(data["load_profile"])

    baseline = annual_load * finance["discom_tariff"]

    hybrid_cost = baseline - np.array(savings)

    years = np.arange(1, len(savings)+1)

    plt.figure(figsize=(10,5))

    plt.plot(years, [baseline/1e7]*len(years),
             label="Baseline DISCOM Cost")

    plt.plot(years, hybrid_cost/1e7,
             label="Hybrid Cost")

    plt.plot(years, savings/1e7,
             label="Savings")

    plt.title("Client Electricity Cost")

    plt.xlabel("Year")
    plt.ylabel("₹ Crore")

    plt.legend()
    plt.grid(True)

    plt.show()

def plot_capex_breakdown(finance):

    capex = finance["capex"]

    labels = ["Solar", "Wind", "BESS", "Transmission"]

    values = [
        capex["solar_capex"],
        capex["wind_capex"],
        capex["bess_capex"],
        capex["transmission_capex"]
    ]

    plt.figure(figsize=(6,6))

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title("CAPEX Breakdown")

    plt.show()

def plot_cumulative_cashflow(finance):

    cashflows = finance["equity_cashflows"]

    cumulative = np.cumsum(cashflows)

    years = np.arange(len(cashflows))

    plt.figure(figsize=(9,5))

    plt.plot(years, cumulative / 1e7)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Cumulative Cashflow (₹ Crore)")
    plt.title("Cumulative Equity Cashflow")

    plt.grid(True)
    plt.show()

# Fixed test configuration
solar = 195.415073395429
wind = 0.0
bess = 164
charge_c = 1.0
discharge_c = 1.0
ppa_cap = 67.5256615562851
target_irr_percent = 20

year1 = energy_engine.evaluate(
    solar_capacity_mw=solar,
    wind_capacity_mw=wind,
    bess_containers=bess,
    charge_c_rate=charge_c,
    discharge_c_rate=discharge_c,
    ppa_capacity_mw=ppa_cap,
    dispatch_priority="solar_first",
    bess_charge_source="solar_only",
)

finance = finance_engine.evaluate(
    year1_results=year1,
    solar_capacity_mw=solar,
    wind_capacity_mw=wind,
    ppa_capacity_mw=ppa_cap,
    target_irr_percent=target_irr_percent,
)

print("\n====== FINANCE TEST ======")

if finance["invalid_solution"]:
    print("❌ Finance engine returned INVALID solution (IRR constraint not achievable)")
else:
    print("Required PPA Tariff:", round(finance["required_ppa_tariff"], 4))
    print("Achieved IRR:", round(finance["achieved_equity_irr"], 2), "%")
    print("Savings NPV:", round(finance["savings_npv"], 2))

print("==========================\n")

plot_tariff_vs_irr(
    finance_engine,
    finance
)

plot_project_cashflows(finance)

plot_client_savings(finance, data)

plot_capex_breakdown(finance)

plot_cumulative_cashflow(finance)