from config_loader import load_config
from data_loader import load_timeseries_data
from energy.year1_engine import Year1Engine
from finance.finance_engine import FinanceEngine

config = load_config()
data = load_timeseries_data(config)

energy_engine = Year1Engine(config, data)
finance_engine = FinanceEngine(config, data)

# Fixed test configuration
solar = 389.273054572568
wind = 0.0
bess = 327
charge_c = 1.0
discharge_c = 1.0
ppa_cap = 134.51327003244
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