from hybrid_plant.config_loader import load_config
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine
from hybrid_plant.finance.augmentation_engine import AugmentationEngine
import numpy as np

config = load_config()
data   = load_timeseries_data(config)
energy_engine  = Year1Engine(config, data)
finance_engine = FinanceEngine(config, data)

params = {
    'solar_capacity_mw':  195.415073395429,
    'wind_capacity_mw':   0.0,
    'bess_containers':    164,
    'charge_c_rate':      1.0,
    'discharge_c_rate':   1.0,
    'ppa_capacity_mw':    67.5256615562851,
    'dispatch_priority':  'solar_first',
    'bess_charge_source': 'solar_only',
}

y1 = energy_engine.evaluate(**params)
print("=== YEAR-1 SIM ===")
print(f"  BESS containers:         {params['bess_containers']}")
print(f"  BESS energy cap (MWh):   {float(y1['energy_capacity_mwh']):.2f}")
print(f"  Charge power (MW):       {float(y1['charge_power_mw']):.2f}")
print(f"  Discharge power (MW):    {float(y1['discharge_power_mw']):.2f}")
print(f"  Solar direct (MWh):      {float(np.sum(y1['solar_direct_pre'])):.1f}")
print(f"  BESS discharge (MWh):    {float(np.sum(y1['discharge_pre'])):.1f}")
print(f"  Total busbar (MWh):      {float(np.sum(y1['solar_direct_pre'])+np.sum(y1['wind_direct_pre'])+np.sum(y1['discharge_pre'])):.1f}")
print(f"  Curtailment (MWh):       {float(np.sum(y1['curtailment_pre'])):.1f}")

# ── WITH AUGMENTATION (full mode) ────────────────────────────────────────
fi_aug = finance_engine.evaluate(
    y1, solar_capacity_mw=params['solar_capacity_mw'],
    wind_capacity_mw=params['wind_capacity_mw'],
    ppa_capacity_mw=params['ppa_capacity_mw'],
    fast_mode=False
)
aug    = fi_aug['augmentation']
ep_aug = fi_aug['energy_projection']
cuf_aug = ep_aug['cuf_per_year']

# ── WITHOUT AUGMENTATION: disable via config monkey-patch ────────────────
from hybrid_plant.finance.energy_projection import EnergyProjection
from hybrid_plant.finance.opex_model import OpexModel
from hybrid_plant.finance.capex_model import CapexModel
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.landed_tariff_model import LandedTariffModel
from hybrid_plant.finance.savings_model import SavingsModel

aug_eng = AugmentationEngine(config)
noaug_result = aug_eng.passthrough_result(params['bess_containers'])
noaug_result['enabled'] = False

class _NoAugEngine:
    """Thin wrapper: same interface as AugmentationEngine but enabled=False."""
    def __init__(self, real_eng):
        self._e = real_eng
        self.enabled = False
        self.trigger_cuf_percent = real_eng.trigger_cuf_percent
        self.restore_pct = real_eng.restore_pct
        self.max_gap_years = real_eng.max_gap_years
        self.min_containers = real_eng.min_containers
        self.cost_per_mwh = real_eng.cost_per_mwh
        self.container_size = real_eng.container_size
        self.project_life = real_eng.project_life
        self.soh_curve = real_eng.soh_curve
    def cohort_soh(self, *a, **kw): return self._e.cohort_soh(*a, **kw)
    def compute_effective(self, *a, **kw): return self._e.compute_effective(*a, **kw)
    def passthrough_result(self, *a, **kw): return self._e.passthrough_result(*a, **kw)

noaug_eng = _NoAugEngine(aug_eng)
proj_noaug = EnergyProjection(
    config=config, data=data, year1_results=y1,
    aug_engine=noaug_eng
).project(fast_mode=False)
ep_noaug = proj_noaug
cuf_noaug = proj_noaug['cuf_per_year']

capex_model  = CapexModel(config)
opex_model   = OpexModel(config)
lcoe_model   = LCOEModel(config)
landed_model = LandedTariffModel(config)
savings_model= SavingsModel(config, data)

bess_mwh   = float(y1['energy_capacity_mwh'])
capex      = capex_model.compute(params['solar_capacity_mw'], params['wind_capacity_mw'], bess_mwh)
total_capex= capex['total_capex']

opex_proj_noaug, opex_bd_noaug = opex_model.compute(
    solar_capacity_mw=params['solar_capacity_mw'],
    wind_capacity_mw=params['wind_capacity_mw'],
    bess_energy_mwh=bess_mwh,
    total_capex=total_capex,
    augmentation_result=proj_noaug['augmentation_result'],
)
lcoe_noaug = lcoe_model.compute(total_capex, opex_proj_noaug, ep_noaug['delivered_pre_mwh'])
landed_noaug = landed_model.compute(
    lcoe_inr_per_kwh=lcoe_noaug['lcoe_inr_per_kwh'],
    ppa_capacity_mw=params['ppa_capacity_mw'],
    busbar_energy_mwh_projection=ep_noaug['delivered_pre_mwh'],
    meter_energy_mwh_projection=ep_noaug['delivered_meter_mwh'],
)
sav_noaug = savings_model.compute(
    landed_tariff_series=landed_noaug['landed_tariff_series'],
    meter_energy_mwh_projection=ep_noaug['delivered_meter_mwh'],
    wacc=lcoe_noaug['wacc'],
)

# ── PRINT YEAR-BY-YEAR COMPARISON TABLE ────────────────────────────────────
CRORE = 1e7
print("\n" + "="*120)
print("YEAR-BY-YEAR COMPARISON: WITH AUG vs WITHOUT AUG")
print("="*120)
print(f"{'Yr':>3}  {'Busbar(noaug)':>14}  {'Busbar(aug)':>12}  {'D%':>6}  "
      f"{'CUF%(noaug)':>12}  {'CUF%(aug)':>10}  "
      f"{'OPEX_noaug Cr':>14}  {'OPEX_aug Cr':>12}  {'Dopex Cr':>10}  "
      f"{'AugPurch Cr':>12}  {'Aug?':>5}")
print("-"*120)
opex_aug_arr = fi_aug['opex_projection']
lts_aug  = fi_aug['landed_tariff_series']
lts_noaug= landed_noaug['landed_tariff_series']

project_life = config.project['project']['project_life_years']
for yr in range(1, project_life+1):
    i = yr - 1
    bus_na = ep_noaug['delivered_pre_mwh'][i]
    bus_a  = ep_aug['delivered_pre_mwh'][i]
    cuf_na = cuf_noaug[i]
    cuf_a  = cuf_aug[i]
    opex_na= opex_proj_noaug[i]/CRORE
    opex_a = opex_aug_arr[i]/CRORE
    purch  = aug['augmentation_purchase_opex'].get(yr, 0)/CRORE
    delta_pct = (bus_a - bus_na)/bus_na*100 if bus_na > 0 else 0
    delta_opex= opex_a - opex_na
    flag   = " AUG" if yr in aug['augmentation_years'] else ""
    print(f"  {yr:>3}  {bus_na:>14.1f}  {bus_a:>12.1f}  {delta_pct:>+6.2f}  "
          f"{cuf_na:>12.3f}  {cuf_a:>10.3f}  "
          f"{opex_na:>14.4f}  {opex_a:>12.4f}  {delta_opex:>+10.4f}  "
          f"{purch:>12.4f}  {flag:>5}")  # noqa: unicode-safe

print("-"*120)
total_bus_na = sum(ep_noaug['delivered_pre_mwh'])
total_bus_a  = sum(ep_aug['delivered_pre_mwh'])
total_opex_na= sum(opex_proj_noaug)/CRORE
total_opex_a = sum(opex_aug_arr)/CRORE
print(f"  {'TOT':>3}  {total_bus_na:>14.1f}  {total_bus_a:>12.1f}  {(total_bus_a-total_bus_na)/total_bus_na*100:>+6.2f}  "
      f"{'':>12}  {'':>10}  "
      f"{total_opex_na:>14.4f}  {total_opex_a:>12.4f}  {total_opex_a-total_opex_na:>+10.4f}  "
      f"{'':>12}  {'':>5}")

print("\n" + "="*80)
print("ECONOMICS SUMMARY")
print("="*80)
lcoe_aug = fi_aug['lcoe_inr_per_kwh']
sv_aug   = fi_aug['savings_breakdown']
sv_noaug = sav_noaug
print(f"  {'Metric':<42}  {'No-Aug':>12}  {'With-Aug':>12}  {'Diff':>12}")
print("-"*80)
print(f"  {'LCOE (Rs/kWh)':<42}  {lcoe_noaug['lcoe_inr_per_kwh']:>12.4f}  {lcoe_aug:>12.4f}  {lcoe_aug - lcoe_noaug['lcoe_inr_per_kwh']:>+12.4f}")
print(f"  {'Landed Tariff Y1 (Rs/kWh)':<42}  {lts_noaug[0]:>12.4f}  {lts_aug[0]:>12.4f}  {lts_aug[0]-lts_noaug[0]:>+12.4f}")
print(f"  {'Landed Tariff Y25 (Rs/kWh)':<42}  {lts_noaug[-1]:>12.4f}  {lts_aug[-1]:>12.4f}  {lts_aug[-1]-lts_noaug[-1]:>+12.4f}")
print(f"  {'DISCOM tariff (Rs/kWh)':<42}  {sv_noaug['discom_tariff']:>12.4f}  {sv_aug['discom_tariff']:>12.4f}  {'(same)':>12}")
print(f"  {'Savings Y1 (Cr Rs)':<42}  {sv_noaug['annual_savings'][0]/CRORE:>12.4f}  {sv_aug['annual_savings'][0]/CRORE:>12.4f}  {(sv_aug['annual_savings'][0]-sv_noaug['annual_savings'][0])/CRORE:>+12.4f}")
print(f"  {'Savings NPV (Cr Rs)':<42}  {sv_noaug['savings_npv']/CRORE:>12.4f}  {sv_aug['savings_npv']/CRORE:>12.4f}  {(sv_aug['savings_npv']-sv_noaug['savings_npv'])/CRORE:>+12.4f}")
print(f"  {'NPV Total Costs (Cr Rs)':<42}  {lcoe_noaug['npv_total_cost']/CRORE:>12.4f}  {fi_aug['lcoe_breakdown']['npv_total_cost']/CRORE:>12.4f}  {(fi_aug['lcoe_breakdown']['npv_total_cost']-lcoe_noaug['npv_total_cost'])/CRORE:>+12.4f}")
print(f"  {'NPV Busbar Energy (GWh)':<42}  {lcoe_noaug['npv_energy_kwh']/1e9:>12.4f}  {fi_aug['lcoe_breakdown']['npv_energy_kwh']/1e9:>12.4f}  {(fi_aug['lcoe_breakdown']['npv_energy_kwh']-lcoe_noaug['npv_energy_kwh'])/1e9:>+12.4f}")

print("\n  --- OPEX detail Y1 vs Y25 ---")
print(f"  {'Item':<32}  {'NoAug-Y1':>10}  {'Aug-Y1':>10}  {'NoAug-Y25':>10}  {'Aug-Y25':>10}")
for key in ['solar_om','wind_om','bess_om','land_lease','insurance','augmentation_purchase']:
    na1 = opex_bd_noaug[0].get(key,0)/CRORE
    a1  = fi_aug['opex_breakdown'][0].get(key,0)/CRORE
    na25= opex_bd_noaug[-1].get(key,0)/CRORE
    a25 = fi_aug['opex_breakdown'][-1].get(key,0)/CRORE
    print(f"  {key:<32}  {na1:>10.4f}  {a1:>10.4f}  {na25:>10.4f}  {a25:>10.4f}")
print(f"  {'bess_installed_mwh':<32}  {opex_bd_noaug[0].get('bess_installed_mwh',0):>10.2f}  {fi_aug['opex_breakdown'][0].get('bess_installed_mwh',0):>10.2f}  {opex_bd_noaug[-1].get('bess_installed_mwh',0):>10.2f}  {fi_aug['opex_breakdown'][-1].get('bess_installed_mwh',0):>10.2f}")

print("\n  --- Cohort detail ---")
for idx, (sy, nc) in enumerate(aug['cohorts']):
    cs = float(config.bess['bess']['container']['size_mwh'])
    print(f"  Cohort {idx}: start_yr={sy}, n={nc}, nameplate={nc*cs:.2f} MWh")
