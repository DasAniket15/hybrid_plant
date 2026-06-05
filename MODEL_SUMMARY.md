# Hybrid RE Plant Model — Technical Summary

> **Purpose of this document**
> This document is a complete technical reference for the `hybrid_plant` simulation-optimisation model. It is written to give a language model (ChatGPT) sufficient context to draft a prompt for the next development phase: converting the model into a **Pyomo algebraic optimisation model** to enable deterministic, constraint-based sizing and augmentation of solar, wind, and BESS capacity.

---

## Table of Contents

1. [What the Model Does](#1-what-the-model-does)
2. [Repository Layout](#2-repository-layout)
3. [Configuration System](#3-configuration-system)
4. [Data Inputs](#4-data-inputs)
5. [Energy Simulation Layer](#5-energy-simulation-layer)
   - 5.1 PlantEngine — 8760-Hour Dispatch
   - 5.2 ToD-Aware BESS Dispatch Logic
   - 5.3 GridInterface — Loss Factor
   - 5.4 MeterLayer — DISCOM Shortfall
   - 5.5 Year1Engine — Orchestrator
6. [Finance Pipeline](#6-finance-pipeline)
   - 6.1 CapexModel
   - 6.2 OpexModel
   - 6.3 EnergyProjection — 25-Year Degradation
   - 6.4 LCOEModel
   - 6.5 LandedTariffModel
   - 6.6 SavingsModel
   - 6.7 FinanceEngine — Orchestrator
7. [Optimisation Layer — SolverEngine](#7-optimisation-layer--solverengine)
8. [Decision Variables](#8-decision-variables)
9. [Constraints](#9-constraints)
10. [Objective Function](#10-objective-function)
11. [Key Mathematical Formulations](#11-key-mathematical-formulations)
12. [Current Limitations Relevant to Pyomo Migration](#12-current-limitations-relevant-to-pyomo-migration)
13. [Future Scope Identified in the Codebase](#13-future-scope-identified-in-the-codebase)
14. [Pyomo Migration — What Needs to Change](#14-pyomo-migration--what-needs-to-change)

---

## 1. What the Model Does

The `hybrid_plant` model sizes and evaluates a grid-connected **hybrid renewable energy plant** — solar PV + wind + BESS (Battery Energy Storage System) — serving a **C&I (commercial & industrial) client** in India. It answers the question:

> *What combination of solar capacity, wind capacity, BESS size, PPA limit, and C-rates minimises the client's landed electricity cost over a 25-year project life, while keeping the project economically viable for the developer?*

The model is currently structured as a **black-box simulation driven by a heuristic optimiser (Optuna TPE)**. For each candidate sizing, it:

1. Simulates 8760 hourly dispatch of solar, wind, and BESS to cover client load.
2. Applies transmission losses to get meter-side delivery.
3. Projects the energy output over 25 years with component degradation.
4. Computes CAPEX, OPEX, LCOE, landed tariff, and client savings NPV.
5. Reports the sizing that maximises client savings NPV subject to feasibility constraints.

**Project context (from `project.yaml`):**
- Location: Upleta, Gujarat, India
- Currency: INR
- Project life: 25 years
- Base year: 2026
- Simulation resolution: hourly (8760 time steps)

---

## 2. Repository Layout

```
hybrid_plant/
├── configs/                     # All YAML configuration files (inputs)
│   ├── project.yaml             # Project identity, life, file paths
│   ├── regulatory.yaml          # Gujarat grid losses, HT/LT split
│   ├── tariffs.yaml             # DISCOM ToD tariff periods and rates
│   ├── bess.yaml                # BESS container specs, efficiency, dispatch
│   ├── finance.yaml             # CAPEX rates, OPEX rates, financing terms
│   └── solver.yaml              # Optuna search space, constraints, n_trials
│
├── data/                        # Time-series and degradation CSVs
│   ├── solar_cuf_8760.csv       # Hourly solar CUF profile (8760 values)
│   ├── wind_cuf_8760.csv        # Hourly wind CUF profile (8760 values)
│   ├── load_profile_8760.csv    # Hourly client load (MWh, 8760 values)
│   ├── solar_efficiency_curve.csv  # Year-wise solar degradation
│   ├── wind_efficiency_curve.csv   # Year-wise wind degradation
│   └── bess_soh_curve.csv       # Year-wise BESS state-of-health
│
├── src/hybrid_plant/
│   ├── config_loader.py         # Loads all YAMLs → FullConfig dataclass
│   ├── data_loader.py           # Loads all CSVs → numpy arrays
│   ├── constants.py             # Unit conversion constants
│   ├── _paths.py                # Project root discovery
│   │
│   ├── energy/
│   │   ├── plant_engine.py      # 8760-hour dispatch simulation (PlantEngine)
│   │   ├── year1_engine.py      # Year-1 orchestrator (Year1Engine)
│   │   ├── grid_interface.py    # Grid loss factor (GridInterface)
│   │   └── meter_layer.py       # DISCOM shortfall computation (MeterLayer)
│   │
│   ├── finance/
│   │   ├── capex_model.py       # Component CAPEX breakdown
│   │   ├── opex_model.py        # 25-year escalating OPEX
│   │   ├── energy_projection.py # 25-year degraded energy delivery
│   │   ├── lcoe_model.py        # NPV-based LCOE
│   │   ├── landed_tariff_model.py # All-in Rs/kWh to client meter
│   │   ├── savings_model.py     # Client savings vs DISCOM baseline
│   │   ├── finance_engine.py    # Finance pipeline orchestrator
│   │   └── _utils.py            # npv(), loan_schedule() helpers
│   │
│   ├── solver/
│   │   └── solver_engine.py     # Optuna TPE wrapper (SolverEngine)
│   │
│   └── run_model.py             # Entry point: runs solver, prints dashboard, saves plots
│
└── outputs/                     # Generated plots
    ├── model_output.png         # 4-panel financial dashboard
    └── day250_dispatch.png      # Single-day BESS dispatch diagnostic
```

**Core abstraction hierarchy (from graph analysis):**
- `FullConfig` (110 edges) — central data bus; everything reads from it
- `PlantEngine` → `Year1Engine` → `FinanceEngine` → `SolverEngine` — the main pipeline
- `LCOEModel`, `CapexModel`, `OpexModel` — finance sub-models

---

## 3. Configuration System

All inputs are in YAML. The `config_loader.load_config()` function discovers the project root, loads all six YAML files, validates them, and returns a **frozen `FullConfig` dataclass** with the following namespaces:

| Namespace | File | What it controls |
|-----------|------|-----------------|
| `config.project` | `project.yaml` | Name, location, life, data file paths, simulation resolution |
| `config.regulatory` | `regulatory.yaml` | State, HT/LT split %, grid loss factors (CTU/STU/wheeling %) per voltage level, banking rules |
| `config.tariffs` | `tariffs.yaml` | DISCOM ToD periods (hours 1-indexed), LT and HT rates in Rs/kWh |
| `config.bess` | `bess.yaml` | Container size (MWh), aux consumption, charge/discharge efficiency, degradation curve file, dispatch window config |
| `config.finance` | `finance.yaml` | CAPEX rates (solar, wind, BESS, transmission), OPEX rates & escalation, financing structure (debt/equity %, interest rate, EMI, ROE, tax rate), regulatory charges (CTU/STU/SLDC/wheeling/electricity tax) |
| `config.solver` | `solver.yaml` | Decision variable bounds, constraint toggles, n_trials, random seed, fast_mode flag |

**Current parameter values (from YAMLs):**

```
Solar CAPEX:         Rs 2.38 Crore/MWp (DC)   AC/DC ratio: 1.398
Wind CAPEX:          Rs 7.0 Crore/MW
BESS CAPEX:          Rs 0.77 Crore/MWh
Transmission CAPEX:  Rs 1.3 Crore/km × 10 km = Rs 13 Crore (fixed)

Solar O&M:           Rs 1.00 Lakh/MWp/yr, 2% escalation
Wind O&M:            Rs 12.00 Lakh/MW/yr, 2% escalation
BESS O&M:            Rs 0.60 Lakh/MWh/yr, no escalation
Land lease:          Rs 1.00 Crore/month, 3% escalation
Insurance:           0.2% of total CAPEX/yr

Debt fraction:       70%,  Interest rate: 8.5%,  Tenure: 25 years
Equity fraction:     30%,  ROE: 20%
Corporate tax rate:  25.17% (WACC computation only)

BESS container size: 5.015 MWh
BESS charge eff:     97.21%
BESS discharge eff:  93.84%
BESS aux:            0.300 MWh/day/container

Grid loss factor:    LT side: (1-0.0337)×(1-0.0749) ≈ 0.8953
                     (HT/LT split = 0%, i.e. 100% LT)

LT ToD tariffs:
  Morning peak  (hours 8–11):  Rs 9.182/kWh
  Solar offpeak (hours 12–15): Rs 8.027/kWh
  Evening peak  (hours 19–22): Rs 9.182/kWh
  Normal        (all others):  Rs 8.687/kWh

Current BESS dispatch config:  discharge_hours [18–23], charge_first = true
```

---

## 4. Data Inputs

Three hourly time series (8760 rows each) are loaded from CSVs:

| Input | File | Unit | Notes |
|-------|------|------|-------|
| Solar CUF | `data/solar_cuf_8760.csv` | fraction [0–1] | Resource profile; multiplied by AC solar capacity to get generation MW |
| Wind CUF | `data/wind_cuf_8760.csv` | fraction [0–1] | Resource profile; multiplied by wind capacity |
| Load profile | `data/load_profile_8760.csv` | MWh per hour | Client demand profile (constant shape; no demand response) |

Three degradation curves (year-indexed):

| Curve | File | Column | Meaning |
|-------|------|--------|---------|
| Solar efficiency | `data/solar_efficiency_curve.csv` | `efficiency` | Year-over-year solar panel degradation multiplier |
| Wind efficiency | `data/wind_efficiency_curve.csv` | `efficiency` | Year-over-year wind turbine performance multiplier |
| BESS SOH | `data/bess_soh_curve.csv` | `soh` | Battery state-of-health multiplier; scales both energy capacity and power limits |

**`operating_value(curve, year)` convention:** Year 1 always returns 1.0 (fresh plant). Year N ≥ 2 returns `curve[N-1]` (end-of-prior-year value), implementing an end-of-year degradation model.

---

## 5. Energy Simulation Layer

### 5.1 PlantEngine — 8760-Hour Dispatch

**File:** `src/hybrid_plant/energy/plant_engine.py`

`PlantEngine.simulate()` is the core computation. It runs a single-pass, hour-by-hour simulation over 8760 time steps. All quantities are at the **busbar (pre-loss)** basis.

**Inputs (decision variables):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `solar_capacity_mw` | float | AC installed solar capacity (MW) |
| `wind_capacity_mw` | float | Wind installed capacity (MW) |
| `bess_containers` | int | Number of physical BESS containers |
| `charge_c_rate` | float | BESS charge C-rate (fraction of energy capacity per hour) |
| `discharge_c_rate` | float | BESS discharge C-rate |
| `ppa_capacity_mw` | float | Contracted PPA export cap (MW) — ceiling on total plant busbar export per hour |
| `dispatch_priority` | str | `"solar_first"` / `"wind_first"` / `"proportional"` |
| `bess_charge_source` | str | `"solar_only"` / `"wind_only"` / `"solar_and_wind"` |
| `loss_factor` | float | Grid loss factor from GridInterface |
| `bess_soh_factor` | float | SOH multiplier (1.0 for Year 1; degrades in EnergyProjection) |

**Derived quantities:**

```
energy_capacity    = bess_containers × container_size × bess_soh_factor
charge_power_cap   = charge_c_rate    × energy_capacity
discharge_power_cap = discharge_c_rate × energy_capacity
solar_generation[h] = solar_capacity_mw × solar_cuf[h]
wind_generation[h]  = wind_capacity_mw  × wind_cuf[h]
required_pre[h]     = load[h] / loss_factor       # load expressed at busbar
```

**Per-hour dispatch sequence (Path A — normal RTC / inside discharge window):**

```
Step 1 — Direct dispatch (solar/wind → load, up to PPA cap)
  solar_direct[h] + wind_direct[h] ≤ PPA_cap
  solar_direct[h] ≤ solar_generation[h]
  wind_direct[h]  ≤ wind_generation[h]
  sum ≤ required_pre[h]   (cannot over-serve load)

Step 2 — BESS charging (from surplus after direct dispatch)
  surplus[h] = generation[h] - direct_dispatch[h]
  charge[h]  = min(surplus, charge_power_cap, energy_capacity - SOC[h])
  SOC[h+1]  += charge[h] × charge_efficiency

Step 3 — Curtailment
  curtailment[h] = total_generation[h] - direct[h] - charge[h]

Step 4 — Auxiliary consumption
  aux[h] = active_containers × (aux_mwh_per_day / 24)
  SOC   -= aux[h]   (only if SOC > 0)

Step 5 — BESS discharge (ToD-aware, inside window only)
  shortfall[h]         = max(load[h] - direct[h] × loss_factor, 0)
  required_discharge   = shortfall / (discharge_eff × loss_factor)
  discharge_raw[h]     = min(required_discharge, available_SOC, discharge_power_cap,
                             PPA_headroom / discharge_eff)
  SOC                 -= discharge_raw[h]
  discharge_pre[h]     = discharge_raw[h] × discharge_efficiency
  discharge_meter[h]   = discharge_pre[h] × loss_factor
```

**Path B — Charge-first (outside discharge window, `charge_first = True`):**
BESS gets generation before direct dispatch. Step order: charge → direct → curtailment → aux. No discharge outside window.

**Path C — Outside window, `charge_first = False`:**
Direct-first but no discharge. Steps 1–4 only.

**Outputs (all hourly arrays, shape `(8760,)`):**

```python
{
  "solar_direct_pre", "wind_direct_pre",        # busbar direct delivery
  "solar_charge_pre", "wind_charge_pre",         # charge source split
  "charge_pre", "charge_loss",                   # BESS charge flows
  "discharge_pre", "discharge_loss",             # BESS discharge flows
  "aux_loss",                                    # parasitic consumption
  "solar_direct_meter", "wind_direct_meter",     # post-loss direct delivery
  "discharge_meter",                             # post-loss discharge
  "plant_export_pre", "curtailment_pre",         # plant totals
  "energy_capacity_mwh",                         # scalar: effective BESS capacity
  "charge_power_mw", "discharge_power_mw",       # scalar: power limits
  "bess_end_soc_mwh",                            # scalar: SOC at year-end
}
```

**Energy conservation invariant (checked in debug mode):**
```
sum(solar_gen + wind_gen) == sum(solar_direct + wind_direct + charge + curtailment)
```

### 5.2 ToD-Aware BESS Dispatch Logic

This is the most complex part of the model. The BESS follows a **heuristic state machine** that ring-fences SOC for high-value dispatch periods, driven by Time-of-Day tariff signals.

**ToD period definitions (0-indexed hours):**

| Period | Hours (0-indexed) | LT Rate (Rs/kWh) |
|--------|------------------|-----------------|
| Morning peak | 7, 8, 9, 10 | 9.182 |
| Solar offpeak | 11, 12, 13, 14 | 8.027 |
| Evening peak | 18, 19, 20, 21 | 9.182 |
| Normal | all others | 8.687 |

**Three SOC reservation variables maintained in the dispatch loop:**

| Variable | Purpose |
|----------|---------|
| `rsrv_evening` | SOC ring-fenced for tonight's evening peak (hod 18–21) |
| `rsrv_morning_next` | SOC ring-fenced for tomorrow's morning peak (hod 7–10) |
| `rsrv_fwd_evening` | Forward estimate of evening need, computed at start of solar window |

**Two trigger points update reservations:**

- **hod = 11** (solar window opens, morning peak just ended):
  - Clear `rsrv_morning_next` (consumed or expired)
  - Project SOC at end of charging window (hod 11–14) by walking forward
  - Set `rsrv_fwd_evening` = min(projected SOC, estimated evening discharge need)

- **hod = 15** (solar window closes, definitive SOC known):
  - Replace `rsrv_fwd_evening` with definitive `rsrv_evening` based on actual SOC
  - Set `rsrv_morning_next` from SOC remaining after `rsrv_evening`

**Available SOC per period determines how much the BESS can discharge:**

```
Morning peak   → available = SOC
Evening peak   → available = max(SOC - rsrv_morning_next, 0)
Solar offpeak  → available = max(SOC - rsrv_fwd_evening, 0)
Normal         → available = max(SOC - rsrv_evening - rsrv_morning_next, 0)
```

**Discharge window override (current config: hours 18–23, charge-first = True):**
When `discharge_hours` is non-empty, the planner only ring-fences SOC for window hours; `re_shortfall` for non-window hours is zeroed. In charge-first mode, generation outside the window is diverted to charge before direct dispatch.

### 5.3 GridInterface — Loss Factor

**File:** `src/hybrid_plant/energy/grid_interface.py`

Computes a blended HT/LT loss factor applied once to all busbar quantities:

```
loss_factor = ht_fraction × (1 - CTU_ht) × (1 - STU_ht) × (1 - wheeling_ht)
            + lt_fraction × (1 - CTU_lt) × (1 - STU_lt) × (1 - wheeling_lt)

Current values (100% LT, Gujarat):
  = 1.0 × (1 - 0.0) × (1 - 0.0337) × (1 - 0.0749)
  ≈ 0.8953
```

This is a **fixed constant** per simulation run. It does not vary by hour.

### 5.4 MeterLayer — DISCOM Shortfall

**File:** `src/hybrid_plant/energy/meter_layer.py`

```
shortfall[h]  = max(load[h] - meter_delivery[h], 0)
annual_discom = sum(shortfall[h] for h in 1..8760)
```

The MeterLayer is deliberately thin — it only computes what the grid must supply.

### 5.5 Year1Engine — Orchestrator

**File:** `src/hybrid_plant/energy/year1_engine.py`

Chains `PlantEngine → GridInterface → MeterLayer` and appends the optional **hourly RE penetration penalty**:

```
re_pen_shortfall[h] = max(load[h] × min_pct - meter_delivery[h], 0) × penalty_mask[h]
annual_re_pen_cost  = sum(re_pen_shortfall[h] × 1000 × tod_tariff[h])
```

The penalty is a soft economic cost (not a hard constraint) — it flows directly into `SavingsModel` to reduce savings NPV, pushing the solver toward configurations that meet the RE floor.

---

## 6. Finance Pipeline

### 6.1 CapexModel

**File:** `src/hybrid_plant/finance/capex_model.py`

```
solar_dc_mwp        = solar_capacity_mw × AC_DC_ratio         (1.398)
solar_capex         = solar_dc_mwp × cost_per_mwp             (Rs 2.38 Crore/MWp)
wind_capex          = wind_capacity_mw × cost_per_mw           (Rs 7.0 Crore/MW)
bess_capex          = bess_energy_mwh × cost_per_mwh           (Rs 0.77 Crore/MWh)
transmission_capex  = length_km × cost_per_km                  (Rs 13 Crore, fixed)
total_capex         = solar_capex + wind_capex + bess_capex + transmission_capex
```

### 6.2 OpexModel

**File:** `src/hybrid_plant/finance/opex_model.py`

Year-t OPEX for each component (escalating components use `(1 + esc)^(t-1)`):

```
solar_om[t]    = solar_dc_mwp × rate_lakh/MWp × LAKH × (1 + 0.02)^(t-1)
wind_om[t]     = wind_mw × rate_lakh/MW × LAKH × (1 + 0.02)^(t-1)
land_lease[t]  = monthly_cost × 12 × CRORE × (1 + 0.03)^(t-1)
bess_om        = bess_mwh × rate_lakh/MWh × LAKH               (no escalation)
solar_trans_om = solar_dc_mwp × rate_lakh/MWp × LAKH           (no escalation)
wind_trans_om  = wind_mw × rate_lakh/MW × LAKH                 (no escalation)
insurance      = total_capex × 0.002                            (no escalation)

total_opex[t]  = sum of all components
```

### 6.3 EnergyProjection — 25-Year Degradation

**File:** `src/hybrid_plant/finance/energy_projection.py`

Two modes:

**Fast mode** (solver trials): Scales Year-1 scalar totals by degradation factors.
```
solar_mwh[t]   = solar_mwh_y1 × solar_eff[t]
wind_mwh[t]    = wind_mwh_y1  × wind_eff[t]
battery_mwh[t] = bess_mwh_y1  × soh[t]
meter_mwh[t]   = (solar + wind + battery)[t] × loss_factor
```

**Full mode** (final reporting): Re-runs `PlantEngine.simulate()` for each of the 25 years with degraded inputs:
```
For year t:
  solar_cap[t]   = base_solar_mw × solar_eff[t]
  wind_cap[t]    = base_wind_mw  × wind_eff[t]
  bess_soh[t]    → passed as bess_soh_factor to PlantEngine
```

This captures non-linear BESS-solar interactions (lower solar → less surplus to charge BESS → different dispatch pattern) that scalar scaling misses.

### 6.4 LCOEModel

**File:** `src/hybrid_plant/finance/lcoe_model.py`

NPV-based LCOE (matches Excel NPV convention: CF at t discounted at `(1+r)^t`, t = 1..N):

```
WACC = D/V × Rd × (1 - Tc) + E/V × Re
     = 0.70 × 0.085 × (1 - 0.2517) + 0.30 × 0.20
     = 0.0446 + 0.060 = 0.1046  (~10.46%)

Debt schedule (fixed-EMI annuity over 25 years):
  EMI = debt_amount × r × (1+r)^n / ((1+r)^n - 1)
  interest[t]   = balance × r
  principal[t]  = EMI - interest[t]
  After debt_tenure: both = 0

ROE schedule: roe_annual = equity_amount × ROE   (constant 25-year stream)

NPV(costs) = NPV(interest) + NPV(principal) + NPV(ROE) + NPV(OPEX)
NPV(energy) = NPV(busbar_kWh[1..25])   (discounted at WACC)

LCOE = NPV(costs) / NPV(energy)   [Rs/kWh]
```

### 6.5 LandedTariffModel

**File:** `src/hybrid_plant/finance/landed_tariff_model.py`

All-in cost to deliver 1 kWh at the client meter:

```
Annual capacity charges (Rs/yr) = (CTU + STU + SLDC) × PPA_MW × 12  [fixed for all 25 years]
  Blended STU + SLDC rate = 119172.80 + 1440 = Rs 120612.80/MW/month

For each year t:
  re_payment[t]      = LCOE × busbar_kWh[t]
  wheeling[t]        = wheeling_rate × meter_kWh[t]   (Rs 0.97/kWh LT)
  elec_tax[t]        = tax_rate × meter_kWh[t]         (Rs 0.0/kWh currently)
  total_cost[t]      = re_payment + capacity_charges + wheeling + elec_tax
  landed_tariff[t]   = total_cost[t] / meter_kWh[t]   [Rs/kWh]
```

### 6.6 SavingsModel

**File:** `src/hybrid_plant/finance/savings_model.py`

Compares hybrid cost against a 100% DISCOM baseline:

```
discom_tariff  = weighted average of ToD rates by hour count
baseline_cost  = annual_load_kWh × discom_tariff  (constant across all 25 years)

For each year t:
  discom_draw_kWh[t] = annual_load_kWh - re_meter_kWh[t]
  hybrid_cost[t]     = re_meter_kWh[t] × landed_tariff[t]
                     + discom_draw_kWh[t] × discom_tariff
                     + annual_re_pen_cost[t]   (if penalty enabled)
  savings[t]         = baseline_cost - hybrid_cost[t]

savings_npv = NPV(savings[1..25], WACC)  ← the optimisation objective
```

### 6.7 FinanceEngine — Orchestrator

**File:** `src/hybrid_plant/finance/finance_engine.py`

Calls sub-models in sequence:
```
CAPEX → OPEX → EnergyProjection → LCOE → LandedTariff → Savings
```

Returns a flat dict with all breakdowns needed by the dashboard.

---

## 7. Optimisation Layer — SolverEngine

**File:** `src/hybrid_plant/solver/solver_engine.py`

Uses **Optuna Tree-structured Parzen Estimator (TPE)** — a Bayesian black-box optimiser.

**Algorithm:** TPE fits two density models over the observed objective values (good and bad trials) and samples new points from the ratio of densities. It is gradient-free and handles mixed integer-continuous search spaces, but it is **not guaranteed to find the global optimum**.

**Execution flow:**
```
1. Create Optuna study (direction="maximize")
2. For each trial (up to n_trials = 1500):
   a. Suggest parameters from search space
   b. Run Year1Engine.evaluate() (energy simulation)
   c. Run FinanceEngine.evaluate(fast_mode=True) (finance, with scalar energy projection)
   d. Check feasibility constraints
   e. Return savings_npv if feasible, else -1e15 (penalty)
3. Re-evaluate best trial with fast_mode=False (full 25-year re-simulation)
4. Return SolverResult with best params and full result dict
```

**Fast mode vs. full mode:** During trials, `EnergyProjection` uses scalar scaling (microseconds per trial). The final best solution always uses full per-year re-simulation for accuracy.

**n_trials = 1500, random_seed = 42, n_jobs = 1 (sequential)**

---

## 8. Decision Variables

**Current scope (actively optimised):**

| Variable | Type | Min | Max | Description |
|----------|------|-----|-----|-------------|
| `solar_capacity_mw` | continuous | 0 | 1000 | AC solar installed capacity |
| `wind_capacity_mw` | continuous | 0 | 1000 | Wind installed capacity |
| `ppa_capacity_mw` | continuous | 0 | 1000 | PPA contracted export cap |
| `bess_containers` | integer | 0 | 1000 | Number of 5.015 MWh containers |
| `bess_charge_c_rate` | continuous | 0 | 1.0 | Charge power as fraction of energy capacity |
| `bess_discharge_c_rate` | continuous | 0 | 1.0 | Discharge power as fraction of energy capacity |

**Derived from containers:**
```
bess_energy_mwh = bess_containers × 5.015
charge_power_mw = charge_c_rate × bess_energy_mwh
discharge_power_mw = discharge_c_rate × bess_energy_mwh
```

**Future scope (fixed in current model):**

| Variable | Current fixed value | Options |
|----------|--------------------|---------| 
| `dispatch_priority` | `solar_first` | `solar_first`, `wind_first`, `proportional` |
| `bess_charge_source` | `solar_only` | `solar_only`, `wind_only`, `solar_and_wind` |

---

## 9. Constraints

All constraints are checked in `SolverEngine._is_feasible()`. Infeasible trials are assigned objective = -1e15 (not pruned from search, to preserve TPE learning).

| Constraint | Enabled by default | Formula |
|------------|-------------------|---------|
| `minimum_savings_npv` | **Yes** | `savings_npv ≥ 0` — project must beat 100% DISCOM |
| `plant_cuf` | No | `min_pct ≤ busbar_MWh / (PPA_MW × 8760) × 100 ≤ max_pct` |
| `minimum_bess_capacity` | No | `bess_energy_mwh ≥ min_mwh` |
| `minimum_bess_discharge` | No | `annual_discharge_MWh ≥ min_annual_mwh` |
| `re_penetration` | No | `meter_MWh / annual_load × 100 ∈ [min, max]` |
| `hourly_re_penetration_penalty` | No | Soft penalty; not a hard gate (see §6.6) |

**Physical constraints enforced inside PlantEngine:**
- `SOC[h] ∈ [0, energy_capacity]` at all hours
- `charge[h] ≤ charge_power_cap`
- `discharge[h] ≤ discharge_power_cap`
- `plant_export[h] ≤ PPA_capacity_mw`
- `solar_direct[h] ≤ solar_generation[h]`
- Energy conservation: generation = direct + charge + curtailment

---

## 10. Objective Function

**Maximise client savings NPV over 25 years:**

```
Objective = NPV( savings[1..25],  WACC )

where:
  savings[t]  = baseline_cost - hybrid_cost[t]
  WACC        ≈ 10.46%  (fixed for current financing structure)
```

This is equivalent to minimising total hybrid cost NPV (since baseline is a constant independent of decision variables).

---

## 11. Key Mathematical Formulations

### Energy balance at the busbar (per hour h):

```
solar_gen[h]     = solar_cap × solar_cuf[h]
wind_gen[h]      = wind_cap  × wind_cuf[h]
total_gen[h]     = solar_gen[h] + wind_gen[h]

direct[h]        = solar_direct[h] + wind_direct[h]   ≤ PPA_cap
direct[h]        ≤ required_pre[h]  = load[h] / loss_factor

charge[h]        = min( surplus[h], charge_power_cap, energy_cap - SOC[h] )
curtailment[h]   = total_gen[h] - direct[h] - charge[h]     ≥ 0

SOC[h+1]         = SOC[h]  +  charge[h] × η_c  −  aux[h]  −  discharge_raw[h]
                   ∈ [0, energy_cap]

discharge_pre[h] = discharge_raw[h] × η_d
load_coverage[h] = direct[h] × loss_factor + discharge_pre[h] × loss_factor
discom_draw[h]   = max( load[h] − load_coverage[h], 0 )
```

### SOC state transition (full form):

```
SOC[h+1] = SOC[h]
          + charge[h] × η_charge
          − ceil(SOC[h] / container_size) × aux_per_hour   (if SOC > 0)
          − discharge_raw[h]

Bounds: 0 ≤ SOC[h] ≤ energy_capacity  ∀h
```

### LCOE (NPV basis):

```
LCOE = NPV( interest + principal + ROE + OPEX ) / NPV( busbar_kWh )

All NPV computed at WACC using Excel convention:
  NPV(series, r) = Σ_{t=1}^{N}  series[t] / (1 + r)^t
```

### Savings NPV (optimisation objective):

```
baseline    = annual_load_kWh × weighted_discom_tariff
hybrid[t]   = meter_kWh[t] × landed_tariff[t]  +  discom_draw_kWh[t] × discom_tariff
savings[t]  = baseline − hybrid[t]
obj         = Σ_{t=1}^{25}  savings[t] / (1 + WACC)^t
```

---

## 12. Current Limitations Relevant to Pyomo Migration

1. **Black-box heuristic optimisation.** Optuna TPE cannot guarantee global optimality. Different random seeds produce different solutions. The 1500-trial budget is a practical limit, not a convergence criterion.

2. **Sequential BESS dispatch loop.** The 8760-hour for-loop in `PlantEngine.simulate()` is fundamentally state-dependent (SOC[h+1] depends on SOC[h]). This is the key reformulation challenge for Pyomo: it must be expressed as a set of simultaneous LP/MILP constraints indexed over the time horizon.

3. **Heuristic ToD reservation planner.** The `rsrv_evening / rsrv_morning_next / rsrv_fwd_evening` state machine is a hand-crafted heuristic that approximates optimal BESS scheduling. In a Pyomo model, this heuristic is replaced by the optimiser itself via price-signal dispatch — the solver naturally chooses when to dispatch given the tariff structure.

4. **Mutual exclusion of charge and discharge.** PlantEngine does not enforce simultaneous charge+discharge prevention explicitly — it is implicit from the sequential logic. A Pyomo MILP must add a binary variable per hour to prevent simultaneous charge and discharge (or rely on LP relaxation if objective pushes against it).

5. **Non-linear objective.** The LCOE formula has a fractional NPV structure: `NPV(costs) / NPV(energy)`. This makes the LCOE itself non-linear. In a Pyomo model, if the objective is savings NPV (not LCOE), this is avoidable — savings NPV is bilinear in sizing variables and tariff terms but can be linearised if LCOE is pre-computed for a fixed financing structure.

6. **Integer variable.** `bess_containers` is an integer (number of physical units). This makes the problem MILP, not pure LP. Relaxing to continuous MWh capacity is a common simplification.

7. **25-year multi-period model.** Full 25-year re-simulation in `EnergyProjection` runs PlantEngine 25 times. In Pyomo, the energy projection would need to be expressed either as a two-stage model (year-1 energy × degradation factor) or as a multi-period LP across all 25 years (computationally expensive).

8. **Dispatch window / charge-first logic.** The binary `charge_first` flag and `discharge_hours` window create conditional branches in the dispatch logic. These become additional binary variables and `if-then` big-M constraints in a MILP.

---

## 13. Future Scope Identified in the Codebase

The following items are explicitly marked `scope: future` in `solver.yaml` or implied by TODO comments and architecture decisions:

| Feature | Current state | Notes |
|---------|--------------|-------|
| `dispatch_priority` | Fixed at `solar_first` | Can be a categorical decision variable (requires categorical branching in Pyomo or 3 binary flags) |
| `bess_charge_source` | Fixed at `solar_only` | Same as above |
| **Augmentation optimisation** | Not implemented | Key next phase: given an existing plant (current capacities as lower bounds), find optimal incremental additions of solar/wind/BESS to maximise savings NPV |
| **Multi-scenario robustness** | Not implemented | Run optimisation across multiple solar/wind resource years; min-max or stochastic objective |
| **Demand response / load shifting** | Not implemented | Load profile is currently fixed; could model load flexibility |
| **Grid export revenue** | Not implemented | Excess generation currently curtailed; Pyomo could optimise grid export vs. self-consumption |
| **Banking / net metering** | Config present in `regulatory.yaml` but not implemented in dispatch | Monthly energy banking rules exist in config structure |
| **Greenfield vs. brownfield** | Not differentiated | Augmentation scenario needs brownfield flag and existing capacity parameters |
| **Multi-plant / portfolio** | Not implemented | Single plant only; future could extend to portfolio optimisation |
| **Transmission capacity constraint** | Fixed 10 km, fixed cost | Could be a decision variable (line rating vs. capacity) |
| **Storage degradation as endogenous variable** | Exogenous SOH curve | Could model SOH as a function of cycle depth and frequency |

---

## 14. Pyomo Migration — What Needs to Change

This section summarises the architectural changes needed to convert the model to a Pyomo algebraic optimisation model. Use this to build the ChatGPT prompt.

### 14.1 Model Type

The Pyomo model will be a **Mixed-Integer Linear Program (MILP)** (or LP relaxation):
- Continuous: `solar_cap`, `wind_cap`, `ppa_cap`, `bess_energy_mwh`, `charge_power`, `discharge_power`
- Integer (or continuous relaxation): `bess_containers` → `bess_energy_mwh`
- Binary (per hour, for charge/discharge mutual exclusion): `u_charge[h]`, `u_discharge[h]` s.t. `u_charge[h] + u_discharge[h] ≤ 1`
- Continuous (per hour, 8760): `solar_direct[h]`, `wind_direct[h]`, `charge[h]`, `discharge[h]`, `SOC[h]`, `curtailment[h]`, `discom_draw[h]`

### 14.2 Decision Variables to Declare

```python
# Sizing (scalar)
model.solar_cap   = Var(domain=NonNegativeReals)
model.wind_cap    = Var(domain=NonNegativeReals)
model.ppa_cap     = Var(domain=NonNegativeReals)
model.bess_energy = Var(domain=NonNegativeReals)   # MWh (relaxed from integer containers)
model.charge_pw   = Var(domain=NonNegativeReals)   # = charge_c_rate × bess_energy
model.discharge_pw= Var(domain=NonNegativeReals)

# Hourly dispatch (indexed over T = range(8760))
model.solar_d     = Var(T, domain=NonNegativeReals)
model.wind_d      = Var(T, domain=NonNegativeReals)
model.charge      = Var(T, domain=NonNegativeReals)
model.discharge   = Var(T, domain=NonNegativeReals)
model.soc         = Var(T, domain=NonNegativeReals)
model.curtail     = Var(T, domain=NonNegativeReals)
model.discom      = Var(T, domain=NonNegativeReals)

# Optional binary for charge/discharge exclusion
model.u_chg       = Var(T, domain=Binary)
model.u_dis       = Var(T, domain=Binary)
```

### 14.3 Constraints to Add

```
# Generation bounds
solar_d[h]  ≤ solar_cap × solar_cuf[h]
wind_d[h]   ≤ wind_cap  × wind_cuf[h]

# PPA cap
solar_d[h] + wind_d[h] + discharge[h] × η_d ≤ ppa_cap

# Load balance (with DISCOM fill)
solar_d[h] × lf + wind_d[h] × lf + discharge[h] × η_d × lf + discom[h] = load[h]

# SOC transition
soc[h+1] = soc[h] + charge[h] × η_c - discharge[h] - aux_loss[h]
soc[0]   = 0   (or periodic boundary)

# SOC bounds
0 ≤ soc[h] ≤ bess_energy

# Power limits
charge[h]    ≤ charge_pw
discharge[h] ≤ discharge_pw
charge_pw    = charge_c_rate × bess_energy    (bilinear → linearisation needed)
discharge_pw = discharge_c_rate × bess_energy (bilinear → linearisation needed)

# Mutual exclusion
charge[h]    ≤ u_chg[h] × charge_pw_max
discharge[h] ≤ u_dis[h] × discharge_pw_max
u_chg[h] + u_dis[h] ≤ 1

# Non-negativity
All variables ≥ 0
discom[h] ≥ 0
curtail[h] ≥ 0

# Energy conservation
solar_d[h] + wind_d[h] + charge[h] + curtail[h] = solar_gen[h] + wind_gen[h]
```

### 14.4 Objective Function

Replace Optuna's black-box search with an algebraic objective:

```python
# Savings NPV = NPV(baseline - hybrid_cost, WACC)
# For Year-1 only (simplification) or full 25-year via parametric degradation:

# Linearised objective (Year-1 approximation):
savings_y1 = baseline_cost - sum(discom[h] × discom_rate[h]) 
           - sum(meter_delivery[h] × landed_tariff_y1)

# Full 25-year objective requires:
#   1. Pre-compute degradation factors d_solar[t], d_wind[t], d_soh[t]
#   2. Scale Year-1 energy by these factors (fast mode parametric approach)
#   3. Express LCOE and landed_tariff as functions of sizing variables and pre-computed energy
```

### 14.5 Handling Non-Linearity

Key non-linear terms that need treatment:

| Non-linearity | Source | Suggested treatment |
|---------------|--------|---------------------|
| `charge_pw = c_rate × bess_energy` | product of two continuous vars | Fix c_rates as parameters; or use McCormick envelopes / piecewise linearisation |
| `LCOE = NPV(cost) / NPV(energy)` | fractional objective | Pre-compute LCOE for given sizing, or use Dinkelbach's iterative algorithm, or express as maximise savings (avoids explicit LCOE) |
| `landed_tariff = total_cost / meter_kWh` | ratio | Similar to LCOE — linearise by expressing savings NPV directly |
| `aux_loss = f(active_containers, SOC)` | `ceil(SOC/container_size)` | Approximate as proportional to SOC; or use binary variables per container |

### 14.6 Augmentation-Specific Extensions

For the augmentation use case (next phase of development), add:

```python
# Existing capacity (fixed lower bounds from current plant)
model.solar_cap_existing   = Param(value=...)   # existing solar AC MW
model.wind_cap_existing    = Param(value=...)   # existing wind MW
model.bess_energy_existing = Param(value=...)   # existing BESS MWh

# Augmentation variables (incremental additions)
model.delta_solar   = Var(domain=NonNegativeReals)
model.delta_wind    = Var(domain=NonNegativeReals)
model.delta_bess    = Var(domain=NonNegativeReals)
model.delta_ppa     = Var(domain=NonNegativeReals)

# Total capacity = existing + augmentation
model.solar_cap     = model.solar_cap_existing + model.delta_solar
model.wind_cap      = model.wind_cap_existing  + model.delta_wind
model.bess_energy   = model.bess_energy_existing + model.delta_bess
model.ppa_cap       = model.ppa_cap_existing   + model.delta_ppa

# Incremental CAPEX (augmentation cost only)
augmentation_capex = delta_solar × ac_dc_ratio × solar_cost_per_mwp
                   + delta_wind  × wind_cost_per_mw
                   + delta_bess  × bess_cost_per_mwh
```

### 14.7 Recommended Pyomo Model Structure

```
pyomo_model/
├── sets.py          # T (hours), Y (years), components
├── params.py        # solar_cuf, wind_cuf, load, degradation factors, rates
├── variables.py     # all Var declarations
├── constraints/
│   ├── energy_balance.py    # per-hour generation = dispatch + charge + curtail
│   ├── soc_dynamics.py      # SOC state transition + bounds
│   ├── ppa_limits.py        # export cap, power limits
│   ├── finance.py           # CAPEX, OPEX, LCOE, tariff, savings
│   └── augmentation.py      # brownfield lower bounds, delta variables
├── objective.py     # maximise savings NPV
└── solve.py         # instantiate model, call solver (GLPK / CBC / Gurobi / HiGHS)
```

---

*This document was auto-generated from the `hybrid_plant` codebase on 2026-06-04.*
*Source: `src/hybrid_plant/` — 27 Python files, ~52k words of code and comments.*
