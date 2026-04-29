# Hybrid RE Plant Model — Technical Design Document

**Version**: post-refactor  
**Audience**: Engineers, Product Stakeholders, Reviewers

---

## 1. Overview

### Purpose

The hybrid plant model sizes and optimises a co-located **Solar + Wind + BESS** (Battery Energy Storage System) plant to maximise electricity cost savings for an industrial client currently purchasing 100 % of its power from the DISCOM (Distribution Company).

### Problem Being Solved

Industrial consumers face rising DISCOM tariffs with Time-of-Day (ToD) premiums during peak hours. A hybrid RE plant can displace DISCOM purchases by generating and storing cheap renewable energy, then dispatching it during high-tariff periods. The model answers:

1. What is the optimal combination of solar capacity, wind capacity, BESS size, and PPA limit?
2. What is the all-in landed cost (Rs/kWh) of RE at the client meter?
3. What are the 25-year savings versus the DISCOM baseline?

---

## 2. System Architecture

### Module Map

```
hybrid_plant/
├── run_model.py            — Entry point: solver → dashboard → plots
├── config_loader.py        — Loads and validates YAML configs → FullConfig
├── data_loader.py          — Loads hourly CSVs and degradation curves
├── constants.py            — Unit conversion constants
├── _paths.py               — Project-root resolution (pyproject.toml sentinel)
│
├── energy/
│   ├── plant_engine.py     — 8760-hour dispatch simulation (busbar basis)
│   ├── grid_interface.py   — Blended HT/LT loss factor → meter delivery
│   ├── meter_layer.py      — DISCOM shortfall at the client meter
│   └── year1_engine.py     — Orchestrates plant → grid → meter (Year 1)
│
├── finance/
│   ├── finance_engine.py   — Finance pipeline orchestrator
│   ├── capex_model.py      — Component-wise CAPEX
│   ├── opex_model.py       — 25-year OPEX projection with escalation
│   ├── lcoe_model.py       — NPV-based LCOE and debt schedule
│   ├── landed_tariff_model.py — All-in landed tariff series
│   ├── savings_model.py    — Client savings vs DISCOM baseline
│   ├── energy_projection.py — 25-year degraded energy delivery
│   └── _utils.py           — Shared financial utilities (NPV)
│
└── solver/
    └── solver_engine.py    — Optuna TPE optimiser over decision variables
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| `PlantEngine` | Physics: dispatch, BESS charge/discharge, curtailment |
| `GridInterface` | Regulatory: blended loss factor from HT/LT split |
| `MeterLayer` | Accounting: DISCOM shortfall at client meter |
| `Year1Engine` | Orchestration: Year-1 plant → grid → meter pipeline |
| `EnergyProjection` | Degradation: 25-year per-year full re-simulation |
| `CapexModel` | Cost: component-wise CAPEX (solar DC, wind, BESS, transmission) |
| `OpexModel` | Cost: escalating 25-year OPEX |
| `LCOEModel` | Finance: NPV(costs)/NPV(energy), debt schedule, WACC |
| `LandedTariffModel` | Finance: absolute annual cost → Rs/kWh at meter |
| `SavingsModel` | Finance: client savings vs DISCOM baseline, NPV |
| `SolverEngine` | Optimisation: Optuna TPE, objective = maximise savings NPV |

---

## 3. Core Concepts

### CUF — Capacity Utilisation Factor

```
CUF (%) = Annual busbar MWh / (Capacity MW × 8760) × 100
```

- **Solar CUF**: Raw solar generation / (solar AC capacity × 8760). Determined by the hourly irradiance profile (`solar_cuf` CSV). Typical Indian value: 20–28 %.
- **Wind CUF**: Raw wind generation / (wind capacity × 8760). Determined by the hourly wind profile (`wind_cuf` CSV). Typical: 30–40 %.
- **Plant CUF**: Total busbar delivery / (PPA capacity × 8760). This is the headline figure used in PPA contracts. It reflects how well the contracted capacity is utilised after dispatch, BESS operation, and curtailment.

### Solar Generation

Each hour `h`:
```
solar_generation[h] = solar_capacity_mw × solar_cuf[h]
```
The `solar_cuf` profile (0–1 fraction) is loaded from a CSV and represents the effective AC output factor per hour, accounting for irradiance, temperature, and inverter losses.

### Wind Generation

```
wind_generation[h] = wind_capacity_mw × wind_cuf[h]
```
Same pattern as solar; the `wind_cuf` profile captures capacity-factor variation across hours.

### BESS — Battery Energy Storage System

The BESS is sized by **number of containers** (each a fixed MWh block) and **C-rates**:

```
energy_capacity_mwh  = bess_containers × container_size_mwh × SOH
charge_power_mw      = charge_c_rate   × energy_capacity_mwh
discharge_power_mw   = discharge_c_rate × energy_capacity_mwh
```

**State of Health (SOH)** degrades annually (loaded from a CSV), reducing both energy capacity and power caps proportionally.

**Efficiency**:
- Charge: `soc += charge_energy × charge_eff`
- Discharge: `discharge_output = discharge_raw × discharge_eff`
- Aux consumption: parasitic draw from active containers each hour

### Degradation

Three independent degradation curves (CSV, end-of-year convention):

| Asset | Parameter | Effect |
|---|---|---|
| Solar | `solar_eff[year]` | AC capacity factor: `solar_mw × eff` |
| Wind | `wind_eff[year]` | Capacity factor: `wind_mw × eff` |
| BESS | `soh[year]` | `bess_soh_factor` passed to PlantEngine |

**Operating value convention**: Year-1 uses `eff = 1.0` (no degradation yet). Year N uses the end-of-Year-(N-1) value from the curve.

### Dispatch Logic

See §5 (Dispatch & System Behavior) for the full hourly sequence.

### Economics

- **LCOE**: NPV of all costs ÷ NPV of busbar energy (kWh). Represents the RE generation cost before grid losses.
- **Landed tariff**: LCOE + capacity charges + wheeling + electricity tax + banking, normalised to Rs/kWh at the client meter.
- **Savings**: `baseline_cost − hybrid_cost`, where baseline = 100 % DISCOM and hybrid = RE cost + DISCOM residual.

---

## 4. Data Flow

```
Configs (YAML)  ──→  FullConfig
CSV profiles    ──→  data dict (solar_cuf, wind_cuf, load_profile)
                          │
                          ▼
                   SolverEngine (Optuna TPE)
                   ├── for each trial:
                   │     params → Year1Engine.evaluate()
                   │                 PlantEngine.simulate()  ← hourly dispatch
                   │                 GridInterface.apply_losses()
                   │                 MeterLayer.compute_shortfall()
                   │           → FinanceEngine.evaluate()
                   │                 CapexModel  → CAPEX
                   │                 OpexModel   → 25-yr OPEX
                   │                 EnergyProjection → 25-yr busbar + meter
                   │                 LCOEModel   → LCOE, WACC, debt schedule
                   │                 LandedTariffModel → Rs/kWh series
                   │                 SavingsModel → annual savings, NPV
                   │           objective = savings_npv
                   └── best_params
                          │
                          ▼
                   Final full evaluation (fast_mode=False)
                          │
                          ▼
                   Dashboard print + plots
```

### Key Inputs

| Input | Source | Unit |
|---|---|---|
| `solar_cuf` | hourly CSV | fraction [0–1] |
| `wind_cuf` | hourly CSV | fraction [0–1] |
| `load_profile` | hourly CSV | MWh/h |
| Solar/wind degradation | CSV | fraction [0–1] per year |
| BESS SOH | CSV | fraction [0–1] per year |
| CAPEX rates | `finance.yaml` | Rs/MWp, Rs/MW, Rs/MWh |
| OPEX rates | `finance.yaml` | Rs Lakh/MWp, Rs Lakh/MW |
| DISCOM ToD tariffs | `tariffs.yaml` | Rs/kWh per period |
| Loss percentages | `regulatory.yaml` | % per grid segment |

### Key Outputs

| Output | Meaning |
|---|---|
| `lcoe_inr_per_kwh` | Levelised cost at busbar |
| `landed_tariff_series[0]` | Year-1 all-in cost at client meter |
| `savings_npv` | 25-year NPV of savings vs DISCOM |
| `annual_savings_year1` | Year-1 cash saving |
| `delivered_meter_mwh` | Annual RE at client meter, each year |
| `plant_export_pre` | Hourly busbar export (8760-element array) |

---

## 5. Dispatch & System Behavior

### Hourly Dispatch Sequence (PlantEngine)

Each of the 8760 hours runs in order:

1. **Direct dispatch** — solar and/or wind → load, up to PPA cap.
   - Priority: `solar_first` | `wind_first` | `proportional` (config).
   - Capped at `ppa_capacity_mw`.

2. **BESS charging** — surplus RE → battery.
   - Source: `solar_only` | `wind_only` | `solar_and_wind` (config).
   - Capped at `charge_power_mw` and remaining headroom `energy_capacity − soc`.

3. **Curtailment** — any remaining RE surplus after charging is wasted.

4. **Auxiliary consumption** — parasitic draw proportional to active containers.

5. **BESS discharge** — ToD-aware, fills shortfall if profitable.

### ToD-Aware Discharge

The BESS prioritises high-tariff hours:

| Period | Hours (0-indexed) | Rate |
|---|---|---|
| Morning peak | 7–10 | ₹9.182/kWh |
| Evening peak | 18–21 | ₹9.182/kWh |
| Normal | 0–6, 15–17, 22–23 | ₹8.687/kWh |
| Solar offpeak | 11–14 | ₹8.027/kWh |

### SOC Reservation Planner

Two fixed trigger points per day prevent cheap hours from consuming SOC needed later:

**hod = 11** (solar window opens):
- Zero out `rsrv_morning_next` (morning peak just ended).
- Compute `rsrv_fwd_evening`: walk the solar charging window (hod 11–14) to estimate post-charging SOC, then reserve enough for tonight's evening peak.
- Effect: prevents cheap offpeak discharge from stealing SOC needed at hod 18–21.

**hod = 15** (solar window closes, actual SOC known):
- Zero `rsrv_fwd_evening` (replaced by definitive value).
- Set `rsrv_evening`: actual reservation for hod 18–21 today.
- Set `rsrv_morning_next`: reservation for hod 7–10 tomorrow, funded from `soc − rsrv_evening`.

**Available SOC per period**:

| Period | Available SOC |
|---|---|
| morning_peak | `soc` (the reservation IS for this window) |
| evening_peak | `max(soc − rsrv_morning_next, 0)` |
| solar_offpeak | `max(soc − rsrv_fwd_evening, 0)` |
| normal | `max(soc − rsrv_evening − rsrv_morning_next, 0)` |

---

## 6. Mathematical Logic

### WACC

```
WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)
```

Where D/V = debt fraction, Rd = debt interest rate, Tc = corporate tax rate, E/V = equity fraction, Re = ROE.

### NPV (Excel-style)

```
NPV = Σ CF_t / (1 + r)^t    for t = 1 … n
```

Implemented via `finance._utils.npv(series, rate)`, shared by both `LCOEModel` and `SavingsModel`.

### LCOE

```
LCOE (Rs/kWh) = NPV(interest + principal + ROE + OPEX) / NPV(busbar kWh)
```

All NPVs discounted at WACC.

### Debt Schedule

Fixed-EMI amortising loan:
```
EMI = P × r × (1+r)^n / ((1+r)^n − 1)
interest_t  = balance_{t-1} × r
principal_t = EMI − interest_t
```

After debt tenure, interest and principal are zero (project continues on equity).

### Landed Tariff

```
landed_t = (LCOE × busbar_kWh_t
          + capacity_charges_annual
          + wheeling_per_kwh × meter_kWh_t
          + electricity_tax_per_kwh × meter_kWh_t
          + banking_per_kwh × banked_kWh_t)
         / meter_kWh_t
```

Capacity charges = `(CTU + STU + SLDC) × PPA_MW × 12`, fixed annually.

### Client Savings

```
baseline_cost = annual_load_kWh × discom_tariff
hybrid_cost_t = (RE_meter_kWh_t × landed_t)
              + (DISCOM_draw_kWh_t × discom_tariff)
savings_t     = baseline_cost − hybrid_cost_t
savings_NPV   = NPV(savings, WACC)
```

---

## 7. Configuration

### YAML Files

| File | Key Parameters |
|---|---|
| `project.yaml` | Project life (25 yr), resolution (hourly), file paths |
| `bess.yaml` | Container size, efficiency (charge/discharge), aux consumption, SOH curve path |
| `finance.yaml` | CAPEX rates, OPEX rates, financing split, interest rate, ROE, tax rate, regulatory charges |
| `tariffs.yaml` | DISCOM ToD periods and rates (LT and HT) |
| `regulatory.yaml` | HT/LT split %, loss percentages (CTU, STU, wheeling) by voltage level |
| `solver.yaml` | Decision variable bounds, n_trials, random_seed, constraints |

### Decision Variables (Solver)

| Variable | Type | Description |
|---|---|---|
| `solar_capacity_mw` | Continuous | AC solar installed capacity |
| `wind_capacity_mw` | Continuous | Wind installed capacity |
| `ppa_capacity_mw` | Continuous | Contracted PPA export limit |
| `bess_containers` | Integer | Number of BESS containers |
| `charge_c_rate` | Continuous (opt.) | BESS charge rate |
| `discharge_c_rate` | Continuous (opt.) | BESS discharge rate |
| `dispatch_priority` | Fixed | `solar_first` / `wind_first` / `proportional` |
| `bess_charge_source` | Fixed | `solar_only` / `wind_only` / `solar_and_wind` |

### Key Sensitivities

- **Solar capacity**: directly scales raw generation; diminishing returns above ~2× PPA due to curtailment.
- **PPA capacity**: tighter cap reduces both generation and capacity charges.
- **BESS containers**: larger battery improves peak shaving but increases CAPEX.
- **C-rates**: higher rates allow faster charge/discharge but don't change energy capacity.

---

## 8. Outputs

### Dashboard (stdout)

Seven sections printed to stdout:

| Section | Content |
|---|---|
| 1 | Optimal configuration (capacities, BESS sizing, C-rates) |
| 2 | CAPEX and financing (debt, equity, WACC, EMI) |
| 3 | Year-1 energy balance (generation, BESS flows, CUF) |
| 4 | LCOE and landed tariff build-up |
| 5 | Client savings (Year-1, NPV, cumulative, payback year) |
| 6 | OPEX breakdown Year 1 vs Year 25 |
| 7 | 25-year projections table |

### Plots

| File | Content |
|---|---|
| `outputs/model_output.png` | 4-panel: savings, tariff, energy mix, OPEX stack |
| `outputs/day250_dispatch.png` | 4-panel BESS dispatch diagnostic for Day 250 |

### Interpreting Results

- **Landed tariff < DISCOM tariff**: the RE plant saves money in Year 1.
- **Savings NPV > 0**: net-present value positive over the project life.
- **Plant CUF**: higher is better; indicates how well the PPA capacity is utilised.
- **Curtailment**: generation wasted because the battery is full and load is met. High curtailment (>30 %) suggests BESS is undersized or PPA is too small.
- **Energy degrades over time**: expected; Year-25 delivery is lower than Year-1 due to degradation.

---

## 9. Edge Cases and Constraints

| Constraint | Handling |
|---|---|
| SOC never negative | `soc = max(soc − discharge_raw, 0)` |
| PPA cap enforced every hour | `direct_pre = min(sd + wd, ppa_capacity_mw)` |
| discharge headroom capped | `discharge_raw ≤ (ppa_cap − direct_pre) / discharge_eff` |
| Zero solar/wind | All arrays remain zero; curtailment/discharge = 0 |
| Zero energy_capacity | `bess_soh_factor = 0` effectively disables BESS |
| Blocked discharge hours | `discharge_mask` zeroes SOC reservations for blocked hods |
| `annual_discom = 0` if RE oversupply | Not possible: shortfall = max(load − delivery, 0) |
| `npv_energy ≤ 0` | `LCOEModel` raises `ValueError` |
| CAPEX negative inputs | `CapexModel` raises `ValueError` |

### Known Simplifications

- Load profile is static across all 25 years (no demand growth model).
- DISCOM tariff is constant in real terms (no escalation in savings model).
- Banking charge is currently a stub (always zero).
- Wind and solar use identical 8760-hour profiles across all 25 years; only the capacity factor scales.

---

## 10. Validation

### Test Coverage

Tests are in `tests/` and run via `pytest`:

| Test Module | Coverage |
|---|---|
| `test_energy.py` | GridInterface, MeterLayer, PlantEngine (energy conservation, PPA cap, SOC, discharge loss), Year1Engine integration, CUF formula |
| `test_finance.py` | CapexModel, OpexModel, LCOEModel (WACC formula), FinanceEngine end-to-end |
| `test_solver.py` | SolverEngine smoke test (marked `slow`; 50-trial run) |

### Smoke Test

`python -m smoke_test.py` validates all 62 invariants across every pipeline layer without the solver, using fixed benchmark parameters.

### Key Invariants Verified

- **Energy conservation**: raw generation = direct + charge + curtailment (to 0.001 MWh)
- **PPA cap**: `plant_export_pre[h] ≤ ppa_capacity_mw + ε` for all h
- **SOC non-negative**: at all times and end of year
- **Discharge loss identity**: `discharge_loss = (discharge_pre / discharge_eff) × (1 − discharge_eff)`
- **Meter ≤ busbar**: after applying loss factor
- **Landed tariff < DISCOM**: confirmed for benchmark parameter sets
- **WACC formula**: verified algebraically against config values
