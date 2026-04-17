# Hybrid Plant — C&I Financial Model

Optimises a hybrid Solar + Wind + BESS plant configuration for a C&I client,
benchmarked against a 100 % DISCOM baseline using an NPV-based LCOE framework.
Includes a cohort-aware BESS Augmentation Engine for 25-year lifecycle planning.

## Project structure

```
hybrid_plant/
├── configs/                          # YAML configuration
│   ├── project.yaml                  # Site, generation sources, load
│   ├── bess.yaml                     # BESS container, efficiency, augmentation
│   ├── finance.yaml                  # CAPEX, OPEX, financing
│   ├── regulatory.yaml               # Grid losses, banking
│   ├── tariffs.yaml                  # DISCOM ToD tariff schedule
│   └── solver.yaml                   # Optuna bounds, constraints, trials
├── data/                             # 8760-hour profiles + degradation curves
│   ├── solar_cuf_8760.csv
│   ├── wind_cuf_8760.csv
│   ├── load_profile_8760.csv
│   ├── solar_efficiency_curve.csv    # End-of-year solar efficiency
│   ├── wind_efficiency_curve.csv     # End-of-year wind efficiency
│   └── bess_soh_curve.csv            # End-of-year BESS SOH
├── outputs/                          # Generated plots (git-ignored)
├── src/
│   └── hybrid_plant/
│       ├── constants.py
│       ├── config_loader.py
│       ├── data_loader.py            # + operating_value helper
│       ├── run_model.py              # Master entry point
│       ├── energy/                   # Hourly dispatch simulation
│       │   ├── plant_engine.py
│       │   ├── grid_interface.py
│       │   ├── meter_layer.py
│       │   └── year1_engine.py
│       ├── finance/                  # LCOE, OPEX, savings pipeline
│       │   ├── capex_model.py
│       │   ├── opex_model.py
│       │   ├── energy_projection.py
│       │   ├── lcoe_model.py
│       │   ├── landed_tariff_model.py
│       │   ├── savings_model.py
│       │   └── finance_engine.py
│       ├── solver/
│       │   └── solver_engine.py      # Optuna TPE optimiser
│       └── augmentation/             # BESS augmentation engine
│           ├── cohort.py
│           ├── cuf_evaluator.py
│           ├── lifecycle_simulator.py
│           └── augmentation_engine.py
├── tests/                            # Pytest suite (72 tests)
│   ├── conftest.py
│   ├── test_energy.py
│   ├── test_finance.py
│   ├── test_solver.py
│   └── test_augmentation.py
└── smoke_test.py                     # Standalone 76-check smoke test
```

## Quick start

```bash
# Install (editable)
pip install -e ".[dev]"

# Run full optimisation + dashboard
python -m hybrid_plant.run_model

# Run tests
pytest tests/                  # all 72 tests (~1 min)
pytest tests/ -m "not slow"    # fast tests only (skips solver suite)

# Run standalone smoke test
python smoke_test.py
```

## Plant CUF — definition

```
CUF (%) = annual busbar MWh / (PPA_MW × 8760) × 100
```

This is the transparent naive formula applied everywhere in the model. It
responds to all three degradation sources (solar, wind, BESS) because busbar
depends on year-t operating values for all three. It matches business
intuition: *"what fraction of the contracted PPA capacity did the plant
actually use?"*

The augmentation engine uses this same formula for trigger and restoration
decisions — no separate "canonical" CUF, no re-runs with `loss_factor = 1`.

## Degradation curve convention

All three curves (`bess_soh_curve.csv`, `solar_efficiency_curve.csv`,
`wind_efficiency_curve.csv`) use the **end-of-year** convention:

- `curve[N]` = residual efficiency/SOH at the END of year N
- During year 1 (install year), the cohort/plant operates at **1.0**
  (no degradation has accumulated yet)
- During year N ≥ 2, the operating value is `curve[N - 1]`

Use `hybrid_plant.data_loader.operating_value(curve, year_or_age)` to look up
the correct operating value for any year or cohort age.

## Augmentation engine — key design points

- **Cohort-based:** each event installs a new "cohort" that ages independently
  from its install year. `cohort.py::CohortRegistry` aggregates active cohorts
  into `(total_containers, blended_soh)` for PlantEngine.
- **Two-pass solver:** Pass 1 optimises the no-augmentation baseline. Pass 2
  (optional, `solver_aware: true`) re-optimises with augmentation lifecycle
  included. Default is `solver_aware: false` — Pass 1 only, followed by
  post-processing the winner through the augmentation engine.
- **Blended trigger:** Plant CUF responds to all three degradation streams.
  A trigger fires when `CUF_t < threshold_CUF - tolerance_pp`.
- **Trigger tolerance:** `trigger_tolerance_pp: 0.0001` in `bess.yaml`
  suppresses float-noise events.
- **Adjusted restoration target:** augmentation cannot replace degraded
  solar/wind. Year-t target =
  `Y1_CUF × (solar_share × solar_eff[t] + wind_share × wind_eff[t] + bess_share)`,
  so the k-search aims at what augmentation can actually achieve.
- **OPEX-only treatment:** augmentation lump cost is a one-time OPEX line item
  in the event year; recurring O&M ladders forward from event year to Y25.
  CAPEX, debt, and depreciation are never altered by augmentation events.

## Dashboard output

`run_model.py` produces a console dashboard with these sections:

- **1** — Optimal configuration
- **2** — CAPEX, financing, WACC, EMI, ROE
- **3** — Year-1 energy balance (raw → busbar → BESS flows → meter → CUFs)
- **4** — LCOE NPV build-up and landed tariff decomposition
- **5** — Client savings vs DISCOM baseline
- **6** — Base OPEX breakdown
- **6b** — Augmentation schedule with adjusted target and ≥Adj?/≥Hard? flags
- **7a** — Per-year energy flows (containers, solar/wind eff, all MWh streams, losses)
- **7b** — Per-year load, DISCOM, RE penetration, CUFs, PPA utilization
- **7c** — Per-year economics (base OPEX, aug lump, aug O&M, landed tariff,
  hybrid cost, savings)

Plus 3 PNG plots saved to `outputs/`:

- `model_output.png` — 4-panel dashboard
- `day250_dispatch.png` — BESS hourly dispatch diagnostic
- `augmentation_dashboard.png` — CUF curve with events, cohort stack, cash impact
