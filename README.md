# Hybrid Plant — C&I Financial Model

Optimises a hybrid Solar + Wind + BESS plant configuration for a C&I client,
benchmarked against a 100 % DISCOM baseline using an NPV-based LCOE framework.

## Project structure

```
hybrid_plant/
├── configs/          # YAML configuration (project, finance, regulatory, solver, …)
├── data/             # 8760-hour time-series CSVs + degradation curves
├── outputs/          # Generated plots and reports (git-ignored)
├── src/
│   └── hybrid_plant/
│       ├── constants.py          # Shared unit-conversion constants
│       ├── config_loader.py      # YAML loader + FullConfig dataclass
│       ├── data_loader.py        # Time-series + degradation curve loader
│       ├── run_model.py          # Master entry point
│       ├── energy/               # Hourly dispatch simulation
│       │   ├── plant_engine.py
│       │   ├── grid_interface.py
│       │   ├── meter_layer.py
│       │   └── year1_engine.py
│       ├── finance/              # LCOE, OPEX, savings pipeline
│       │   ├── capex_model.py
│       │   ├── opex_model.py
│       │   ├── energy_projection.py
│       │   ├── lcoe_model.py
│       │   ├── landed_tariff_model.py
│       │   ├── savings_model.py
│       │   └── finance_engine.py
│       └── solver/
│           └── solver_engine.py  # Optuna TPE optimiser
└── tests/
    ├── conftest.py               # Shared pytest fixtures
    ├── test_energy.py
    ├── test_finance.py
    ├── test_solver.py
    └── validate_solver.py
```

## Quick start

```bash
# Install (editable)
pip install -e ".[dev]"

# Run full optimisation + dashboard
python -m hybrid_plant.run_model

# Run tests
pytest tests/
```
