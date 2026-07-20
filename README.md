# Hybrid Plant — C&I Financial Model

Sizes and dispatches a hybrid Solar + Wind + BESS plant for a C&I client in India,
benchmarked against a 100 % DISCOM baseline. A Pyomo MILP chooses capacities and
hourly dispatch; the finance stack evaluates the result on an NPV-based LCOE and
landed-tariff framework.

**Objective: maximise client savings NPV** — not developer IRR.

## Project structure

```
hybrid_plant/
├── configs/                          # YAML configuration
│   ├── project.yaml                  # Site, generation sources, load
│   ├── bess.yaml                     # BESS container, efficiency, degradation
│   ├── finance.yaml                  # CAPEX, OPEX, financing
│   ├── regulatory.yaml               # Grid losses, banking
│   ├── tariffs.yaml                  # DISCOM ToD tariff schedule
│   └── solver.yaml                   # Engine switch, decision-variable bounds,
│                                     #   optional constraints, solver settings
├── data/                             # 8760-hour profiles + degradation curves
│   ├── solar_cuf_8760.csv
│   ├── wind_cuf_8760.csv
│   ├── load_profile_8760.csv
│   ├── solar_efficiency_curve.csv    # End-of-year solar efficiency
│   ├── wind_efficiency_curve.csv     # End-of-year wind efficiency
│   └── bess_soh_curve.csv            # End-of-year BESS SOH
├── docs/                             # See docs/README.md for the index
├── outputs/                          # Generated dashboards + plots (git-ignored)
├── src/
│   └── hybrid_plant/
│       ├── constants.py
│       ├── config_loader.py
│       ├── data_loader.py            # + operating_value helper
│       ├── run_model.py              # Master entry point
│       ├── optimise/                 # ── Pyomo optimisation layer (production)
│       │   ├── config.py             # OptModelConfig dataclass
│       │   ├── params.py             # FullConfig + CSVs → OptParams
│       │   ├── sets.py               # Time-index construction
│       │   ├── variables.py          # Sizing + hourly dispatch variables
│       │   ├── constraints/          # balance, soc, allocation, ppa, optional
│       │   ├── objective.py          # savings_npv objective
│       │   ├── build.py              # Assembles the ConcreteModel
│       │   ├── solve.py              # Solver driver (HiGHS)
│       │   ├── verify.py             # Post-solve invariant checks
│       │   ├── report.py             # LCOE + landed tariff reporting
│       │   ├── dashboard.py          # Self-contained HTML dashboards
│       │   └── pipeline.py           # Cutover driver — LP + finance reporting
│       ├── energy/                   # Hourly dispatch simulation (RTC heuristic)
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
│       └── legacy/                   # Deprecated — pre-migration Optuna search
│           └── solver_engine.py
├── tests/                            # Pytest suite (276 tests)
│   ├── conftest.py
│   ├── test_energy.py
│   ├── test_finance.py
│   ├── test_solver.py                # Legacy Optuna engine (all slow-marked)
│   └── optimise/                     # Pyomo layer suite
└── smoke_test.py                     # Standalone smoke test
```

## Quick start

```bash
# Install (editable, with dev tooling)
pip install -e ".[dev]"

# Run optimisation + write dashboards
python -m hybrid_plant.run_model

# Tests
pytest tests -m "not slow"     # 220 fast tests (~6 s)
pytest tests                   # all 276, including full-horizon solves

# Lint
ruff check src tests
```

The solver is HiGHS, pulled in as the `highspy` wheel — no separate system
install. `optuna` is **not** a runtime dependency; it is needed only for the
legacy engine (`pip install -e ".[legacy]"`).

> **Git worktrees do not isolate this project.**
> `_paths.find_project_root()` walks up from the *installed package file*, not
> the working directory. With an editable install, code and configs always
> resolve to the repo that was `pip install -e`'d — so a test run inside a
> worktree silently reads the main checkout's `configs/` and `src/`, and
> comparing two worktrees measures the same thing twice. To test another
> revision, either install it into its own venv or change the file in place.

## Optimisation engine

`solver.yaml → solver.engine` selects the path:

| Value | Engine | Status |
|---|---|---|
| `pyomo` | Deterministic single-year MILP over sizing + 8760-hour dispatch | **Default, production** |
| `optuna` | Optuna TPE search over sizing, RTC heuristic dispatch | Deprecated, cross-check only |

The Pyomo model is a pure LP except for one integer variable (BESS container
count). It co-optimises sizing and hourly dispatch, which the Optuna path could
not do — that search sized the plant and then dispatched it with a fixed
round-the-clock heuristic.

Specification: [`docs/design/pyomo-migration-design.md`](docs/design/pyomo-migration-design.md).
Docstrings across `optimise/` cite its section numbers ("design §2.4").

## Reported numbers

The LP objective and the reported client savings are deliberately different
quantities, and `pipeline.py` surfaces three figures rather than one:

- **`lp_objective_npv`** — the LP's own ToD-valued objective. An optimisation
  proxy, not a financial statement.
- **finance `savings_npv`** — `FinanceEngine` evaluated on the LP's dispatch.
  **This is the reported headline.**
- **`oracle_rtc_npv`** — `FinanceEngine` on the RTC heuristic dispatch at the
  same sizing. The heuristic floor; the gap to it is the value of optimal
  dispatch.

Definitional gaps between the LP objective and the FinanceEngine savings
(ToD-vs-flat valuation ~1 %, aux netting ~0.5 %) are model-definition
differences, not errors.

## Plant CUF — definition

```
CUF (%) = annual busbar MWh / (PPA_MW × 8760) × 100
```

The transparent naive formula, applied everywhere. It responds to all three
degradation sources (solar, wind, BESS) because busbar depends on year-t
operating values for all three. It matches business intuition: *"what fraction
of the contracted PPA capacity did the plant actually use?"*

## Degradation curve convention

All three curves (`bess_soh_curve.csv`, `solar_efficiency_curve.csv`,
`wind_efficiency_curve.csv`) use the **end-of-year** convention:

- `curve[N]` = residual efficiency/SOH at the END of year N
- During year 1 (install year), the plant operates at **1.0**
  (no degradation has accumulated yet)
- During year N ≥ 2, the operating value is `curve[N - 1]`

Use `hybrid_plant.data_loader.operating_value(curve, year_or_age)` to look up
the correct operating value for any year or cohort age.

## Outputs

Written to `outputs/` (git-ignored):

- `dashboard_executive.html` — one page: KPI cards + headline charts
- `dashboard_detailed.html` — dense engineering board: KPI band, sizing,
  dispatch chart with ToD-period overlay, per-year tables

The legacy `optuna` engine instead writes `model_output.png` (4-panel figure)
and `day250_dispatch.png` (BESS dispatch diagnostic).

## Verification

The Pyomo layer checks itself at four levels (design §11): parameter
construction, constraint feasibility, post-solve physical invariants
(`verify.py`), and end-to-end economics against the FinanceEngine. Invariant
failures raise rather than warn — a solve that violates energy balance or SOC
continuity is a bug, not a result.

## Current scope notes

- **Solar + wind + BESS are all live.** Reference optimum is roughly
  `S=71.8 MW`, `W=95.3 MW`, `P=56.4 MW`, 19 BESS containers, LP objective
  ~1019 Cr. The slow tests pin these, so they will fail loudly if a config edit
  moves the optimum — that is the intent.
- **The `wind_capacity_mw.max` bound is load-bearing.** Setting it to 0 forces a
  solar-only portfolio that loses diurnal complementarity with the load: the
  optimizer compensates with ~3x the solar and ~8x the BESS and still lands at
  roughly half the client savings. Change it deliberately, not incidentally.
- **BESS augmentation is not on this branch.** The cohort-based 25-year
  augmentation engine is preserved at tag `archive/augmentation-v3` and is
  planned to be rebuilt on top of the single-mode LP.
