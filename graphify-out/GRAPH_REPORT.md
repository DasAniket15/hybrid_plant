# Graph Report - .  (2026-04-21)

## Corpus Check
- Corpus is ~14,937 words - fits in a single context window. You may not need a graph.

## Summary
- 274 nodes · 499 edges · 26 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 225 edges (avg confidence: 0.57)
- Token cost: 9,800 input · 2,100 output

## Community Hubs (Navigation)
- [[_COMMUNITY_CAPEX Model|CAPEX Model]]
- [[_COMMUNITY_Config & Dataclasses|Config & Dataclasses]]
- [[_COMMUNITY_Energy Simulation Core|Energy Simulation Core]]
- [[_COMMUNITY_Solver & Entry Point|Solver & Entry Point]]
- [[_COMMUNITY_Grid & Meter Layer|Grid & Meter Layer]]
- [[_COMMUNITY_LCOE Financial Model|LCOE Financial Model]]
- [[_COMMUNITY_Energy Projection|Energy Projection]]
- [[_COMMUNITY_Financial Utilities|Financial Utilities]]
- [[_COMMUNITY_Solver Validation|Solver Validation]]
- [[_COMMUNITY_Data Loading|Data Loading]]
- [[_COMMUNITY_Config Loading|Config Loading]]
- [[_COMMUNITY_Unit Constants|Unit Constants]]
- [[_COMMUNITY_BESS Dispatch Control|BESS Dispatch Control]]
- [[_COMMUNITY_Projection Modes|Projection Modes]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Finance Sub-models|Finance Sub-models]]
- [[_COMMUNITY_Test Suite Energy|Test Suite Energy]]
- [[_COMMUNITY_Test Suite Finance|Test Suite Finance]]
- [[_COMMUNITY_Test Suite Solver|Test Suite Solver]]
- [[_COMMUNITY_OPEX Model|OPEX Model]]
- [[_COMMUNITY_Savings Model|Savings Model]]
- [[_COMMUNITY_Landed Tariff Model|Landed Tariff Model]]
- [[_COMMUNITY_README & Docs|README & Docs]]
- [[_COMMUNITY_Solver Internals|Solver Internals]]
- [[_COMMUNITY_Benchmark Params|Benchmark Params]]
- [[_COMMUNITY_Misc Utilities|Misc Utilities]]

## God Nodes (most connected - your core abstractions)
1. `FullConfig` - 70 edges
2. `PlantEngine` - 24 edges
3. `FinanceEngine` - 24 edges
4. `LCOEModel` - 23 edges
5. `Year1Engine` - 20 edges
6. `CapexModel` - 19 edges
7. `SolverEngine` - 18 edges
8. `GridInterface` - 17 edges
9. `OpexModel` - 17 edges
10. `EnergyProjection` - 13 edges

## Surprising Connections (you probably didn't know these)
- `LCOEModel` --implements--> `NPV-based LCOE Framework`  [INFERRED]
  src\hybrid_plant\finance\lcoe_model.py → README.md
- `SavingsModel` --implements--> `100% DISCOM Baseline Benchmark`  [INFERRED]
  src\hybrid_plant\finance\savings_model.py → README.md
- `Resolve a project-relative path string to an absolute ``Path``.` --uses--> `FullConfig`  [INFERRED]
  src\hybrid_plant\data_loader.py → src\hybrid_plant\config_loader.py
- `Read the first column of a header-less CSV as a float64 array.      Parameters` --uses--> `FullConfig`  [INFERRED]
  src\hybrid_plant\data_loader.py → src\hybrid_plant\config_loader.py
- `Raise if *array* does not contain exactly 8760 values.` --uses--> `FullConfig`  [INFERRED]
  src\hybrid_plant\data_loader.py → src\hybrid_plant\config_loader.py

## Hyperedges (group relationships)
- **Year-1 Plant-Grid-Meter Pipeline** — plant_engine_PlantEngine, grid_interface_GridInterface, meter_layer_MeterLayer [EXTRACTED 1.00]
- **ToD-Aware BESS Dispatch System** — plant_engine_tod_dispatch, plant_engine_soc_reservations, plant_engine_dispatch_mask [EXTRACTED 0.95]
- **Finance Pipeline Orchestration** — finance_engine_FinanceEngine, lcoe_model_LCOEModel, landed_tariff_model_LandedTariffModel, savings_model_SavingsModel, opex_model_OpexModel [EXTRACTED 1.00]
- **Solver Optimisation Loop** — solver_engine_SolverEngine, solver_engine_tpe, solver_engine_evaluate, finance_engine_evaluate [EXTRACTED 1.00]
- **Config and Data Bootstrap** — config_loader_load_config, data_loader_load_timeseries_data, paths_find_project_root [EXTRACTED 0.95]

## Communities

### Community 0 - "CAPEX Model"
Cohesion: 0.07
Nodes (16): CapexModel, capex_model.py ────────────── Computes project CAPEX broken down by component., Calculates total project CAPEX and a per-component breakdown.      Parameters, Parameters         ----------         solar_capacity_mw        : AC solar inst, OpexModel, opex_model.py ───────────── Projects annual OPEX across the full project lifet, Computes a 25-year annual OPEX projection with per-component detail.      Para, Parameters         ----------         solar_capacity_mw : AC solar installed c (+8 more)

### Community 1 - "Config & Dataclasses"
Cohesion: 0.08
Nodes (29): FullConfig, Immutable bundle of all project configuration namespaces., config(), data(), finance_engine(), conftest.py ─────────── Shared pytest fixtures for the hybrid_plant test suite, Fully loaded and validated FullConfig (loaded once per session)., Time-series + degradation curve data dict (loaded once per session). (+21 more)

### Community 2 - "Energy Simulation Core"
Cohesion: 0.1
Nodes (23): energy_engine(), Shared Year1Engine instance., Load a degradation CSV into a {year: value} dict., GridInterface, Blended HT/LT Loss Factor, grid_interface.py ───────────────── Computes the blended HT/LT grid loss facto, Translates plant-busbar export (pre-loss) to client-meter delivery     (post-lo, MeterLayer (+15 more)

### Community 3 - "Solver & Entry Point"
Cohesion: 0.09
Nodes (14): run_model.py ──────────── Master entry point for the hybrid RE plant model., Fast Mode (Scalar Energy Projection), solver_engine.py ──────────────── Optimisation layer — wraps Optuna TPE to sea, Map an Optuna trial to a complete parameter set., Run energy + finance engines for a given parameter set., Return True if savings_npv meets the configured minimum., Execute the optimisation study.          Parameters         ----------, Structured output of a completed optimisation run. (+6 more)

### Community 4 - "Grid & Meter Layer"
Cohesion: 0.07
Nodes (9): Scale busbar export by the loss factor to produce meter delivery.          Par, Compute hourly DISCOM shortfall.          Parameters         ----------, test_energy.py ────────────── Unit and integration tests for the energy simula, result(), TestGridInterface, TestMeterLayer, TestPlantEngineSolarOnly, TestYear1EngineSolarOnly (+1 more)

### Community 5 - "LCOE Financial Model"
Cohesion: 0.13
Nodes (10): LCOEModel, lcoe_model.py ───────────── NPV-based Levelised Cost of Energy (LCOE)., Parameters         ----------         total_capex                   : Total pr, Computes the project LCOE and supporting financial schedules.      WACC is com, WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re), Excel-style NPV: series[0] is Year 1, discounted at t = 1.              NPV =, Fixed-EMI amortising loan schedule.          Returns two lists of length ``pro, WACC Computation (+2 more)

### Community 6 - "Energy Projection"
Cohesion: 0.14
Nodes (11): EnergyProjection, _load_curve(), energy_projection.py ──────────────────── Projects annual energy delivery acro, Return annual energy totals across the 25-year project lifetime.          Para, Fast path: scale Year-1 scalar totals by annual degradation factors.         Ru, Full path: re-simulate each of the 25 project years with that year's         de, Runs a per-year full plant simulation to produce an accurate 25-year     energy, Run the full finance pipeline for a given plant configuration.          Parame (+3 more)

### Community 7 - "Financial Utilities"
Cohesion: 0.31
Nodes (12): compute_cuf(), compute_payback_year(), cr(), pct(), print_section1(), print_section2(), print_section3(), print_section4() (+4 more)

### Community 8 - "Solver Validation"
Cohesion: 0.35
Nodes (9): cr(), validate_solver.py ────────────────── Full 4-layer solver validation script., run_engines(), sep(), _sweep(), validate_benchmark(), validate_convergence(), validate_physical() (+1 more)

### Community 9 - "Data Loading"
Cohesion: 0.27
Nodes (9): _load_csv_column(), load_timeseries_data(), data_loader.py ────────────── Loads all time-series CSVs (8760-hour profiles), Resolve a project-relative path string to an absolute ``Path``., Read the first column of a header-less CSV as a float64 array.      Parameters, Raise if *array* does not contain exactly 8760 values., Load all time-series profiles and degradation curves defined in     ``project.y, _resolve() (+1 more)

### Community 10 - "Config Loading"
Cohesion: 0.32
Nodes (7): load_config(), _load_yaml(), config_loader.py ──────────────── Loads all YAML configuration files and bundl, Load a single YAML file and return its contents as a dict., Run lightweight sanity checks on a freshly loaded config.      Raises     ---, Discover the project root, load all YAML configs, validate, and return     a ``, _validate()

### Community 11 - "Unit Constants"
Cohesion: 1.0
Nodes (1): constants.py ──────────── Shared physical and financial unit-conversion consta

### Community 12 - "BESS Dispatch Control"
Cohesion: 1.0
Nodes (2): BESS Dispatch Mask, ToD-Aware BESS Dispatch Logic

### Community 13 - "Projection Modes"
Cohesion: 1.0
Nodes (2): Fast Scalar Projection Mode, Full Per-Year Resimulation Mode

### Community 14 - "Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Finance Sub-models"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Test Suite Energy"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Test Suite Finance"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Test Suite Solver"
Cohesion: 1.0
Nodes (1): SOC Reservation State Variables

### Community 19 - "OPEX Model"
Cohesion: 1.0
Nodes (1): NPV Helper (Excel-style)

### Community 20 - "Savings Model"
Cohesion: 1.0
Nodes (1): Debt Amortisation Schedule

### Community 21 - "Landed Tariff Model"
Cohesion: 1.0
Nodes (1): OPEX Escalation Logic

### Community 22 - "README & Docs"
Cohesion: 1.0
Nodes (1): Optuna TPE Sampler

### Community 23 - "Solver Internals"
Cohesion: 1.0
Nodes (1): Hybrid Plant README

### Community 24 - "Benchmark Params"
Cohesion: 1.0
Nodes (1): BENCHMARK Params (validate_solver)

### Community 25 - "Misc Utilities"
Cohesion: 1.0
Nodes (1): Unit Conversion Constants

## Knowledge Gaps
- **34 isolated node(s):** `config_loader.py ──────────────── Loads all YAML configuration files and bundl`, `Load a single YAML file and return its contents as a dict.`, `Immutable bundle of all project configuration namespaces.`, `Run lightweight sanity checks on a freshly loaded config.      Raises     ---`, `Discover the project root, load all YAML configs, validate, and return     a ``` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Unit Constants`** (2 nodes): `constants.py ──────────── Shared physical and financial unit-conversion consta`, `constants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BESS Dispatch Control`** (2 nodes): `BESS Dispatch Mask`, `ToD-Aware BESS Dispatch Logic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Projection Modes`** (2 nodes): `Fast Scalar Projection Mode`, `Full Per-Year Resimulation Mode`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Finance Sub-models`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Suite Energy`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Suite Finance`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Suite Solver`** (1 nodes): `SOC Reservation State Variables`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `OPEX Model`** (1 nodes): `NPV Helper (Excel-style)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Savings Model`** (1 nodes): `Debt Amortisation Schedule`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Landed Tariff Model`** (1 nodes): `OPEX Escalation Logic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `README & Docs`** (1 nodes): `Optuna TPE Sampler`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Solver Internals`** (1 nodes): `Hybrid Plant README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Benchmark Params`** (1 nodes): `BENCHMARK Params (validate_solver)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Utilities`** (1 nodes): `Unit Conversion Constants`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FullConfig` connect `Config & Dataclasses` to `CAPEX Model`, `Energy Simulation Core`, `Solver & Entry Point`, `Grid & Meter Layer`, `LCOE Financial Model`, `Energy Projection`, `Data Loading`, `Config Loading`?**
  _High betweenness centrality (0.441) - this node is a cross-community bridge._
- **Why does `FinanceEngine` connect `Config & Dataclasses` to `CAPEX Model`, `Energy Simulation Core`, `Solver & Entry Point`, `LCOE Financial Model`, `Energy Projection`, `Solver Validation`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `result()` connect `Grid & Meter Layer` to `Energy Simulation Core`, `Energy Projection`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 67 inferred relationships involving `FullConfig` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `Resolve a project-relative path string to an absolute ``Path``.`) actually correct?**
  _`FullConfig` has 67 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `PlantEngine` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `run_model.py ──────────── Master entry point for the hybrid RE plant model.`) actually correct?**
  _`PlantEngine` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `FinanceEngine` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `run_model.py ──────────── Master entry point for the hybrid RE plant model.`) actually correct?**
  _`FinanceEngine` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `LCOEModel` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `FinanceEngine`) actually correct?**
  _`LCOEModel` has 15 INFERRED edges - model-reasoned connections that need verification._