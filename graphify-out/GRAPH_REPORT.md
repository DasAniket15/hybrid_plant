# Graph Report - C:\Users\Aniket4.Das\OneDrive - Reliance Corporate IT Park Limited\Documents\Models\Excel Solutioning - GitHub\hybrid_plant  (2026-06-04)

## Corpus Check
- 27 files · ~51,789 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 300 nodes · 598 edges · 18 communities detected
- Extraction: 46% EXTRACTED · 54% INFERRED · 0% AMBIGUOUS · INFERRED: 321 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `FullConfig` - 110 edges
2. `PlantEngine` - 35 edges
3. `FinanceEngine` - 34 edges
4. `Year1Engine` - 32 edges
5. `GridInterface` - 22 edges
6. `LCOEModel` - 22 edges
7. `CapexModel` - 20 edges
8. `SolverEngine` - 19 edges
9. `OpexModel` - 18 edges
10. `TestFinanceEngineIntegration` - 16 edges

## Surprising Connections (you probably didn't know these)
- `LCOEModel` --implements--> `NPV-based LCOE Framework`  [INFERRED]
  C:\Users\Aniket4.Das\OneDrive - Reliance Corporate IT Park Limited\Documents\Models\Excel Solutioning - GitHub\hybrid_plant\src\hybrid_plant\finance\lcoe_model.py → README.md
- `SavingsModel` --implements--> `100% DISCOM Baseline Benchmark`  [INFERRED]
  C:\Users\Aniket4.Das\OneDrive - Reliance Corporate IT Park Limited\Documents\Models\Excel Solutioning - GitHub\hybrid_plant\src\hybrid_plant\finance\savings_model.py → README.md
- `FullConfig` --uses--> `Resolve a project-relative path string to an absolute ``Path``.`  [INFERRED]
  src\hybrid_plant\config_loader.py → src\hybrid_plant\data_loader.py
- `Resolve a project-relative path string to an absolute ``Path``.` --uses--> `FullConfig`  [INFERRED]
  C:\Users\Aniket4.Das\OneDrive - Reliance Corporate IT Park Limited\Documents\Models\Excel Solutioning - GitHub\hybrid_plant\src\hybrid_plant\data_loader.py → src\hybrid_plant\config_loader.py
- `Read the first column of a header-less CSV as a float64 array.      Parameters` --uses--> `FullConfig`  [INFERRED]
  C:\Users\Aniket4.Das\OneDrive - Reliance Corporate IT Park Limited\Documents\Models\Excel Solutioning - GitHub\hybrid_plant\src\hybrid_plant\data_loader.py → src\hybrid_plant\config_loader.py

## Hyperedges (group relationships)
- **Year-1 Plant-Grid-Meter Pipeline** — plant_engine_PlantEngine, grid_interface_GridInterface, meter_layer_MeterLayer [EXTRACTED 1.00]
- **ToD-Aware BESS Dispatch System** — plant_engine_tod_dispatch, plant_engine_soc_reservations, plant_engine_dispatch_mask [EXTRACTED 0.95]
- **Finance Pipeline Orchestration** — finance_engine_FinanceEngine, lcoe_model_LCOEModel, landed_tariff_model_LandedTariffModel, savings_model_SavingsModel, opex_model_OpexModel [EXTRACTED 1.00]
- **Solver Optimisation Loop** — solver_engine_SolverEngine, solver_engine_tpe, solver_engine_evaluate, finance_engine_evaluate [EXTRACTED 1.00]
- **Config and Data Bootstrap** — config_loader_load_config, data_loader_load_timeseries_data, paths_find_project_root [EXTRACTED 0.95]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (35): _load_curve(), energy_projection.py ──────────────────── Projects annual energy delivery acro, Load a degradation CSV into a {year: value} dict., Return annual energy totals across the 25-year project lifetime.          Para, Load a degradation CSV into a {year: value} dict., Fast path: scale Year-1 scalar totals by annual degradation factors.         Ru, Full path: re-simulate each of the 25 project years with that year's         de, Runs a per-year full plant simulation to produce an accurate 25-year     energy (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (26): CapexModel, Calculates total project CAPEX and a per-component breakdown.      Parameters, EnergyProjection, FinanceEngine, finance_engine.py ───────────────── Top-level finance pipeline orchestrator., Orchestrates the full LCOE-based finance pipeline for a given     plant configu, Run the full finance pipeline for a given plant configuration.          Parame, Run the full finance pipeline for a given plant configuration.          Parame (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (22): energy_engine(), Shared Year1Engine instance., First year where cumulative savings turns positive., solver_engine.py ──────────────── Optimisation layer — wraps Optuna TPE to sea, Map an Optuna trial to a complete parameter set., Run energy + finance engines for a given parameter set., Return True if all configured constraints are satisfied., Execute the optimisation study.          Parameters         ---------- (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (31): capex_model.py ────────────── Computes project CAPEX broken down by component., Parameters         ----------         solar_capacity_mw        : AC solar inst, FullConfig, Immutable bundle of all project configuration namespaces., Read the first column of a header-less CSV as a float64 array.      Parameters, Raise if *array* does not contain exactly 8760 values., Load all time-series profiles and degradation curves defined in     ``project.y, landed_tariff_model.py ────────────────────── Computes the annual landed tarif (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (8): Scale busbar export by the loss factor to produce meter delivery.          Par, Compute hourly DISCOM shortfall.          Parameters         ----------, test_energy.py ────────────── Unit and integration tests for the energy simula, TestGridInterface, TestMeterLayer, TestPlantEngineSolarOnly, TestYear1EngineSolarOnly, TestYear1EngineSolarWind

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (10): CUF formula: busbar_mwh / (capacity_mw × 8760) × 100      Plant CUF uses PPA c, Solar CUF for an Indian site should be in [15%, 35%]., Plant CUF derived from simulation busbar totals stays in [20%, 80%]., TestCUF, _config_no_penalty(), energy_engine(), finance(), finance_engine() (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (14): load_config(), _load_yaml(), config_loader.py ──────────────── Loads all YAML configuration files and bundl, Load a single YAML file and return its contents as a dict., Run lightweight sanity checks on a freshly loaded config.      Raises     ---, Discover the project root, load all YAML configs, validate, and return     a ``, _validate(), config() (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (14): _load_csv_column(), load_timeseries_data(), operating_value(), data_loader.py ────────────── Loads all time-series CSVs (8760-hour profiles), Load all time-series profiles and degradation curves defined in     ``project.y, Resolve a project-relative path string to an absolute ``Path``., Resolve a project-relative path string to an absolute ``Path``., Read the first column of a header-less CSV as a float64 array.      Parameters (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (14): compute_cuf(), compute_payback_year(), cr(), pct(), print_section1(), print_section2(), print_section3(), print_section4() (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (5): Parameters         ----------         total_capex                   : Total pr, Fixed-EMI amortising loan schedule.          Returns two lists of length ``pro, npv(), finance/_utils.py ───────────────── Shared financial utility functions used ac, Excel-style NPV: series[0] is Year 1, discounted at t = 1.          NPV = Σ se

### Community 10 - "Community 10"
Cohesion: 0.5
Nodes (3): _period_weighted_avg(), savings_model.py ──────────────── Computes client electricity cost savings ver, _weighted_discom_tariff()

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): constants.py ──────────── Shared physical and financial unit-conversion consta

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Hybrid Plant README

## Knowledge Gaps
- **45 isolated node(s):** `config_loader.py ──────────────── Loads all YAML configuration files and bundl`, `Load a single YAML file and return its contents as a dict.`, `Immutable bundle of all project configuration namespaces.`, `Run lightweight sanity checks on a freshly loaded config.      Raises     ---`, `Discover the project root, load all YAML configs, validate, and return     a ``` (+40 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 11`** (2 nodes): `constants.py`, `constants.py ──────────── Shared physical and financial unit-conversion consta`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `WACC = (D/V × Rd × (1 − Tc)) + (E/V × Re)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Hybrid Plant README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FullConfig` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.531) - this node is a cross-community bridge._
- **Why does `FinanceEngine` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `result()` connect `Community 0` to `Community 4`, `Community 5`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 107 inferred relationships involving `FullConfig` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `Resolve a project-relative path string to an absolute ``Path``.`) actually correct?**
  _`FullConfig` has 107 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `PlantEngine` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `run_model.py ──────────── Master entry point for the hybrid RE plant model.`) actually correct?**
  _`PlantEngine` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `FinanceEngine` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `run_model.py ──────────── Master entry point for the hybrid RE plant model.`) actually correct?**
  _`FinanceEngine` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `Year1Engine` (e.g. with `smoke_test.py ───────────── Self-contained smoke test using only stdlib + nump` and `run_model.py ──────────── Master entry point for the hybrid RE plant model.`) actually correct?**
  _`Year1Engine` has 27 INFERRED edges - model-reasoned connections that need verification._