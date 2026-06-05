# Pyomo Migration — Session Handoff & Implementation Log

**Purpose:** Full handoff so this work can resume on another machine. Read this
top-to-bottom before continuing. Authoritative spec is `PYOMO_MIGRATION_DESIGN.md`
(do not redesign; raise spec issues rather than coding around them).

**Date of handoff:** 2026-06-05
**Branch:** `feature/pyomo-migration` (off `main`). Never commit to `main`.
**Repo location:** OneDrive-synced folder, so committed history + working tree
sync to the home PC. Python 3.13.7, Windows, pytest-9.

---

## 1. Goal & ground rules

Replace the Optuna `SolverEngine` with a deterministic **Pyomo MILP** that
maximises client savings NPV. Model is an LP with exactly **one integer**
(BESS container count `nb`).

Hard rules (from the task brief):
- **Do NOT modify anything under `src/hybrid_plant/` except, as the very last
  step (Step 8), a single switch in `run_model.py`.** The existing engines
  (PlantEngine, Year1Engine, FinanceEngine, LCOEModel, LandedTariffModel,
  SavingsModel, EnergyProjection) are the **validation oracle** — keep intact.
- New code lives in `src/hybrid_plant/optimise/` (the §9 layout).
- Keep the Optuna `SolverEngine` intact behind a flag (don't delete).
- **Nothing hardcoded** — every parameter read from YAML/CSV via the existing
  `config_loader.FullConfig` / `data_loader`.
- Commit at the end of each completed, test-passing phase. Local commits only;
  **do not push or open a PR** unless explicitly asked.
- Solver: HiGHS via Pyomo `appsi_highs` (in-memory). CBC kept as cross-check
  fallback (Step 7). `pyomo>=6.7`, `highspy>=1.7` added to `pyproject.toml`
  (installed; pyomo 6.10.1).
- After EACH step: STOP, run tests, commit, report, and **wait for user
  confirmation** before the next step.

---

## 2. Commit state (git log on feature/pyomo-migration)

```
c580a4d Step 4: Unfix sizing -> first true optimisation + Layer 3 validation
303646b Reformulate aux from financial cost to energy-level netting (D8 update)
dae844b Step 3: savings_npv objective + report.py + Layer 2 finance parity
fa53c2b Step 2: Single-year LP with fixed sizing + Layer 1 physics validation
82fdc74 Step 1: Scaffold optimise package and implement params.py
5bfed12 (Step 4's parent — pre-existing main work)
```

**Step 5 is IMPLEMENTED but NOT yet committed** and its **slow tests were never
successfully run to completion** (killed mid-solve twice — see §7). A WIP commit
of the Step 5 code may have been made at handoff (check `git log`); if so it is
clearly labelled WIP and the slow-test validation is still PENDING.

---

## 3. What each completed step delivered

### Step 1 (82fdc74) — Scaffold + params
- Created `src/hybrid_plant/optimise/` package with the full §9 module skeleton.
- `config.py`: `OptModelConfig` frozen dataclass — `horizon` ("single"/"full"),
  `solver_name` ("appsi_highs"/"cbc"), `scale_money=1e-7` (INR→Cr),
  `scale_energy=1e-3` (MWh→GWh; these are **unit constants from design §6-8**,
  not yet applied in computation — see §10), `augmentation=False`.
- `params.py`: `OptParams` frozen dataclass + `build_params(config, data)`.
  Carries every model parameter + precomputed constants:
  `df[]`, `A_N`, `A_n`, `emi_factor`, `phi (Φ)`, `G_solar_om/G_wind_om/G_land`
  (escalated-OPEX discount sums), `G_bess_om/G_solar_trans/G_wind_trans/
  G_insurance` (= A_N), `D_s/D_w/D_b` (discounted-degradation sums),
  `d_s/d_w/d_b[]` (per-year operating values via `data_loader.operating_value`).
- **83 reconciliation tests** in `tests/optimise/test_params.py` — every field
  vs FullConfig; `phi` cross-checked vs LCOEModel NPV; `G_solar_om` vs OpexModel
  NPV; degradation arrays vs `operating_value` per year. All pass.

Key config facts confirmed by reading the YAMLs:
- C-rate: `solver.yaml` exposes `bess_charge_c_rate`/`bess_discharge_c_rate`.
  params reads `fixed_value` if present else `max` (currently `max=1.0`).
- `tenure = 25 = project_life` in current config, so `A_n == A_N`.
- aux: `bess.yaml` `auxiliary_consumption_mwh_per_day: 0.300` → `aux_pc = 0.300/24`.
- loss factor `lf ≈ 0.86` from `GridInterface` (regulatory losses product).

### Step 2 (fa53c2b) — Single-year LP, fixed sizing, Layer 1 physics
- `sets.py` `add_sets`; `variables.py` (sizing S/W/P/nb + dispatch
  sd/wd/chg/dis/soc/ddraw, all NonNegative; `nb` integer; `E_b = nb·cs` Expression).
- Constraints C1–C10 across `allocation.py` (C1,C2), `balance.py` (C3; C4 via
  domain), `ppa.py` (C5), `soc.py` (C6 dynamics with soc[-1]=0, C8 cap,
  C9/C10 C-rate power caps).
- `build.py` `build_single_year_model(..., fixed_sizing, objective)`.
- `solve.py` HiGHS `appsi_highs` driver + `extract_dispatch`.
- **Layer 1 (19 tests)**: C1–C5/C9/C10 hold on imposed PlantEngine dispatch;
  D8 aux divergence quantified; LP feasibility. Documented finding: with the
  Step-2 placeholder "maximize RE delivery" objective the LP delivers ~10% more
  than PlantEngine **only because** PlantEngine was in evening-window mode; in
  RTC mode the gap is **+0.28%** (within the 0.5% Layer-1 tolerance).

### Step 3 (dae844b) — savings_npv objective + report.py + Layer 2
- `objective.py` `add_savings_npv_objective` — design §2.4 single-year factored
  form: `lf·1000·(D_s·Σnet_tod·sd + D_w·Σnet_tod·wd + D_b·Σnet_tod·η_d·dis)
  − total_capex·Φ − NPV(opex) − cap_rate·P·12·A_N − NPV(aux)`, where
  `net_tod = tod − wheel − tax`. Also kept `add_maximize_re_delivery_objective`
  (Step-2 placeholder).
- `report.py` `compute_report(sizing, config, data, fast_mode)` — delegates to
  the existing FinanceEngine so LCOE/landed tariff are **outputs only** (D2).
- **Layer 2 (18 tests)**: every cost component matches FinanceEngine ≤0.1%;
  **Finding A** (cost-plus cancellation) verified to 9e-15 with flat tariff;
  report.py LCOE/savings/landed match FinanceEngine to 1e-6.

### Aux reformulation (303646b) — USER-REQUESTED deviation from naive D8
The original D8 modelled aux as a financial cost at the **full DISCOM tariff**
(`aux_pc·Σtod·1000`), which wrongly included wheeling and ignored the loss
factor. The user directed: **aux is consumed at the plant busbar, before grid
export — net it at the ENERGY level, with no wheeling/tax and no loss on the aux
energy itself.** Implemented:
- `balance.py` C3 → `lf·(sd + wd + η_d·dis − nb·aux_pc) + ddraw = load`
  (aux subtracted *inside* the lf bracket).
- `objective.py` aux term → `nb · lf·aux_pc·Σnet_tod·1000 · A_N` (priced at
  net_tod = tod−wheel−tax, scaled by lf).
- NPV(aux) dropped 100.7 Cr → 80.1 Cr. Apples-to-apples LP vs PlantEngine RTC:
  meter +0.14%, savings_npv −0.72% — consistent.

### Step 4 (c580a4d) — Free single-year MILP + Layer 3
- `build.py` free sizing (unfix S/W/P/nb). Solves in **~34 s**.
- **Optimum:** Solar 71.76 MW, Wind 95.33 MW, PPA 56.39 MW, BESS 19 containers
  (95.3 MWh) → **savings_npv 1019 Cr**, serving 85.8% of load.
- **Layer 3 (13 tests)**: global-optimum (free 1019 ≥ benchmark-fixed 176, same
  objective, +844 Cr right-sizing); oracle (FinanceEngine RTC full re-sim
  = 1001 Cr, within **1.79%**); LCOE 3.94 INR/kWh; difference report.
- **CRITICAL FINDING — Optuna degeneracy:** the genuine 1500-trial Optuna run
  (seed 42, current config) found **ZERO feasible solutions** — it samples
  oversized plants (950 MW wind, 599 containers) with negative savings, all
  failing the `minimum_savings_npv ≥ 0` gate; its flat-tariff objective is
  ToD-blind (design Finding B). Reference saved to
  `tests/optimise/optuna_reference.json`. So Optuna is **not** a useful numeric
  benchmark; the FinanceEngine oracle is the real Layer-3 check.
- **C-rate divergence (documented):** `solver.yaml` marks C-rates `scope:current`
  so **Optuna optimises them**, while Pyomo **fixes C-rate=1.0 per D3**. Benchmark
  sizing uses 1.0, so Layer 3B stays clean. Flag for cutover.

---

## 4. Step 5 (IN PROGRESS) — Full 25-year mode

### Design choices (confirmed with user via AskUserQuestion)
1. **Unify via `TimeContext`** (chosen over parallel builders).
2. **Mark slow, run once** (chosen over running the heavy solve every test).

### Core idea — flat time index unifies both horizons
Both modes use one flat dispatch index `H = 0…n_steps−1`
(single: 8760; full: 8760×25 = 219,000). With a flat index the SOC recursion
`soc[t] = soc[t-1] + η_c·chg[t] − dis[t]` (soc[-1]=0) is identical in both modes,
so the **year-to-year SOC carryover (§3.4) is automatic** — no special-casing.

`TimeContext` (in `sets.py`) carries the per-timestep things that differ:
- `hour_of[t]` → 0…8759 (indexes the 8760-length cuf/load/tod arrays)
- `year_of[t]` → 1…25
- `deg_s/deg_w/deg_b[t]` → degradation factor on capacity bounds.
  **Single mode = all 1.0** (degradation enters the *objective* via D_s/D_w/D_b);
  **full mode = d_*[year(t)]** (degradation enters the *bounds*, §3.5).
- `disc[t]` → `df[year(t)]` (full-mode objective discount; NaN/unused in single).

### Files changed in Step 5 (all in `src/hybrid_plant/optimise/`)
- **`sets.py`**: added `TimeContext` dataclass + `build_time_context(params,
  horizon)`; `add_sets(model, n_steps)` (dropped the unused Y RangeSet).
- **`constraints/allocation.py|balance.py|ppa.py|soc.py`**: each builder now
  takes `tc: TimeContext`. Degradation folded into precomputed float coefficients
  (`coef = deg_*[t]·cuf_*[hour_of[t]]`), load indexed by `hour_of[t]`, C8/C9/C10
  scale `E_b` by `deg_b[t]`. **Single mode is numerically unchanged** (×1.0 exact,
  identity hour map) — verified: all 38 single-mode tests still pass.
- **`objective.py`**: added `add_savings_npv_objective_full(model, params, tc)` —
  revenue `Σ_t disc[t]·lf·1000·net_tod[hour(t)]·(sd+wd+η_d·dis)`, degradation in
  the bounds, cost side identical to single, aux discounts at A_N. Single-mode
  objective left untouched.
- **`build.py`**: `build_model(opt_cfg, ...)` toggle on `opt_cfg.horizon`;
  `_assemble_common` (shared sets+vars+C1–C10); `build_single_year_model` and
  `build_full_model`.
- **`solve.py`**: `extract_dispatch` infers length from `model.H`. Added
  **`solve_relax_and_snap(model, opt_cfg)`** — see §7.

### Tests written: `tests/optimise/test_full_horizon.py` (NEW, untracked)
- **Fast (14 tests, ALL PASS):** TimeContext array construction; horizon toggle;
  a **mini 2-year × 24-hour** model built from a hand-made TimeContext that
  verifies SOC carryover across the year boundary, degraded capacity bounds,
  hour-of-year load in C3, and objective self-consistency (numpy vs Pyomo) —
  all solved instantly.
- **Slow (`@pytest.mark.slow`, NOT yet validated):** free full solve via
  relax-and-snap; SOC carryover across all 24 boundaries; degraded per-year
  bounds; C3 balance; objective self-consistency; FinanceEngine full-mode oracle
  (≤5%); single-vs-full degradation relationship; relax-and-snap gap <1%.

---

## 5. KEY EMPIRICAL FINDINGS (carry these forward)

1. **Full mode is correct, validated by oracle.** At the benchmark sizing,
   Pyomo full-mode = **278 Cr** vs FinanceEngine full 25-yr re-sim = **273 Cr**
   (**2.1%**). 
2. **Single-vs-full "degradation error" (design §10 Step 5) — understood:**
   single mode scales year-1 *delivery* by the generation-degradation factor
   `d[y]`, but delivery is buffered by the **non-degrading load/PPA caps** — so
   for an oversized, heavily-curtailed plant single mode **underestimates** badly
   (benchmark: 176 vs 273 Cr, ~35% low), while at the **right-sized optimum**
   (low curtailment) it is near-exact (~2%). Both modes valid per D6: full =
   high fidelity, single = fast approximation accurate near the optimum.
3. **Runtime is the open problem.** Full model = **1,314,004 vars,
   1,752,000 constraints**. Build ~17 s. A single fixed-sizing LP solves in
   **~505–577 s** — dual simplex, IPM-no-crossover, IPM-crossover all ~the same,
   so it is **structural model size, not solver method or conditioning**. The
   **free-sizing** LP is ~1.4× slower per solve (larger basis; S/W/P free). So
   relax-and-snap (2 free LPs) ran **>25 min of CPU** and was killed twice.

---

## 6. RUNTIME / SOLVER STRATEGY decided

- Straight MILP branch-and-bound on the free full model is too slow / unbounded
  in practice → switched to **relax-and-snap** (design §5):
  1. relax `nb`→continuous, solve LP (upper bound);
  2. snap `nb`→`round`, fix it; 3. re-solve dispatch LP (feasible).
  Reports `snap_gap_frac` for auditable near-optimality. Implemented in
  `solve.py::solve_relax_and_snap`. The full-mode test fixture uses it.
- **Even relax-and-snap is ~2×~700 s ≈ 24 min CPU** for the free model and was
  not completed. **OPEN: full-mode free solve runtime.** Options to pursue
  (Step 7 hardening, or sooner if needed):
  - Apply the §6-8 **unit scaling** (`scale_money`, `scale_energy` already on
    OptModelConfig but NOT wired into build/objective). NOTE: the *constraint*
    matrix is already well-conditioned (entries ~0.01–50); scaling mainly helps
    the objective range, so it may not fix the LP solve time — but untested.
  - Run with **`tee=True`** so HiGHS streams iteration progress (this run was
    blind — `solve()` uses `tee=False`, so we had no ETA).
  - Reduce columns (e.g. eliminate `ddraw` via a C3 inequality) — modest, since
    rows (1.75M) dominate simplex time, not columns.
  - Temporal aggregation / representative periods — changes fidelity, NOT in
    current scope without spec approval.
  - Accept single-mode (34 s) as the production optimizer; full mode as periodic
    high-fidelity validation only.

---

## 7. CURRENT UNCOMMITTED WORKING-TREE STATE (read carefully)

`git status --short` at handoff showed:
```
 M configs/bess.yaml                              <-- NOT mine; see below
 M data/load_profile_8760.csv                     <-- pre-existing, NOT mine
 M graphify-out/GRAPH_REPORT.md                   <-- pre-existing, NOT mine
 M src/hybrid_plant/optimise/build.py             <-- Step 5
 M src/hybrid_plant/optimise/constraints/allocation.py  <-- Step 5
 M src/hybrid_plant/optimise/constraints/balance.py     <-- Step 5
 M src/hybrid_plant/optimise/constraints/ppa.py         <-- Step 5
 M src/hybrid_plant/optimise/constraints/soc.py         <-- Step 5
 M src/hybrid_plant/optimise/objective.py         <-- Step 5
 M src/hybrid_plant/optimise/sets.py              <-- Step 5
 M src/hybrid_plant/optimise/solve.py             <-- Step 5
?? MODEL_SUMMARY.md                               <-- pre-existing untracked
?? PYOMO_MIGRATION_DESIGN.md                      <-- the spec (untracked)
?? tests/optimise/test_full_horizon.py            <-- Step 5 tests
?? PYOMO_MIGRATION_HANDOFF.md                     <-- this file
```

### ⚠️ `configs/bess.yaml` anomaly
It changed `discharge_hours: [18,19,20,21,22,23] → []` and
`charge_first: true → false` (i.e. switched PlantEngine to **normal RTC**).
**I did not edit this file** — my probes only `copy.deepcopy(config.bess)` and
modified the copy in memory. It was unmodified at Step 1 and appeared modified
during the session. Likely an external/manual edit or sync artifact.
- It does **NOT** affect the Pyomo model (params reads only efficiency/
  container/aux from bess.yaml, not the dispatch block).
- It **does** change the default config used by PlantEngine/FinanceEngine and the
  conftest benchmark (now RTC instead of evening-window).
- **DECIDE before committing:** revert it (`git checkout configs/bess.yaml`) to
  restore evening-window, or keep RTC intentionally. It was left **unstaged**.

---

## 8. HOW TO RESUME STEP 5 (do this first on the home PC)

1. `cd` into the repo; confirm branch: `git branch --show-current` →
   `feature/pyomo-migration`. If Step 5 code was WIP-committed, it's already
   there; otherwise the working-tree changes from §7 should have synced.
2. Sanity: `pip show pyomo highspy` (reinstall if missing:
   `pip install pyomo highspy`). Confirm `python -c "import highspy"`.
3. **Fast regression (should be quick, must pass):**
   ```
   python -m pytest tests/optimise/ -m "not slow" -q
   ```
   Expect: params (83) + dispatch single (Part A) + finance (Layer2A/B/D) +
   full_horizon fast (14) all green. ~30-60 s.
4. **Decide the bess.yaml question** (§7) before running oracle-based tests, since
   the oracle tests `deepcopy` to RTC anyway but conftest benchmark uses the file.
5. **Run the slow full-horizon validation ONCE** (will take ~25–40 min; keep the
   machine awake — see §11). Recommend enabling solver streaming first by
   temporarily passing `tee=True`, or just run blind:
   ```
   python -m pytest tests/optimise/test_full_horizon.py -m slow -v -s
   ```
   Expected (from the partial/probe runs): free optimum near the single-mode
   optimum but degradation-adjusted; relax-and-snap gap <1%; oracle agreement
   within 5%; single-vs-full documents the degradation error. **Do not weaken a
   tolerance to pass — diagnose if off.**
6. If the slow solve is impractically long, pursue §6 options (start with
   `tee=True` for visibility, then consider eliminating `ddraw`).
7. When slow tests pass: run the FULL suite, then **commit Step 5** with a clear
   message, and report to the user. Then await confirmation for Step 6.

---

## 9. PENDING STEPS (6–8, from design §10)

- **Step 6** — Optional `solver.yaml` constraint toggles, each individually
  switchable with a regression test: `minimum_savings_npv` (report-only default),
  `plant_cuf`, `minimum_bess_capacity`, `minimum_bess_discharge`,
  `re_penetration`, `hourly_re_penetration_penalty`. Goes in
  `constraints/optional.py` (currently a stub). `ddraw`/meter delivery feed the
  RE-penetration constraints.
- **Step 7** — Solver hardening: HiGHS↔CBC agreement (single-year); `verify.py`
  post-solve invariants (no simultaneous charge+discharge confirming D7;
  0≤SOC≤E_b; per-hour energy conservation; export≤PPA; ddraw≥0); **runtime
  profiling vs §12 + the full-mode runtime work from §6**; scaling tuning.
- **Step 8** — Cutover: add the `run_model.py` switch (Optuna behind a flag);
  greenfield-equivalence augmentation test (existing=0 reproduces Phase-1).

Module stubs still inert: `verify.py`, `augmentation.py`, `constraints/optional.py`.

---

## 10. DESIGN DECISIONS LOCKED (do not relitigate)

D1 hourly ToD valuation · D2 LCOE/landed tariff reporting-only · D3 C-rate fixed
param · D4 energy-only BESS cost · D5 charge=solar_only (Phase 1) · D6 dual
horizon toggle · D7 no charge/discharge binary (efficiency enforces exclusivity;
verify post-solve) · **D8 aux = grid-fed, now ENERGY-LEVEL netted per user (see
§3)** · D9 commissioning-zero SOC, continuous carryover, free terminal ·
D10 PPA capacity is a variable (clamp to fix) · D11 integer nb is the only
integrality · D12 Year-1 = fresh = 1.0 operating value · D13 banking/export
future scope (C3/SOC structured to admit an export var + mutual-exclusion binary).

Unit scaling (scale_money=1e-7, scale_energy=1e-3) are on `OptModelConfig` but
**not yet applied in computation** — they were scaffolded for §6-8 and would be
inverted at report time. Revisit in Step 7.

---

## 11. ENVIRONMENT / GOTCHAS

- **Windows + PowerShell.** Bash tool mangles `$_`; use the PowerShell tool for
  process management. Unicode `−`/`×` in `print()` crash on cp1252 — use ASCII
  `-`/`x` in test prints.
- **Sleep settings were CHANGED on the work machine** to keep the long solve
  alive: AC and DC `standby-timeout` set to 0, AC `hibernate-timeout` set to 0.
  **Originals were 45 min (both AC & DC standby); DC hibernate was already 0.**
  RESTORE on the work machine (these are machine-local, won't sync):
  ```
  powercfg /change standby-timeout-ac 45
  powercfg /change standby-timeout-dc 45
  powercfg /change hibernate-timeout-ac 0
  ```
- **The big solve is NOT checkpointed** — a sleep/power-loss mid-solve loses it
  (a benign S3 suspend survived once, but don't rely on it). Keep awake; lid-open.
- **No solver progress visibility** this session because `solve()` ran HiGHS with
  `tee=False`. For long runs, stream with `tee=True`.
- Reference data: `tests/optimise/optuna_reference.json` (the zero-feasible
  Optuna best, seed 42, current config).

---

## 12. KEY NUMBERS QUICK-REFERENCE

| Quantity | Value |
|---|---|
| Single-year free MILP solve time | ~34 s |
| Single-year optimum | S 71.76 / W 95.33 / PPA 56.39 MW / 19 cont (95.3 MWh) |
| Single-year optimum savings_npv | 1019 Cr (oracle RTC full: 1001 Cr, 1.79%) |
| Benchmark sizing (conftest SOLAR_WIND) | S 190.45 / W 116.13 / PPA 120.63 / 120 cont |
| Benchmark savings_npv: single / full / oracle | 176 / 278 / 273 Cr |
| Full model size | 1,314,004 vars · 1,752,000 constraints |
| Full LP solve (fixed sizing) | ~505–577 s (any method) |
| Full LP solve (free sizing) | ~700 s each (×2 for relax-and-snap) |
| NPV(aux) after energy-level reform | 80.1 Cr (was 100.7 Cr at full tariff) |
| Optuna 1500-trial feasible count | 0 (degenerate) |
| Tests passing | params 83, single-dispatch 19, finance 18, layer3 13, full-fast 14 |
