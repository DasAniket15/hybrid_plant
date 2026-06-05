# Pyomo Migration — Technical Design Document

**Project:** `hybrid_plant` — grid-connected C&I hybrid RE sizing (Solar + Wind + BESS + PPA), India
**Phase:** 1 — Replace the Optuna `SolverEngine` with a deterministic Pyomo optimisation model
**Objective:** Maximise client savings NPV (customer-savings-driven only)
**Status:** Design — approved for authoring; no production code until this document is approved
**Date:** 2026-06-05

---

## 0. Reading guide / decisions locked in review

This document is the contract for implementation. Every modelling choice below was settled in the preceding architecture review. The decisive ones:

| # | Decision | Consequence |
|---|----------|-------------|
| D1 | Objective values delivered RE and the DISCOM baseline at **hourly ToD rates** | Battery has a genuine economic reason to dispatch into peaks; still linear |
| D2 | **LCOE and landed tariff are reporting-only** (they cancel in NPV) | The "core non-linearities" disappear from the optimisation |
| D3 | **C-rate is a fixed user parameter** | Removes the `c_rate × E_b` bilinearity |
| D4 | **Energy-only BESS cost** | No power cost ⇒ C-rate has nothing to trade off (consistent with D3) |
| D5 | **Charge source = solar_only** (Phase 1), structured to become a variable later | One allocation constraint |
| D6 | **Dual horizon mode** with a toggle: single representative year ↔ full 25×8760 | One parameterised builder |
| D7 | **No charge/discharge binary** — mutual exclusion is enforced by round-trip efficiency in the objective; verified post-solve | Model stays a pure LP (+1 integer) |
| D8 | **Aux = fixed per installed container**, modelled as a grid-fed parasitic cost | Linear; robust against the commissioning-zero SOC start |
| D9 | **SOC: commissioning-zero**, no cyclic constraint. Single-year: `SOC[0]=0`, free end. Full: `SOC=0` at commissioning, continuous carryover across years, free terminal | Physically exact; LP-feasible |
| D10 | **PPA capacity is an optimisation variable** (clamp bounds to fix it) | Capacity charges give PPA a real cost |
| D11 | **Integer container count** retained | Model is a MILP with exactly one general-integer variable |
| D12 | Degradation uses the **Year-1 = fresh = 1.0** `operating_value` convention | Matches current model |
| D13 | Banking/grid export is **possible future scope** | SOC + balance designed to accept an export var and the mutual-exclusion binary later |

---

## 1. Architecture review

### 1.1 What the current system does

```
SolverEngine (Optuna TPE, 1500 trials)
  └─ for each candidate sizing:
       Year1Engine.evaluate()            # 8760-h heuristic dispatch (PlantEngine)
         → GridInterface (scalar loss)   # lf ≈ 0.8953
         → MeterLayer (DISCOM shortfall)
       FinanceEngine.evaluate(fast|full) # CAPEX→OPEX→EnergyProjection→LCOE→LandedTariff→Savings
       feasibility gate (savings_npv ≥ 0)
  └─ re-evaluate best with full 25-year re-simulation
```

The optimiser is a black-box sampler wrapped around a simulation. Dispatch is a **hand-built ToD reservation heuristic** (`rsrv_evening / rsrv_morning_next / rsrv_fwd_evening`) that ring-fences SOC for peak windows. Sizing search is gradient-free and non-globally-optimal.

### 1.2 The two findings that reshape the migration

**Finding A — the financing is cost-plus, so the LCOE/landed-tariff ratios cancel in the objective.**
Tracing the code: `savings.re_cost_t = re_meter_kWh_t × landed_t` and `landed_t = total_cost_t / meter_kWh_t`, so `re_cost_t = total_cost_t` exactly. In NPV, `Σ_t LCOE·busbar_kWh_t·df_t = LCOE·NPV(busbar_kWh) = NPV(total project cost)`. The ratio LCOE and the ratio landed-tariff never survive into the objective. They become **post-solve reporting metrics**. This removes the only genuine objective non-linearities the brief anticipated.

**Finding B — the current objective is ToD-blind; removing the heuristic would expose that.**
`SavingsModel` values every delivered kWh at a single flat weighted-average DISCOM tariff. The ToD structure lives only inside the heuristic planner, not in the economics. If we removed the heuristic and kept the flat objective, the optimiser would have zero incentive to time-shift the battery. **Decision D1 fixes this** by moving the avoided-cost valuation to hourly ToD rates (baseline included, for consistency), which is what makes dispatch optimisation economically meaningful — and it stays linear.

### 1.3 Resulting model class

After D1–D13, the model is a **Mixed-Integer Linear Program** whose *only* integrality is the BESS container count. With that one variable relaxed it is a pure LP. There is **no bilinearity, no fractional objective, no MINLP**. This is the central conclusion of the review: the migration is far safer than a literal reading of the simulation suggests.

---

## 2. Mathematical reformulation strategy

### 2.1 Sets

| Set | Description | Cardinality |
|-----|-------------|-------------|
| `H` | hours in a representative year | 8760 |
| `Y` | project years | 25 |
| `Yopt` | years the dispatch is *explicitly optimised* over | `{1}` (single mode) or `Y` (full mode) |

The horizon toggle (D6) selects `Yopt`. The model builder is identical otherwise.

### 2.2 Parameters (all from existing YAML/CSV — nothing hardcoded)

- `cuf_s[h], cuf_w[h]` — solar/wind capacity factors `[0,1]`
- `load[h]` — client load (MWh)
- `lf` — scalar grid loss factor (≈0.8953)
- `η_c = 0.9721`, `η_d = 0.9384` — charge/discharge efficiency
- `cs = 5.015` — container size (MWh)
- `aux_pc = 0.300/24` — aux MWh per hour per container
- `crc, crd` — fixed charge/discharge C-rates (user params, D3)
- `tod[h]` — blended hourly DISCOM ToD tariff (INR/kWh), via existing `_build_hourly_discom_tariff`
- `wheel, tax` — wheeling & electricity-tax rates (INR/kWh)
- `cap_rate` — blended (CTU+STU+SLDC) INR/MW/month
- `d_s[y], d_w[y], d_b[y]` — degradation `operating_value` per year (Year 1 = 1.0, D12)
- `w` — WACC (constant); `df[y] = (1+w)^(-y)`
- CAPEX: `solar_rate` (INR/MWp DC), `ac_dc` (1.398), `wind_rate`, `bess_rate` (INR/MWh), `trans_fixed`
- OPEX rates + escalations; financing (`debt_frac, eq_frac, r, tenure, ROE, tax_rate`)

**Precomputed scalar constants** (computed once in Python, not by the solver):
- Annuity factors: `A_N = Σ_{t=1..25} df[t]`, `A_n = Σ_{t=1..tenure} df[t]`
- EMI factor: `a = r(1+r)^n/((1+r)^n − 1)`
- **Financing recovery factor** `Φ = debt_frac·a·A_n + eq_frac·ROE·A_N` ⇒ `NPV(financing) = total_capex·Φ`
- Escalated-OPEX discount sums `G_esc = Σ_t df[t](1+esc)^(t−1)` per escalation rate
- **Discounted-degradation sums** (single-year mode): `D_s = Σ_t df[t]·d_s[t]`, `D_w = Σ_t df[t]·d_w[t]`, `D_b = Σ_t df[t]·d_b[t]`

### 2.3 Decision variables

**Sizing (scalar):**

| Var | Domain | Meaning |
|-----|--------|---------|
| `S` | ℝ≥0 | AC solar capacity (MW) |
| `W` | ℝ≥0 | wind capacity (MW) |
| `P` | ℝ≥0 | PPA export cap (MW) |
| `n_b` | ℤ≥0 | BESS container count (**the only integer**) |

Derived (linear expressions, not free vars): `E_b = n_b·cs` (year-1 energy capacity, MWh); `total_capex = S·ac_dc·solar_rate + W·wind_rate + E_b·bess_rate + trans_fixed`.

**Hourly dispatch** — indexed `H` (single mode) or `Yopt×H` (full mode). All ℝ≥0:

| Var | Meaning |
|-----|---------|
| `sd[·]` | solar → client load, direct (busbar, pre-loss) |
| `wd[·]` | wind → client load, direct (busbar, pre-loss) |
| `chg[·]` | solar → battery (busbar, pre charge-efficiency) |
| `dis[·]` | battery → out (energy removed from SOC, pre discharge-efficiency) |
| `soc[·]` | state of charge (MWh) |
| `ddraw[·]` | residual DISCOM draw at meter (MWh) |

`curt[·]` (curtailment) is a reporting expression, not a variable (it is implied slack).
Convention (matches PlantEngine): busbar discharge delivery `= η_d·dis`; SOC gains `η_c·chg`.

### 2.4 Objective — derivation to a linear form

Start from `SavingsModel` with the hourly-ToD correction (D1). Per year `t`, with all energy in kWh (×1000):

```
baseline_t      = Σ_h load[h]·tod[h]·1000                 (constant in t; load fixed)
discom_cost_t   = Σ_h ddraw[h,t]·tod[h]·1000
re_cost_t       = re_payment_t + cap_t + (wheel+tax)·Σ_h re_meter[h,t]·1000
savings_t       = baseline_t − discom_cost_t − re_cost_t − penalty_t
```

Because `load[h] − ddraw[h,t] = re_meter[h,t]` (no export; delivered ≤ load), the baseline and DISCOM terms collapse:

```
baseline_t − discom_cost_t = Σ_h re_meter[h,t]·tod[h]·1000
```

and `Σ_t df[t]·re_payment_t = NPV(financing) + NPV(opex)` (cost-plus cancellation, Finding A). The full objective:

```
max  savings_npv =
     Σ_t df[t]·Σ_h re_meter[h,t]·tod[h]·1000          # avoided DISCOM, hourly-valued
   − Σ_t df[t]·(wheel+tax)·1000·Σ_h re_meter[h,t]      # grid charges on delivered RE
   − total_capex·Φ                                     # NPV(financing): debt service + ROE
   − NPV(opex)                                          # linear in S, W, n_b, capex
   − cap_rate·P·12·A_N                                  # NPV(capacity charges)
   − NPV(aux cost)                                      # Σ_t df[t]·Σ_h n_b·aux_pc·tod[h]·1000
   − NPV(penalty)                                       # optional RE-penetration penalty
```

with `re_meter[h,t] = lf·(sd[h,t] + wd[h,t] + η_d·dis[h,t])`.

**Single-year mode** uses the factorisation that degradation is hour-independent, so the year sum and hour sum separate per stream:

```
avoided = lf·1000·[ (Σ_h tod[h]·sd[h])·D_s
                  + (Σ_h tod[h]·wd[h])·D_w
                  + (Σ_h tod[h]·η_d·dis[h])·D_b ]
grid_chg = (wheel+tax)·1000·lf·[ (Σ_h sd[h])·D_s + (Σ_h wd[h])·D_w + (Σ_h η_d·dis[h])·D_b ]
```

This keeps **hourly ToD value *and* correct per-stream degradation** in a single 8760-term linear expression — strictly better than the current `fast_mode`, which scales annual totals and loses ToD entirely.

**Full mode** evaluates the double sum `Σ_t df[t]·Σ_h …` directly over `Y×H`, with per-year degraded generation and battery capacity (§3.5). Same objective, more terms.

Every term is linear in the decision variables. ∎

### 2.5 NPV(opex) expansion (all linear)

```
NPV(opex) =  S·ac_dc·solar_om_rate·G_0.02            (solar O&M, 2% esc)
           + W·wind_om_rate·G_0.02                    (wind O&M, 2% esc)
           + land_lease_monthly·12·G_0.03             (land, 3% esc — constant)
           + E_b·bess_om_rate·A_N                      (BESS O&M, no esc)
           + S·ac_dc·solar_trans_om_rate·A_N          (no esc)
           + W·wind_trans_om_rate·A_N                  (no esc)
           + total_capex·0.002·A_N                     (insurance, no esc)
```

---

## 3. Full constraint inventory

All constraints below are linear. Indexed over `H` (single) or `Yopt×H` (full); the year index is dropped for readability.

### 3.1 Generation allocation
```
C1  sd[h] + chg[h] ≤ S·cuf_s[h]          # solar serves direct + charge (solar_only source, D5)
C2  wd[h]          ≤ W·cuf_w[h]          # wind serves direct only (no wind charging in Phase 1)
```
*(D5 future-proofing: a `charge_w[h]` term and a source-selection parameter slot into C1/C2 with no structural change.)*

### 3.2 Client load balance (no export — D13 leaves room to add one)
```
C3  lf·(sd[h] + wd[h] + η_d·dis[h]) + ddraw[h] = load[h]
C4  ddraw[h] ≥ 0                          # ⇒ delivered RE ≤ load (no over-serve, no export)
```
*(Banking/export future: add `exp[h] ≥ 0` to the LHS and an export-revenue term to the objective; C3 becomes `… + ddraw − exp = load`. SOC/discharge structures are unchanged.)*

### 3.3 PPA export cap
```
C5  sd[h] + wd[h] + η_d·dis[h] ≤ P
```

### 3.4 SOC dynamics & limits (D9 boundary conditions)
```
C6  soc[h] = soc[h−1] + η_c·chg[h] − dis[h]        # interior hours
C7  soc[0] = 0                                       # commissioning-zero (single & full year-1)
C8  soc[h] ≤ E_b
C9  chg[h] ≤ crc·E_b                                 # charge power cap (crc fixed ⇒ linear)
C10 dis[h] ≤ crd·E_b                                 # discharge power cap
```
- **Single-year mode:** terminal `soc[H_last]` is **free** (no cyclic constraint).
- **Full mode:** SOC is one continuous chain across the 25×8760 horizon — `soc[y, 0] = soc[y−1, H_last]` for `y ≥ 2`, `soc[1,0]=0`, terminal `soc[25, H_last]` free. Energy capacity and power caps use the degraded `E_b·d_b[y]` (§3.5).

Aux is **not** in the SOC balance (D8) — it is a grid-fed cost term in the objective, which keeps C6 feasible at the commissioning-zero start (no "owe aux at empty SOC" infeasibility).

### 3.5 Degradation coupling (full mode only)
For year `y`, replace capacities by their degraded values inside C1, C2, C8, C9, C10:
```
C1' sd[y,h] + chg[y,h] ≤ S·d_s[y]·cuf_s[h]
C2' wd[y,h]            ≤ W·d_w[y]·cuf_w[h]
C8' soc[y,h] ≤ E_b·d_b[y]
C9' chg[y,h] ≤ crc·E_b·d_b[y]
C10' dis[y,h] ≤ crd·E_b·d_b[y]
```
Single-year mode uses `d_·[1]=1.0` (fresh) and applies degradation only in the objective via `D_s, D_w, D_b`.

### 3.6 Optional constraints (ported from `solver.yaml`, all linear, default off)

| Toggle | Constraint |
|--------|-----------|
| `minimum_savings_npv` | Report-only by default (don't make the model infeasible when the optimum is negative); optional hard floor `savings_npv ≥ v` |
| `plant_cuf` | `min ≤ Σ busbar_MWh / (P·8760)·100 ≤ max` |
| `minimum_bess_capacity` | `E_b ≥ min_mwh` |
| `minimum_bess_discharge` | `Σ_h η_d·dis[h] ≥ min_annual_mwh` |
| `re_penetration` | `min ≤ Σ re_meter / Σ load·100 ≤ max` |
| `hourly_re_penetration_penalty` | soft penalty term in objective (already linear; hour-masked) |

### 3.7 Bounds (from `solver.yaml`)
`0 ≤ S,W,P ≤ 1000`; `0 ≤ n_b ≤ 1000` integer. Bound-clamping `P` is how the user fixes PPA (D10).

---

## 4. Identification of all non-linearities

| Candidate (from brief / summary §14.5) | Status after design |
|---|---|
| `LCOE = NPV(cost)/NPV(energy)` (fractional) | **Eliminated** — cancels in NPV; reporting-only (D2) |
| `landed = total_cost/meter_kWh` (ratio) | **Eliminated** — cancels in savings; reporting-only (D2) |
| `charge_power = c_rate·E_b` (bilinear) | **Eliminated** — C-rate fixed parameter (D3) |
| charge/discharge mutual exclusion (binary) | **Not needed** — efficiency dominance makes simultaneity strictly suboptimal (D7); verified post-solve |
| `aux = ceil(SOC/cs)·aux_pc` (integer step, SOC-gated) | **Eliminated** — fixed per-container grid-fed cost (D8) |
| Integer `n_b` | **Retained** — the sole integrality; MILP |
| degradation × dispatch coupling | **Linear** — degradation enters as constant coefficients (objective in single mode; capacity bounds in full mode) |

**Net: one general-integer variable. No bilinear, fractional, or nonconvex terms anywhere.**

---

## 5. LP vs MILP vs MINLP trade-offs

- **MINLP** (what a literal port would produce): rejected. The non-linearities that would force it (LCOE ratio, c_rate bilinearity, aux ceil) are all design-eliminated above. No reason to incur global-MINLP solver fragility.
- **MILP** (chosen): exactly one general-integer variable (`n_b ∈ [0,1000]`). Branch-and-bound on a single integer is trivial; the LP relaxation is tight and the rounding gap is ≤ one container (≈5.015 MWh).
- **Pure LP** (relaxation): available as a mode — relax `n_b` to continuous `E_b`, solve, then snap to `round(E_b/cs)` and re-solve the dispatch LP with `n_b` fixed to confirm. For a 1000-container range this is effectively exact and the fastest path. **Recommendation: solve as MILP by default (cheap here); offer the relax-and-snap LP for the full 25-year horizon if MILP branch time ever bites.**

---

## 6 & 7 & 8. Recommended solver stack

**Primary: HiGHS** (open-source, actively maintained, dual-simplex + interior-point LP and a strong MILP B&B). Reasons:
- Handles the single-year LP (~60k vars) in low single-digit seconds and the full 25-year LP (~1.5M vars) in seconds-to-minutes.
- One integer variable ⇒ MILP overhead is negligible.
- First-class Pyomo support via the `appsi_highs` in-memory interface (avoids file round-trips — important at 1.5M vars).

**Fallback: CBC** (open-source) — for cross-validation that the optimum is solver-independent. Slower on the large LP but fine for the single-year model.

**Commercial (not required, optional):** Gurobi/CPLEX would shave the full-horizon LP to seconds and are drop-in via Pyomo, but the **architecture will not depend on them**. They are a performance option only.

**Numerical conditioning (important):** raw INR values are ~1e7–1e10 (crores) while energy terms are ~1e0–1e4. Solve in **scaled units — lakh/crore INR and GWh** — to keep the constraint matrix well-conditioned for HiGHS. Scaling factors are applied at build and inverted at report.

---

## 9. Proposed Pyomo architecture

```
src/hybrid_plant/optimise/                 # new package, parallel to solver/
├── __init__.py
├── config.py            # OptModelConfig: horizon mode, solver, toggles, scaling, augmentation
├── sets.py              # H, Y, Yopt construction
├── params.py            # load YAML/CSV → scaled Pyomo Params + precomputed constants (Φ, A_N, D_s…)
├── variables.py         # sizing + hourly Var declarations (mode-aware indexing)
├── constraints/
│   ├── allocation.py    # C1, C2  (generation split; source-selection-ready)
│   ├── balance.py       # C3, C4  (client load; export-ready)
│   ├── ppa.py           # C5
│   ├── soc.py           # C6–C10 (+ full-mode carryover & degraded caps)
│   ├── sizing.py        # capex/opex linear expressions, bounds
│   └── optional.py      # CUF / RE-pen / min-BESS toggles
├── objective.py         # savings_npv (single- and full-mode forms)
├── build.py             # assemble ConcreteModel from OptModelConfig (the toggle lives here)
├── solve.py             # appsi_highs driver; status handling; solution extraction
├── report.py            # POST-SOLVE: LCOE, landed tariff, dashboards (reporting-only metrics)
├── verify.py            # post-solve assertions (no simultaneous c/d; SOC bounds; energy conservation)
└── augmentation.py      # brownfield deltas (Phase 2 stub — present, inert)
```

**Design principles:**
- **One builder, two horizons.** `build.py` reads `OptModelConfig.horizon ∈ {single, full}` and sets `Yopt`; constraint modules are written index-generically so neither mode is special-cased.
- **Reporting strictly separated from optimisation.** `report.py` recomputes LCOE and landed tariff *from the optimal sizing* using the existing `LCOEModel`/`LandedTariffModel` so dashboards are unchanged — these never enter the model (D2).
- **`SolverEngine` is replaced, not edited.** `run_model.py` gains a switch to call `optimise.solve` instead of `solver_engine`. The energy/finance engines remain as the **validation oracle** (§11), not deleted.
- **Augmentation seams pre-cut.** Sizing vars are written as `existing + delta` with `existing` defaulting to 0 (greenfield), so Phase 2 only flips parameters (§14).

---

## 10. Migration strategy (incremental, each step validated)

1. **Scaffold + params.** Build `params.py`, confirm every value reconciles 1:1 with `FullConfig` (unit test against `config_loader`). No model yet.
2. **Single-year LP, dispatch only.** Variables + C1–C10 with sizing **fixed** to a known Optuna solution. Validate energy accounting against PlantEngine on *imposed* dispatch (§11 Layer 1).
3. **Add finance objective (reporting parity).** Wire §2.4 objective; with fixed sizing + fixed annual energy, confirm `savings_npv`, LCOE, landed tariff match FinanceEngine within tolerance (§11 Layer 2).
4. **Unfix sizing → first true optimisation.** Solve single-year MILP. Compare optimum to Optuna best (§11 Layer 3).
5. **Full 25-year mode.** Add `Y` indexing, carryover SOC, degraded caps. Validate full-mode objective against single-year-scaled within expected degradation error.
6. **Optional constraints + RE penalty.** Port toggles; regression-test each.
7. **Solver hardening.** HiGHS↔CBC cross-check; scaling tuning; runtime profiling.
8. **Cutover.** `run_model.py` switch; keep Optuna path behind a flag for one release as a safety net.

---

## 11. Validation strategy

The optimiser will **not** reproduce PlantEngine's *dispatch hour-by-hour* — PlantEngine uses a heuristic, Pyomo finds the optimum. So validation targets the **physics and finance math**, not dispatch identity.

**Layer 1 — Physics replication (constraint correctness).**
Fix sizing to a reference case and **fix the dispatch variables to PlantEngine's hourly outputs** (`sd, wd, chg, dis`). Assert Pyomo's constraints are satisfied and its derived quantities — `soc[h]`, meter delivery, curtailment, busbar/meter annual totals — match PlantEngine within tolerance. Known, *documented* divergences: aux treatment (D8, grid-fed vs DC-drain) and SOC boundary (D9). Quantify both; expect < 0.5% on annual energy.

**Layer 2 — Finance replication (objective math).**
Feed an identical annual energy projection into both Pyomo's objective expression and `FinanceEngine`. Assert `savings_npv`, `lcoe`, `landed_tariff`, capacity/wheeling/tax components match within **0.1%**. This isolates the cost-plus cancellation (Finding A) and proves the linear objective equals the original pipeline.

**Layer 3 — Optimality comparison.**
Run the full Pyomo optimisation. Assert `savings_npv(Pyomo) ≥ savings_npv(Optuna best)` (Pyomo may legitimately win via better dispatch + global optimum). Produce a **difference report**: sizing deltas, dispatch-pattern deltas, and the economic driver of each. Every material difference must be explainable (typically: Pyomo exploits hourly ToD that the flat-tariff Optuna objective never saw).

**Layer 4 — Property/invariant tests.** No simultaneous charge+discharge (D7 verification); `0 ≤ soc ≤ E_b`; energy conservation per hour; `export ≤ P`; `ddraw ≥ 0`.

**Layer 5 — Regression.** Golden-output snapshots (objective, sizing, key dispatch stats) under fixed seed/config; CI fails on drift beyond tolerance.

### Acceptance criteria (migration "done")
- Layer 1 ≤ 0.5% energy divergence, fully attributed to D8/D9.
- Layer 2 ≤ 0.1% on all financial metrics.
- Layer 3: `Pyomo ≥ Optuna`, every difference explained.
- Layer 4: all invariants hold (incl. zero simultaneous c/d, confirming D7).
- Runtime within §12 targets.
- HiGHS and CBC agree on the optimum (single-year) within tolerance.

---

## 12. Runtime expectations

| Configuration | Size | Expected (HiGHS) |
|---|---|---|
| Single-year LP relaxation | ~60k vars, ~45k constraints | 1–5 s |
| Single-year MILP (1 integer) | same + B&B | seconds |
| Full 25-year LP | ~1.5M vars | ~0.5–5 min |
| Full 25-year MILP | + B&B on 1 integer | minutes (relax-and-snap if needed) |

Baseline for comparison: current Optuna run ≈ 2–3 min (fast mode) / 20–40 min (full re-simulation), and only *heuristically* optimal. Pyomo is comparable-or-faster **and** globally optimal for the stated objective.

---

## 13. Technical risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Full-mode MILP branch time | Low | Relax-and-snap LP (§5); single integer keeps gap tiny |
| Numerical conditioning (INR magnitude) | Medium | Scaled units (crore/GWh) from the start (§6–8) |
| Aux/SOC divergence from PlantEngine confuses validation | Medium | Pre-declared as known divergences; quantified in Layer 1; validation-only flag can replicate PlantEngine's DC-drain + zero-start/free-end for apples-to-apples |
| Dispatch degeneracy (alternate optimal schedules) | Medium | Objective value is unique; dispatch need not be. Report on aggregates, not hour identity. Optional tiny curtailment tie-break penalty if a canonical schedule is wanted |
| Hourly ToD changes sizing vs Optuna materially | Expected | This is *intended* (Finding B); covered by Layer-3 explainability, not treated as an error |
| Banking/export future forces rebalancing | Low | C3/SOC already structured to admit an export var + mutual-exclusion binary (D13) |
| `tod[h]` array construction mismatch | Low | Reuse existing `_build_hourly_discom_tariff`; unit-test against it |
| LP relaxation yields fractional containers users dislike | Low | MILP default returns integer; relax-and-snap documented |

---

## 14. Future augmentation readiness (Phase 2)

The architecture is built so Phase 2 is **parameter changes, not restructuring**:

- **Brownfield sizing:** every sizing variable is `total = existing + delta`. Phase 1 sets `existing = 0`. Phase 2 sets `existing` from the installed plant and optimises `delta_solar, delta_wind, delta_bess, delta_ppa ≥ 0`. Lives in `augmentation.py`, inert in Phase 1.
- **Incremental CAPEX:** the objective's `total_capex` term already references the *total* capacity; Phase 2 swaps in an **incremental** CAPEX expression (charge only `delta`) and treats existing financing as sunk — a localized change in `sizing.py`/`objective.py`.
- **Charge-source variable (D5):** C1/C2 already isolate the source; promoting `solar_only` to a decision needs a source-fraction variable, not a rebuild.
- **Banking/grid export (D13):** C3 admits an `exp[h]` term; the objective admits an export-revenue term; **and the mutual-exclusion binary (§ formulation in review) drops into `soc.py`** because export breaks the surplus/shortfall exclusivity that currently guarantees no simultaneous c/d. This is the one future feature that converts the LP to a MILP-with-many-binaries, so it is explicitly isolated.
- **Multi-year mode is already the vehicle** for augmentation timing and any future vintage/replacement modelling.

---

## 15. Testing strategy

- **Unit (per module):** params reconciliation vs `FullConfig`; each constraint block builds and is feasible on a toy 24-h instance; objective expression equals a hand-computed value on a 3-hour fixture.
- **Integration:** full build in both horizon modes; solve to optimality; status assertions.
- **Property-based:** invariants from Layer 4 across randomized feasible sizings.
- **Oracle parity:** Layers 1–2 automated against PlantEngine/FinanceEngine.
- **Solver-agnostic:** HiGHS vs CBC optimum equality (single-year).
- **Edge cases:** `n_b = 0` (no battery), `W = 0` (solar-only), `S = 0` (wind-only), PPA binding vs slack, RE-penalty on/off, all optional constraints individually.
- **Regression:** golden snapshots in CI.

---

## 16. Success criteria

Phase 1 is complete when:

1. The Pyomo model **replaces** `SolverEngine` in `run_model.py` and produces the full dashboard (sizing, dispatch stats, LCOE, landed tariff, savings NPV) with no regression in reporting.
2. **Layer 1** physics parity ≤ 0.5% (divergence fully attributed to D8/D9).
3. **Layer 2** finance parity ≤ 0.1% on all metrics — proving the linear objective ≡ the original pipeline.
4. **Layer 3** `savings_npv(Pyomo) ≥ savings_npv(Optuna)` with a complete, explainable difference report.
5. **Layer 4** invariants hold — in particular **zero simultaneous charge/discharge**, empirically confirming D7.
6. Runtime within §12; HiGHS-only (no commercial dependency); HiGHS↔CBC agreement.
7. Both horizon modes operate behind the toggle (D6).
8. Augmentation seams (§14) present and inert, with a passing greenfield-equivalence test (`existing = 0` reproduces Phase-1 results).

---

## Appendix A — Objective, one-line summary

> **Maximise:** hourly-ToD-valued RE delivered to the client (× discounted per-stream degradation) − grid charges on that RE − cost-plus recovery of CAPEX financing − OPEX − PPA capacity charges − aux − optional RE penalty.
>
> Linear in `{S, W, P, n_b}` and the hourly dispatch. MILP with one integer. LCOE and landed tariff are **outputs**, not inputs.

## Appendix B — Open items requiring no further input (recorded for traceability)

- Aux modelled as grid-fed (conservative vs PlantEngine's DC-drain); validated as immaterial in Layer 1.
- `minimum_savings_npv` is report-only by default to avoid spurious infeasibility.
- Dispatch degeneracy handled by reporting aggregates; optional tie-break penalty available if a canonical schedule is desired.
