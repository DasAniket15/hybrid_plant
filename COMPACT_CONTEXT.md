## 🗜️ COMPACT CONTEXT — hybrid_plant Pyomo migration, Step 5 slow-test unblock

**Session summary:** Resumed Step 5 (full 25-year horizon mode) on branch `feature/pyomo-migration`. Fast tests (121) all pass. Three attempts to run the slow full-horizon solve were killed; the root cause of both the convergence failure and the pre-solve hangs is now fully diagnosed. No new commits were made this session — `solve.py` is reverted to clean. Awaiting user approval on the proposed two-part fix before proceeding.

---

**Key context & decisions:**

- **Repo / branch / rules:** `C:\Users\pc\Documents\GitHub Repositories\hybrid_plant`, branch `feature/pyomo-migration`. Hard rules: do NOT touch `src/hybrid_plant/` except `optimise/`; do NOT push; commit only after tests pass; wait for user confirmation between steps. Authoritative spec: `PYOMO_MIGRATION_DESIGN.md`. Handoff log: `PYOMO_MIGRATION_HANDOFF.md`.

- **Step status:** Steps 1–4 committed and green. Step 5 code WIP-committed at `d2f8d1c` (full 25-year model, `TimeContext`, `solve_relax_and_snap`). Slow tests (`tests/optimise/test_full_horizon.py -m slow`, 10 tests) were **never validated** — this session confirmed why.

- **Session fix #0 (done):** Editable install was pointing to a stale `.claude/worktrees/` path. Fixed with `pip install -e .`. Fast suite now passes in 1.5 s.

- **Bottleneck #1 — APPSI compile time (measured):** `build_model(full)` = 9.2 s (fine). `solver.set_instance(model)` = **217.8 s ≈ 3.6 min** (slow, unavoidable per call). `solve_relax_and_snap` creates a **new solver per `solve()` call**, so Phase 1 + Phase 2 = **2 × 217.8 s = 7.3 min** of Python overhead before HiGHS even runs. This explains why LP stats appeared at ~7 min into run 1.

- **Bottleneck #2 — HiGHS convergence (unresolved):** Raw-INR objective range is **[5e2, 9e7]** (5 orders of magnitude). HiGHS 1.14.0 explicitly warns "excessively large costs; consider `user_objective_scale = -7`". Dual simplex oscillated for 33+ min (1.09M iterations, 178k primal infeasibilities, no convergence). This is the same LP solve that the handoff (§6) said "may not fix LP solve time — but untested."

- **Three failed scaling attempts this session (all reverted):**
  1. `model.obj.set_value(_orig_expr * 1e-7)` inside `solve()` → APPSI re-walks a `ProductExpression(SumExpression[219k], const)` each call → O(n²) hang, 30+ min, zero HiGHS output.
  2. `solver.options["simplex_scale_strategy"] = 4` → also caused 18+ min pre-solve hang (mechanism unclear; possibly option interaction with APPSI update logic).
  3. Reverted to clean `solve.py` (no options, no expression manipulation).

- **`tests/optimise/test_full_horizon.py` line 297:** Still has `tee=True` added this session (`solve_relax_and_snap(model, opt_cfg, tee=True)`). This is intentional for solver visibility but is not committed. Revert to `False` before the Step 5 commit.

- **`configs/bess.yaml`:** `discharge_hours: []`, `charge_first: false` — intentional RTC switch, committed at `5f73bea`. Do NOT revert.

---

**Proposed fix (needs user approval — not yet implemented):**

**Part 1 — Scale objective at BUILD TIME in `objective.py`**
Multiply every INR-valued coefficient by `1e-7` *during expression construction* (not after), so APPSI sees a properly-scaled `LinearExpression` from the start. Divide `obj_val` by `1e7` in `solve.py` when extracting via `pyo.value(model.obj)`. No expression wrapping, no re-compilation overhead, no hang.

```python
# objective.py — add at top of each add_savings_npv_objective* function
_S = 1e-7  # TODO Step 7: replace with opt_cfg.scale_money

# Every INR coefficient (e.g. lf * 1000 * net_tod[h]) gets premultiplied:
coef = _S * lf * 1000.0 * net_tod[tc.hour_of[t]] * tc.disc[t]
```

```python
# solve.py — when extracting obj_val:
_S = 1e-7  # must match objective.py
obj_val = float(pyo.value(model.obj)) / _S
```

**Part 2 — Reuse APPSI solver across relax-and-snap phases**
Refactor `solve_relax_and_snap` to create ONE solver and pass it into both `solve()` calls. Phase 2 then does only an incremental HiGHS model update (fix `nb`) rather than a full 217.8 s recompile. Saves one full compile cycle.

Expected total runtime after both fixes: **~30–45 min** (3.6 min APPSI + ~15 min Phase 1 HiGHS + incremental update + ~10 min Phase 2 HiGHS).

---

**Current file states (nothing committed this session):**

| File | State |
|---|---|
| `src/hybrid_plant/optimise/solve.py` | **CLEAN** — reverted to original (no options, no expression manipulation) |
| `tests/optimise/test_full_horizon.py` | `tee=True` at line 297 (not committed; revert before Step 5 commit) |
| `slow_test_output.txt` | Partial run-1 output (untracked, can delete) |

---

**Open threads / next steps:**

- **Awaiting user go/no-go** on the two-part fix above (Part 1 only, or Part 1 + Part 2).
- Once approved: implement → fast regression → launch slow tests → if all 10 pass → revert `tee=True` → full suite → commit Step 5 → report → wait for Step 6 confirmation.
- After Step 5 commit, update `PYOMO_MIGRATION_HANDOFF.md` to document: APPSI compile = 217.8 s/call, objective scaling implemented in `objective.py`, solver reuse in `solve_relax_and_snap`.
- Step 7 work still needed: proper `scale_money` wiring via `opt_cfg`, HiGHS↔CBC cross-check, `verify.py` post-solve invariants.
- Sleep settings on this machine were changed for long solves (handoff §11). Restore after Step 5: `powercfg /change standby-timeout-ac 45` etc.
