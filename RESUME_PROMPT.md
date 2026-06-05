# Resume Prompt — Pyomo Migration

Copy-paste the block below into a fresh Claude Code session on this repo to
continue the Pyomo migration. Detail lives in `PYOMO_MIGRATION_HANDOFF.md`
(handoff log) and `PYOMO_MIGRATION_DESIGN.md` (authoritative spec).

---

## Prompt (copy everything in the block)

```text
I'm resuming an in-progress Pyomo migration in this repo (hybrid_plant). A
previous session handed off mid-Step-5.

Before doing anything, read these two files in full — they are authoritative:
1. PYOMO_MIGRATION_HANDOFF.md — the session handoff log (start here; section 8
   has the exact resume procedure)
2. PYOMO_MIGRATION_DESIGN.md — the approved technical spec (don't redesign;
   raise spec issues instead of coding around them)

Context: branch is feature/pyomo-migration. Step 5 (full 25-year mode) is
implemented and WIP-committed (d2f8d1c) but its slow tests were never
validated — the 25x8760 solve was killed during a machine switch. Steps 1-4
are committed and passing.

Hard rules (also in the handoff): do not modify anything under
src/hybrid_plant/ except the new optimise/ package (and, only at Step 8, one
switch in run_model.py); the other engines are the validation oracle. Nothing
hardcoded. Local commits only — do not push or open a PR. Work the gated steps
one at a time: after each, run tests, commit, report, and wait for my
confirmation before the next.

Note on configs/bess.yaml: the change to discharge_hours: [] and
charge_first: false is INTENTIONAL — I disabled the evening-only / charge-first
BESS dispatch so PlantEngine runs in normal RTC mode. Keep it. Do NOT revert.

Do NOT start coding yet. First: (a) confirm the branch and that pyomo/highspy
import; (b) run `pytest tests/optimise/ -m "not slow" -q` and report results;
(c) give me your plan to finish Step 5 (run the slow full-horizon suite,
expected ~25-40 min) and flag the full-mode runtime problem from handoff
section 6. Then stop and wait for me.
```

---

## Practical notes (for you, not part of the prompt)

- **Verify the sync first.** Before pasting, confirm this machine has commit
  `d2f8d1c`: `git log --oneline -3` on `feature/pyomo-migration`. If OneDrive
  hasn't finished syncing, wait. If this PC is a *separate clone* (not the same
  OneDrive folder), `git pull` the branch first (the previous session was told
  not to push, so confirm the branch actually arrived).
- **Keep the machine awake** for the slow solve (~25-40 min). The earlier
  session disabled then restored sleep on the *work* machine; do the same here
  if needed (`powercfg /change standby-timeout-ac 0`, restore after).
- **Don't want to wait on the long solve right away?** Add to the prompt:
  "For now just do the fast checks; we'll run the slow suite later."
- The `~/.claude` memory from the other machine does not travel — everything
  needed is in the two repo docs above.
