# Documentation index

## `design/`

Specifications that still govern the code.

- [`pyomo-migration-design.md`](design/pyomo-migration-design.md) — authoritative
  spec for the Pyomo optimisation model: sets, variables, constraints, objective,
  and the Layer 1–4 verification scheme. Section numbers referenced from
  docstrings throughout `src/hybrid_plant/optimise/` (e.g. "design §2.4",
  "design §11 Layer 4") point here. Still current.

## `reference/`

Background material describing the model as a whole.

- [`model-summary.md`](reference/model-summary.md) — technical walkthrough of the
  energy and finance stack. **Written before the Pyomo migration**, to brief an
  LLM on the pre-migration state. The energy/finance chapters remain accurate;
  the optimisation chapters describe the Optuna `SolverEngine`, which is now
  `hybrid_plant.legacy` and no longer the production path. Read alongside
  `design/pyomo-migration-design.md`, which supersedes it on optimisation.

## `history/`

Records of how the code reached its current state. Not maintained.

- [`pyomo-migration-handoff.md`](history/pyomo-migration-handoff.md) — step-by-step
  implementation log for the migration, Steps 1–8. Useful for reconstructing why
  a given decision was made; not a description of current behaviour.
