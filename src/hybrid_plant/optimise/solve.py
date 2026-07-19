"""
optimise/solve.py
─────────────────
Solver driver for the Pyomo model.

Primary: HiGHS via Pyomo ``appsi_highs`` (in-memory, avoids file round-trips).
Fallback: CBC for cross-validation (Step 7).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from hybrid_plant.optimise.config import OptModelConfig


def _solve_once(
    solver: Any,
    model:  pyo.ConcreteModel,
    tee:    bool = False,
) -> dict[str, Any]:
    """
    Solve *model* with an already-created solver instance.

    Un-scales obj_val by ``model._obj_scale`` if set (full-mode objective
    stores 1e-7 there so HiGHS sees a well-conditioned range; single-year
    models leave the attribute absent, defaulting to 1.0).

    An infeasible model (e.g. an over-tight optional PPA constraint) is
    reported as ``status="infeasible"`` rather than raising — the appsi_highs
    interface otherwise throws when it cannot load a solution.
    """
    import math

    try:
        results = solver.solve(model, tee=tee)
    except RuntimeError as exc:
        if "feasible solution was not found" in str(exc).lower():
            return {"status": "infeasible", "obj_val": math.nan, "wall_sec": math.nan}
        raise
    status  = str(results.solver.termination_condition)

    obj_scale = getattr(model, "_obj_scale", 1.0)
    try:
        obj_val = float(pyo.value(model.obj)) / obj_scale
    except Exception:
        obj_val = math.nan

    try:
        wall_sec = float(results.solver.wall_time)
    except AttributeError:
        wall_sec = float("nan")

    return {"status": status, "obj_val": obj_val, "wall_sec": wall_sec}


def solve(
    model:   pyo.ConcreteModel,
    opt_cfg: OptModelConfig,
    tee:     bool = False,
) -> dict[str, Any]:
    """
    Solve *model* and return a status + timing dict.

    Parameters
    ----------
    model   : assembled ConcreteModel (from ``build.py``)
    opt_cfg : OptModelConfig — selects solver via ``solver_name``
    tee     : if True, stream solver output to stdout

    Returns
    -------
    dict
        status   : str   — ``"optimal"`` | ``"infeasible"`` | other
        obj_val  : float — objective value in original INR units (or NaN)
        wall_sec : float — solver wall-clock time in seconds
    """
    solver_name = opt_cfg.solver_name

    if solver_name == "appsi_highs":
        solver = pyo.SolverFactory("appsi_highs")
    elif solver_name == "cbc":
        solver = pyo.SolverFactory("cbc")
    else:
        raise ValueError(f"Unsupported solver: {solver_name!r}. Use 'appsi_highs' or 'cbc'.")

    return _solve_once(solver, model, tee)


def solve_relax_and_snap(
    model:   pyo.ConcreteModel,
    opt_cfg: OptModelConfig,
    tee:     bool = False,
) -> dict[str, Any]:
    """
    Solve a MILP by LP-relaxation + snap (design §5).

    For this problem the only integrality is the BESS container count ``nb``,
    and the LP relaxation is tight (the rounding gap is ≤ one container ≈ cs
    MWh).  Rather than pay branch-and-bound on a 1.75M-row full-horizon model,
    this routine:

      1. Relaxes ``nb`` to a continuous variable and solves the LP            (bound)
      2. Snaps ``nb`` to the nearest integer and fixes it
      3. Re-solves the now-pure dispatch LP                                   (feasible)

    The relaxation objective is an upper bound on the true MILP optimum; the
    snapped re-solve is a feasible value.  Their difference is reported as
    ``snap_gap_frac`` so the near-optimality is auditable.  Two clean LP solves
    instead of unbounded branching — deterministic and ~2× a single LP solve.

    Returns
    -------
    dict
        status        : termination condition of the final (snapped) solve
        obj_val       : feasible savings_npv at the snapped integer nb
        relaxed_obj   : LP-relaxation objective (upper bound)
        nb_relaxed    : continuous nb from the relaxation
        nb_snapped    : int(round(nb_relaxed)) — the fixed value
        snap_gap_frac : (relaxed_obj − obj_val) / |relaxed_obj|  (≥ 0, small)
        wall_sec      : combined wall time of both solves
    """
    import math

    solver_name = opt_cfg.solver_name
    if solver_name == "appsi_highs":
        solver = pyo.SolverFactory("appsi_highs")
    elif solver_name == "cbc":
        solver = pyo.SolverFactory("cbc")
    else:
        raise ValueError(f"Unsupported solver: {solver_name!r}. Use 'appsi_highs' or 'cbc'.")

    # ── Phase 1: LP relaxation (nb continuous) ───────────────────────────────
    orig_domain = model.nb.domain
    was_fixed   = model.nb.is_fixed()
    if was_fixed:
        model.nb.unfix()
    model.nb.domain = pyo.NonNegativeReals

    r1 = _solve_once(solver, model, tee=tee)
    nb_relaxed  = float(pyo.value(model.nb))
    relaxed_obj = r1["obj_val"]

    # ── Phase 2: snap nb to nearest integer, re-solve dispatch LP ────────────
    # Keep nb in the CONTINUOUS domain (NonNegativeReals) — only fix its value.
    # Changing domain (continuous→integer) triggers an O(n²) APPSI expression
    # re-walk across all 657k constraints that reference E_b=nb*cs, hanging for
    # hours.  A bound-only fix (lb=ub=nb_snapped) is an O(1) APPSI update and
    # preserves the Phase-1 LP basis for a warm-started Phase-2 solve.
    nb_snapped = int(round(nb_relaxed))
    model.nb.fix(nb_snapped)

    r2 = _solve_once(solver, model, tee=tee)

    # Restore domain and unfix for the caller.
    model.nb.domain = orig_domain
    model.nb.unfix()

    w1 = r1.get("wall_sec", float("nan"))
    w2 = r2.get("wall_sec", float("nan"))
    wall = (0.0 if math.isnan(w1) else w1) + (0.0 if math.isnan(w2) else w2)

    gap = abs(relaxed_obj - r2["obj_val"]) / max(abs(relaxed_obj), 1.0)

    return {
        "status":        r2["status"],
        "obj_val":       r2["obj_val"],
        "relaxed_obj":   relaxed_obj,
        "nb_relaxed":    nb_relaxed,
        "nb_snapped":    nb_snapped,
        "snap_gap_frac": gap,
        "wall_sec":      wall,
    }


def extract_dispatch(
    model:   pyo.ConcreteModel,
    n_hours: int | None = None,
) -> dict[str, Any]:
    """
    Pull solved variable values out of *model* into numpy arrays.

    Parameters
    ----------
    model   : solved ConcreteModel
    n_hours : number of timesteps to extract.  Defaults to the length of
              ``model.H`` (8760 single-mode, 219000 full-mode).

    Returns
    -------
    dict with keys: sd, wd, chg, dis, soc, ddraw — each np.ndarray shape (n_hours,)
    """
    import numpy as np

    if n_hours is None:
        n_hours = len(model.H)

    def _arr(var: pyo.Var) -> "np.ndarray":
        return np.array([pyo.value(var[h]) for h in range(n_hours)], dtype=np.float64)

    out = {
        "sd":    _arr(model.sd),
        "wd":    _arr(model.wd),
        "chg":   _arr(model.chg),
        "dis":   _arr(model.dis),
        "soc":   _arr(model.soc),
        "ddraw": _arr(model.ddraw),
    }
    # D5 split: wind-sourced charge portion, present only for wind/both sources.
    if hasattr(model, "chg_w"):
        out["chg_w"] = _arr(model.chg_w)
    return out
