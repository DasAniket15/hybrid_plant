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
        obj_val  : float — objective value (or NaN if not solved)
        wall_sec : float — solver wall-clock time in seconds
    """
    solver_name = opt_cfg.solver_name

    if solver_name == "appsi_highs":
        solver = pyo.SolverFactory("appsi_highs")
    elif solver_name == "cbc":
        solver = pyo.SolverFactory("cbc")
    else:
        raise ValueError(f"Unsupported solver: {solver_name!r}. Use 'appsi_highs' or 'cbc'.")

    results = solver.solve(model, tee=tee)

    tc = results.solver.termination_condition
    status = str(tc)

    try:
        obj_val = float(pyo.value(model.obj))
    except Exception:
        import math
        obj_val = math.nan

    try:
        wall_sec = float(results.solver.wall_time)
    except AttributeError:
        wall_sec = float("nan")

    return {"status": status, "obj_val": obj_val, "wall_sec": wall_sec}


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

    # ── Phase 1: LP relaxation (nb continuous) ───────────────────────────────
    orig_domain = model.nb.domain
    was_fixed   = model.nb.is_fixed()
    if was_fixed:
        model.nb.unfix()
    model.nb.domain = pyo.NonNegativeReals

    r1 = solve(model, opt_cfg, tee=tee)
    nb_relaxed  = float(pyo.value(model.nb))
    relaxed_obj = r1["obj_val"]

    # ── Phase 2: snap nb to nearest integer, re-solve dispatch LP ────────────
    nb_snapped = int(round(nb_relaxed))
    model.nb.domain = orig_domain
    model.nb.fix(nb_snapped)

    r2 = solve(model, opt_cfg, tee=tee)

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

    return {
        "sd":    _arr(model.sd),
        "wd":    _arr(model.wd),
        "chg":   _arr(model.chg),
        "dis":   _arr(model.dis),
        "soc":   _arr(model.soc),
        "ddraw": _arr(model.ddraw),
    }
