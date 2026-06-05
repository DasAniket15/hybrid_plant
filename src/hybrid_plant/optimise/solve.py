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


def extract_dispatch(
    model:  pyo.ConcreteModel,
    n_hours: int = 8760,
) -> dict[str, Any]:
    """
    Pull solved variable values out of *model* into numpy arrays.

    Returns
    -------
    dict with keys: sd, wd, chg, dis, soc, ddraw — each np.ndarray shape (n_hours,)
    """
    import numpy as np

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
