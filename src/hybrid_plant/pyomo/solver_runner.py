"""
solver_runner.py
────────────────
Calls HiGHS (or CBC/GLPK) via Pyomo, checks solution status, and returns
raw variable value arrays.

Returns a SolveResult dataclass with:
  - status          : str  ("optimal" | "feasible" | "infeasible" | ...)
  - termination     : str  (solver termination condition string)
  - obj_value       : float  (objective value)
  - var_values      : dict   (variable name → value or {index → value})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyomo.environ as pyo


@dataclass
class SolveResult:
    status:       str
    termination:  str
    obj_value:    float
    var_values:   dict[str, Any] = field(default_factory=dict)


def solve_model(
    model: pyo.ConcreteModel,
    params: dict[str, Any],
) -> SolveResult:
    """
    Solve model using the solver configured in params.

    Parameters
    ----------
    model   : built ConcreteModel from model_builder.build_model()
    params  : parameter dict from parameter_builder.build_parameters()

    Returns
    -------
    SolveResult

    Raises
    ------
    RuntimeError if solver returns infeasible or error status.
    """
    solver_name     = params.get("pyomo_solver", "highs")
    time_limit      = params.get("time_limit_seconds", 7200)
    mip_gap         = params.get("mip_gap", 0.005)

    # Resolve solver factory name
    if solver_name.lower() in ("highs", "appsi_highs"):
        factory_name = "appsi_highs"
    elif solver_name.lower() == "cbc":
        factory_name = "cbc"
    elif solver_name.lower() == "glpk":
        factory_name = "glpk"
    else:
        factory_name = solver_name

    solver = pyo.SolverFactory(factory_name)
    if not solver.available():
        raise RuntimeError(
            f"Solver '{factory_name}' is not available. "
            "Install highspy (pip install highspy) for HiGHS support."
        )

    # Set solver options
    if factory_name == "appsi_highs":
        solver.options["time_limit"]     = time_limit
        solver.options["mip_rel_gap"]    = mip_gap
        solver.options["log_to_console"] = False
    elif factory_name == "cbc":
        solver.options["seconds"]  = time_limit
        solver.options["ratio"]    = mip_gap
    elif factory_name == "glpk":
        solver.options["tmlim"]    = time_limit
        solver.options["mipgap"]   = mip_gap

    result = solver.solve(model, tee=False, load_solutions=True)

    status      = str(result.solver.status).lower()
    termination = str(result.solver.termination_condition).lower()

    if termination in ("infeasible", "infeasibleorunbounded"):
        raise RuntimeError(
            f"Solver returned {termination}. "
            "The model may be over-constrained (e.g. CUF maintenance infeasible "
            "without augmentation). Check solver.yaml pyomo.cuf_maintenance_enabled."
        )
    if termination == "error":
        raise RuntimeError(f"Solver error: status={status}, termination={termination}")

    obj_value = float(pyo.value(model.obj))

    var_values = _extract_var_values(model)

    return SolveResult(
        status=status,
        termination=termination,
        obj_value=obj_value,
        var_values=var_values,
    )


def _extract_var_values(model: pyo.ConcreteModel) -> dict[str, Any]:
    """Extract all variable values from a solved model into plain Python dicts."""
    out: dict[str, Any] = {}
    for vardata in model.component_objects(pyo.Var, active=True):
        name = vardata.name
        if vardata.is_indexed():
            out[name] = {idx: pyo.value(vardata[idx]) for idx in vardata}
        else:
            out[name] = pyo.value(vardata)
    return out
