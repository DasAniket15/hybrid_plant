"""
optimise/verify.py
──────────────────
Post-solve invariant checks (design §11 Layer 4).

The LP/MILP already *encodes* these as constraints, so on a correctly-solved
model every check passes to solver tolerance.  verify runs them again on the
extracted solution as an independent guard against:
  - a future refactor silently dropping or mis-indexing a constraint,
  - a degenerate optimum that satisfies the LP but violates an intent that is
    only implicit (D7 charge/discharge exclusivity — enforced by efficiency,
    not by a hard row),
  - the report-only ``minimum_savings_npv`` viability gate, which is deliberately
    NOT an LP row (see constraints/optional.py) and must be checked here.

Invariants
──────────
nonneg           sd, wd, chg, dis, soc, ddraw ≥ 0
soc_bounds       0 ≤ soc[t] ≤ E_b·deg_b[t]                      (C8)
charge_cap       chg[t] ≤ crc·E_b·deg_b[t]                      (C9)
discharge_cap    dis[t] ≤ crd·E_b·deg_b[t]                      (C10)
soc_dynamics     soc[t] = soc[t-1] + η_c·chg[t] − dis[t]        (C6, soc[-1]=0)
solar_alloc      sd[t] + chg[t] ≤ S·deg_s[t]·cuf_s[hour]        (C1)
wind_alloc       wd[t] ≤ W·deg_w[t]·cuf_w[hour]                 (C2)
ppa_cap          sd[t] + wd[t] + η_d·dis[t] ≤ P                 (C5)
load_balance     lf·(sd+wd+η_d·dis − nb·aux) + ddraw = load     (C3)
no_simultaneous  ¬(chg[t] > 0 ∧ dis[t] > 0)                     (D7, soft intent)
min_savings_npv  obj_val ≥ min_value            (report-only gate, if enabled)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyomo.environ as pyo

from hybrid_plant.optimise.params import OptParams
from hybrid_plant.optimise.sets import TimeContext
from hybrid_plant.optimise.solve import extract_dispatch


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InvariantResult:
    name:          str
    ok:            bool
    max_violation: float   # 0.0 when ok; else the largest breach magnitude
    worst_index:   int     # timestep of the worst breach (-1 if not indexed)
    detail:        str


@dataclass(frozen=True)
class VerifyReport:
    ok:      bool
    results: tuple[InvariantResult, ...]

    def failures(self) -> list[InvariantResult]:
        return [r for r in self.results if not r.ok]

    def assert_ok(self) -> None:
        """Raise AssertionError listing every failed invariant."""
        if self.ok:
            return
        lines = [f"  - {r.name}: {r.detail} (max_violation={r.max_violation:.3e}"
                 f"{'' if r.worst_index < 0 else f', t={r.worst_index}'})"
                 for r in self.failures()]
        raise AssertionError("Post-solve invariant check failed:\n" + "\n".join(lines))

    def __str__(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        rows = [f"[{'ok ' if r.ok else 'BAD'}] {r.name:<16} "
                f"max_violation={r.max_violation:.3e}"
                f"{'' if r.worst_index < 0 else f' @t={r.worst_index}'}"
                for r in self.results]
        return f"verify: {head}\n" + "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Violation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _worst(viol: np.ndarray) -> tuple[float, int]:
    """Largest positive violation and its index (0.0, -1 if none)."""
    if viol.size == 0:
        return 0.0, -1
    i = int(np.argmax(viol))
    v = float(viol[i])
    return (v, i) if v > 0.0 else (0.0, -1)


def _le(lhs: np.ndarray, rhs: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """Per-element amount by which lhs exceeds rhs beyond tolerance (≥ 0)."""
    slack = atol + rtol * np.abs(rhs)
    return np.maximum(lhs - rhs - slack, 0.0)


def _eq(lhs: np.ndarray, rhs: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    slack = atol + rtol * np.abs(rhs)
    return np.maximum(np.abs(lhs - rhs) - slack, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core check (pure — operates on arrays, testable with synthetic input)
# ─────────────────────────────────────────────────────────────────────────────

def check_invariants(
    dispatch: dict[str, np.ndarray],
    sizing:   dict[str, float],
    params:   OptParams,
    tc:       TimeContext,
    *,
    obj_val:  float | None = None,
    atol:     float = 1e-5,
    rtol:     float = 1e-6,
) -> VerifyReport:
    """
    Verify a solved dispatch against every model invariant.

    Parameters
    ----------
    dispatch : dict with keys sd, wd, chg, dis, soc, ddraw (np arrays, len n_steps)
    sizing   : dict with keys S, W, P, nb
    obj_val  : savings NPV in INR (for the minimum_savings_npv gate); the gate is
               skipped when None.
    """
    sd, wd  = dispatch["sd"], dispatch["wd"]
    chg, dis = dispatch["chg"], dispatch["dis"]
    soc, ddraw = dispatch["soc"], dispatch["ddraw"]
    # D5 charge split: wind portion of chg (zeros when charging is solar_only).
    chg_w = dispatch.get("chg_w", np.zeros_like(chg))

    S = float(sizing["S"]); W = float(sizing["W"])
    P = float(sizing["P"]); nb = float(sizing["nb"])
    E_b = nb * params.cs

    lf, eta_c, eta_d = params.lf, params.eta_c, params.eta_d
    crc, crd, aux    = params.crc, params.crd, params.aux_pc
    deg_b = tc.deg_b
    hour  = tc.hour_of
    cap   = E_b * deg_b                       # (n_steps,)

    results: list[InvariantResult] = []

    def add(name: str, viol: np.ndarray, detail: str) -> None:
        v, i = _worst(np.asarray(viol, dtype=np.float64))
        results.append(InvariantResult(name, v == 0.0, v, i if v > 0 else -1, detail))

    # nonneg
    neg = np.concatenate([np.maximum(-x - atol, 0.0)
                          for x in (sd, wd, chg, dis, soc, ddraw)])
    add("nonneg", neg, "a dispatch variable is negative")

    # soc bounds (C8) — lower 0, upper cap
    add("soc_bounds",
        np.maximum(_le(soc, cap, atol, rtol), np.maximum(-soc - atol, 0.0)),
        "soc outside [0, E_b*deg_b]")

    # power caps (C9/C10)
    add("charge_cap",    _le(chg, crc * cap, atol, rtol), "chg exceeds crc*E_b*deg_b")
    add("discharge_cap", _le(dis, crd * cap, atol, rtol), "dis exceeds crd*E_b*deg_b")

    # soc dynamics (C6), soc[-1] = 0
    soc_prev = np.empty_like(soc)
    soc_prev[0] = 0.0
    soc_prev[1:] = soc[:-1]
    add("soc_dynamics", _eq(soc, soc_prev + eta_c * chg - dis, atol, rtol),
        "soc recursion broken")

    # allocation (C1/C2); with the D5 split, solar carries chg-chg_w and wind
    # carries chg_w (chg_w = 0 for solar_only reduces to the base form).
    add("solar_alloc", _le(sd + chg - chg_w, S * tc.deg_s * params.cuf_s[hour], atol, rtol),
        "sd+(chg-chg_w) exceeds solar availability")
    add("wind_alloc", _le(wd + chg_w, W * tc.deg_w * params.cuf_w[hour], atol, rtol),
        "wd+chg_w exceeds wind availability")

    # ppa cap (C5)
    add("ppa_cap", _le(sd + wd + eta_d * dis, np.full_like(sd, P), atol, rtol),
        "export exceeds PPA cap P")

    # load balance (C3)
    lhs = lf * (sd + wd + eta_d * dis - nb * aux) + ddraw
    add("load_balance", _eq(lhs, params.load[hour], atol, rtol),
        "C3 load balance residual")

    # no simultaneous charge+discharge (D7) — soft intent, larger tol
    both = np.minimum(chg, dis)
    add("no_simultaneous", np.maximum(both - 1e-4, 0.0),
        "simultaneous charge and discharge")

    # minimum_savings_npv gate (report-only)
    cfg = params.opt_constraints
    if cfg.min_savings_npv_enabled and obj_val is not None:
        shortfall = max(cfg.min_savings_npv_value - obj_val, 0.0)
        results.append(InvariantResult(
            "min_savings_npv", shortfall <= 0.0, float(shortfall), -1,
            f"savings_npv {obj_val:.3e} < min {cfg.min_savings_npv_value:.3e}"))

    ok = all(r.ok for r in results)
    return VerifyReport(ok=ok, results=tuple(results))


# ─────────────────────────────────────────────────────────────────────────────
# Model-facing entry point
# ─────────────────────────────────────────────────────────────────────────────

def verify_solution(
    model:  pyo.ConcreteModel,
    params: OptParams,
    tc:     TimeContext,
    *,
    atol:   float = 1e-5,
    rtol:   float = 1e-6,
) -> VerifyReport:
    """
    Extract the solved solution from *model* and verify every invariant.

    The savings-NPV gate uses ``pyo.value(model.obj)`` un-scaled by
    ``model._obj_scale`` (INR), matching solve.py's ``obj_val``.
    """
    dispatch = extract_dispatch(model, n_hours=tc.n_steps)
    sizing = {
        "S":  float(pyo.value(model.S)),
        "W":  float(pyo.value(model.W)),
        "P":  float(pyo.value(model.P)),
        "nb": float(pyo.value(model.nb)),
    }
    obj_scale = getattr(model, "_obj_scale", 1.0)
    try:
        obj_val: float | None = float(pyo.value(model.obj)) / obj_scale
    except Exception:
        obj_val = None

    return check_invariants(
        dispatch, sizing, params, tc, obj_val=obj_val, atol=atol, rtol=rtol
    )
