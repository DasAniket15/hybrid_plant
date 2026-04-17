"""
oversize_optimizer.py
─────────────────────
Post-processing sweep to find the optimal number of upfront oversize BESS
containers after Pass-1 solver selects B* containers.

Role in the pipeline
────────────────────
Pass-1 selects B* containers to maximise Year-1 savings NPV with no
degradation model.  Because the lifecycle starts at exactly the trigger
threshold CUF (no headroom), SOH drop from Year 1 to Year 2 (~0.005)
immediately fires an augmentation event — which is not economically ideal.

This module sweeps extra = 0, 1, 2, … and evaluates:
  • initial_containers = B* + extra
  • restoration_target_cuf = Y1 CUF with B* + extra (always > threshold)
  • full 25-year lifecycle with minimum-k search and payback filter

The candidate that maximises savings_npv (strict improvement by more than
``tolerance`` Rs) is selected.  Tie-breaking favours lower extra (less
upfront spend for equivalent NPV) via strict ``>`` comparison.

Termination
───────────
Patience-based early stop: if ``patience`` consecutive candidates fail to
improve savings_npv by more than ``tolerance``, the sweep terminates.
A hard cap of ``max_extra_containers`` prevents runaway in degenerate cases.

Invariants
──────────
• extra is always a non-negative integer.
• params is never mutated — evaluate_scenario() receives initial_containers
  as a separate argument.
• All evaluations use fast_mode=False (full re-simulation); the sweep is a
  one-time post-processing step where accuracy beats speed.
• The sweep always evaluates extra=0 first (no oversize baseline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OversizeResult:
    """
    Output of a completed oversize sweep.

    Attributes
    ----------
    best_extra                : int   — number of extra containers above B*
    best_initial_containers   : int   — B* + best_extra (total initial cohort)
    best_result               : dict  — full evaluate_scenario() output for the winner
    sweep_log                 : list[dict] — one entry per candidate evaluated:
                                  {extra, initial_containers, npv, n_events,
                                   n_skipped, total_aug_cost_rs, terminated_reason}
    """
    best_extra:              int
    best_initial_containers: int
    best_result:             dict[str, Any]
    sweep_log:               list[dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_oversize(
    augmentation_engine:   Any,
    base_params:           dict[str, Any],
    threshold_cuf:         float,
    max_extra_containers:  int   = 500,
    patience:              int   = 3,
    tolerance:             float = 1e3,
) -> OversizeResult:
    """
    Sweep extra = 0, 1, 2, … to find the upfront container count that
    maximises lifecycle savings_npv for the given Pass-1 solution.

    Parameters
    ----------
    augmentation_engine   : AugmentationEngine — constructed with Pass-1
                            trigger_threshold_cuf and pass1_lcoe.
    base_params           : C* from Pass 1 (solver's best_params dict).
    threshold_cuf         : pre-oversize Y1 CUF (fixed reference; not used
                            internally but kept for traceability in sweep_log).
    max_extra_containers  : hard safety cap on extra; sweep stops if reached.
    patience              : consecutive non-improving steps before early stop.
    tolerance             : Rs; NPV improvement must exceed this to count as
                            "better".  Tie-breaking favours lower extra.

    Returns
    -------
    OversizeResult
    """
    base_containers = int(base_params["bess_containers"])
    sweep_log: list[dict[str, Any]] = []

    def _evaluate(extra: int) -> dict[str, Any]:
        """Run full evaluate_scenario for B* + extra initial containers."""
        initial = base_containers + extra
        return augmentation_engine.evaluate_scenario(
            params             = base_params,
            initial_containers = initial,
            fast_mode          = False,
        )

    def _log_entry(extra: int, result: dict, reason: str | None) -> dict:
        fi  = result["finance"]
        aug = fi.get("augmentation", {})
        return {
            "extra":              extra,
            "initial_containers": base_containers + extra,
            "npv":                fi["savings_npv"],
            "n_events":           aug.get("n_events", 0),
            "n_skipped":          aug.get("n_skipped", 0),
            "total_aug_cost_rs":  (aug.get("total_lump_cost_rs", 0.0)
                                   + aug.get("total_om_cost_rs", 0.0)),
            "terminated_reason":  reason,
        }

    # ── Evaluate extra = 0 (no oversizing) ───────────────────────────────────
    logger.info("Oversize sweep: evaluating extra=0 (baseline, no oversizing).")
    result0 = _evaluate(0)
    best_npv    = result0["finance"]["savings_npv"]
    best_extra  = 0
    best_result = result0
    consecutive_nonimproving = 0

    sweep_log.append(_log_entry(0, result0, None))
    logger.info(
        "Oversize sweep: extra=0  initial=%d  npv=%.2f Cr",
        base_containers, best_npv / 1e7,
    )

    # ── Sweep extra = 1, 2, … ─────────────────────────────────────────────────
    extra = 0
    while True:
        extra += 1

        if extra > max_extra_containers:
            logger.warning(
                "Oversize sweep: hit hard cap extra=%d — stopping.", extra
            )
            sweep_log[-1]["terminated_reason"] = f"hard_cap_{max_extra_containers}"
            break

        logger.info(
            "Oversize sweep: evaluating extra=%d (initial=%d) …",
            extra, base_containers + extra,
        )
        result = _evaluate(extra)
        npv    = result["finance"]["savings_npv"]

        if npv > best_npv + tolerance:
            # Genuine improvement — update best and reset patience counter
            best_npv    = npv
            best_extra  = extra
            best_result = result
            consecutive_nonimproving = 0
            terminated_reason = None
            logger.info(
                "Oversize sweep: extra=%d  npv=%.2f Cr  ← NEW BEST",
                extra, npv / 1e7,
            )
        else:
            consecutive_nonimproving += 1
            logger.info(
                "Oversize sweep: extra=%d  npv=%.2f Cr  "
                "(non-improving %d/%d)",
                extra, npv / 1e7, consecutive_nonimproving, patience,
            )
            if consecutive_nonimproving >= patience:
                terminated_reason = f"patience_{patience}"
                sweep_log.append(_log_entry(extra, result, terminated_reason))
                logger.info(
                    "Oversize sweep: stopping — patience=%d exhausted at extra=%d.",
                    patience, extra,
                )
                break
            terminated_reason = None

        sweep_log.append(_log_entry(extra, result, terminated_reason))

    logger.info(
        "Oversize sweep complete: best_extra=%d  best_initial=%d  best_npv=%.2f Cr  "
        "candidates_evaluated=%d",
        best_extra, base_containers + best_extra, best_npv / 1e7, len(sweep_log),
    )

    return OversizeResult(
        best_extra              = best_extra,
        best_initial_containers = base_containers + best_extra,
        best_result             = best_result,
        sweep_log               = sweep_log,
    )