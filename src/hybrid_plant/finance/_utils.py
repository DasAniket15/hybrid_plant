"""
finance/_utils.py
─────────────────
Shared financial utility functions used across finance sub-modules.
"""

from __future__ import annotations


def npv(series: list[float], rate: float) -> float:
    """
    Excel-style NPV: series[0] is Year 1, discounted at t = 1.

        NPV = Σ series[t] / (1 + rate)^(t+1)   for t = 0 … n-1
    """
    return sum(v / (1 + rate) ** (t + 1) for t, v in enumerate(series))
