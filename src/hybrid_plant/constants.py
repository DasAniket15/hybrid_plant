"""
constants.py
────────────
Shared physical and financial unit-conversion constants.

Importing these rather than scattering magic numbers throughout the codebase
makes intent explicit and eliminates silent inconsistencies.
"""

from __future__ import annotations

# ── Unit conversions ──────────────────────────────────────────────────────────

MWH_TO_KWH: float = 1_000.0        # 1 MWh = 1,000 kWh
KWH_TO_MWH: float = 1.0 / 1_000.0

LAKH_TO_RS: float  = 1e5           # 1 Lakh  = 100,000 Rs
CRORE_TO_RS: float = 1e7           # 1 Crore = 10,000,000 Rs

PERCENT_TO_DECIMAL: float = 0.01   # divide percent values by 100

# ── Time ─────────────────────────────────────────────────────────────────────

HOURS_PER_DAY:  int = 24
HOURS_PER_YEAR: int = 8_760

# ── Months ───────────────────────────────────────────────────────────────────

MONTHS_PER_YEAR: int = 12
