"""
optimise/verify.py
──────────────────
Post-solve invariant assertions (§11 Layer 4):
  - No simultaneous charge+discharge (D7)
  - 0 ≤ SOC ≤ E_b
  - Per-hour energy conservation
  - export ≤ PPA cap
  - ddraw ≥ 0
Stub — implemented in Step 7.
"""
