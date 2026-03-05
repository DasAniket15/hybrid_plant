import numpy as np


# -------------------------------------------------
# NPV
# -------------------------------------------------

def npv(rate, cashflows):
    """
    Compute NPV given discount rate and list of cashflows.
    rate should be decimal (e.g., 0.10 for 10%)
    """
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


# -------------------------------------------------
# IRR (Robust Bisection Method)
# -------------------------------------------------

def irr(cashflows, tol=1e-6, max_iter=1000):
    """
    Compute IRR using bisection method.
    Returns IRR as decimal.
    """

    low = -0.9999
    high = 1.0

    npv_low = npv(low, cashflows)
    npv_high = npv(high, cashflows)

    if npv_low * npv_high > 0:
        raise ValueError("IRR not bracketed. Cashflows may not have valid IRR.")

    for _ in range(max_iter):
        mid = (low + high) / 2
        npv_mid = npv(mid, cashflows)

        if abs(npv_mid) < tol:
            return mid

        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    raise RuntimeError("IRR calculation did not converge.")


# -------------------------------------------------
# Root Finder for Tariff
# -------------------------------------------------

def solve_for_tariff(target_irr, irr_function, low=0.0, high=20.0, tol=1e-6, max_iter=100):
    """
    Solve for tariff such that irr_function(tariff) = target_irr
    irr_function must return equity IRR (decimal).
    """

    for _ in range(max_iter):
        mid = (low + high) / 2
        irr_mid = irr_function(mid)

        if abs(irr_mid - target_irr) < tol:
            return mid

        if irr_mid < target_irr:
            low = mid
        else:
            high = mid

    raise RuntimeError("Tariff solver did not converge.")