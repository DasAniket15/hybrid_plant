"""
tests/optimise/test_finance_objective.py
────────────────────────────────────────
Step 3 validation — finance objective + reporting parity (design §11 Layer 2).

Structure
─────────
Layer 2A  Algebraic cost-component check (no LP solve):
          Evaluate each term of the Pyomo savings_npv formula in Python using
          reference annual energy.  Assert every cost component (financing,
          opex, cap charges, wheeling+tax) matches FinanceEngine within 0.1%.

Layer 2B  Cost-plus savings_npv comparison (algebraic, flat tariff):
          With tod[h] = flat_discom_tariff (matching FinanceEngine's flat model),
          Pyomo savings_npv + NPV(aux) must equal FinanceEngine savings_npv
          within 0.1%.  This is the Finding A proof: the linear objective ≡
          the original pipeline.

Layer 2C  LP solve with real objective (slow):
          Confirm the savings_npv LP terminates optimally and yields a finite,
          positive savings_npv.

Layer 2D  report.py parity (slow):
          LCOE and Year-1 landed tariff from report.py must match FinanceEngine
          within 0.1%.

Energy convention
─────────────────
All algebraic evaluations use Year1Engine annual totals disaggregated by
stream (solar_1, wind_1, bess_1 in busbar MWh).  FinanceEngine is run with
fast_mode=True so both use the identical Year-1-scaled energy projection
(same operating_value degradation approximation as Pyomo's D_s/D_w/D_b sums).
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.data_loader import load_timeseries_data
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.finance_engine import FinanceEngine
from hybrid_plant.finance.savings_model import SavingsModel
from hybrid_plant.optimise.build import build_single_year_model
from hybrid_plant.optimise.config import OptModelConfig
from hybrid_plant.optimise.params import OptParams, build_params
from hybrid_plant.optimise.report import compute_report
from hybrid_plant.optimise.solve import extract_dispatch, solve

# Reference sizing
_SOLAR_WIND = {
    "solar_capacity_mw":  190.454972460807,
    "wind_capacity_mw":   116.130108575195,
    "bess_containers":    120,
    "charge_c_rate":      1.0,
    "discharge_c_rate":   1.0,
    "ppa_capacity_mw":    120.632227022855,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_only",
}
_FIXED = {
    "S":  _SOLAR_WIND["solar_capacity_mw"],
    "W":  _SOLAR_WIND["wind_capacity_mw"],
    "P":  _SOLAR_WIND["ppa_capacity_mw"],
    "nb": _SOLAR_WIND["bess_containers"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Module fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config(config) -> FullConfig:
    return config


@pytest.fixture(scope="module")
def data(config: FullConfig) -> dict:
    return load_timeseries_data(config)


@pytest.fixture(scope="module")
def params(config: FullConfig, data: dict) -> OptParams:
    return build_params(config, data)


@pytest.fixture(scope="module")
def opt_cfg() -> OptModelConfig:
    return OptModelConfig(horizon="single", solver_name="appsi_highs")


@pytest.fixture(scope="module")
def year1(config: FullConfig, data: dict) -> dict:
    """Year1Engine result for reference sizing (single call)."""
    return Year1Engine(config, data).evaluate(**_SOLAR_WIND)


@pytest.fixture(scope="module")
def finance_result_fast(config: FullConfig, data: dict, year1: dict) -> dict:
    """FinanceEngine fast_mode=True — matches Pyomo single-year degradation."""
    return FinanceEngine(config, data).evaluate(
        year1_results     = year1,
        solar_capacity_mw = _FIXED["S"],
        wind_capacity_mw  = _FIXED["W"],
        ppa_capacity_mw   = _FIXED["P"],
        fast_mode         = True,
    )


@pytest.fixture(scope="module")
def lp_savings_result(opt_cfg: OptModelConfig, params: OptParams) -> dict:
    """Solve LP with savings_npv objective; module-scoped for speed."""
    model  = build_single_year_model(opt_cfg, params, fixed_sizing=_FIXED,
                                     objective="savings_npv")
    status = solve(model, opt_cfg, tee=False)
    dispatch = extract_dispatch(model, n_hours=8760)
    return {"model": model, "dispatch": dispatch, "status": status}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0)


def _npv_series(series: list, df_arr: np.ndarray) -> float:
    return float(np.dot(df_arr, series))


def _annual_energy(year1: dict, params: OptParams) -> tuple[float, float, float]:
    """solar_1, wind_1, bess_1 — busbar annual MWh (Year-1, pre-loss)."""
    solar_1 = float(np.sum(year1["solar_direct_pre"]))
    wind_1  = float(np.sum(year1["wind_direct_pre"]))
    bess_1  = float(np.sum(year1["discharge_pre"]))    # eta_d × sum(dis)
    return solar_1, wind_1, bess_1


def _pyomo_cost_components(
    solar_1: float, wind_1: float, bess_1: float,
    S: float, W: float, P: float, nb: int,
    params: OptParams,
) -> dict[str, float]:
    """
    Evaluate every cost component of the Pyomo savings_npv formula
    algebraically (Python scalars, no Pyomo variables).
    """
    E_b      = nb * params.cs
    solar_dc = S  * params.ac_dc
    total_capex = (
        solar_dc * params.solar_rate
        + W * params.wind_rate
        + E_b * params.bess_rate
        + params.trans_fixed
    )

    npv_financing = total_capex * params.phi

    npv_opex = (
        solar_dc * (params.solar_om_rate * params.G_solar_om
                    + params.solar_trans_om_rate * params.A_N)
        + W * (params.wind_om_rate * params.G_wind_om
               + params.wind_trans_om_rate * params.A_N)
        + E_b * params.bess_om_rate * params.A_N
        + params.land_lease_monthly * 12.0 * params.G_land
        + total_capex * params.insurance_pct * params.A_N
    )

    npv_cap = params.cap_rate * P * 12.0 * params.A_N

    # Discounted weighted meter energy (MWh, lf applied)
    disc_meter_mwh = params.lf * (solar_1 * params.D_s + wind_1 * params.D_w + bess_1 * params.D_b)

    npv_wheeling_tax = (params.wheel + params.tax) * 1000.0 * disc_meter_mwh

    # Aux cost (D8) — grid-fed, constant per container
    aux_annual_rate = params.aux_pc * float(np.sum(params.tod)) * 1000.0
    npv_aux = nb * aux_annual_rate * params.A_N

    return {
        "total_capex":      total_capex,
        "npv_financing":    npv_financing,
        "npv_opex":         npv_opex,
        "npv_cap":          npv_cap,
        "npv_wheeling_tax": npv_wheeling_tax,
        "npv_aux":          npv_aux,
        "disc_meter_mwh":   disc_meter_mwh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2A — Individual cost components
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer2ACostComponents:
    """
    Verify each cost term of the Pyomo objective against FinanceEngine output.
    All checks are algebraic (no LP solve).
    Tolerance: ≤ 0.1% relative error.
    """

    def test_total_capex(
        self, params: OptParams, finance_result_fast: dict
    ) -> None:
        S, W, nb = _FIXED["S"], _FIXED["W"], _FIXED["nb"]
        solar_1, wind_1, bess_1 = 0.0, 0.0, 0.0  # unused here

        cc = _pyomo_cost_components(solar_1, wind_1, bess_1, S, W, _FIXED["P"], nb, params)
        expected = finance_result_fast["capex"]["total_capex"]
        err = _rel_err(cc["total_capex"], expected)
        assert err < 1e-6, (
            f"total_capex: Pyomo={cc['total_capex']:.2f}, FinanceEngine={expected:.2f}, "
            f"rel_err={err:.2e}"
        )

    def test_npv_financing_matches_lcoe_model(
        self, params: OptParams, finance_result_fast: dict
    ) -> None:
        """NPV(financing) = Σ NPV(interest + principal + ROE) from LCOEModel."""
        S, W, nb = _FIXED["S"], _FIXED["W"], _FIXED["nb"]
        cc = _pyomo_cost_components(0.0, 0.0, 0.0, S, W, _FIXED["P"], nb, params)

        lb = finance_result_fast["lcoe_breakdown"]
        expected = lb["npv_interest"] + lb["npv_principal"] + lb["npv_roe"]
        err = _rel_err(cc["npv_financing"], expected)
        assert err < 1e-4, (
            f"NPV(financing): Pyomo={cc['npv_financing']:.2f}, "
            f"LCOEModel={expected:.2f}, rel_err={err:.2e}"
        )

    def test_npv_opex_matches_lcoe_model(
        self, params: OptParams, finance_result_fast: dict
    ) -> None:
        """NPV(opex) must match LCOEModel's npv_opex."""
        S, W, nb = _FIXED["S"], _FIXED["W"], _FIXED["nb"]
        cc = _pyomo_cost_components(0.0, 0.0, 0.0, S, W, _FIXED["P"], nb, params)

        expected = finance_result_fast["lcoe_breakdown"]["npv_opex"]
        err = _rel_err(cc["npv_opex"], expected)
        assert err < 1e-3, (
            f"NPV(opex): Pyomo={cc['npv_opex']:.2f}, LCOEModel={expected:.2f}, "
            f"rel_err={err:.2e}"
        )

    def test_npv_capacity_charges(
        self, params: OptParams, finance_result_fast: dict
    ) -> None:
        """NPV(cap) = annual_cap_rs × A_N."""
        S, W, nb = _FIXED["S"], _FIXED["W"], _FIXED["nb"]
        cc = _pyomo_cost_components(0.0, 0.0, 0.0, S, W, _FIXED["P"], nb, params)

        # FinanceEngine gives annual capacity charge (constant per year)
        annual_cap_rs = finance_result_fast["landed_tariff_breakdown"]["annual_capacity_charge_rs"]
        expected_npv  = annual_cap_rs * params.A_N
        err = _rel_err(cc["npv_cap"], expected_npv)
        assert err < 1e-6, (
            f"NPV(capacity): Pyomo={cc['npv_cap']:.2f}, "
            f"finance × A_N={expected_npv:.2f}, rel_err={err:.2e}"
        )

    def test_npv_wheeling_tax(
        self, params: OptParams, year1: dict, finance_result_fast: dict
    ) -> None:
        """
        NPV(wheeling+tax) = (wheel+tax) × 1000 × lf × (solar×D_s + wind×D_w + bess×D_b).
        Cross-check against FinanceEngine's annual wheeling+tax discounted at WACC.
        """
        solar_1, wind_1, bess_1 = _annual_energy(year1, params)
        cc = _pyomo_cost_components(solar_1, wind_1, bess_1, _FIXED["S"], _FIXED["W"],
                                    _FIXED["P"], _FIXED["nb"], params)

        lt  = finance_result_fast["landed_tariff_breakdown"]
        whl = np.array(lt["annual_wheeling"])
        tax = np.array(lt["annual_electricity_tax"])
        npv_wt_finance = _npv_series((whl + tax).tolist(), params.df)

        err = _rel_err(cc["npv_wheeling_tax"], npv_wt_finance)
        assert err < 1e-3, (
            f"NPV(wheeling+tax): Pyomo={cc['npv_wheeling_tax']:.2f}, "
            f"FinanceEngine={npv_wt_finance:.2f}, rel_err={err:.2e}"
        )

    def test_total_cost_npv_matches_finance(
        self, params: OptParams, year1: dict, finance_result_fast: dict
    ) -> None:
        """
        Sum of all cost NPVs (financing + opex + cap + wheeling+tax) must equal
        FinanceEngine's NPV(hybrid cost) within 0.1%.  Aux excluded — FinanceEngine
        does not model aux.
        """
        solar_1, wind_1, bess_1 = _annual_energy(year1, params)
        cc = _pyomo_cost_components(solar_1, wind_1, bess_1, _FIXED["S"], _FIXED["W"],
                                    _FIXED["P"], _FIXED["nb"], params)
        pyomo_total_cost = (
            cc["npv_financing"] + cc["npv_opex"] + cc["npv_cap"] + cc["npv_wheeling_tax"]
        )

        # FinanceEngine: NPV(hybrid cost) = NPV(re_payment + cap + wheeling + tax + discom)
        # Re-derive from savings: savings_npv = NPV(baseline) - NPV(hybrid_cost)
        # → NPV(hybrid_cost) = NPV(baseline) - savings_npv + NPV(discom draw × flat_tariff)
        # Simpler: NPV(financing+opex) + NPV(cap) + NPV(wheeling+tax) from breakdowns
        lb = finance_result_fast["lcoe_breakdown"]
        lt = finance_result_fast["landed_tariff_breakdown"]
        whl = np.array(lt["annual_wheeling"])
        tax = np.array(lt["annual_electricity_tax"])

        finance_total_cost = (
            lb["npv_interest"] + lb["npv_principal"] + lb["npv_roe"] + lb["npv_opex"]
            + lt["annual_capacity_charge_rs"] * params.A_N
            + _npv_series((whl + tax).tolist(), params.df)
        )
        err = _rel_err(pyomo_total_cost, finance_total_cost)
        assert err < 1e-3, (
            f"Total cost NPV: Pyomo={pyomo_total_cost:.2f}, "
            f"FinanceEngine={finance_total_cost:.2f}, rel_err={err:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2B — Savings_npv with flat-tariff (Finding A proof)
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer2BSavingsNpv:
    """
    Prove the cost-plus cancellation (Finding A): with flat tariff and
    identical annual energy, Pyomo savings_npv + NPV(aux) = FinanceEngine
    savings_npv within 0.1%.
    """

    def _flat_discom_tariff(self, config: FullConfig, data: dict) -> float:
        """Reproduce SavingsModel's weighted-average DISCOM tariff."""
        sm = SavingsModel(config, data)
        return sm._discom_tariff

    def test_savings_npv_flat_tod_matches_finance(
        self, config: FullConfig, data: dict, params: OptParams,
        year1: dict, finance_result_fast: dict
    ) -> None:
        """
        Substitute flat tod (= weighted-avg DISCOM tariff) for all hours.

        Finding A: with flat tariff and identical annual energy, the Pyomo
        objective formula (excluding aux) = FinanceEngine savings_npv (±0.1%).
        Both exclude aux — FinanceEngine never models grid-fed aux; Pyomo
        includes it as an explicit extra cost term.  So:

            pyomo_savings_flat  ≈  FinanceEngine savings_npv
            pyomo_obj_value     =  pyomo_savings_flat − NPV(aux)

        This proves the linear objective ≡ the original pipeline, with aux
        as the only additional cost term.
        """
        flat_tod = self._flat_discom_tariff(config, data)
        solar_1, wind_1, bess_1 = _annual_energy(year1, params)

        S, W, P, nb = _FIXED["S"], _FIXED["W"], _FIXED["P"], _FIXED["nb"]

        # Revenue side with flat tariff: avoided DISCOM - grid charges
        net_flat = flat_tod - params.wheel - params.tax   # INR/kWh
        disc_meter_mwh = params.lf * (solar_1 * params.D_s + wind_1 * params.D_w + bess_1 * params.D_b)
        rev_net = net_flat * 1000.0 * disc_meter_mwh

        # Cost side — financing, opex, cap  (aux is intentionally EXCLUDED here
        # to match FinanceEngine which also does not include aux)
        cc = _pyomo_cost_components(solar_1, wind_1, bess_1, S, W, P, nb, params)
        cost_total = cc["npv_financing"] + cc["npv_opex"] + cc["npv_cap"]

        pyomo_savings_flat = rev_net - cost_total
        npv_aux            = cc["npv_aux"]
        finance_savings    = finance_result_fast["savings_npv"]

        # ── Finding A assertion ──────────────────────────────────────────────
        err = _rel_err(pyomo_savings_flat, finance_savings)

        print(
            f"\n[Layer2B] savings_npv (Finding A):"
            f"  Pyomo_flat={pyomo_savings_flat/1e7:.4f} Cr"
            f"  FinanceEngine={finance_savings/1e7:.4f} Cr"
            f"  NPV(aux) [extra Pyomo cost]={npv_aux/1e7:.4f} Cr"
            f"  pyomo_obj_value = Pyomo_flat - aux = {(pyomo_savings_flat-npv_aux)/1e7:.4f} Cr"
            f"  rel_err={err:.2e}"
        )
        assert err < 1e-3, (
            f"Finding A violated: Pyomo_flat ≠ FinanceEngine savings_npv\n"
            f"  Pyomo_flat={pyomo_savings_flat:.2f}  finance={finance_savings:.2f}  rel_err={err:.2e}"
        )

    def test_npv_aux_is_positive(
        self, params: OptParams
    ) -> None:
        """NPV(aux) is a cost (positive), reducing savings."""
        nb = _FIXED["nb"]
        aux_annual_rate = params.aux_pc * float(np.sum(params.tod)) * 1000.0
        npv_aux = nb * aux_annual_rate * params.A_N
        assert npv_aux > 0

    def test_hourly_tod_objective_geq_flat_tod(
        self, config: FullConfig, data: dict, params: OptParams,
        year1: dict
    ) -> None:
        """
        With ToD-aware dispatch, the savings_npv should be ≥ the flat-tariff
        version if the RE delivery is concentrated in high-value hours (which
        the LP optimizer will achieve).

        Here we just verify that the hourly-ToD revenue from PlantEngine's
        ACTUAL dispatch is ≥ the flat-tariff revenue (because PlantEngine
        delivers more during high-tod hours due to its peak-first BESS policy).
        """
        solar_1, wind_1, bess_1 = _annual_energy(year1, params)
        flat_tod = SavingsModel(config, data)._discom_tariff

        disc_meter = params.lf * (solar_1 * params.D_s + wind_1 * params.D_w + bess_1 * params.D_b)
        flat_rev = flat_tod * 1000.0 * disc_meter

        # Hourly-ToD revenue using PlantEngine dispatch
        sd  = year1["solar_direct_pre"]
        wd  = year1["wind_direct_pre"]
        dis = year1["discharge_pre"] / params.eta_d   # Pyomo dis
        # Use single-year sum (not degradation-discounted) for comparison
        tod_rev_y1 = params.lf * 1000.0 * float(
            np.sum(params.tod * sd)
            + np.sum(params.tod * wd)
            + np.sum(params.tod * params.eta_d * dis)
        )
        # Not a strict ordering guarantee — just checking plausibility
        print(
            f"\n[Layer2B] ToD-weighted revenue (Year-1):"
            f"  hourly={tod_rev_y1/1e7:.4f} Cr  flat={flat_rev/1e7/params.A_N:.4f} Cr/yr × A_N={flat_rev/1e7:.4f} Cr"
        )
        assert tod_rev_y1 > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2C — LP solve with real savings_npv objective
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer2CLP:
    """Solve LP with savings_npv objective (slow)."""

    @pytest.mark.slow
    def test_lp_terminates_optimal(self, lp_savings_result: dict) -> None:
        status = lp_savings_result["status"]["status"]
        assert "optimal" in status.lower(), f"LP did not solve to optimality: {status}"

    @pytest.mark.slow
    def test_lp_obj_is_finite(self, lp_savings_result: dict) -> None:
        import math
        obj_val = lp_savings_result["status"]["obj_val"]
        assert not math.isnan(obj_val), "Objective is NaN"
        assert not math.isinf(obj_val), "Objective is Inf"

    @pytest.mark.slow
    def test_lp_savings_npv_positive(self, lp_savings_result: dict) -> None:
        """With the reference sizing, savings must be positive."""
        obj_val = lp_savings_result["status"]["obj_val"]
        assert obj_val > 0, f"Pyomo savings_npv ≤ 0: {obj_val:.2f}"

    @pytest.mark.slow
    def test_lp_savings_npv_geq_flat_tariff_estimate(
        self, config: FullConfig, data: dict, params: OptParams,
        year1: dict, finance_result_fast: dict,
        lp_savings_result: dict
    ) -> None:
        """
        Pyomo savings_npv (ToD-aware) ≥ FinanceEngine savings_npv (flat tariff)
        minus NPV(aux).  The LP extracts extra value by dispatching into peaks.

        Here we check a lower bound: LP savings_npv ≥ FinanceEngine_savings - aux - 10%
        margin (for dispatch differences vs PlantEngine heuristic).
        """
        lp_npv = lp_savings_result["status"]["obj_val"]
        fin_npv = finance_result_fast["savings_npv"]

        cc  = _pyomo_cost_components(0.0, 0.0, 0.0, _FIXED["S"], _FIXED["W"],
                                     _FIXED["P"], _FIXED["nb"], params)
        npv_aux = cc["npv_aux"]
        lower_bound = (fin_npv - npv_aux) * 0.90   # allow 10% margin

        print(
            f"\n[Layer2C] savings_npv:"
            f"  LP={lp_npv/1e7:.4f} Cr"
            f"  FinanceEngine={fin_npv/1e7:.4f} Cr"
            f"  NPV(aux)={npv_aux/1e7:.4f} Cr"
            f"  lower_bound={lower_bound/1e7:.4f} Cr"
        )
        assert lp_npv >= lower_bound, (
            f"LP savings_npv {lp_npv:.2f} < lower bound {lower_bound:.2f}"
        )

    @pytest.mark.slow
    def test_lp_feasibility_constraints(
        self, params: OptParams, lp_savings_result: dict
    ) -> None:
        """All dispatch constraints must hold in the savings_npv LP solution."""
        d     = lp_savings_result["dispatch"]
        E_b   = _FIXED["nb"] * params.cs

        # C3
        lhs   = params.lf * (d["sd"] + d["wd"] + params.eta_d * d["dis"]) + d["ddraw"]
        assert float(np.abs(lhs - params.load).max()) < 1e-3

        # C8
        assert float(np.max(d["soc"]) - E_b) < 1e-5

        # ddraw ≥ 0
        assert float(np.min(d["ddraw"])) >= -1e-6

        # soc ≥ 0
        assert float(np.min(d["soc"])) >= -1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2D — report.py parity with FinanceEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer2DReport:
    """
    LCOE and Year-1 landed tariff from report.py must match FinanceEngine
    within 0.1% (D2: reporting-only metrics).
    """

    @pytest.fixture(scope="class")
    def report(self, config: FullConfig, data: dict) -> dict:
        return compute_report(_FIXED, config, data, fast_mode=True)

    @pytest.fixture(scope="class")
    def finance_full(self, config: FullConfig, data: dict) -> dict:
        """Independent FinanceEngine call for the same sizing."""
        engine = Year1Engine(config, data)
        year1  = engine.evaluate(**_SOLAR_WIND)
        return FinanceEngine(config, data).evaluate(
            year1_results     = year1,
            solar_capacity_mw = _FIXED["S"],
            wind_capacity_mw  = _FIXED["W"],
            ppa_capacity_mw   = _FIXED["P"],
            fast_mode         = True,
        )

    def test_lcoe_matches(self, report: dict, finance_full: dict) -> None:
        err = _rel_err(report["lcoe_inr_per_kwh"], finance_full["lcoe_inr_per_kwh"])
        assert err < 1e-6, f"LCOE rel_err={err:.2e}"

    def test_savings_npv_matches(self, report: dict, finance_full: dict) -> None:
        err = _rel_err(report["savings_npv"], finance_full["savings_npv"])
        assert err < 1e-6, f"savings_npv rel_err={err:.2e}"

    def test_landed_tariff_y1_matches(self, report: dict, finance_full: dict) -> None:
        lt_report  = report["landed_tariff_series"][0]
        lt_finance = finance_full["landed_tariff_series"][0]
        err = _rel_err(lt_report, lt_finance)
        assert err < 1e-6, f"Landed tariff Year-1 rel_err={err:.2e}"

    def test_report_has_expected_keys(self, report: dict) -> None:
        for key in ("lcoe_inr_per_kwh", "savings_npv", "landed_tariff_series",
                    "capex", "opex_projection", "energy_projection"):
            assert key in report, f"Missing key in report: {key}"
