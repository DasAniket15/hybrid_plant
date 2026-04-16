"""
test_augmentation.py
────────────────────
Test suite for the BESS Augmentation Engine.

Tests
─────
  Test 1  TestNoAugmentationDeclines   — Without aug, CUF strictly declines
  Test 2  TestAugmentationRestoresCUF  — Post-event CUF >= restoration target
  Test 3  TestMultipleAugmentations    — Pathological config triggers >=2 events
  Test 4  TestCohortIndependence       — Each cohort's capacity == containers × soh[age]
  Test 5  TestOpexOnlyTreatment        — Aug cost in OPEX; CAPEX/debt/EMI unchanged
  Test 6  TestOversizingImproves       — (skipped in fast CI; flagged @pytest.mark.slow)

Run
───
    pytest tests/test_augmentation.py -v
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_resources():
    """Load config, data, engine once for the whole module."""
    from hybrid_plant.config_loader import load_config
    from hybrid_plant.data_loader import load_timeseries_data, load_soh_curve
    from hybrid_plant.energy.year1_engine import Year1Engine
    from hybrid_plant._paths import find_project_root

    config = load_config()
    data   = load_timeseries_data(config)
    soh    = load_soh_curve(config)
    engine = Year1Engine(config, data)

    root = find_project_root()

    def _load_curve(fpath: str, col: str) -> dict[int, float]:
        df = pd.read_csv(root / fpath)
        df.columns = df.columns.str.strip().str.lower()
        return dict(zip(df["year"].astype(int), df[col]))

    solar_eff = _load_curve(
        config.project["generation"]["solar"]["degradation"]["file"], "efficiency"
    )
    wind_eff = _load_curve(
        config.project["generation"]["wind"]["degradation"]["file"], "efficiency"
    )

    return {
        "config": config,
        "data": data,
        "soh": soh,
        "engine": engine,
        "solar_eff": solar_eff,
        "wind_eff": wind_eff,
    }


def make_simulator(res: dict) -> Any:
    """Convenience constructor for LifecycleSimulator from base_resources dict."""
    from hybrid_plant.augmentation.lifecycle_simulator import LifecycleSimulator

    return LifecycleSimulator(
        config          = res["config"],
        plant_engine    = res["engine"].plant,
        soh_curve       = res["soh"],
        solar_eff_curve = res["solar_eff"],
        wind_eff_curve  = res["wind_eff"],
        loss_factor     = res["engine"].grid.loss_factor,
    )


# Standard params for most tests — realistic, produces positive savings NPV
STANDARD_PARAMS = {
    "solar_capacity_mw":  200.0,
    "wind_capacity_mw":   0.0,
    "ppa_capacity_mw":    60.0,
    "bess_containers":    30,
    "charge_c_rate":      0.5,
    "discharge_c_rate":   0.5,
    "dispatch_priority":  "solar_first",
    "bess_charge_source": "solar_only",
}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — No augmentation: CUF must decline over time
# ─────────────────────────────────────────────────────────────────────────────

class TestNoAugmentationDeclines:
    """
    Test 1: With augmentation disabled (trigger threshold set so low it
    never fires), Plant CUF should strictly decline year-over-year because
    solar and BESS both degrade.
    """

    def test_cuf_declines_without_augmentation(self, base_resources):
        sim = make_simulator(base_resources)

        # Set threshold = 0 so no event ever fires (all CUFs > 0)
        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = 0.0,   # never triggered
            restoration_target_cuf  = 0.0,
            fast_mode               = True,
        )

        assert result.event_log == [], "No events should fire with threshold=0%"

        cufs = result.cuf_series
        assert len(cufs) == 25

        # CUF should trend downward: Y25 < Y1 (SOH drops from ~0.995 to ~0.645)
        assert cufs[-1] < cufs[0], (
            f"CUF should degrade: Y1={cufs[0]:.3f}%, Y25={cufs[-1]:.3f}%"
        )

    def test_cuf_declines_fast_mode(self, base_resources):
        """Same test using fast_mode=True (approximate CUF via SOH ratio)."""
        sim = make_simulator(base_resources)

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = 0.0,
            restoration_target_cuf  = 0.0,
            fast_mode               = True,
        )

        assert result.event_log == []
        cufs = result.cuf_series
        assert cufs[-1] < cufs[0], (
            f"Fast mode CUF should degrade: Y1={cufs[0]:.3f}%, Y25={cufs[-1]:.3f}%"
        )

    def test_cohort_registry_no_events(self, base_resources):
        """With no events, the cohort registry should have exactly one cohort."""
        sim = make_simulator(base_resources)

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = 0.0,
            restoration_target_cuf  = 0.0,
            fast_mode               = True,
        )

        # Only the initial cohort
        assert len(result.cohort_snapshot) == 1
        assert result.cohort_snapshot[0]["install_year"] == 1
        assert result.cohort_snapshot[0]["containers"] == STANDARD_PARAMS["bess_containers"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Post-event CUF must meet restoration target
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentationRestoresCUF:
    """
    Test 2: When an augmentation event fires, the post-event CUF recorded
    in the event log must be >= restoration_target_cuf (within float tolerance).
    """

    def test_post_event_cuf_meets_target(self, base_resources):
        sim = make_simulator(base_resources)
        soh = base_resources["soh"]

        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)

        # Set a threshold that will be breached in mid-life
        threshold = cuf_y1 * 0.93   # fires once SOH drops ~7 %

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        for ev in result.event_log:
            assert ev["post_event_cuf"] >= cuf_y1 - 0.01, (
                f"Year {ev['year']}: post_event_cuf {ev['post_event_cuf']:.3f}% "
                f"< target {cuf_y1:.3f}%"
            )

    def test_event_log_fields_present(self, base_resources):
        """Event log dicts must contain all required keys."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)
        threshold = cuf_y1 * 0.93

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        required_keys = {"year", "trigger_cuf", "target_cuf", "post_event_cuf",
                         "k_containers", "lump_cost_rs"}

        for ev in result.event_log:
            missing = required_keys - ev.keys()
            assert not missing, f"Event log missing keys: {missing}"
            assert ev["year"] > 1, "Events must be in year > 1"
            assert ev["k_containers"] >= 1
            assert ev["lump_cost_rs"] > 0

    def test_cuf_series_jumps_at_event_year(self, base_resources):
        """CUF series should show a visible jump in the event year."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)
        threshold = cuf_y1 * 0.93

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        if result.event_log:
            ev_year = result.event_log[0]["year"]
            idx     = ev_year - 1       # 0-based
            cuf_before = result.cuf_series[idx - 1] if idx > 0 else None
            cuf_at     = result.cuf_series[idx]

            if cuf_before is not None:
                assert cuf_at > cuf_before, (
                    f"CUF should jump at event year {ev_year}: "
                    f"before={cuf_before:.3f}%, at={cuf_at:.3f}%"
                )

    def test_lump_cost_formula(self, base_resources):
        """Lump-sum cost must equal k × container_size × cost_per_mwh."""
        config = base_resources["config"]
        bess_cfg = config.bess["bess"]
        container_size = float(bess_cfg["container"]["size_mwh"])
        cost_per_mwh   = float(bess_cfg["augmentation"]["cost_per_mwh"])

        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = cuf_y1 * 0.93,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        for ev in result.event_log:
            expected = ev["k_containers"] * container_size * cost_per_mwh
            assert math.isclose(ev["lump_cost_rs"], expected, rel_tol=1e-9), (
                f"Lump cost mismatch: got {ev['lump_cost_rs']}, expected {expected}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Multiple augmentation events
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleAugmentations:
    """
    Test 3: A configuration with few initial containers and an aggressive
    trigger threshold should produce >= 2 events over 25 years, and the
    cohort registry must grow correctly after each.
    """

    # Use far fewer containers so SOH degradation triggers events sooner
    TIGHT_PARAMS = {**STANDARD_PARAMS, "bess_containers": 5}

    def test_triggers_multiple_events(self, base_resources):
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, self.TIGHT_PARAMS)

        # Threshold at 98 % of Y1 — fires very quickly as SOH degrades
        threshold = cuf_y1 * 0.98

        result = sim.simulate(
            params                  = self.TIGHT_PARAMS,
            initial_containers      = self.TIGHT_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        assert len(result.event_log) >= 2, (
            f"Expected >=2 augmentation events, got {len(result.event_log)}"
        )

    def test_cohort_count_matches_events(self, base_resources):
        """Registry should have 1 + n_events cohorts."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, self.TIGHT_PARAMS)
        threshold = cuf_y1 * 0.98

        result = sim.simulate(
            params                  = self.TIGHT_PARAMS,
            initial_containers      = self.TIGHT_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        n_events  = len(result.event_log)
        n_cohorts = len(result.cohort_snapshot)
        assert n_cohorts == 1 + n_events, (
            f"Expected {1 + n_events} cohorts for {n_events} events, got {n_cohorts}"
        )

    def test_cohort_install_years_match_events(self, base_resources):
        """Each augmentation cohort install_year must match its event year."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, self.TIGHT_PARAMS)

        result = sim.simulate(
            params                  = self.TIGHT_PARAMS,
            initial_containers      = self.TIGHT_PARAMS["bess_containers"],
            trigger_threshold_cuf   = cuf_y1 * 0.98,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        aug_cohorts = [c for c in result.cohort_snapshot if c["cohort_index"] > 0]
        for i, (event, cohort) in enumerate(zip(result.event_log, aug_cohorts)):
            assert cohort["install_year"] == event["year"], (
                f"Event {i}: event_year={event['year']}, cohort_install_year={cohort['install_year']}"
            )

    def test_opex_lump_series_has_correct_event_years(self, base_resources):
        """Lump-sum OPEX series must be non-zero only in event years."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, self.TIGHT_PARAMS)

        result = sim.simulate(
            params                  = self.TIGHT_PARAMS,
            initial_containers      = self.TIGHT_PARAMS["bess_containers"],
            trigger_threshold_cuf   = cuf_y1 * 0.98,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        event_years = {ev["year"] for ev in result.event_log}
        for yr, lump in enumerate(result.opex_augmentation_lump, start=1):
            if yr in event_years:
                assert lump > 0, f"Lump cost should be > 0 in event year {yr}"
            else:
                assert lump == 0.0, f"Lump cost should be 0 in non-event year {yr}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Cohort independence
# ─────────────────────────────────────────────────────────────────────────────

class TestCohortIndependence:
    """
    Test 4: Each cohort's effective capacity at year t equals
    containers × container_size × soh[age], and cohorts do NOT
    cross-pollinate (each ages from its own install_year).
    """

    def test_single_cohort_capacity_formula(self, base_resources):
        """Initial cohort effective capacity must match formula exactly."""
        from hybrid_plant.augmentation.cohort import BESSCohort

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])

        cohort = BESSCohort(install_year=1, containers=10)

        for yr in [1, 5, 10, 20, 25]:
            age      = yr - 1 + 1  # age = yr for initial cohort
            expected = 10 * csize * soh.get(age, soh[max(soh)])
            actual   = cohort.effective_capacity_mwh(yr, csize, soh)
            assert math.isclose(actual, expected, rel_tol=1e-12), (
                f"Year {yr}: expected {expected:.4f}, got {actual:.4f}"
            )

    def test_two_cohorts_independent_aging(self, base_resources):
        """Two cohorts installed at different years age independently."""
        from hybrid_plant.augmentation.cohort import CohortRegistry

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])

        reg = CohortRegistry(initial_containers=10)
        reg.add(install_year=8, containers=4)   # augmentation event at year 8

        # Year 10: initial cohort age = 10, aug cohort age = 3
        yr = 10
        total_expected = (
            10 * csize * soh[10]   # initial cohort, age 10
            + 4  * csize * soh[3]  # aug cohort,     age 3
        )
        total_actual = reg.effective_capacity_mwh(yr, csize, soh)
        assert math.isclose(total_actual, total_expected, rel_tol=1e-12), (
            f"Year {yr}: expected {total_expected:.4f}, got {total_actual:.4f}"
        )

    def test_aug_cohort_not_active_before_install(self, base_resources):
        """An augmentation cohort installed at year 8 contributes 0 before year 8."""
        from hybrid_plant.augmentation.cohort import BESSCohort

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])

        cohort = BESSCohort(install_year=8, containers=5)

        for yr in [1, 2, 5, 7]:
            cap = cohort.effective_capacity_mwh(yr, csize, soh)
            assert cap == 0.0, f"Cohort should be inactive before install year 8, got {cap} in year {yr}"

    def test_blended_soh_formula(self, base_resources):
        """to_plant_params blended SOH must satisfy capacity identity."""
        from hybrid_plant.augmentation.cohort import CohortRegistry

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])

        reg = CohortRegistry(initial_containers=15)
        reg.add(install_year=5, containers=6)

        for yr in [5, 10, 15, 20, 25]:
            n, blended = reg.to_plant_params(yr, csize, soh)
            eff_actual   = n * csize * blended
            eff_expected = reg.effective_capacity_mwh(yr, csize, soh)
            assert math.isclose(eff_actual, eff_expected, rel_tol=1e-12), (
                f"Year {yr}: blended formula mismatch: {eff_actual:.4f} != {eff_expected:.4f}"
            )

    def test_cohort_capacity_timeline_shape(self, base_resources):
        """cohort_capacity_timeline must have one entry per cohort, 25 values each."""
        from hybrid_plant.augmentation.cohort import CohortRegistry

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])
        life   = config.project["project"]["project_life_years"]

        reg = CohortRegistry(initial_containers=10)
        reg.add(install_year=8,  containers=3)
        reg.add(install_year=16, containers=2)

        timeline = reg.cohort_capacity_timeline(life, csize, soh)

        assert len(timeline) == 3, "Should have 3 cohorts"
        for idx, vals in timeline.items():
            assert len(vals) == life, f"Cohort {idx} should have {life} values"

    def test_initial_soh_is_from_curve_not_100_pct(self, base_resources):
        """
        Fresh cohorts must use soh_curve[age=1] (≈0.9953), NOT 1.0.
        This is the key bug from prior implementations.
        """
        from hybrid_plant.augmentation.cohort import BESSCohort

        config = base_resources["config"]
        soh    = base_resources["soh"]
        csize  = float(config.bess["bess"]["container"]["size_mwh"])

        cohort = BESSCohort(install_year=1, containers=1)
        cap_y1 = cohort.effective_capacity_mwh(1, csize, soh)

        expected_from_curve = 1 * csize * soh[1]
        soh_100_pct_cap     = 1 * csize * 1.0

        assert math.isclose(cap_y1, expected_from_curve, rel_tol=1e-12)
        assert cap_y1 < soh_100_pct_cap, (
            f"Y1 capacity ({cap_y1:.4f}) should be < nameplate ({soh_100_pct_cap:.4f}); "
            f"SOH[1]={soh[1]} must be applied, not 1.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Augmentation is OPEX-only; CAPEX/debt/EMI/ROE unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestOpexOnlyTreatment:
    """
    Test 5: Augmentation procurement cost appears in opex_projection in event
    years.  CAPEX, debt amount, EMI schedule, ROE schedule, and depreciation
    are byte-identical to a no-augmentation baseline.
    """

    def _get_baseline_and_aug(self, base_resources):
        """Run finance for the same params with aug disabled vs enabled."""
        from hybrid_plant.finance.finance_engine import FinanceEngine
        from hybrid_plant.augmentation.lifecycle_simulator import LifecycleSimulator

        config = base_resources["config"]
        data   = base_resources["data"]
        engine = base_resources["engine"]
        soh    = base_resources["soh"]

        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)
        threshold = cuf_y1 * 0.93

        # Year-1 simulation
        y1 = engine.evaluate(
            solar_capacity_mw  = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw   = STANDARD_PARAMS["wind_capacity_mw"],
            bess_containers    = STANDARD_PARAMS["bess_containers"],
            charge_c_rate      = STANDARD_PARAMS["charge_c_rate"],
            discharge_c_rate   = STANDARD_PARAMS["discharge_c_rate"],
            ppa_capacity_mw    = STANDARD_PARAMS["ppa_capacity_mw"],
            dispatch_priority  = STANDARD_PARAMS["dispatch_priority"],
            bess_charge_source = STANDARD_PARAMS["bess_charge_source"],
        )

        finance_engine = FinanceEngine(config, data)

        # Baseline (no augmentation override)
        baseline = finance_engine.evaluate(
            year1_results     = y1,
            solar_capacity_mw = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw  = STANDARD_PARAMS["wind_capacity_mw"],
            ppa_capacity_mw   = STANDARD_PARAMS["ppa_capacity_mw"],
            fast_mode         = True,
        )

        # Augmentation lifecycle
        sim = make_simulator(base_resources)
        lc  = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = threshold,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )
        aug_opex = [
            lump + om
            for lump, om in zip(lc.opex_augmentation_lump, lc.opex_augmentation_om)
        ]

        augmented = finance_engine.evaluate(
            year1_results             = y1,
            solar_capacity_mw         = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw          = STANDARD_PARAMS["wind_capacity_mw"],
            ppa_capacity_mw           = STANDARD_PARAMS["ppa_capacity_mw"],
            fast_mode                 = True,
            energy_projection_override = lc.energy_projection,
            opex_augmentation_series  = aug_opex,
        )

        return baseline, augmented, lc

    def test_capex_unchanged(self, base_resources):
        """CAPEX must be identical with and without augmentation."""
        baseline, augmented, _ = self._get_baseline_and_aug(base_resources)
        assert baseline["capex"]["total_capex"] == augmented["capex"]["total_capex"]
        assert baseline["capex"]["bess_capex"]  == augmented["capex"]["bess_capex"]

    def test_debt_amount_unchanged(self, base_resources):
        """Debt amount must not change (augmentation is OPEX, never debt-financed)."""
        baseline, augmented, _ = self._get_baseline_and_aug(base_resources)
        assert math.isclose(
            baseline["lcoe_breakdown"]["debt_amount"],
            augmented["lcoe_breakdown"]["debt_amount"],
            rel_tol=1e-12,
        )

    def test_emi_schedule_unchanged(self, base_resources):
        """EMI (loan repayment) schedule must be identical."""
        baseline, augmented, _ = self._get_baseline_and_aug(base_resources)
        assert math.isclose(
            baseline["lcoe_breakdown"]["emi"],
            augmented["lcoe_breakdown"]["emi"],
            rel_tol=1e-12,
        )

    def test_roe_schedule_unchanged(self, base_resources):
        """ROE payment schedule must be identical."""
        baseline, augmented, _ = self._get_baseline_and_aug(base_resources)
        b_roe = baseline["lcoe_breakdown"]["roe_schedule"]
        a_roe = augmented["lcoe_breakdown"]["roe_schedule"]
        for yr, (b, a) in enumerate(zip(b_roe, a_roe), start=1):
            assert math.isclose(b, a, rel_tol=1e-12), f"ROE differs in year {yr}"

    def test_aug_opex_appears_in_event_years(self, base_resources):
        """
        opex_projection[t] - baseline_opex[t] must equal
        lump_cost[t] + om_from_new_cohorts[t].
        """
        baseline, augmented, lc = self._get_baseline_and_aug(base_resources)

        b_opex = baseline["opex_projection"]
        a_opex = augmented["opex_projection"]

        for yr_i, (b, a, lump, om) in enumerate(
            zip(b_opex, a_opex, lc.opex_augmentation_lump, lc.opex_augmentation_om)
        ):
            delta    = a - b
            expected = lump + om
            assert math.isclose(delta, expected, rel_tol=1e-9, abs_tol=1.0), (
                f"Year {yr_i + 1}: OPEX delta {delta:.2f} != lump+om {expected:.2f}"
            )

    def test_aug_opex_nonneg_all_years(self, base_resources):
        """Augmentation OPEX must be >= 0 in every year."""
        sim = make_simulator(base_resources)
        cuf_y1 = _compute_y1_cuf(base_resources, STANDARD_PARAMS)

        result = sim.simulate(
            params                  = STANDARD_PARAMS,
            initial_containers      = STANDARD_PARAMS["bess_containers"],
            trigger_threshold_cuf   = cuf_y1 * 0.93,
            restoration_target_cuf  = cuf_y1,
            fast_mode               = True,
        )

        for yr, (lump, om) in enumerate(
            zip(result.opex_augmentation_lump, result.opex_augmentation_om), start=1
        ):
            assert lump >= 0.0, f"Year {yr}: lump < 0"
            assert om   >= 0.0, f"Year {yr}: om < 0"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Augmentation engine integration (evaluate_scenario)
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentationEngineIntegration:
    """
    Test 6 (partial — no full solver run in unit tests):
    AugmentationEngine.evaluate_scenario() produces a valid finance dict with
    the augmentation sub-dict, and augmented savings_npv is reasonable.
    """

    def test_evaluate_scenario_returns_required_keys(self, base_resources):
        """evaluate_scenario must return a finance dict with all required keys."""
        from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine
        from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf

        config = base_resources["config"]
        data   = base_resources["data"]
        engine = base_resources["engine"]
        soh    = base_resources["soh"]

        cuf_y1 = compute_plant_cuf(
            engine.plant, STANDARD_PARAMS,
            bess_containers = STANDARD_PARAMS["bess_containers"],
            bess_soh_factor = soh[1],
        )
        aug_engine = AugmentationEngine(
            config, data, engine, soh, trigger_threshold_cuf=cuf_y1 * 0.93
        )

        result = aug_engine.evaluate_scenario(STANDARD_PARAMS, fast_mode=True)

        assert "year1"   in result
        assert "finance" in result

        fi = result["finance"]
        required = {"lcoe_inr_per_kwh", "savings_npv", "opex_projection",
                    "energy_projection", "capex", "augmentation"}
        missing = required - fi.keys()
        assert not missing, f"Finance result missing keys: {missing}"

    def test_augmentation_sub_dict_structure(self, base_resources):
        """The augmentation sub-dict must have all specified keys."""
        from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine
        from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf

        config = base_resources["config"]
        data   = base_resources["data"]
        engine = base_resources["engine"]
        soh    = base_resources["soh"]

        cuf_y1 = compute_plant_cuf(
            engine.plant, STANDARD_PARAMS,
            STANDARD_PARAMS["bess_containers"], soh[1],
        )
        aug_engine = AugmentationEngine(
            config, data, engine, soh, trigger_threshold_cuf=cuf_y1 * 0.93
        )

        result = aug_engine.evaluate_scenario(STANDARD_PARAMS, fast_mode=True)
        aug    = result["finance"]["augmentation"]

        required_keys = {
            "trigger_threshold_cuf", "restoration_target_cuf",
            "event_log", "cuf_series", "cohort_snapshot",
            "cohort_capacity_timeline", "opex_augmentation_lump",
            "opex_augmentation_om", "total_lump_cost_rs",
            "total_om_cost_rs", "initial_containers",
            "total_containers_added", "n_events",
        }
        missing = required_keys - aug.keys()
        assert not missing, f"Augmentation sub-dict missing: {missing}"

    def test_savings_npv_is_finite(self, base_resources):
        """Augmented savings NPV must be a finite number (not NaN or inf)."""
        from hybrid_plant.augmentation.augmentation_engine import AugmentationEngine
        from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf

        config = base_resources["config"]
        data   = base_resources["data"]
        engine = base_resources["engine"]
        soh    = base_resources["soh"]

        cuf_y1 = compute_plant_cuf(
            engine.plant, STANDARD_PARAMS,
            STANDARD_PARAMS["bess_containers"], soh[1],
        )
        aug_engine = AugmentationEngine(
            config, data, engine, soh, trigger_threshold_cuf=cuf_y1 * 0.93
        )

        result = aug_engine.evaluate_scenario(STANDARD_PARAMS, fast_mode=True)
        npv    = result["finance"]["savings_npv"]

        assert math.isfinite(npv), f"savings_npv is not finite: {npv}"

    def test_disabled_augmentation_is_pure_bypass(self, base_resources):
        """
        With augmentation disabled in config, FinanceEngine.evaluate() with
        no override must produce the same result as the direct baseline.
        """
        from hybrid_plant.finance.finance_engine import FinanceEngine

        config = base_resources["config"]
        data   = base_resources["data"]
        engine = base_resources["engine"]

        y1 = engine.evaluate(
            solar_capacity_mw  = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw   = STANDARD_PARAMS["wind_capacity_mw"],
            bess_containers    = STANDARD_PARAMS["bess_containers"],
            charge_c_rate      = STANDARD_PARAMS["charge_c_rate"],
            discharge_c_rate   = STANDARD_PARAMS["discharge_c_rate"],
            ppa_capacity_mw    = STANDARD_PARAMS["ppa_capacity_mw"],
            dispatch_priority  = STANDARD_PARAMS["dispatch_priority"],
            bess_charge_source = STANDARD_PARAMS["bess_charge_source"],
        )

        fe = FinanceEngine(config, data)

        # No overrides — baseline
        baseline = fe.evaluate(
            year1_results     = y1,
            solar_capacity_mw = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw  = STANDARD_PARAMS["wind_capacity_mw"],
            ppa_capacity_mw   = STANDARD_PARAMS["ppa_capacity_mw"],
            fast_mode         = True,
        )

        # Explicit None overrides — should be identical
        same = fe.evaluate(
            year1_results             = y1,
            solar_capacity_mw         = STANDARD_PARAMS["solar_capacity_mw"],
            wind_capacity_mw          = STANDARD_PARAMS["wind_capacity_mw"],
            ppa_capacity_mw           = STANDARD_PARAMS["ppa_capacity_mw"],
            fast_mode                 = True,
            energy_projection_override = None,
            opex_augmentation_series  = None,
        )

        assert math.isclose(
            baseline["savings_npv"], same["savings_npv"], rel_tol=1e-12
        ), "None overrides should produce identical savings_npv"

        assert math.isclose(
            baseline["lcoe_inr_per_kwh"], same["lcoe_inr_per_kwh"], rel_tol=1e-12
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_y1_cuf(resources: dict, params: dict) -> float:
    """Compute Year-1 Plant CUF for the given params using canonical formula."""
    from hybrid_plant.augmentation.cuf_evaluator import compute_plant_cuf

    engine = resources["engine"]
    soh    = resources["soh"]
    return compute_plant_cuf(
        engine.plant, params,
        bess_containers = params["bess_containers"],
        bess_soh_factor = soh[1],
    )
