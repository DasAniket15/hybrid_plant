"""
lifecycle.py
────────────
Cohort-aware 25-year plant + finance simulator.

Runs the standard hourly dispatch simulation once per project year with
per-year degraded capacities, but — unlike the single-battery
``EnergyProjection`` — computes the BESS effective capacity from an
evolving list of independently-aging cohorts.

Augmentation events are optionally injected during the per-year loop,
either by a CUF trigger (see ``sizing.find_augmentation_size``) or by a
fixed user-provided schedule. Each augmentation installs a new cohort and
adds a one-time OPEX charge in its installation year.

Integration with the existing finance pipeline
──────────────────────────────────────────────
The lifecycle simulator produces the same ``energy_projection`` dict that
``EnergyProjection.project()`` returns, so it slots into the downstream
LCOE → landed-tariff → savings pipeline without any changes to those
models. CAPEX is computed from the INITIAL BESS size only (augmentation
cost is OPEX, not CAPEX — a hard rule of this design).

Augmentation OPEX injection
───────────────────────────
    OPEX_rs[year − 1] += augmentation_size_mwh × cost_per_mwh

The cost per MWh is read from ``bess.yaml → bess.augmentation.cost_per_mwh``.
The raw augmentation CAPEX-equivalent is treated as a single-year OPEX hit
(never depreciated, never added to asset base).

Plant CUF definition
────────────────────
We reuse the existing ``compute_cuf`` from ``run_model`` verbatim and apply
it to the busbar plant delivery against the PPA contracted capacity:

    plant_cuf = busbar_delivered_mwh / (ppa_capacity_mw × 8760) × 100

This mirrors the plant CUF reported in the existing dashboard and is
sensitive to BESS degradation (degraded BESS → less discharge → lower
busbar delivery).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from hybrid_plant._paths import find_project_root
from hybrid_plant.augmentation.cohort import CohortManager
from hybrid_plant.config_loader import FullConfig
from hybrid_plant.constants import HOURS_PER_YEAR
from hybrid_plant.energy.year1_engine import Year1Engine
from hybrid_plant.finance.capex_model import CapexModel
from hybrid_plant.finance.landed_tariff_model import LandedTariffModel
from hybrid_plant.finance.lcoe_model import LCOEModel
from hybrid_plant.finance.opex_model import OpexModel
from hybrid_plant.finance.savings_model import SavingsModel


# ─────────────────────────────────────────────────────────────────────────────
# CUF (reuses the exact formula from run_model.compute_cuf — do not change)
# ─────────────────────────────────────────────────────────────────────────────

def plant_cuf_from_busbar(busbar_mwh: float, ppa_capacity_mw: float) -> float | None:
    """
    Plant CUF (%) = busbar_mwh / (ppa_capacity_mw × 8760) × 100.

    Returns None if ppa_capacity_mw ≤ 0, matching the existing compute_cuf
    semantics. Deliberately re-expressed here to avoid a circular import
    with run_model, but the formula is identical.
    """
    if ppa_capacity_mw <= 0:
        return None
    return busbar_mwh / (ppa_capacity_mw * HOURS_PER_YEAR) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# LifecycleSimulator
# ─────────────────────────────────────────────────────────────────────────────

class LifecycleSimulator:
    """
    Cohort-aware 25-year simulation + finance evaluation.

    The simulator reuses ``PlantEngine.simulate`` (via the shared
    ``Year1Engine`` instance) and the standard finance sub-models so that
    the augmentation pipeline produces results directly comparable with the
    base optimiser's output.

    Parameters
    ----------
    config       : FullConfig
    data         : dict   — output of ``load_timeseries_data``
    year1_engine : Year1Engine  — reused for its PlantEngine and grid loss factor
    """

    def __init__(
        self,
        config:       FullConfig,
        data:         dict[str, Any],
        year1_engine: Year1Engine,
    ) -> None:
        self._config = config
        self._data   = data
        self._year1  = year1_engine

        # ── BESS container & augmentation cost config ────────────────────────
        bess_cfg       = config.bess["bess"]
        aug_cfg        = bess_cfg.get("augmentation", {}) or {}

        self.container_size: float = float(bess_cfg["container"]["size_mwh"])
        self.aug_cost_per_mwh: float = float(aug_cfg.get("cost_per_mwh", 0.0))
        self.aug_min_containers: int = int(aug_cfg.get("minimum_augmentation_containers", 1))

        # ── Degradation curves (same files / format as EnergyProjection) ─────
        root = find_project_root()
        self._solar_eff = self._load_curve(
            root / config.project["generation"]["solar"]["degradation"]["file"],
            column="efficiency",
        )
        self._wind_eff = self._load_curve(
            root / config.project["generation"]["wind"]["degradation"]["file"],
            column="efficiency",
        )
        self._soh_curve = self._load_curve(
            root / bess_cfg["degradation"]["file"],
            column="soh",
        )

        self.project_life: int = int(config.project["project"]["project_life_years"])

        # ── Finance sub-models (own instances so we don't reach into the
        # shared FinanceEngine's private attributes) ─────────────────────────
        self._capex_model   = CapexModel(config)
        self._opex_model    = OpexModel(config)
        self._lcoe_model    = LCOEModel(config)
        self._landed_model  = LandedTariffModel(config)
        self._savings_model = SavingsModel(config, data)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_curve(path: Path, column: str) -> dict[int, float]:
        """Load a degradation CSV into a {year: value} dict."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        if "year" not in df.columns:
            raise ValueError(f"'year' column not found in {path}")
        col = column.lower()
        if col not in df.columns:
            raise ValueError(f"'{column}' column not found in {path}")
        return dict(zip(df["year"].astype(int), df[col]))

    # ─────────────────────────────────────────────────────────────────────────

    def make_cohort_manager(self, initial_containers: int) -> CohortManager:
        """Construct a CohortManager seeded with the initial cohort."""
        cm = CohortManager(self.container_size, self._soh_curve)
        cm.add_initial(int(initial_containers))
        return cm

    def soh_curve(self) -> dict[int, float]:
        """Expose the SoH curve for reuse (e.g. by the sizing search)."""
        return dict(self._soh_curve)

    def solar_eff(self) -> dict[int, float]:
        return dict(self._solar_eff)

    def wind_eff(self) -> dict[int, float]:
        return dict(self._wind_eff)

    # ─────────────────────────────────────────────────────────────────────────

    def simulate_year(
        self,
        sim_params: dict[str, Any],
        year:       int,
        cm:         CohortManager,
    ) -> dict[str, Any]:
        """
        Run a full 8760-hour plant simulation for project year ``year`` with
        the current cohort-aggregated BESS state and per-year solar/wind
        degradation.

        Matches EnergyProjection._project_full's approach: it scales the
        nameplate AC/wind capacity by the degradation curve and passes an
        aggregate SoH factor to the PlantEngine. Other sim parameters
        (C-rates, PPA cap, dispatch rules) are held constant across years.
        """
        solar_eff = self._solar_eff.get(year, 1.0)
        wind_eff  = self._wind_eff.get(year, 1.0)

        total_containers = cm.total_containers(year)
        aggregate_soh    = cm.aggregate_soh_factor(year)

        # Edge case: if no containers active, pass 0 — PlantEngine handles
        # zero-BESS correctly (just no charge/discharge). We still pass the
        # aggregate SoH (which will be 0.0) so capacity resolves to 0.
        return self._year1.plant.simulate(
            solar_capacity_mw  = sim_params["solar_capacity_mw"] * solar_eff,
            wind_capacity_mw   = sim_params["wind_capacity_mw"]  * wind_eff,
            bess_containers    = total_containers,
            bess_soh_factor    = aggregate_soh,
            charge_c_rate      = sim_params["charge_c_rate"],
            discharge_c_rate   = sim_params["discharge_c_rate"],
            ppa_capacity_mw    = sim_params["ppa_capacity_mw"],
            dispatch_priority  = sim_params["dispatch_priority"],
            bess_charge_source = sim_params["bess_charge_source"],
            loss_factor        = sim_params["loss_factor"],
        )

    # ─────────────────────────────────────────────────────────────────────────

    def run_lifecycle(
        self,
        sim_params:          dict[str, Any],
        initial_containers:  int,
        threshold_cuf:       float | None = None,
        augmentation_years:  Iterable[int] | None = None,
        trigger_callback=None,
    ) -> dict[str, Any]:
        """
        Run a full 25-year cohort-aware simulation with optional augmentation.

        Parameters
        ----------
        sim_params : dict
            Plant dispatch parameters (same keys ``Year1Engine.sim_params``
            produces): solar_capacity_mw, wind_capacity_mw, charge_c_rate,
            discharge_c_rate, ppa_capacity_mw, dispatch_priority,
            bess_charge_source, loss_factor.
        initial_containers : int
            Number of containers installed at project start (may equal or
            exceed the base solver's choice — the oversizing lever).
        threshold_cuf : float or None
            If provided, a CUF trigger is checked each year. A CUF below
            threshold causes the ``trigger_callback`` (or the default
            ``find_augmentation_size``) to decide how many containers to
            install.
        augmentation_years : iterable[int] or None
            If provided, forces an augmentation in each listed year using
            the configured minimum container count. May coexist with
            ``threshold_cuf`` (both triggers can fire in the same year, but
            only one augmentation event is created — the larger of the two
            sizings wins).
        trigger_callback : callable or None
            Optional override for the sizing search. Signature:
                callback(cm, year, sim_params, threshold_cuf, simulator)
                    → containers_to_add (int)
            If None, the default CUF-restoring search is used.

        Returns
        -------
        dict — see the return block at the end of this method.
        """
        cm = self.make_cohort_manager(initial_containers)

        # Pre-allocate per-year outputs — mirrors EnergyProjection shape
        n = self.project_life
        solar_arr    = np.zeros(n)
        wind_arr     = np.zeros(n)
        battery_arr  = np.zeros(n)
        pre_arr      = np.zeros(n)
        meter_arr    = np.zeros(n)
        cuf_arr      = np.zeros(n)
        eff_mwh_arr  = np.zeros(n)
        containers_arr = np.zeros(n, dtype=int)
        aug_opex_arr = np.zeros(n)

        aug_events: list[dict[str, Any]] = []
        forced_years = set(int(y) for y in (augmentation_years or []))

        for i, year in enumerate(range(1, n + 1)):
            # 1. Cohort state (pre-augmentation for this year)
            containers_arr[i] = cm.total_containers(year)
            eff_mwh_arr[i]    = cm.effective_mwh(year)

            # 2. Plant sim for this year with current cohorts
            yr = self.simulate_year(sim_params, year, cm)
            s = float(np.sum(yr["solar_direct_pre"]))
            w = float(np.sum(yr["wind_direct_pre"]))
            b = float(np.sum(yr["discharge_pre"]))
            busbar = s + w + b
            meter  = busbar * sim_params["loss_factor"]

            solar_arr[i]   = s
            wind_arr[i]    = w
            battery_arr[i] = b
            pre_arr[i]     = busbar
            meter_arr[i]   = meter

            # 3. Pre-augmentation plant CUF (basis for trigger decision)
            cuf = plant_cuf_from_busbar(busbar, sim_params["ppa_capacity_mw"])
            cuf_arr[i] = cuf if cuf is not None else 0.0

            # 4. Trigger augmentation? Two pathways can fire:
            #    (a) CUF threshold trigger
            #    (b) fixed year trigger
            #    The augmented cohort becomes active the NEXT year (cohorts
            #    with installation_year == Y are active iff Y' > Y). So its
            #    effect restores CUF in year Y+1 onwards — which is the
            #    physical behaviour of a mid-year install.
            containers_to_add = 0
            feasible_flag:    bool = True  # True unless the CUF-trigger
                                            # search couldn't meet threshold

            cuf_trigger = (
                threshold_cuf is not None
                and cuf is not None
                and cuf < threshold_cuf
                and year < n  # augmenting in the final year has no forward effect
            )
            if cuf_trigger:
                if trigger_callback is not None:
                    cb_result = trigger_callback(
                        cm, year, sim_params, threshold_cuf, self
                    )
                    # Accept either int or (int, bool) for backward-compat
                    if isinstance(cb_result, tuple):
                        containers_to_add, feasible_flag = int(cb_result[0]), bool(cb_result[1])
                    else:
                        containers_to_add, feasible_flag = int(cb_result), True
                else:
                    # Lazy import to avoid circular dep at module load time
                    from hybrid_plant.augmentation.sizing import find_augmentation_size
                    containers_to_add, feasible_flag = find_augmentation_size(
                        cohort_manager   = cm,
                        trigger_year     = year,
                        sim_params       = sim_params,
                        threshold_cuf    = threshold_cuf,
                        simulator        = self,
                        min_containers   = self.aug_min_containers,
                    )

            if year in forced_years and year < n:
                # Fixed-mode request: at least the configured minimum.
                # If the CUF trigger already asked for more, keep the larger.
                # Fixed mode is considered "feasible" by construction — the
                # user explicitly asked for an install in this year.
                if containers_to_add < self.aug_min_containers:
                    containers_to_add = self.aug_min_containers
                    feasible_flag = True

            if containers_to_add > 0:
                mwh_added = containers_to_add * self.container_size
                cm.add_augmentation(installation_year=year, containers=containers_to_add)
                aug_opex = mwh_added * self.aug_cost_per_mwh
                aug_opex_arr[i] = aug_opex
                aug_events.append({
                    "year":            year,
                    "containers":      containers_to_add,
                    "mwh":             mwh_added,
                    "opex_rs":         aug_opex,
                    "pre_aug_cuf":     cuf,
                    "threshold_cuf":   threshold_cuf,
                    "feasible":        feasible_flag,
                })

        return {
            # Energy projection — same keys as EnergyProjection.project()
            "energy_projection": {
                "solar_direct_mwh":     solar_arr,
                "wind_direct_mwh":      wind_arr,
                "battery_mwh":          battery_arr,
                "delivered_pre_mwh":    pre_arr,
                "delivered_meter_mwh":  meter_arr,
            },
            # Lifecycle diagnostics
            "annual_cuf":                cuf_arr,
            "annual_effective_mwh":      eff_mwh_arr,
            "annual_total_containers":   containers_arr,
            "augmentation_events":       aug_events,
            "augmentation_opex_rs":      aug_opex_arr,
            "cohorts":                   cm.snapshot(),
            "cohort_manager":            cm,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def run_finance(
        self,
        sim_params:         dict[str, Any],
        initial_containers: int,
        lifecycle_result:   dict[str, Any],
    ) -> dict[str, Any]:
        """
        Feed the lifecycle energy projection + augmentation OPEX into the
        existing finance pipeline (CAPEX → OPEX → LCOE → landed tariff →
        savings). CAPEX uses the INITIAL BESS size only — augmentation
        containers do not add to the asset base, per the design rules.

        Returns a dict whose key layout matches ``FinanceEngine.evaluate()``
        so downstream reporting / tests can consume it uniformly. Two
        additional keys are added for augmentation diagnostics:
            augmentation_opex_projection_rs : ndarray (25,)
            augmentation_events             : list[dict]
        """
        # ── 1. CAPEX (INITIAL BESS ONLY — augmentations are OPEX) ──────────
        initial_bess_mwh = float(initial_containers) * self.container_size
        capex = self._capex_model.compute(
            sim_params["solar_capacity_mw"],
            sim_params["wind_capacity_mw"],
            initial_bess_mwh,
        )
        total_capex = capex["total_capex"]

        # ── 2. OPEX (standard) + augmentation OPEX injection ────────────────
        base_opex_projection, base_opex_breakdown = self._opex_model.compute(
            sim_params["solar_capacity_mw"],
            sim_params["wind_capacity_mw"],
            initial_bess_mwh,
            total_capex,
        )

        aug_opex_rs = lifecycle_result["augmentation_opex_rs"]
        opex_projection: list[float] = []
        opex_breakdown:  list[dict[str, Any]] = []
        for i, (base_val, base_row) in enumerate(zip(base_opex_projection, base_opex_breakdown)):
            aug_val = float(aug_opex_rs[i])
            total   = float(base_val) + aug_val
            row = dict(base_row)
            row["augmentation"] = aug_val
            row["total"]        = total
            opex_projection.append(total)
            opex_breakdown.append(row)

        # ── 3. Energy projection from lifecycle (not EnergyProjection) ─────
        projection = lifecycle_result["energy_projection"]
        busbar_mwh = projection["delivered_pre_mwh"]
        meter_mwh  = projection["delivered_meter_mwh"]

        # ── 4. LCOE ────────────────────────────────────────────────────────
        lcoe_result = self._lcoe_model.compute(
            total_capex, opex_projection, busbar_mwh,
        )
        lcoe = lcoe_result["lcoe_inr_per_kwh"]
        wacc = lcoe_result["wacc"]

        # ── 5. Landed tariff ───────────────────────────────────────────────
        landed_result = self._landed_model.compute(
            lcoe_inr_per_kwh             = lcoe,
            ppa_capacity_mw              = sim_params["ppa_capacity_mw"],
            busbar_energy_mwh_projection = busbar_mwh,
            meter_energy_mwh_projection  = meter_mwh,
        )

        # ── 6. Client savings ──────────────────────────────────────────────
        savings_result = self._savings_model.compute(
            landed_tariff_series        = landed_result["landed_tariff_series"],
            meter_energy_mwh_projection = meter_mwh,
            wacc                        = wacc,
        )

        return {
            # Primary outputs — same keys as FinanceEngine.evaluate()
            "lcoe_inr_per_kwh":        lcoe,
            "landed_tariff_series":    landed_result["landed_tariff_series"],
            "annual_savings_year1":    savings_result["annual_savings_year1"],
            "savings_npv":             savings_result["savings_npv"],
            # Supporting
            "wacc":                    wacc,
            "capex":                   capex,
            "opex_projection":         opex_projection,
            "opex_breakdown":          opex_breakdown,
            "energy_projection":       projection,
            "lcoe_breakdown":          lcoe_result,
            "landed_tariff_breakdown": landed_result,
            "savings_breakdown":       savings_result,
            # Augmentation additions
            "augmentation_opex_projection_rs": aug_opex_rs,
            "augmentation_events":             lifecycle_result["augmentation_events"],
        }
