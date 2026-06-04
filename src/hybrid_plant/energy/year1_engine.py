"""
year1_engine.py
───────────────
Top-level energy engine for Year-1 simulation.

Orchestrates three layers in sequence:
    PlantEngine   → hourly busbar dispatch (pre-loss)
    GridInterface → applies blended HT/LT loss factor
    MeterLayer    → computes DISCOM shortfall at the meter
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.energy.grid_interface import GridInterface
from hybrid_plant.energy.meter_layer import MeterLayer
from hybrid_plant.energy.plant_engine import PlantEngine


def _build_penalty_mask(config: FullConfig, n_hours: int = 8760) -> np.ndarray:
    """
    Build an n_hours-length boolean mask for RE penetration penalty hours.

    True  → hour is subject to the penalty.
    False → hour is exempt (shortfall incurs no cost).

    ``penalty_hours`` in solver.yaml uses 1-indexed hours (1 = midnight–1 am,
    24 = 11 pm–midnight), matching the tariffs.yaml convention.
    An empty list means all hours are penalised.
    """
    hrep          = (
        config.solver["solver"]
        .get("constraints", {})
        .get("hourly_re_penetration_penalty", {})
    )
    penalty_hours = hrep.get("penalty_hours", [])

    if not penalty_hours:
        return np.ones(n_hours, dtype=bool)

    mask_24 = np.zeros(24, dtype=bool)
    for h1 in penalty_hours:
        mask_24[(int(h1) - 1) % 24] = True

    n_days    = n_hours // 24
    remainder = n_hours % 24
    return np.concatenate([np.tile(mask_24, n_days), mask_24[:remainder]])


def _build_hourly_discom_tariff(config: FullConfig, n_hours: int = 8760) -> np.ndarray:
    """
    Build an n_hours-length blended DISCOM ToD tariff array (INR/kWh).

    Each element is the HT/LT-weighted tariff for that hour's ToD period:
        blended[h] = lt_fraction × lt_rate[period(h)]
                   + ht_fraction × ht_rate[period(h)]

    The 24-hour pattern is tiled to fill n_hours.
    Hours in tariffs.yaml are 1-indexed (1 = midnight–1 am, 24 = 11 pm–midnight).
    """
    ht_frac   = config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"] / 100.0
    lt_frac   = 1.0 - ht_frac
    lt_perds  = config.tariffs["discom"]["lt"]["tod_periods"]
    ht_perds  = config.tariffs["discom"]["ht"]["tod_periods"]

    # Map 1-indexed hour → blended tariff (INR/kWh)
    h_to_rate: dict[int, float] = {}
    for pp in lt_perds.values():
        lt_rate = pp["rate_inr_per_kwh"]
        for h1 in pp["hours"]:
            ht_rate = next(
                (hp["rate_inr_per_kwh"] for hp in ht_perds.values() if h1 in hp["hours"]),
                0.0,
            )
            h_to_rate[h1] = lt_frac * lt_rate + ht_frac * ht_rate

    tariff_24   = np.array([h_to_rate.get(h1, 0.0) for h1 in range(1, 25)])
    n_days      = n_hours // 24
    remainder   = n_hours % 24
    return np.concatenate([np.tile(tariff_24, n_days), tariff_24[:remainder]])


class Year1Engine:
    """
    Runs a full Year-1 hourly simulation and returns a unified result dict
    combining plant, grid, and meter outputs.

    Parameters
    ----------
    config : FullConfig
    data   : dict   — loaded by ``data_loader.load_timeseries_data``
    """

    def __init__(self, config: FullConfig, data: dict[str, Any]) -> None:
        self.plant = PlantEngine(config, data)
        self.grid  = GridInterface(config)
        self.meter = MeterLayer(data)

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        """
        Run plant → grid → meter pipeline for a given set of decision variables.

        All ``kwargs`` are forwarded verbatim to ``PlantEngine.simulate``
        (solar_capacity_mw, wind_capacity_mw, bess_containers, …).
        The ``loss_factor`` is injected automatically from ``GridInterface``.

        Returns
        -------
        dict
            Merged result dict from all three layers.
        """
        plant_results = self.plant.simulate(loss_factor=self.grid.loss_factor, **kwargs)
        grid_results  = self.grid.apply_losses(plant_results["plant_export_pre"])
        meter_results = self.meter.compute_shortfall(grid_results["meter_delivery"])

        re_pen_shortfall_mwh, annual_re_pen_shortfall_mwh, annual_re_pen_cost_inr = (
            self._compute_re_pen_shortfall(grid_results["meter_delivery"])
        )

        # Store all simulation parameters so EnergyProjection can re-run
        # the full dispatch for each year without threading them separately.
        sim_params = {**kwargs, "loss_factor": self.grid.loss_factor}

        return {
            **plant_results,
            **grid_results,
            **meter_results,
            "re_pen_shortfall_mwh":        re_pen_shortfall_mwh,
            "annual_re_pen_shortfall_mwh": annual_re_pen_shortfall_mwh,
            "annual_re_pen_cost_inr":      annual_re_pen_cost_inr,
            "sim_params":                  sim_params,
        }

    def _compute_re_pen_shortfall(
        self, meter_delivery: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """
        Compute hourly RE penetration shortfall against the configured floor.

        Penalty rate per hour = blended ToD DISCOM tariff (HT/LT mix from
        regulatory.yaml) for the period that hour falls in.

        Returns
        -------
        shortfall_mwh    : np.ndarray  hourly shortfall at meter (MWh)
        annual_shortfall : float       annual sum of shortfall (MWh)
        annual_cost_inr  : float       annual penalty cost (INR)
                           = sum_h( shortfall_mwh[h] × 1000 × tod_tariff[h] )
        All three are zero when the constraint is disabled.
        """
        config = self.plant.config
        hrep   = (
            config.solver["solver"]
            .get("constraints", {})
            .get("hourly_re_penetration_penalty", {})
        )
        load = self.plant.data["load_profile"]

        if not hrep.get("enabled", False):
            zeros = np.zeros(len(load))
            return zeros, 0.0, 0.0

        min_pct        = hrep.get("min_percent", 0.0) / 100.0
        shortfall_raw  = np.maximum(load * min_pct - meter_delivery, 0.0)
        mask           = _build_penalty_mask(config, len(load))
        shortfall      = np.where(mask, shortfall_raw, 0.0)

        tariff_8760 = _build_hourly_discom_tariff(config, len(load))
        annual_cost = float(np.sum(shortfall * tariff_8760)) * 1_000.0  # MWh → kWh

        return shortfall, float(np.sum(shortfall)), annual_cost