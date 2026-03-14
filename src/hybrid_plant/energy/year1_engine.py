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

from hybrid_plant.config_loader import FullConfig
from hybrid_plant.energy.grid_interface import GridInterface
from hybrid_plant.energy.meter_layer import MeterLayer
from hybrid_plant.energy.plant_engine import PlantEngine


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

        return {**plant_results, **grid_results, **meter_results}
