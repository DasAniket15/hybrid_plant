from .plant_engine import PlantEngine
from .grid_interface import GridInterface
from .meter_layer import MeterLayer


class Year1Engine:

    def __init__(self, config, data):
        self.plant = PlantEngine(config, data)
        self.grid = GridInterface(config)
        self.meter = MeterLayer(data)

    def evaluate(self, **kwargs):

        plant_results = self.plant.simulate(
            loss_factor=self.grid.loss_factor,
            **kwargs
        )

        grid_results = self.grid.apply_losses(
            plant_results["plant_export_pre"]
        )

        meter_results = self.meter.compute_shortfall(
            grid_results["meter_delivery"]
        )

        return {
            **plant_results,
            **grid_results,
            **meter_results
        }