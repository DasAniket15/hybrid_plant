import numpy as np


class MeterLayer:

    def __init__(self, data):
        self.load = data["load_profile"]

    def compute_shortfall(self, meter_delivery):
        shortfall = np.maximum(self.load - meter_delivery, 0)

        return {
            "shortfall": shortfall,
            "annual_discom": np.sum(shortfall)
        }