import numpy as np


class GridInterface:

    def __init__(self, config):
        reg = config.regulatory["regulatory"]

        split = reg["connection"]["ht_lt_split_percent"] / 100

        ht = reg["losses"]["ht_side"]
        lt = reg["losses"]["lt_side"]

        self.loss_factor = (
            split * (1 - ht["ctu_percent"]/100)
                    * (1 - ht["stu_percent"]/100)
                    * (1 - ht["wheeling_percent"]/100)
            +
            (1 - split) * (1 - lt["ctu_percent"]/100)
                        * (1 - lt["stu_percent"]/100)
                        * (1 - lt["wheeling_percent"]/100)
        )

    def apply_losses(self, plant_export_pre):
        meter = plant_export_pre * self.loss_factor

        return {
            "meter_delivery": meter,
            "annual_meter_delivery": np.sum(meter),
            "loss_factor": self.loss_factor
        }