import numpy as np
import pandas as pd


class EnergyProjection:
    """
    Projects annual energy across the full project lifetime using:
        - Year-1 dispatch results (solar direct, wind direct, BESS discharge)
        - Solar efficiency degradation curve
        - Wind efficiency degradation curve
        - BESS SOH degradation curve
        - Grid loss factor (passed in directly from GridInterface, no recomputation)

    Returns both:
        delivered_pre_mwh   : busbar energy (pre-loss)  — used in LCOE denominator
        delivered_meter_mwh : meter energy  (post-loss) — used in savings calculation
    """

    def __init__(
        self,
        config,
        data,
        year1_results,
        solar_capacity_mw,
        wind_capacity_mw,
        loss_factor,
    ):
        self.config      = config
        self.data        = data
        self.year1       = year1_results
        self.solar_cap   = solar_capacity_mw
        self.wind_cap    = wind_capacity_mw
        self.loss_factor = loss_factor

        self.project_life = config.project["project"]["project_life_years"]

        self.solar_eff_curve = self._load_curve(
            config.project["generation"]["solar"]["degradation"]["file"],
            column="efficiency",
        )
        self.wind_eff_curve = self._load_curve(
            config.project["generation"]["wind"]["degradation"]["file"],
            column="efficiency",
        )
        self.soh_curve = self._load_curve(
            config.bess["bess"]["degradation"]["file"],
            column="SOH",
        )

    # ------------------------------------------------------------------

    def _load_curve(self, path, column):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        if "year" not in df.columns:
            raise ValueError(f"'year' column not found in {path}")
        if column.lower() not in df.columns:
            raise ValueError(f"'{column}' column not found in {path}")

        return dict(zip(df["year"], df[column.lower()]))

    # ------------------------------------------------------------------

    def project(self):

        # Year-1 actuals from hourly simulation
        solar_direct_1 = float(np.sum(self.year1["solar_direct_pre"]))
        wind_direct_1  = float(np.sum(self.year1["wind_direct_pre"]))
        battery_1      = float(np.sum(self.year1["discharge_pre"]))

        delivered_pre_projection   = []
        delivered_meter_projection = []
        solar_direct_projection    = []
        wind_direct_projection     = []
        battery_projection         = []

        for year in range(1, self.project_life + 1):

            solar_eff = self.solar_eff_curve.get(year, 1.0)
            wind_eff  = self.wind_eff_curve.get(year, 1.0)
            soh       = self.soh_curve.get(year, 1.0)

            solar_direct_t = solar_direct_1 * solar_eff
            wind_direct_t  = wind_direct_1  * wind_eff
            battery_t      = battery_1      * soh

            delivered_pre_t   = solar_direct_t + wind_direct_t + battery_t
            delivered_meter_t = delivered_pre_t * self.loss_factor

            solar_direct_projection.append(solar_direct_t)
            wind_direct_projection.append(wind_direct_t)
            battery_projection.append(battery_t)
            delivered_pre_projection.append(delivered_pre_t)
            delivered_meter_projection.append(delivered_meter_t)

        return {
            "solar_direct_mwh":     np.array(solar_direct_projection),
            "wind_direct_mwh":      np.array(wind_direct_projection),
            "battery_mwh":          np.array(battery_projection),
            "delivered_pre_mwh":    np.array(delivered_pre_projection),   # busbar, LCOE denominator
            "delivered_meter_mwh":  np.array(delivered_meter_projection), # at meter, savings calc
        }