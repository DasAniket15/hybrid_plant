import numpy as np
import os
import pandas as pd


class EnergyProjection:
    """
    Builds 25-year annual energy projection using:

        - Solar efficiency curve
        - Wind efficiency curve
        - BESS SOH curve
        - Physical CUF profiles for raw generation

    This avoids reconstructing raw generation from dispatch outputs.
    """

    def __init__(
        self,
        config,
        data,
        year1_results,
        solar_capacity_mw,
        wind_capacity_mw,
    ):

        self.config = config
        self.data = data
        self.year1 = year1_results

        self.solar_capacity = solar_capacity_mw
        self.wind_capacity = wind_capacity_mw

        self.project_life = config.project["project"]["project_life_years"]

        self.loss_factor = self._compute_loss_factor()

        self.solar_eff_curve = self._load_curve(
            config.project["generation"]["solar"]["degradation"]["file"],
            column="efficiency"
        )

        self.wind_eff_curve = self._load_curve(
            config.project["generation"]["wind"]["degradation"]["file"],
            column="efficiency"
        )

        self.soh_curve = self._load_curve(
            config.bess["bess"]["degradation"]["file"],
            column="SOH"
        )

    # -------------------------------------------------

    def _resolve_path(self, relative_path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, relative_path)

    # -------------------------------------------------

    def _load_curve(self, path, column):

        df = pd.read_csv(path)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()

        if "year" not in df.columns:
            raise ValueError(f"'year' column not found in {path}")

        if column.lower() not in df.columns:
            raise ValueError(f"'{column}' column not found in {path}")

        return dict(zip(df["year"], df[column.lower()]))

    # -------------------------------------------------

    def _compute_loss_factor(self):

        losses = self.config.regulatory["regulatory"]["losses"]
        ht_lt_split = self.config.regulatory["regulatory"]["connection"]["ht_lt_split_percent"]

        ht_loss = (
            losses["ht_side"]["ctu_percent"]
            + losses["ht_side"]["stu_percent"]
            + losses["ht_side"]["wheeling_percent"]
        ) / 100

        lt_loss = (
            losses["lt_side"]["ctu_percent"]
            + losses["lt_side"]["stu_percent"]
            + losses["lt_side"]["wheeling_percent"]
        ) / 100

        weighted_loss = (
            (ht_lt_split / 100) * ht_loss
            + ((100 - ht_lt_split) / 100) * lt_loss
        )

        return 1 - weighted_loss

    # -------------------------------------------------

    def project(self):

        solar_cuf = self.data["solar_cuf"]
        wind_cuf = self.data["wind_cuf"]

        # -----------------------------
        # Year-1 RAW from physics
        # -----------------------------

        solar_raw_1 = np.sum(self.solar_capacity * solar_cuf)
        wind_raw_1 = np.sum(self.wind_capacity * wind_cuf)

        solar_direct_1 = np.sum(self.year1["solar_direct_pre"])
        wind_direct_1 = np.sum(self.year1["wind_direct_pre"])

        battery_1 = np.sum(self.year1["discharge_pre"])

        raw_projection = []
        direct_projection = []
        battery_projection = []
        delivered_pre_projection = []
        delivered_meter_projection = []

        for year in range(1, self.project_life + 1):

            solar_eff = self.solar_eff_curve.get(year, 1.0)
            wind_eff = self.wind_eff_curve.get(year, 1.0)
            soh = self.soh_curve.get(year, 1.0)

            # Degraded raw generation
            solar_raw_t = solar_raw_1 * solar_eff
            wind_raw_t = wind_raw_1 * wind_eff

            # Degraded direct component
            solar_direct_t = solar_direct_1 * solar_eff
            wind_direct_t = wind_direct_1 * wind_eff

            # Degraded battery contribution
            battery_t = battery_1 * soh

            delivered_pre_t = solar_direct_t + wind_direct_t + battery_t
            delivered_meter_t = delivered_pre_t * self.loss_factor

            raw_projection.append(solar_raw_t + wind_raw_t)
            direct_projection.append(solar_direct_t + wind_direct_t)
            battery_projection.append(battery_t)
            delivered_pre_projection.append(delivered_pre_t)
            delivered_meter_projection.append(delivered_meter_t)

        return {
            "raw_mwh": np.array(raw_projection),
            "direct_mwh": np.array(direct_projection),
            "battery_mwh": np.array(battery_projection),
            "delivered_pre_mwh": np.array(delivered_pre_projection),
            "delivered_meter_mwh": np.array(delivered_meter_projection),
        }