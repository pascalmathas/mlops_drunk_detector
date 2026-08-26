import polars as pl
import warnings

from configs.settings import PreprocessingConfig, run_config


class Preprocessing:

    def __init__(self):
        self.config = PreprocessingConfig()

    def combine_data(self, data: dict) -> pl.DataFrame:
        accelerometer_data = data["accelerometer"]
        tac_data = data["tac"]
        phone_types = data["phone_types"]

        accelerometer_data = accelerometer_data.sort(["pid", "time"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            merged = accelerometer_data.join_asof(
                tac_data,
                on="time",
                by="pid",
                strategy="nearest",
                tolerance=self.config.max_time_difference_ms,
            )

        merged = merged.filter(pl.col("TAC_Reading").is_not_null())
        merged = merged.join(phone_types, on="pid", how="left")

        return merged

    def add_labels(self, data: pl.DataFrame) -> pl.DataFrame:
        threshold = self.config.intoxication_threshold
        data = data.with_columns((pl.col("TAC_Reading") >= threshold).alias("label"))
        return data

    def sample_data(self, data: pl.DataFrame) -> pl.DataFrame:
        sampled_data = data.sample(fraction=run_config.sample_rate, seed=run_config.random_state)
        return sampled_data
