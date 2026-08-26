import os
import pickle
import tempfile
import typing
import polars as pl
import mlflow

from configs.settings import run_config, PathsConfig


class _FeatureExtractorData:
    def __init__(self):
        self._phonetype_mapping: typing.Optional[dict[str, int]] = {}

    def is_set(self) -> bool:
        return True if self._phonetype_mapping and len(self._phonetype_mapping) > 0 else False

    def save(self, directory_path: str):
        with open(os.path.join(directory_path, PathsConfig.phonetype_mapping_file_name), "wb") as f:
            pickle.dump(self._phonetype_mapping, f)

    def load_from_mlflow(self, run_id: str):
        with tempfile.TemporaryDirectory() as directory_name:
            mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
            if mlflow_experiment is None:
                raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")

            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=run_config.run_name,
                dst_path=directory_name,
                tracking_uri=run_config.mlflow_tracking_uri,
            )

            artifact_folder = os.path.join(directory_name, run_config.run_name)

            with open(os.path.join(artifact_folder, PathsConfig.phonetype_mapping_file_name), "rb") as f:
                self._phonetype_mapping = pickle.load(f)

    def save_to_mlflow(self, run_id: str):
        with tempfile.TemporaryDirectory() as directory_name:
            mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
            if mlflow_experiment is None:
                raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")

            self.save(directory_name)
            mlflow.log_artifacts(directory_name, run_config.run_name)


class FeatureExtractor:
    def __init__(self):
        self._state = _FeatureExtractorData()

    def save_to_mlflow(self, run_id: str):
        self._state.save_to_mlflow(run_id)

    def load_from_mlflow(self, run_id: str):
        self._state.load_from_mlflow(run_id)

    def get_features(self, data: pl.DataFrame) -> pl.DataFrame:
        data = self._set_windows(data)
        data = self._aggregate_features(data)

        is_inference_time = self._state.is_set()
        return self._get_features_inference(data) if is_inference_time else self._get_features_training(data)

    def _get_features_training(self, data: pl.DataFrame) -> pl.DataFrame:
        self._set_fit_phonetype_mapping(data)
        data = self._get_encoded_phonetype(data)
        return data

    def _get_features_inference(self, data: pl.DataFrame) -> pl.DataFrame:
        data = self._get_encoded_phonetype(data)
        return data

    def _set_windows(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns(
            (pl.col("pid") + "_" + (pl.col("time") // run_config.window).cast(pl.Utf8)).alias("window_id")
        )
        return data

    def _aggregate_features(self, data: pl.DataFrame) -> pl.DataFrame:
        features = []

        features.append(pl.col("x").mean().alias("x_mean"))
        features.append(pl.col("x").std().alias("x_std"))
        features.append(pl.col("x").min().alias("x_min"))
        features.append(pl.col("x").max().alias("x_max"))

        features.append(pl.col("y").mean().alias("y_mean"))
        features.append(pl.col("y").std().alias("y_std"))
        features.append(pl.col("y").min().alias("y_min"))
        features.append(pl.col("y").max().alias("y_max"))

        features.append(pl.col("z").mean().alias("z_mean"))
        features.append(pl.col("z").std().alias("z_std"))
        features.append(pl.col("z").min().alias("z_min"))
        features.append(pl.col("z").max().alias("z_max"))

        features.append(pl.col("pid").first().alias("pid"))
        features.append(pl.col("phonetype").first().alias("phonetype"))

        if "label" in data.columns:
            features.append(pl.col("label").max().alias("label"))

        return data.group_by("window_id").agg(features)

    def _set_fit_phonetype_mapping(self, data: pl.DataFrame):
        unique_types = data["phonetype"].unique().sort().to_list()
        self._state._phonetype_mapping = {val: index for index, val in enumerate(unique_types)}

    def _get_encoded_phonetype(self, data: pl.DataFrame) -> pl.DataFrame:
        if not self._state.is_set():
            raise ValueError("Phonetype mapping is not set in FeatureExtractor.")

        return data.with_columns(
            pl.col("phonetype").replace(self._state._phonetype_mapping, default=-1).cast(pl.Int64)
        )
