import click
import json
import mlflow
import polars as pl
from pathlib import Path
from mlflow.entities import RunStatus
import logging

from src.dataloader import DataLoader
from src.preprocessing import Preprocessing
from src.feature_extractor import FeatureExtractor
from src.model import Model
from configs.settings import PathsConfig, run_config, ModelConfig
from src.mlflow_utils import (
    create_mlflow_experiment_if_not_exist,
    create_mlflow_run_if_not_exists,
    get_latest_run_id,
    save_artifacts_to_mlflow,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--preprocess", is_flag=True, help="Run Preprocessing step.")
@click.option("--feat-eng", is_flag=True, help="Run Feature Engineering step.")
@click.option("--training", is_flag=True, help="Run Training step.")
def main(preprocess: bool, feat_eng: bool, training: bool):

    if preprocess:
        logger.info("Started preprocessing...")
        dataloader = DataLoader()
        data_dict = dataloader.load_all()

        preprocessor = Preprocessing()
        data = preprocessor.combine_data(data_dict)
        data = preprocessor.add_labels(data)

        PathsConfig.preprocessing_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(PathsConfig.preprocessing_data_path)
        logger.info(f"Preprocessing complete. Saved to {PathsConfig.preprocessing_data_path}")

    if feat_eng:
        logger.info("Starting Feature Engineering...")
        data = pl.read_parquet(PathsConfig.preprocessing_data_path)

        feature_extractor = FeatureExtractor()
        data = feature_extractor.get_features(data)

        preprocessor = Preprocessing()
        data = preprocessor.sample_data(data)

        PathsConfig.features_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(PathsConfig.features_data_path)

        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)

        run_id = get_latest_run_id(run_config.run_name)
        feature_extractor.save_to_mlflow(run_id)

        logger.info(
            f"Feature Engineering complete. Saved to {PathsConfig.features_data_path} and logged to MLflow."
        )

    if training:
        logger.info("Starting Training...")
        data = pl.read_parquet(PathsConfig.features_data_path)

        model = Model()
        model.train_model(data.to_pandas())
        scores = model.get_cv_scores(data.to_pandas())
        print(f"CV Scores: {scores}")

        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)

        run_id = get_latest_run_id(run_config.run_name)
        save_artifacts_to_mlflow({ModelConfig.model_name: model, "cv_scores": scores}, run_id)

        telemetry_data = {
            "intoxicated": {
                "false": data.filter(pl.col("label") == 0).height,
                "true": data.filter(pl.col("label") == 1).height,
            }
        }
        telemetry_path = Path(PathsConfig.telemetry_training_data_path)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(telemetry_path, "w") as file:
            json.dump(telemetry_data, file)

        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
        logger.info("Training complete and logged to MLflow.")


if __name__ == "__main__":
    main()
