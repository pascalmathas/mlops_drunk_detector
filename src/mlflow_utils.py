import pickle
import tempfile
import typing

import mlflow

from configs.settings import run_config


def create_mlflow_experiment_if_not_exist():
    if not mlflow.get_experiment_by_name(run_config.experiment_name):
        mlflow.create_experiment(run_config.experiment_name)


def create_mlflow_run_if_not_exists(run_name: str):
    mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
    if mlflow.search_runs([mlflow_experiment.experiment_id], f"attributes.run_name = '{run_name}'").empty:
        mlflow.start_run(run_name=run_config.run_name, experiment_id=mlflow_experiment.experiment_id)


def get_latest_run_id(run_name: str) -> str:
    mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
    if mlflow_experiment is None:
        raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")
    runs = mlflow.search_runs([mlflow_experiment.experiment_id], f"attributes.run_name = '{run_name}'")

    if runs.empty:
        raise RuntimeError(f"Run with name {run_name} is not found in MLFlow.")
    if len(runs[runs["end_time"].isna()]) > 1:
        raise RuntimeError("MLFlow has multiple unfinished runs. Expected one or none unfinished run.")
    if runs[runs["end_time"].isna()].empty:
        latest_run = runs.loc[runs["end_time"].idxmax()]
    else:
        latest_run = runs[runs["end_time"].isna()].iloc[0]

    return latest_run["run_id"]


def save_artifacts_to_mlflow(artifacts: dict[str, typing.Any], run_id: str):
    mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
    if mlflow_experiment is None:
        raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")

    for artifact_name, artifact in artifacts.items():
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = f"{temporary_directory}/{artifact_name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
            mlflow.log_artifact(artifact_path, f"{run_config.run_name}", run_id)
