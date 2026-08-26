import pickle
import pandas as pd
import tempfile
import polars as pl
import mlflow
import os
from sklearn.model_selection import cross_validate, GroupKFold
from xgboost import XGBClassifier

from configs.settings import ModelConfig, run_config, PathsConfig


class Model:
    def __init__(self):
        self._model = None
        self._cv_scores = None

    @staticmethod
    def _get_model_object() -> XGBClassifier:
        return XGBClassifier(
            max_depth=ModelConfig.max_depth,
            learning_rate=ModelConfig.learning_rate,
            scale_pos_weight=ModelConfig.scale_pos_weight,
            eval_metric="logloss",
            n_jobs=2,
        )

    @staticmethod
    def _get_x_y(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        groups = data["pid"]
        y = data["label"]
        x = data.drop(["label", "window_id", "pid"], axis=1)

        return x, y, groups

    def train_model(self, data: pd.DataFrame) -> None:
        x, y, groups = self._get_x_y(data)
        classifier = self._get_model_object()
        group_kfold = GroupKFold(n_splits=run_config.num_folds)

        self._cv_scores = cross_validate(
            classifier,
            x,
            y,
            cv=group_kfold,
            groups=groups,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"],
        )

        self._model = classifier.fit(x, y)

    def get_cv_scores(self, data: pd.DataFrame) -> dict[str, float]:
        return self._cv_scores

    def save_model(self):
        with open(f"{PathsConfig.model_path}{ModelConfig.model_name}.pkl", "wb") as f:
            pickle.dump(self._model, f)

    def load_model_from_mlflow(self, run_id: str):
        mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")

        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                dst_path=temp_dir,
                tracking_uri=run_config.mlflow_tracking_uri,
            )
            artifacts_path = temp_dir + "/" + run_config.run_name
            model_file_name = ModelConfig.model_name + ".pkl"
            if model_file_name not in os.listdir(artifacts_path):
                raise RuntimeError(f"Model {ModelConfig.model_name} is not among MLFLow artifacts.")
            with open(artifacts_path + "/" + model_file_name, "rb") as file:
                self._model = pickle.load(file)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ["label", "window_id", "pid"]
        existing_cols_to_drop = [col for col in cols_to_drop if col in features.columns]
        features = features.drop(existing_cols_to_drop, axis=1)

        return self._model.predict(features)
