import mock
import pandas as pd
import pytest
from src.mlflow_utils import (
    create_mlflow_experiment_if_not_exist,
    create_mlflow_run_if_not_exists,
    save_artifacts_to_mlflow,
)
from src.feature_extractor import _FeatureExtractorData, FeatureExtractor


class MockRunConfig:
    experiment_name = "test_experiment"
    run_name = "test_run"
    mlflow_tracking_uri = "http://test:8080"


def test_create_mlflow_experiment_if_not_exist():
    with mock.patch("src.mlflow_utils.run_config", MockRunConfig):
        with mock.patch("src.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            create_mlflow_experiment_if_not_exist()
            mock_mlflow.create_experiment.assert_called_once_with(MockRunConfig.experiment_name)

            mock_mlflow.reset_mock()
            mock_mlflow.get_experiment_by_name.return_value = mock.Mock()
            create_mlflow_experiment_if_not_exist()
            mock_mlflow.create_experiment.assert_not_called()


def test_create_mlflow_run_if_not_exists():
    with mock.patch("src.mlflow_utils.run_config", MockRunConfig):
        with mock.patch("src.mlflow_utils.mlflow") as mock_mlflow:
            mock_experiment = mock.Mock()
            mock_experiment.experiment_id = "exp_id_1"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            mock_mlflow.search_runs.return_value = pd.DataFrame()
            create_mlflow_run_if_not_exists("test_run")
            mock_mlflow.start_run.assert_called_once_with(
                run_name=MockRunConfig.run_name, experiment_id="exp_id_1"
            )

            mock_mlflow.start_run.reset_mock()
            mock_mlflow.search_runs.return_value = pd.DataFrame({"run_id": ["existing_run"]})
            create_mlflow_run_if_not_exists("test_run")
            mock_mlflow.start_run.assert_not_called()


def test_save_artifacts_to_mlflow():
    with mock.patch("src.mlflow_utils.run_config", MockRunConfig):
        with mock.patch("src.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            with pytest.raises(RuntimeError):
                save_artifacts_to_mlflow({}, "run_id")

            mock_mlflow.get_experiment_by_name.return_value = mock.Mock()
            artifacts = {"model": "foo"}

            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("pickle.dump"):
                    save_artifacts_to_mlflow(artifacts, "run_id_123")

            mock_mlflow.log_artifact.assert_called_once()


def test_feature_extractor_data_save_to_mlflow():
    with mock.patch("src.feature_extractor.run_config", MockRunConfig):
        with mock.patch("src.feature_extractor.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            feature_extractor_data = _FeatureExtractorData()
            with pytest.raises(RuntimeError):
                feature_extractor_data.save_to_mlflow("run_id")

            mock_mlflow.get_experiment_by_name.return_value = mock.Mock()
            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("pickle.dump"):
                    feature_extractor_data.save_to_mlflow("run_id_123")
                    mock_mlflow.log_artifacts.assert_called_once()


def test_feature_extractor_save_to_mlflow():
    with mock.patch("src.feature_extractor.run_config", MockRunConfig):
        with mock.patch("src.feature_extractor.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            feature_extractor = FeatureExtractor()
            with pytest.raises(RuntimeError):
                feature_extractor.save_to_mlflow("run_id")

            mock_mlflow.get_experiment_by_name.return_value = mock.Mock()
            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("pickle.dump"):
                    feature_extractor.save_to_mlflow("run_id_123")
                    mock_mlflow.log_artifacts.assert_called_once()
