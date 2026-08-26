import pytest
import mock
import polars as pl
import pandas as pd
import numpy as np

from src.api import app


@pytest.fixture()
def test_app():
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture()
def client(test_app):
    return test_app.test_client()


@pytest.fixture()
def runner(test_app):
    return test_app.test_cli_runner()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert b"OK" in response.data


def test_reload_model(client):
    with mock.patch("src.api.get_model") as mock_get_model:
        with mock.patch("src.api.get_feature_extractor_loaded") as mock_feature_extractor:
            response = client.post("/reload")
            assert response.status_code == 200
            mock_get_model.cache_clear.assert_called()
            mock_feature_extractor.cache_clear.assert_called()
            assert mock_get_model.called
            assert mock_feature_extractor.called


def test_reload_invalid_input(client):
    response = client.get("/reload")
    assert response.status_code == 405


def test_predict(client):
    with mock.patch("src.api.get_model") as mock_get_model:
        with mock.patch("src.api.get_feature_extractor_loaded") as mock_feature_extractor:
            with mock.patch("src.api.data_quality_counter") as mock_data_quality_counter:
                mock_model = mock.Mock()
                mock_model.predict.return_value = np.array([1])
                mock_get_model.return_value = mock_model

                mock_features = mock.Mock()
                mock_features.get_features.return_value = pl.DataFrame({"feature1": [1.0]})
                mock_feature_extractor.return_value = mock_features

                request_data = {
                    "x": [10.1, 0.2, 0.3],
                    "y": [0.4, 10.5, 0.6],
                    "z": [0.7, 0.8, 10.9],
                    "pid": "test_pid",
                    "phonetype": "iPhone",
                    "time": [1.0, 2.0, 3.0],
                }

                response = client.post("/predict", json=request_data)
                assert response.status_code == 200
                assert response.json == {"predictions": [True], "num_windows": 1, "intoxicated": True}
                mock_model.predict.assert_called_once()
                mock_features.get_features.assert_called_once()

                mock_data_quality_counter.labels.assert_any_call(quality_rule="high_x_value")
                mock_data_quality_counter.labels.assert_any_call(quality_rule="high_y_value")
                mock_data_quality_counter.labels.assert_any_call(quality_rule="high_z_value")

                assert mock_data_quality_counter.labels.return_value.inc.call_count == 3


def test_webhook(client):
    response = client.post("/webhook", json={"alert": "test"})
    assert response.status_code == 200
    assert b"OK" in response.data
