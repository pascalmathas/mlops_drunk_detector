import pandas as pd
import pytest
from mock import mock

from src.model import Model


def test_model_get_x_y():
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "label": [1, 0, 0, 1],
                    "window_id": ["1_0", "1_1", "2_0", "2_1"],
                    "pid": [1, 1, 2, 2],
                    "amount": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "expected_x": pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0]}),
            "expected_y": pd.Series([1, 0, 0, 1], name="label"),
            "expected_groups": pd.Series([1, 1, 2, 2], name="pid"),
        }
    ]

    for case in cases:
        out_x, out_y, out_groups = Model._get_x_y(case["data"])
        pd.testing.assert_frame_equal(out_x, case["expected_x"])
        pd.testing.assert_series_equal(out_y, case["expected_y"])
        pd.testing.assert_series_equal(out_groups, case["expected_groups"])


def test_model_train_model():
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "label": [1, 0, 0, 1, 1, 0],
                    "window_id": ["1_0", "1_1", "2_0", "2_1", "3_0", "3_1"],
                    "pid": [1, 1, 2, 3, 4, 5],
                    "x_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "y_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            )
        }
    ]

    for case in cases:
        model = Model()
        model.train_model(case["data"])
        assert model._model is not None
        assert model._cv_scores is not None


def test_model_save_model():
    class MockPathsConfig:
        model_path = "foo/"

    class MockModelConfig:
        model_name = "bar"

    with mock.patch("src.model.PathsConfig", MockPathsConfig()):
        with mock.patch("src.model.ModelConfig", MockModelConfig()):
            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("pickle.dump"):
                    model = Model()
                    model._model = mock.Mock()
                    model.save_model()


def test_model_predict():
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "label": [1, 0, 0, 1],
                    "window_id": ["1_0", "1_1", "2_0", "2_1"],
                    "pid": [1, 1, 2, 2],
                    "x_mean": [1.0, 2.0, 3.0, 4.0],
                    "y_mean": [1.0, 2.0, 3.0, 4.0],
                }
            )
        }
    ]

    for case in cases:
        model = Model()
        model._model = mock.Mock()
        model._model.predict.return_value = [1, 0, 0, 1]
        predictions = model.predict(case["data"])
        assert predictions is not None
