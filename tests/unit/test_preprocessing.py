import tempfile

import mock
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.preprocessing import Preprocessing


def test_combine_data():
    data = {
        "accelerometer": pl.DataFrame(
            {
                "pid": [1, 1, 2, 2],
                "time": [100, 200, 150, 250],
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        "tac": pl.DataFrame(
            {
                "pid": [1, 1, 2, 2],
                "time": [105, 205, 155, 255],
                "TAC_Reading": [0.01, 0.02, 0.03, 0.04],
            }
        ),
        "phone_types": pl.DataFrame(
            {
                "pid": [1, 2],
                "phone_type": ["iPhone", "Android"],
            }
        ),
    }
    expected = pl.DataFrame(
        {
            "pid": [1, 1, 2, 2],
            "time": [100, 200, 150, 250],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "z": [1.0, 2.0, 3.0, 4.0],
            "TAC_Reading": [0.01, 0.02, 0.03, 0.04],
            "phone_type": ["iPhone", "iPhone", "Android", "Android"],
        }
    )

    with tempfile.TemporaryDirectory() as temporary_directory:

        class MockPreprocessingConfig:
            max_time_difference_ms = 100

        with mock.patch("configs.settings.PreprocessingConfig", MockPreprocessingConfig):
            preprocessing = Preprocessing()
            out = preprocessing.combine_data(data)
            assert_frame_equal(out, expected)


def test_add_labels():
    data = pl.DataFrame(
        {
            "pid": [1, 1, 2, 2],
            "TAC_Reading": [0.01, 0.08, 0.03, 0.09],
        }
    )
    expected = pl.DataFrame(
        {
            "pid": [1, 1, 2, 2],
            "TAC_Reading": [0.01, 0.08, 0.03, 0.09],
            "label": [False, True, False, True],
        }
    )

    with tempfile.TemporaryDirectory() as temporary_directory:

        class MockPreprocessingConfig:
            intoxication_threshold = 0.08

        with mock.patch("configs.settings.PreprocessingConfig", MockPreprocessingConfig):
            preprocessing = Preprocessing()
            out = preprocessing.add_labels(data)
            assert_frame_equal(out, expected)


def test_sample_data():
    data = pl.DataFrame(
        {
            "pid": [1, 1, 2, 2],
            "TAC_Reading": [0.01, 0.08, 0.03, 0.09],
        }
    )

    with tempfile.TemporaryDirectory() as temporary_directory:

        class MockRunConfig:
            sample_rate = 0.5
            random_state = 42

        with mock.patch("configs.settings.run_config", MockRunConfig):
            preprocessing = Preprocessing()
            out = preprocessing.sample_data(data)
            assert len(out) <= len(data)
            assert set(out.columns) == set(data.columns)
