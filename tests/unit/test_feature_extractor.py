import os
import tempfile

import pytest
from mock import mock
import polars as pl
from polars.testing import assert_frame_equal

from src.feature_extractor import _FeatureExtractorData, FeatureExtractor


def test_feature_extractor_data_save():
    class MockPathsConfig:
        phonetype_mapping_file_name = "foo1"

    feature_extractor_data = _FeatureExtractorData()
    with mock.patch("src.feature_extractor.PathsConfig", MockPathsConfig()):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_extractor_data.save(temporary_directory)
            assert {
                MockPathsConfig.phonetype_mapping_file_name,
            } == set(os.listdir(temporary_directory))


def test_feature_extractor_data_is_set():
    feature_extractor_data = _FeatureExtractorData()
    assert feature_extractor_data.is_set() is False
    feature_extractor_data._phonetype_mapping = {"iPhone": 0}
    assert feature_extractor_data.is_set() is True


def test_feature_extractor_get_features_training_time():
    with mock.patch("src.feature_extractor.FeatureExtractor._get_features_training") as mock_get_feat:
        feature_extractor = FeatureExtractor()
        feature_extractor.get_features(
            pl.DataFrame(
                {"pid": ["1"], "time": [100], "x": [1.0], "y": [1.0], "z": [1.0], "phonetype": ["iPhone"]}
            )
        )
        mock_get_feat.assert_called_once()

    with mock.patch("src.feature_extractor.FeatureExtractor._get_features_training") as mock_get_feat:
        feature_extractor = FeatureExtractor()
        feature_extractor._state._phonetype_mapping = {}
        feature_extractor.get_features(
            pl.DataFrame(
                {"pid": ["1"], "time": [100], "x": [1.0], "y": [1.0], "z": [1.0], "phonetype": ["iPhone"]}
            )
        )
        mock_get_feat.assert_called_once()


def test_feature_extractor_get_features_inference_time():
    with mock.patch("src.feature_extractor.FeatureExtractor._get_features_inference") as mock_get_feat:
        feature_extractor = FeatureExtractor()
        feature_extractor._state._phonetype_mapping = {"iPhone": 0}
        feature_extractor.get_features(
            pl.DataFrame(
                {"pid": ["1"], "time": [100], "x": [1.0], "y": [1.0], "z": [1.0], "phonetype": ["iPhone"]}
            )
        )
        mock_get_feat.assert_called_once()


def test_feature_extractor_set_windows():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "pid": ["1", "1", "2", "2"],
                    "time": [100, 200, 150, 250],
                    "x": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "expected": pl.DataFrame(
                {
                    "pid": ["1", "1", "2", "2"],
                    "time": [100, 200, 150, 250],
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "window_id": ["1_0", "1_0", "2_0", "2_0"],
                }
            ),
        }
    ]

    class MockRunConfig:
        window = 1000

    for case in cases:
        feature_extractor = FeatureExtractor()
        with mock.patch("src.feature_extractor.run_config", MockRunConfig()):
            out = feature_extractor._set_windows(case["data"])
            assert_frame_equal(out, case["expected"])


def test_feature_extractor_aggregate_features():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "window_id": ["1_0", "1_0", "2_0", "2_0"],
                    "pid": [1, 1, 2, 2],
                    "time": [100, 200, 150, 250],
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "y": [1.0, 2.0, 3.0, 4.0],
                    "z": [1.0, 2.0, 3.0, 4.0],
                    "phonetype": ["iPhone", "iPhone", "Android", "Android"],
                    "label": [False, True, False, True],
                }
            ),
            "expected": pl.DataFrame(
                {
                    "window_id": ["1_0", "2_0"],
                    "x_mean": [1.5, 3.5],
                    "x_std": [0.7071067811865476, 0.7071067811865476],
                    "x_min": [1.0, 3.0],
                    "x_max": [2.0, 4.0],
                    "y_mean": [1.5, 3.5],
                    "y_std": [0.7071067811865476, 0.7071067811865476],
                    "y_min": [1.0, 3.0],
                    "y_max": [2.0, 4.0],
                    "z_mean": [1.5, 3.5],
                    "z_std": [0.7071067811865476, 0.7071067811865476],
                    "z_min": [1.0, 3.0],
                    "z_max": [2.0, 4.0],
                    "pid": [1, 2],
                    "phonetype": ["iPhone", "Android"],
                    "label": [True, True],
                }
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        out = feature_extractor._aggregate_features(case["data"])
        assert_frame_equal(out, case["expected"], check_row_order=False)


def test_feature_extractor_set_fit_phonetype_mapping():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "phonetype": ["iPhone", "Android", "Android", "Samsung"],
                }
            ),
            "expected": {"Android": 0, "Samsung": 1, "iPhone": 2},
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_fit_phonetype_mapping(case["data"])
        assert feature_extractor._state._phonetype_mapping == case["expected"]


def test_feature_extractor_get_encoded_phonetype():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "phonetype": ["iPhone", "Android", "Android", "Samsung"],
                }
            ),
            "expected": pl.DataFrame(
                {
                    "phonetype": [2, 0, 0, 1],
                }
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._state._phonetype_mapping = {"Android": 0, "Samsung": 1, "iPhone": 2}
        out = feature_extractor._get_encoded_phonetype(case["data"])
        assert_frame_equal(out, case["expected"])


def test_feature_extractor_get_encoded_phonetype_not_set():
    feature_extractor = FeatureExtractor()
    with pytest.raises(ValueError):
        feature_extractor._get_encoded_phonetype(pl.DataFrame([{}]))


def test_feature_extractor_get_features_training():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "window_id": ["1_0", "2_0"],
                    "pid": [1, 2],
                    "x_mean": [1.5, 3.5],
                    "y_mean": [1.5, 3.5],
                    "z_mean": [1.5, 3.5],
                    "phonetype": ["iPhone", "Android"],
                }
            ),
            "expected": pl.DataFrame(
                {
                    "window_id": ["1_0", "2_0"],
                    "pid": [1, 2],
                    "x_mean": [1.5, 3.5],
                    "y_mean": [1.5, 3.5],
                    "z_mean": [1.5, 3.5],
                    "phonetype": [1, 0],
                }
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        out = feature_extractor._get_features_training(case["data"])
        assert_frame_equal(out, case["expected"])


def test_feature_extractor_get_features_inference():
    cases = [
        {
            "data": pl.DataFrame(
                {
                    "window_id": ["1_0", "2_0"],
                    "pid": [1, 2],
                    "x_mean": [1.5, 3.5],
                    "y_mean": [1.5, 3.5],
                    "z_mean": [1.5, 3.5],
                    "phonetype": ["iPhone", "Android"],
                }
            ),
            "expected": pl.DataFrame(
                {
                    "window_id": ["1_0", "2_0"],
                    "pid": [1, 2],
                    "x_mean": [1.5, 3.5],
                    "y_mean": [1.5, 3.5],
                    "z_mean": [1.5, 3.5],
                    "phonetype": [1, 0],
                }
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._state._phonetype_mapping = {"Android": 0, "iPhone": 1}
        out = feature_extractor._get_features_inference(case["data"])
        assert_frame_equal(out, case["expected"])
