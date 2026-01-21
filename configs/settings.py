import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_tac_data_path: Path = Path(os.getenv("RAW_TAC_DATA_PATH", "data/raw_tac"))
    clean_tac_data_path: Path = Path(os.getenv("CLEAN_TAC_DATA_PATH", "data/clean_tac"))
    accelerometer_data_path: Path = Path(
        os.getenv("ACCELEROMETER_DATA_PATH", "data/all_accelerometer_data_pids_13.csv")
    )
    phone_type_data_path: Path = Path(os.getenv("PHONE_TYPE_DATA_PATH", "data/phone_types.csv"))
    pids_data_path: Path = Path(os.getenv("PIDS_DATA_PATH", "data/pids.txt"))

    model_path: str = os.getenv("MODEL_PATH", "models/")

    preprocessing_data_path: Path = Path(
        os.getenv("PREPROCESSING_DATA_PATH", "data/intermediate/preprocessing/data.parquet")
    )
    features_data_path: Path = Path(
        os.getenv("FEATURES_DATA_PATH", "data/intermediate/features/data.parquet")
    )

    phonetype_mapping_file_name: str = os.getenv("PHONETYPE_MAPPING_FILE_NAME", "phonetype_mapping.pkl")

    telemetry_training_data_path: str = os.getenv(
        "TELEMETRY_TRAINING_DATA_PATH", "data/telemetry/data_dist.json"
    )
    telemetry_live_data_path: str = os.getenv(
        "TELEMETRY_LIVE_DATA_PATH", "data/telemetry/live_data_dist.json"
    )


@dataclass(init=False, frozen=True)
class PreprocessingConfig:
    # These values are taken from https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking
    # 0.08 or higher is considered intoxicated
    intoxication_threshold: float = float(os.getenv("INTOXICATION_THRESHOLD", "0.08"))
    # 30 minutes in milliseconds for 30min intervals
    max_time_difference_ms: int = int(os.getenv("MAX_TIME_DIFFERENCE_MS", "1800000"))


@dataclass(frozen=True)
class _RunConfig:
    sample_rate: float = float(os.getenv("SAMPLE_RATE", "1.0"))
    num_folds: int = int(os.getenv("NUM_FOLDS", "5"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

    window: int = int(os.getenv("WINDOW", "1000"))

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080")
    experiment_name: str = os.getenv("EXPERIMENT_NAME", "BarCrawl_Experiment_A")
    run_name: str = os.getenv("RUN_NAME", "XGboost_run1")
    app_name: str = os.getenv("APP_NAME", "drunk_detector")


@dataclass(frozen=True)
class TelemetryConfig:
    accel_lower_bound: float = float(os.getenv("ACCEL_LOWER_BOUND", "-4.0"))
    accel_upper_bound: float = float(os.getenv("ACCEL_UPPER_BOUND", "4.0"))
    num_instances_for_live_dist: int = int(os.getenv("NUM_INSTANCES_FOR_LIVE_DIST", "100"))
    epsilon: float = float(os.getenv("EPSILON", "0.0001"))
    push_gateway_uri: str = os.getenv("PUSH_GATEWAY_URI", "http://prometheus_push_gateway:9091")


@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = os.getenv("MODEL_NAME", "drunk_detector")
    max_depth: int = int(os.getenv("MAX_DEPTH", "6"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.1"))
    scale_pos_weight: float = float(os.getenv("SCALE_POS_WEIGHT", "1.0"))


run_config = _RunConfig()
