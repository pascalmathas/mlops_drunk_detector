from functools import cache
import logging
import mlflow
import polars as pl
from flask import Flask, request, jsonify
from http import HTTPStatus
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter

from configs.settings import run_config, ModelConfig, TelemetryConfig
from src.feature_extractor import FeatureExtractor
from src.mlflow_utils import get_latest_run_id
from src.model import Model

logger = logging.getLogger(__name__)


@cache
def get_model() -> Model:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    model = Model()
    model.load_model_from_mlflow(run_id)
    return model


@cache
def get_feature_extractor_loaded() -> FeatureExtractor:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    feature_extractor = FeatureExtractor()
    feature_extractor.load_from_mlflow(run_id)
    return feature_extractor


logger.info("Loading cache.")
get_model()
get_feature_extractor_loaded()

logger.info("Starting flask app.")
app = Flask(__name__)
metrics = PrometheusMetrics(app, defaults_prefix=run_config.app_name)
metrics.info(
    f"{run_config.app_name}_model_version",
    "Model version information",
    experiment_name=run_config.experiment_name,
    run_name=run_config.run_name,
    model_name=ModelConfig.model_name,
)

pred_counter = Counter(
    f"{run_config.app_name}_predictions", "Count of predictions by class", labelnames=["pred"]
)
phonetype_counter = Counter(
    f"{run_config.app_name}_phonetypes", "Count of predictions by phonetype", labelnames=["phonetype"]
)
for phonetype in ["iPhone", "Samsung Galaxy", "Google Pixel", "Xiaomi", "OnePlus"]:
    phonetype_counter.labels(phonetype=phonetype).inc(0)

data_quality_counter = Counter(
    f"{run_config.app_name}_data_quality", "Count of data quality issues", labelnames=["quality_rule"]
)
data_quality_counter.labels(quality_rule="high_x_value").inc(0)
data_quality_counter.labels(quality_rule="high_y_value").inc(0)
data_quality_counter.labels(quality_rule="high_z_value").inc(0)
data_quality_counter.labels(quality_rule="missing_time_field").inc(0)
data_quality_counter.labels(quality_rule="array_length_mismatch").inc(0)


@app.route("/health", methods=["GET"])
@metrics.do_not_track()
def health():
    logging.debug("Health endpoint pinged.")
    return "OK", HTTPStatus.OK


@app.route("/webhook", methods=["POST"])
@metrics.do_not_track()
def webhook():
    logging.info("Received alert from Alertmanager.")
    logging.info(request.json)
    return "OK", HTTPStatus.OK


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received prediction request at /predict.")
    data_json = request.json

    if "time" not in data_json:
        data_quality_counter.labels(quality_rule="missing_time_field").inc()
        logging.error("Data error: no time fiedls.")
        return jsonify({"error": "Missing required field: time"}), HTTPStatus.BAD_REQUEST

    if len(data_json["x"]) != len(data_json["time"]):
        data_quality_counter.labels(quality_rule="array_length_mismatch").inc()
        logging.error("Arrays not of equal length")
        return jsonify({"error": "All arrays (x, y, z) must have same length"}), HTTPStatus.BAD_REQUEST

    for val in data_json["x"]:
        if not (TelemetryConfig.accel_lower_bound <= val <= TelemetryConfig.accel_upper_bound):
            data_quality_counter.labels(quality_rule="high_x_value").inc()
            logging.warning(f"Data quality issue: high x-value {val} detected.")
            break

    for val in data_json["y"]:
        if not (TelemetryConfig.accel_lower_bound <= val <= TelemetryConfig.accel_upper_bound):
            data_quality_counter.labels(quality_rule="high_y_value").inc()
            logging.warning(f"Data quality issue: high y-value {val} detected.")
            break

    for val in data_json["z"]:
        if not (TelemetryConfig.accel_lower_bound <= val <= TelemetryConfig.accel_upper_bound):
            data_quality_counter.labels(quality_rule="high_z_value").inc()
            logging.warning(f"Data quality issue: high z-value {val} detected.")
            break

    data = pl.DataFrame(
        {
            "x": data_json["x"],
            "y": data_json["y"],
            "z": data_json["z"],
            "pid": [str(data_json["pid"])] * len(data_json["x"]),
            "phonetype": [data_json["phonetype"]] * len(data_json["x"]),
            "time": data_json["time"],
        }
    )

    logging.info("Calculating features for the request.")
    feature_extractor = get_feature_extractor_loaded()
    features = feature_extractor.get_features(data)

    logging.info("Features are ready. Calculating prediction.")
    model = get_model()
    predictions = model.predict(features.to_pandas())

    for prediction in predictions:
        pred_counter.labels(pred="true" if prediction else "false").inc()
    phonetype_counter.labels(phonetype=data_json["phonetype"]).inc()

    logging.info(f"Prediction complete. {len(predictions)} window(s) predicted.")
    return jsonify(
        {
            "predictions": [bool(p) for p in predictions],
            "num_windows": len(predictions),
            "intoxicated": bool(any(predictions)),
        }
    )


@app.route("/reload", methods=["POST"])
def reload_model():
    get_model.cache_clear()
    get_feature_extractor_loaded.cache_clear()
    get_model()
    get_feature_extractor_loaded()

    return "OK", HTTPStatus.OK
