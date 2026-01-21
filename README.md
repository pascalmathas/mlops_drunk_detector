# Drunk Detector

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache-Airflow-017CEE.svg)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600.svg)](https://xgboost.ai/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800.svg)](https://grafana.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-ready ML pipeline for detecting intoxication from smartphone accelerometer data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Apache Airflow                                  │
│                         (DAG Orchestration :4242)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
   │ Preprocess  │     →     │  Feature    │     →     │   Train     │
   │    Data     │           │ Engineering │           │   Model     │
   └─────────────┘           └─────────────┘           └─────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MLflow (:8080)                                      │
│                    (Model Registry & Artifacts)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Flask API (:5001)                                   │
│                       (Prediction Service)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
┌───────────────────┐                                 ┌───────────────────┐
│    Prometheus     │                                 │   Alertmanager    │
│     (:9090)       │────────────────────────────────▶│     (:9093)       │
└───────────────────┘                                 └───────────────────┘
          │                                                     │
          ▼                                                     ▼
┌───────────────────┐                                 ┌───────────────────┐
│     Grafana       │                                 │  Webhook Alerts   │
│     (:3000)       │                                 │                   │
└───────────────────┘                                 └───────────────────┘
```

## Screenshots

| Airflow DAG | MLflow Experiments | Grafana Dashboard |
|:-----------:|:------------------:|:-----------------:|
| ![Airflow](docs/images/airflow.png) | ![MLflow](docs/images/mlflow.png) | ![Grafana](docs/images/grafana.png) |

## Project Structure

```
mlops_drunk_detector/
├── src/                          # core application code
│   ├── api.py                    # flask rest api
│   ├── pipeline.py               # training pipeline
│   ├── model.py                  # xgboost model wrapper
│   ├── feature_extractor.py      # feature engineering
│   ├── preprocessing.py          # data preprocessing
│   ├── dataloader.py             # data loading utilities
│   ├── drift_calculation.py      # psi drift detection
│   └── mlflow_utils.py           # mlflow integration
├── infra/                        # infrastructure
│   ├── docker-compose.yaml       # full stack orchestration
│   ├── build/dockerfile          # container definition
│   ├── airflow/dags/             # airflow dag definitions
│   ├── grafana/                  # dashboards & provisioning
│   └── telemetry/                # prometheus & alertmanager config
├── configs/                      # configuration
│   └── settings.py               # environment-based settings
├── tests/                        # test suite
│   ├── unit/                     # unit tests
│   └── infra/                    # integration tests
├── scripts/                      # utility scripts
├── data/                         # data directory
└── models/                       # trained model storage
```

## Model Performance

**Dataset:** [UCI Bar Crawl: Detecting Heavy Drinking](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking)

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| Training Samples | 304,951 |
| Class Distribution | 71.5% sober / 28.5% intoxicated |
| Intoxication Threshold | TAC ≥ 0.08 |
| Validation Strategy | GroupKFold (5 folds, grouped by participant) |
| Features | 12 aggregated accelerometer features per 1s window |

## API Documentation

**Base URL:** `http://localhost:5001`

### Predict Request

```json
{
  "x": [0.12, -0.34, 0.56],
  "y": [0.78, -0.91, 0.23],
  "z": [0.45, -0.67, 0.89],
  "time": [1000, 1010, 1020],
  "pid": "participant_001",
  "phonetype": "iPhone"
}
```

### Predict Response

```json
{
  "predictions": [false, false, true],
  "num_windows": 10,
  "intoxicated": true
}
```

## Run the App

### Download Data

Download manually from [UCI Machine Learning - Bar Crawl: Detecting Heavy Drinking](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking) or run:

```bash
bash scripts/download_data.sh
```

### Install Environment Locally

Required to run tests.

```bash
conda create --name mlops_drunk_detector python=3.13 -y
conda activate mlops_drunk_detector
pip install -r requirements.txt
```

### Configure Settings

Edit configuration or leave defaults:

```bash
cp .env.example .env
```

### Run Drunk Detector

Docker Daemon should be running.

```bash
make build
docker compose -f infra/docker-compose.yaml up -d
```

### Run Training Pipeline

Wait until 10/10 containers are running before proceeding.

1. Navigate to Apache Airflow: [http://localhost:4242](http://localhost:4242)

2. Run the `bar_crawl_training` DAG. This will:
   - Preprocess the data
   - Build the features
   - Train the XGBoost model
   - Reload the API with the new model

3. View model artifacts in MLflow: [http://localhost:8080](http://localhost:8080)

### Simulate Traffic

**Low traffic:**
```bash
bash scripts/simulate_low_traffic.sh
```

**High traffic:**
```bash
bash scripts/simulate_high_traffic.sh
```

**Continuous traffic:**
```bash
bash scripts/simulate_continuous_traffic.sh
```

**Unusual traffic:**
```bash
bash scripts/simulate_data_quality_warnings.sh
```

### Monitor Model

Grafana Dashboard: [http://localhost:3000](http://localhost:3000)

### Close Drunk Detector

```bash
docker compose -f infra/docker-compose.yaml down
```

### Run Tests

```bash
conda activate mlops_drunk_detector
make test
```

## Authors

[@pascalmathas](https://github.com/pascalmathas)

## Acknowledgements

Code and structure follows examples given in the University of Amsterdam course Engineering Production-Ready ML/AI Systems.

Course was given by: Kemal Yesilbek. Github: [@kemalty](https://github.com/kemalty).
