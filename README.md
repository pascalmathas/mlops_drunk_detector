# Drunk detector via accelerometer data

Machine learning / artificial intelligence production-ready drunk detector model.

---

## Run the app

### Download Data

Download manually [UCI Machine Learning - Bar Crawl: Detecting Heavy Drinking](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking)

or run:

```bash
cd drunk_detector
bash scripts/download_data.sh
```

### Install environment locally (required to run tests)

```bash
cd drunk_detector
conda create --name drunk_detector python=3.13 -y
conda activate drunk_detector
pip install -r requirements.txt
```

### Configure Settings

Edit configuration for system or leave defaults, then run:

```bash
cd drunk_detector
cp .env.example .env
```

### Run Drunk Detector

**Docker Daemon** should be running.

```bash
cd drunk_detector
make build
docker compose -f infra/docker-compose.yaml up -d
```

### Run Training Pipeline

**Note:** Wait until 10/10 containers are running before proceeding.

1. **Navigate to Apache Airflow:** [http://localhost:4242](http://localhost:4242)

2. **Run the `bar_crawl_training` DAG.** This will:
   - Preprocess the data
   - Build the features
   - Train the XGBoost model
   - Reload the API with the new model

3. **View model artifacts in MLflow:** [http://localhost:8080](http://localhost:8080)

### Simulate Traffic

**Low traffic**:
```bash
cd drunk_detector
bash scripts/simulate_low_traffic.sh
```

**High traffic**:
```bash
cd drunk_detector
bash scripts/simulate_high_traffic.sh
```

**Continuous traffic**:
```bash
cd drunk_detector
bash scripts/simulate_continuous_traffic.sh
```

**Unusual traffic**:
```bash
cd drunk_detector
bash scripts/simulate_data_quality_warnings.sh
```

### Monitor Model

Grafana Dashboard: [http://localhost:3000](http://localhost:3000)

### Close Drunk Detector

```bash
docker compose -f infra/docker-compose.yaml down
```

### Run tests

```bash
cd drunk_detector
conda activate drunk_detector
make test
```

---

## Authors

@pascalmathas

---

### Acknowledgements

Code, and code structure follows examples given in the University of Amsterdam course Engineering Production-Ready ML/AI Systems.

Course was given by: Kemal Yesilbek. Github: @kemalty.
