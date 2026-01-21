from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import timedelta
import pendulum

local_tz = pendulum.timezone("Europe/Amsterdam")

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "bar_crawl_training",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 11, 28, tz=local_tz),
    schedule="59 23 * * *",
    catchup=False,
)

log_datetime_start_task = BashOperator(task_id="log_datetime_start", bash_command="date", dag=dag)

preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/pipeline.py --preprocess",
    image="drunk-detector:latest",
    network_mode="infra_default",
    task_id="preprocessing_bar_crawl_data",
    mounts=[Mount(source="drunk_detector_data", target="/drunk_detector/data", type="volume")],
    mount_tmp_dir=False,
    dag=dag,
)

feat_eng_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/pipeline.py --feat-eng",
    image="drunk-detector:latest",
    network_mode="infra_default",
    task_id="building_features",
    mounts=[Mount(source="drunk_detector_data", target="/drunk_detector/data", type="volume")],
    mount_tmp_dir=False,
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/pipeline.py --training",
    image="drunk-detector:latest",
    network_mode="infra_default",
    task_id="training_XGboost",
    mounts=[Mount(source="drunk_detector_data", target="/drunk_detector/data", type="volume")],
    mount_tmp_dir=False,
    dag=dag,
)

reload_api_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="curl -f -X POST http://app:5001/reload",
    image="curlimages/curl:latest",
    network_mode="infra_default",
    task_id="restarting",
    mount_tmp_dir=False,
    dag=dag,
)

log_datetime_end_task = BashOperator(task_id="log_datetime_end", bash_command="date", dag=dag)

(
    log_datetime_start_task
    >> preprocessing_task
    >> feat_eng_task
    >> model_training_task
    >> reload_api_task
    >> log_datetime_end_task
)
