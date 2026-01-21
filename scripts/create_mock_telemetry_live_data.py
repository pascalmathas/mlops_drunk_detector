from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from configs.settings import PathsConfig


def main():
    np.random.seed(42)
    start_date = datetime(2025, 1, 1, 0, 0)

    timestamps = []
    intoxicated_labels = []

    current_time = start_date
    for _ in range(120):
        current_time += timedelta(hours=np.random.randint(1, 7))
        timestamps.append(current_time)

        intoxicated_labels.append(np.random.random() < 0.30)

    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "intoxicated": intoxicated_labels,
        }
    )
    predictions.to_json(PathsConfig.telemetry_live_data_path, orient="records")


if __name__ == "__main__":
    main()
