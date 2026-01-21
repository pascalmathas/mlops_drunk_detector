import polars as pl
from configs.settings import PathsConfig


class DataLoader:

    def __init__(self):
        self.config = PathsConfig()

    def load_accelerometer_data(self) -> pl.DataFrame:
        accelerometer_path = self.config.accelerometer_data_path
        if not accelerometer_path.exists():
            raise FileNotFoundError(f"Accelerometer data not found: {accelerometer_path}")
        return pl.read_csv(accelerometer_path)

    def load_tac_data(self) -> pl.DataFrame:
        tac_directory = self.config.clean_tac_data_path
        if not tac_directory.exists():
            raise FileNotFoundError(f"TAC directory not found: {tac_directory}")

        tac_files = list(tac_directory.glob("*_clean_TAC.csv"))
        if not tac_files:
            raise FileNotFoundError(f"No TAC files found in {tac_directory}")

        data_pids = []
        for file in tac_files:
            pid = file.stem.replace("_clean_TAC", "")
            data_pid = (
                pl.scan_csv(file)
                .with_columns([pl.lit(pid).alias("pid"), (pl.col("timestamp") * 1000).alias("time")])
                .select(["time", "pid", "TAC_Reading"])
            )
            data_pids.append(data_pid)

        data = pl.concat(data_pids).collect()
        data = data.sort(["pid", "time"])
        return data

    def load_phone_types(self) -> pl.DataFrame:
        phone_type_path = self.config.phone_type_data_path
        if not phone_type_path.exists():
            raise FileNotFoundError(f"Phone types file not found: {phone_type_path}")
        return pl.read_csv(phone_type_path)

    def load_all(self) -> dict[str, pl.DataFrame]:
        return {
            "accelerometer": self.load_accelerometer_data(),
            "tac": self.load_tac_data(),
            "phone_types": self.load_phone_types(),
        }
