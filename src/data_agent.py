import pandas as pd
from pathlib import Path
from . import config


class DataAgent:
    """
    Responsible for reading raw data and writing Bronze/Silver tables.
    """

    def __init__(self, raw_path: str = "data/raw_transactions.csv"):
        self.raw_path = raw_path

    def run(self) -> None:
        df = pd.read_csv(self.raw_path, parse_dates=["date"])

        # Very light cleaning – you can extend this with more business rules
        df = df.dropna()
        df = df[df["price"] > 0]

        Path(config.BRONZE_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.BRONZE_PATH, index=False)

        # Silver: add week and day-of-week features
        df["dow"] = df["date"].dt.dayofweek
        df["week"] = df["date"].dt.isocalendar().week.astype(int)

        df.to_csv(config.SILVER_PATH, index=False)
        print(f"[DataAgent] Bronze → {config.BRONZE_PATH}, Silver → {config.SILVER_PATH}")
