import pandas as pd
from pathlib import Path
from . import config


class FeatureAgent:
    """
    Aggregates Silver into a Gold feature table per product/region/week.
    """

    def run(self) -> None:
        df = pd.read_csv(config.SILVER_PATH)

        agg = (
            df.groupby(["product_id", "region", "week"], as_index=False)
              .agg(
                  avg_price=("price", "mean"),
                  units_sold=("units_sold", "sum"),
                  promo_ratio=("promo_flag", "mean"),
              )
        )

        agg = agg.sort_values(["product_id", "region", "week"])
        agg["lag_units_sold"] = agg.groupby(
            ["product_id", "region"]
        )["units_sold"].shift(1)

        # drop first-week rows with no lag
        agg.dropna(inplace=True)

        Path(config.GOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(config.GOLD_PATH, index=False)
        print(f"[FeatureAgent] Gold features → {config.GOLD_PATH}")
