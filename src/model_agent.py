import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow

from . import config


class ModelAgent:
    """
    Trains a RandomForest demand model and logs metrics to MLflow.
    """

    def run(self) -> None:
        df = pd.read_csv(config.GOLD_PATH)
        X = df[["avg_price", "promo_ratio", "lag_units_sold"]]
        y = df["units_sold"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with mlflow.start_run(run_name="rf_demand_model"):
            model = RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            Path("models").mkdir(exist_ok=True)
            joblib.dump(model, config.MODEL_PATH)

            print(f"[ModelAgent] MAE={mae:.3f}, R²={r2:.3f}")
            print(f"[ModelAgent] Model saved → {config.MODEL_PATH}")
