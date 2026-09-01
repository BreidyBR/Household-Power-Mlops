"""
Monitoreo de desempeño del modelo sobre batches de producción.

Calcula MAE_t y RMSE_t usando el ground truth disponible
en cada batch temporal.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.monitoring.production_batches import (
    TARGET,
    load_monitoring_data,
    split_reference_and_production,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_model.joblib"
)


def load_model():
    """Carga el modelo final guardado localmente."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta src/training/train.py primero."
        )

    return joblib.load(MODEL_PATH)


def evaluate_batch(model, batch):
    """
    Calcula MAE_t y RMSE_t para un batch de producción.

    El target se usa únicamente como ground truth para evaluar
    el desempeño del modelo.
    """

    X = batch.drop(columns=TARGET)
    y_true = batch[TARGET]

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    y_pred = model.predict(X)

    mae_t = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse_t = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "mae_t": float(mae_t),
        "rmse_t": float(rmse_t),
    }


def main():
    df = load_monitoring_data()

    _, batch_1, batch_2, batch_3 = (
        split_reference_and_production(df)
    )

    model = load_model()

    batches = {
        "PRODUCTION_BATCH_1": batch_1,
        "PRODUCTION_BATCH_2": batch_2,
        "PRODUCTION_BATCH_3": batch_3,
    }

    print("=== MODEL MONITORING ===")

    for batch_name, batch in batches.items():
        metrics = evaluate_batch(
            model,
            batch,
        )

        print()
        print(batch_name)
        print(f"MAE_t:  {metrics['mae_t']:.4f} kW")
        print(f"RMSE_t: {metrics['rmse_t']:.4f} kW")


if __name__ == "__main__":
    main()