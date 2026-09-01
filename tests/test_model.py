"""
Día 9 — Pruebas de modelo: input válido -> forecast válido, y
comportamiento ante input inválido. Carga el modelo real registrado en el
Model Registry de MLflow (household-power-forecaster, Día 8) — no un
modelo de juguete — para probar exactamente lo que la API sirve.

Uso:
    pytest tests/test_model.py -v
"""
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.training.train import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME  # noqa: E402

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "rollmean_3h", "rollstd_3h", "rollmean_6h", "rollstd_6h",
    "rollmean_24h", "rollstd_24h", "rollmean_168h", "rollstd_168h",
]

VALID_INPUT = {
    "hour": 20, "day_of_week": 5, "month": 12, "is_weekend": 1,
    "hour_sin": -1.0, "hour_cos": 0.0,
    "lag_1h": 1.5, "lag_2h": 1.4, "lag_3h": 1.3, "lag_24h": 1.6, "lag_48h": 1.5, "lag_168h": 1.4,
    "rollmean_3h": 1.4, "rollstd_3h": 0.1, "rollmean_6h": 1.3, "rollstd_6h": 0.2,
    "rollmean_24h": 1.1, "rollstd_24h": 0.3, "rollmean_168h": 1.0, "rollstd_168h": 0.3,
}


def _model_available():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        for stage in ("Production", "Staging"):
            if client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage]):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="No hay modelo registrado en MLflow. Corre 'python src/training/train.py' primero.",
)


@pytest.fixture(scope="module")
def model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    for stage in ("Production", "Staging"):
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
        if versions:
            return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{stage}")
    pytest.skip("No hay modelo registrado disponible.")


def test_valid_input_returns_finite_forecast(model):
    X = pd.DataFrame([VALID_INPUT])[FEATURE_COLUMNS]
    forecast = model.predict(X)[0]
    assert np.isfinite(forecast)


def test_forecast_within_physically_plausible_range(model):
    """Global_active_power real observado va de 0 a ~11 kW (ver notebook Parte I)."""
    X = pd.DataFrame([VALID_INPUT])[FEATURE_COLUMNS]
    forecast = model.predict(X)[0]
    assert -0.5 < forecast < 20, f"Pronóstico fuera de rango plausible: {forecast}"


def test_missing_feature_raises(model):
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "lag_1h"}
    X = pd.DataFrame([incomplete])
    with pytest.raises((ValueError, KeyError)):
        model.predict(X[[c for c in FEATURE_COLUMNS if c in X.columns]])


def test_non_numeric_feature_raises(model):
    bad_input = dict(VALID_INPUT)
    bad_input["hour"] = "veinte"  # tipo incorrecto, simula input corrupto
    X = pd.DataFrame([bad_input])[FEATURE_COLUMNS]
    with pytest.raises((ValueError, TypeError)):
        model.predict(X)
