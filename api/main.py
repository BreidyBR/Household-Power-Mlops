"""
Día 9 — API de inferencia (FastAPI), sección M del enunciado.

Sirve el modelo YA REGISTRADO en el Model Registry de MLflow (Día 8,
`household-power-forecaster`, stage Production). Esta API no entrena ni
reentrena nada: solo carga el modelo ganador y lo expone para consumo
externo.

Endpoints:
    GET  /health       — estado del servicio y si el modelo cargó correctamente
    GET  /metrics       — métricas operativas del servicio
    GET  /model-info    — qué modelo/versión/stage está sirviendo ahora mismo
    POST /predict       — recibe las 20 features y devuelve el pronóstico a 24h

Uso local:
    uvicorn api.main:app --reload --port 8000
"""
import os
import time
import joblib


import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.monitoring.system_monitor import SystemMonitor

REGISTERED_MODEL_NAME = "household-power-forecaster"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

app = FastAPI(
    title="Household Power Forecasting API",
    description="Pronostica Global_active_power (kW) a 24 horas vista.",
    version="1.0.0",
)

system_monitor = SystemMonitor()

# Etapa desde la que se sirve el modelo. Se intenta primero Production
# (la promoción final del Día 8); si no existe, se cae a Staging, para que
# la API siga siendo útil en un entorno de desarrollo donde todavía no se
# ha hecho la promoción final.
MODEL_STAGES_TO_TRY = ("Production", "Staging")

_model = None
_model_version_info = None  # dict con version/stage/run_id del modelo cargado


class PredictRequest(BaseModel):
    """
    Las 20 features que consume el modelo, en el mismo orden y con el mismo
    significado definidos en src/features/build_features.py. Deben
    construirse con ese mismo módulo sobre el histórico reciente del hogar.
    """

    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    lag_1h: float
    lag_2h: float
    lag_3h: float
    lag_24h: float
    lag_48h: float
    lag_168h: float
    rollmean_3h: float
    rollstd_3h: float
    rollmean_6h: float
    rollstd_6h: float
    rollmean_24h: float
    rollstd_24h: float
    rollmean_168h: float
    rollstd_168h: float


class PredictResponse(BaseModel):
    forecast: float
    horizon: str = "24h"
    model_version: str


def load_production_model():
    """
    Carga el modelo desde una ruta local si MODEL_PATH está definida
    (por ejemplo, dentro de Docker). Si no, utiliza el Model Registry
    de MLflow como en el entorno local de desarrollo.
    """

    global _model, _model_version_info

    model_path = os.getenv("MODEL_PATH")

    if model_path:
        _model = joblib.load(model_path)
        _model_version_info = {
            "version": os.getenv("MODEL_VERSION", "docker"),
            "stage": "Production",
            "run_id": "local-artifact",
        }
        print(f"Modelo cargado desde archivo local: {model_path}")
        return

    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    for stage in MODEL_STAGES_TO_TRY:
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME,
            stages=[stage],
        )

        if versions:
            mv = versions[0]
            _model = mlflow.sklearn.load_model(
                f"models:/{REGISTERED_MODEL_NAME}/{stage}"
            )
            _model_version_info = {
                "version": mv.version,
                "stage": stage,
                "run_id": mv.run_id,
            }

            print(
                f"Modelo cargado: "
                f"{REGISTERED_MODEL_NAME} v{mv.version} "
                f"(stage={stage})"
            )
            return

    raise RuntimeError(
        f"No hay ningún modelo registrado en stage "
        f"{MODEL_STAGES_TO_TRY} para "
        f"'{REGISTERED_MODEL_NAME}'. "
        f"Corre src/training/train.py primero."
    )


@app.on_event("startup")
def startup_event():
    try:
        load_production_model()
    except Exception as e:  # noqa: BLE001
        # No se tumba el proceso: /health reportará not_ready y /predict
        # devolverá 503, en vez de que la API ni siquiera arranque.
        print(f"ADVERTENCIA: no se pudo cargar el modelo al iniciar: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "not_ready",
        "model_loaded": _model is not None,
    }

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Mide latencia y resultado de cada solicitud atendida por la API."""

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency = time.perf_counter() - start_time

        if request.url.path != "/metrics":
            system_monitor.record_request(
                latency_seconds=latency,
                status_code=500,
            )
        raise

    latency = time.perf_counter() - start_time

    if request.url.path != "/metrics":
        system_monitor.record_request(
            latency_seconds=latency,
            status_code=response.status_code,
        )

    return response

@app.get("/metrics")
def metrics():
    """Expone las métricas operativas acumuladas del servicio."""
    return system_monitor.get_metrics()

@app.get("/model-info")
def model_info():
    if _model_version_info is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ver /health.")
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "version": _model_version_info["version"],
        "stage": _model_version_info["stage"],
        "run_id": _model_version_info["run_id"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ver /health.")

    try:
        X = pd.DataFrame([req.model_dump()])
        # Reordenar columnas al orden exacto con el que se entrenó el modelo,
        # en vez de confiar en el orden en que llegaron en el JSON.
        if hasattr(_model, "feature_names_in_"):
            X = X[_model.feature_names_in_]
        forecast = float(_model.predict(X)[0])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Input inválido para el modelo: {e}") from e

    return PredictResponse(forecast=forecast, model_version=str(_model_version_info["version"]))
