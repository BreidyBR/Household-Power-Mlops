"""
Día 9 — Pruebas de API: Request -> HTTP 200 -> Response schema válida, y
comportamiento ante input inválido.

Uso:
    pytest tests/test_api.py -v
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.main import app  # noqa: E402

VALID_PAYLOAD = {
    "hour": 20, "day_of_week": 5, "month": 12, "is_weekend": 1,
    "hour_sin": -1.0, "hour_cos": 0.0,
    "lag_1h": 1.5, "lag_2h": 1.4, "lag_3h": 1.3, "lag_24h": 1.6, "lag_48h": 1.5, "lag_168h": 1.4,
    "rollmean_3h": 1.4, "rollstd_3h": 0.1, "rollmean_6h": 1.3, "rollstd_6h": 0.2,
    "rollmean_24h": 1.1, "rollstd_24h": 0.3, "rollmean_168h": 1.0, "rollstd_168h": 0.3,
}


@pytest.fixture(scope="module")
def client():
    # TestClient debe usarse como context manager para que dispare el
    # evento de startup (donde se carga el modelo desde MLflow) -- si se
    # instancia "a secas", el startup nunca corre y /predict siempre
    # respondería 503 aunque exista un modelo registrado.
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_model_info_matches_registry(client):
    resp = client.get("/model-info")
    if resp.status_code == 503:
        pytest.xfail("No hay modelo registrado en MLflow todavía.")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "household-power-forecaster"
    assert "version" in body
    assert "stage" in body


def test_predict_valid_input_returns_200_with_valid_schema(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    if resp.status_code == 503:
        pytest.xfail("No hay modelo registrado en MLflow todavía (corre src/training/train.py).")
    assert resp.status_code == 200
    body = resp.json()
    assert "forecast" in body and isinstance(body["forecast"], float)
    assert body["horizon"] == "24h"
    assert "model_version" in body


def test_predict_missing_field_returns_422(client):
    incomplete = VALID_PAYLOAD.copy()
    del incomplete["lag_1h"]
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["hour"] = "veinte"  # tipo incorrecto, sección Q del enunciado
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_empty_body_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
