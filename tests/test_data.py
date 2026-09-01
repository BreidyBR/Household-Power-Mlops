"""
Día 9 — Pruebas de datos: esquema, tipos, rangos, missing, variables
obligatorias. Corren sobre el dataset supervisado final
(data/processed/features_hourly.parquet), que es lo que el modelo y la
API realmente consumen.

Uso:
    pytest tests/test_data.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.training.train import DATA_PATH  # noqa: E402

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "rollmean_3h", "rollstd_3h", "rollmean_6h", "rollstd_6h",
    "rollmean_24h", "rollstd_24h", "rollmean_168h", "rollstd_168h",
]
TARGET_COLUMN = "target"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Corre 'python src/features/build_features.py' antes de los tests de datos.",
)


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(DATA_PATH)


def test_required_columns_present(df):
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    assert not missing, f"Faltan columnas obligatorias: {missing}"


def test_no_missing_values(df):
    """El dropna() de build_features.py debe garantizar 0 missing en el dataset final."""
    total_missing = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum()
    assert total_missing == 0, f"Hay {total_missing} valores faltantes en el dataset supervisado"


def test_feature_dtypes_are_numeric(df):
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} no es numérica ({df[col].dtype})"


def test_calendar_features_within_expected_ranges(df):
    assert df["hour"].between(0, 23).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["month"].between(1, 12).all()
    assert df["is_weekend"].isin([0, 1]).all()
    assert df["hour_sin"].between(-1.0001, 1.0001).all()
    assert df["hour_cos"].between(-1.0001, 1.0001).all()


def test_target_and_lags_non_negative(df):
    """Global_active_power (y por tanto target/lags/rollmean) nunca es negativo (ver Día 4, regla target_non_negative)."""
    lag_and_target_cols = [c for c in FEATURE_COLUMNS if c.startswith(("lag_", "rollmean_"))] + [TARGET_COLUMN]
    for col in lag_and_target_cols:
        assert (df[col] >= 0).all(), f"{col} tiene valores negativos (físicamente imposible)"


def test_temporal_index_is_sorted_and_unique(df):
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
