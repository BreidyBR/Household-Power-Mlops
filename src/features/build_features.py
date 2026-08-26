from pathlib import Path

import numpy as np
import pandas as pd


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Archivo generado por la etapa de limpieza (src/cleaning/clean.py)
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "household_power_cleaned.parquet"

# Directorio y archivo de salida de esta etapa
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "features_hourly.parquet"

# Variable objetivo del proyecto
TARGET_COL = "Global_active_power"

# Horizonte de predicción: se pronostica el consumo 24 horas hacia adelante.
# Justificado en notebooks/03_feature_engineering.ipynb (sección 4) y coincide
# con el ciclo diario dominante (autocorrelación fuerte en el lag 24h,
# ver notebooks/02_eda.ipynb) y corresponde a un caso de uso realista
# ("consumo de mañana a esta hora").
HORIZON_HOURS = 24

# Lags (en horas) usados como features. Justificados en
# notebooks/03_feature_engineering.ipynb (sección 6): cada uno coincide con
# un pico real de autocorrelación encontrado en el EDA, no son valores
# arbitrarios (1-3h: dependencia de corto plazo; 24h/48h: ciclo diario;
# 168h: ciclo semanal).
LAGS_HOURS = [1, 2, 3, 24, 48, 168]

# Ventanas (en horas) para rolling mean/std. Mismo criterio que los lags.
ROLLING_WINDOWS_HOURS = [3, 6, 24, 168]

# Duración máxima (en minutos) de una racha de valores faltantes que se
# considera segura para interpolar linealmente, como último recurso de la
# imputación en cascada (ver build_hourly_series). Mismo valor que
# GAP_LIMIT_MINUTES en src/cleaning/clean.py.
GAP_LIMIT_MINUTES = 60


def load_interim():
    """Carga el dataset limpio generado por src/cleaning/clean.py."""

    if not INTERIM_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el dataset limpio. Ejecuta primero src/cleaning/clean.py."
        )

    return pd.read_parquet(INTERIM_FILE)


def build_hourly_series(df_interim):
    """
    Agrega el dataset limpio (nivel minuto) a frecuencia horaria y aplica la
    imputación en cascada de las horas sin ningún dato.

    Frecuencia horaria justificada en notebooks/03_feature_engineering.ipynb
    (sección 2): conserva el patrón intradía y coincide con las unidades ya
    definidas para lags/rolling.

    Imputación justificada en notebooks/03_feature_engineering.ipynb (sección
    3): sin imputar antes de construir lags/rolling, el missing se propaga de
    1.22% a  aproximadamente 5.10% de las filas (porque el lag más largo, 168h, arrastra
    cualquier hora faltante hasta 168 filas hacia adelante). Se usa una
    estrategia estacional en cascada, no interpolación lineal directa:

        1. valor de 168h antes (mismo día/hora de la semana anterior)
        2. si también falta, valor de 24h antes (mismo momento del día anterior)
        3. si también falta, interpolación temporal (último recurso)

    Devuelve también `filled_via_168h`, una máscara booleana de las horas que
    se imputaron con el método (1). La usan los chequeos de leakage para no
    confundir esa coincidencia esperada con una fuga de datos real.
    """

    hourly = df_interim[TARGET_COL].resample("h").mean()
    missing_mask = hourly.isna()

    filled = hourly.fillna(hourly.shift(168))
    filled_via_168h = missing_mask & filled.notna()

    filled = filled.fillna(hourly.shift(24))
    filled = filled.interpolate(method="time")

    return filled, filled_via_168h


def add_calendar_features(features, index):
    """
    Agrega variables de calendario, respaldadas por los patrones cuantificados
    en notebooks/02_eda.ipynb: hour/hour_sin/hour_cos por el patrón intradía,
    day_of_week/is_weekend por el patrón semanal, month por la estacionalidad
    anual.
    """

    features = features.copy()
    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["month"] = index.month
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)

    # Codificación cíclica de la hora: evita la discontinuidad artificial
    # entre 23h y 0h que tendría usar la hora como número entero.
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

    return features


def add_lag_features(features, series, lags=LAGS_HOURS):
    """Agrega columnas lag_Nh = valor de la serie N horas antes de cada fila."""

    features = features.copy()
    for lag in lags:
        features[f"lag_{lag}h"] = series.shift(lag)
    return features


def add_rolling_features(features, series, windows=ROLLING_WINDOWS_HOURS):
    """
    Agrega rolling mean/std sobre `series` desplazada un paso (shift(1))
    antes de aplicar el rolling. Esto es lo que garantiza que la ventana en
    el instante t resuma horas ANTERIORES a t y nunca incluya el propio
    valor de t (evita data leakage).
    """

    features = features.copy()
    shifted = series.shift(1)
    for w in windows:
        features[f"rollmean_{w}h"] = shifted.rolling(w).mean()
        features[f"rollstd_{w}h"] = shifted.rolling(w).std()
    return features


def add_target(features, series, horizon=HORIZON_HOURS):
    """Agrega la columna target = valor de la serie `horizon` horas después de cada fila."""

    features = features.copy()
    features["target"] = series.shift(-horizon)
    return features


def verify_no_leakage(features, series, filled_via_168h, horizon=HORIZON_HOURS, lags=LAGS_HOURS):
    """
    Verificación explícita de que ninguna feature usa información no
    disponible en el momento de la predicción (ver
    notebooks/03_feature_engineering.ipynb, sección 8).

    Se excluyen de la comprobación de lag_168h las filas que fueron
    imputadas con el valor de 168h antes (build_hourly_series): para esas
    filas, series[t] == series[t-168] es un resultado esperado de la
    imputación, no una fuga de datos.
    """

    sample_t = features.index[len(features) // 4]

    for lag in lags:
        expected = series.loc[sample_t - pd.Timedelta(hours=lag)]
        actual = features.loc[sample_t, f"lag_{lag}h"]
        assert np.isclose(actual, expected), f"lag_{lag}h no corresponde a t-{lag}h"

    expected_target = series.loc[sample_t + pd.Timedelta(hours=horizon)]
    assert np.isclose(features.loc[sample_t, "target"], expected_target), (
        "target no corresponde a t+{horizon}h"
    )

    contemporaneous = series.loc[features.index]
    imputed_via_168h = filled_via_168h.reindex(features.index, fill_value=False)

    for col in [c for c in features.columns if c.startswith(("lag_", "rollmean_", "rollstd_"))]:
        mask = ~imputed_via_168h if col == "lag_168h" else pd.Series(True, index=features.index)
        exact_match = (features.loc[mask, col] == contemporaneous.loc[mask]).mean()
        assert exact_match < 0.01, f"{col} coincide sospechosamente con el valor contemporaneo de t"

    print("Verificacion de leakage completada: sin problemas detectados.")


def build_features(df_interim):
    """Pipeline completo: dataset limpio (nivel minuto) -> dataset supervisado listo para entrenar."""

    hourly_filled, filled_via_168h = build_hourly_series(df_interim)

    features = pd.DataFrame(index=hourly_filled.index)
    features = add_calendar_features(features, features.index)
    features = add_lag_features(features, hourly_filled)
    features = add_rolling_features(features, hourly_filled)
    features = add_target(features, hourly_filled)

    verify_no_leakage(features, hourly_filled, filled_via_168h)

    dataset = features.dropna()
    return dataset


def main():
    """Punto de entrada del proceso de feature engineering."""

    print("=== INICIO DE FEATURE ENGINEERING ===")

    df_interim = load_interim()
    print(f"Dataset limpio cargado: {df_interim.shape[0]:,} filas")

    dataset = build_features(df_interim)
    print(f"Dataset supervisado final: {dataset.shape[0]:,} filas, {dataset.shape[1]} columnas")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_FILE)

    print(f"Dataset guardado en: {PROCESSED_FILE}")
    print("=== FEATURE ENGINEERING COMPLETADO ===")


if __name__ == "__main__":
    main()
