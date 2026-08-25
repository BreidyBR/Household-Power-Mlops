from pathlib import Path

import numpy as np
import pandas as pd


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Archivo generado por el pipeline de ingesta
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"

# Directorio y archivo de salida de esta etapa
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_FILE = INTERIM_DIR / "household_power_cleaned.parquet"

# Variables eléctricas que deben convertirse a tipo numérico
NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Duración máxima (en minutos) de una racha de valores faltantes que se
# considera aceptable para interpolación temporal dentro de este pipeline.
# El umbral se definió a partir del análisis documentado en
# notebooks/01_data_quality_cleaning.ipynb (Parte II, sección 3).
#
# Las rachas de hasta 60 minutos representan una fracción muy pequeña del
# volumen total de datos faltantes, por lo que se adopta este umbral como una
# estrategia conservadora. Las rachas más largas se mantienen como NaN para
# evitar reconstruir períodos prolongados sin observaciones reales.
GAP_LIMIT_MINUTES = 60


def load_raw():
    """Carga el dataset crudo como texto, preservando su representación original."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el dataset raw. Ejecuta primero src/ingestion/ingest.py."
        )

    return pd.read_csv(RAW_FILE, sep=";", dtype=str)


def build_temporal_index(df):
    """Construye el índice temporal a partir de Date y Time, y descarta ambas columnas."""

    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    df = df.set_index("datetime").sort_index()
    df = df.drop(columns=["Date", "Time"])

    return df


def convert_numeric_types(df):
    """Convierte las variables eléctricas a tipo numérico, tratando '?' como valor faltante."""

    df = df.copy()

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].replace("?", np.nan)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric)

    return df


def compute_gap_sizes(missing_mask):
    """Para cada fila faltante, calcula la duración (en minutos) de la racha a la que pertenece."""

    gap_id = missing_mask.ne(missing_mask.shift()).cumsum()
    gap_sizes = missing_mask.groupby(gap_id).transform("sum")

    return gap_sizes


def interpolate_short_gaps(df, gap_limit_minutes=GAP_LIMIT_MINUTES):
    """
    Interpola linealmente (basado en tiempo) únicamente las rachas de valores
    faltantes cuya duración es <= gap_limit_minutes. Las rachas más largas se
    dejan intencionalmente como NaN: su tratamiento se define en Feature
    Engineering, en conjunto con la frecuencia de resampling elegida ahí.
    """

    df = df.copy()

    # Las siete variables faltan simultáneamente (ver notebook, Parte I),
    # por lo que una sola columna basta para identificar las rachas.
    missing_mask = df["Global_active_power"].isna()
    gap_sizes = compute_gap_sizes(missing_mask)

    interpolated = df[NUMERIC_COLUMNS].interpolate(method="time")
    fillable_mask = missing_mask & (gap_sizes <= gap_limit_minutes)

    df.loc[fillable_mask, NUMERIC_COLUMNS] = interpolated.loc[fillable_mask, NUMERIC_COLUMNS]

    return df


def clean(df_raw):
    """Aplica el pipeline completo de limpieza sobre el dataset crudo."""

    df = build_temporal_index(df_raw)
    df = convert_numeric_types(df)
    df = interpolate_short_gaps(df)

    return df


def validate_clean_dataset(df, df_raw):
    """Verificaciones mínimas de que la limpieza no corrompió la estructura del dataset."""

    assert df.shape[0] == df_raw.shape[0], "La limpieza no debe eliminar filas."
    assert df.index.is_monotonic_increasing, "El índice temporal debe ser estrictamente creciente."
    assert df.index.duplicated().sum() == 0, "No debe haber timestamps duplicados."
    assert (df[NUMERIC_COLUMNS] < 0).sum().sum() == 0, "No deben existir valores negativos."

    print("Validación del dataset limpio completada.")
    print(f"Filas: {df.shape[0]:,}")
    print(f"Missing restante (rachas largas, sin imputar): {df['Global_active_power'].isna().sum():,}")


def main():
    """Punto de entrada del proceso de limpieza."""

    print("=== INICIO DEL PIPELINE DE LIMPIEZA ===")

    df_raw = load_raw()
    print(f"Dataset raw cargado: {df_raw.shape[0]:,} filas")

    df_clean = clean(df_raw)
    validate_clean_dataset(df_clean, df_raw)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(INTERIM_FILE)

    print(f"Dataset limpio guardado en: {INTERIM_FILE}")
    print("=== LIMPIEZA COMPLETADA ===")


if __name__ == "__main__":
    main()
