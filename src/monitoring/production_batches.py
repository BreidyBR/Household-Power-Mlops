"""
Construcción de referencia y lotes temporales de producción.

La referencia corresponde al 85% histórico usado durante desarrollo
(train + validation). El 15% final se reserva como simulación de
producción y se divide cronológicamente en tres batches.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_hourly.parquet"
)

TARGET = "target"

REFERENCE_RATIO = 0.85
N_PRODUCTION_BATCHES = 3


def load_monitoring_data():
    """Carga el dataset horario preparado para entrenamiento y monitoreo."""

    df = pd.read_parquet(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{TARGET}'."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "El índice temporal debe estar ordenado de forma ascendente."
        )

    return df


def split_reference_and_production(df):
    """
    Divide cronológicamente el dataset en:

    - Reference: 85% histórico.
    - Production Batch 1
    - Production Batch 2
    - Production Batch 3

    No se realiza ningún shuffle para evitar fuga temporal.
    """

    reference_end = int(len(df) * REFERENCE_RATIO)

    reference = df.iloc[:reference_end].copy()
    production = df.iloc[reference_end:].copy()

    batch_size = len(production) // N_PRODUCTION_BATCHES

    batch_1 = production.iloc[:batch_size].copy()
    batch_2 = production.iloc[batch_size:2 * batch_size].copy()
    batch_3 = production.iloc[2 * batch_size:].copy()

    return reference, batch_1, batch_2, batch_3


def describe_split(reference, batch_1, batch_2, batch_3):
    """Imprime tamaños y rangos temporales para validar la división."""

    groups = {
        "REFERENCE": reference,
        "PRODUCTION_BATCH_1": batch_1,
        "PRODUCTION_BATCH_2": batch_2,
        "PRODUCTION_BATCH_3": batch_3,
    }

    for name, data in groups.items():
        print(
            f"{name}: "
            f"{len(data):,} filas | "
            f"{data.index.min()} -> {data.index.max()}"
        )


def main():
    df = load_monitoring_data()

    reference, batch_1, batch_2, batch_3 = (
        split_reference_and_production(df)
    )

    print("=== MONITORING DATA SPLIT ===")
    print(f"Dataset total: {len(df):,} filas")
    print()

    describe_split(
        reference,
        batch_1,
        batch_2,
        batch_3,
    )

    total_after_split = (
        len(reference)
        + len(batch_1)
        + len(batch_2)
        + len(batch_3)
    )

    print()
    print(
        "Filas conservadas después del split:",
        total_after_split,
    )

    if total_after_split != len(df):
        raise RuntimeError(
            "La división perdió o duplicó filas."
        )

    if not (
        reference.index.max() < batch_1.index.min()
        < batch_2.index.min()
        < batch_3.index.min()
    ):
        raise RuntimeError(
            "Los grupos no respetan el orden temporal."
        )

    print("Split temporal validado correctamente.")


if __name__ == "__main__":
    main()