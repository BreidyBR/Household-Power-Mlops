"""
Monitoreo de calidad de datos en producción.

Este módulo:
1. Simula contaminación sobre una COPIA de un batch de producción.
2. Detecta problemas de calidad.
3. Decide si debe continuar, advertir o bloquear.
4. Registra los resultados en un reporte JSON.

El dataset original nunca se modifica.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.monitoring.production_batches import (
    load_monitoring_data,
    split_reference_and_production,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "quality_report.json"
)

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"


def contaminate_batch(batch):
    """
    Introduce problemas de calidad sobre una copia del batch.

    Simula:
    - missing value
    - fila duplicada
    - outlier extremo
    - datatype incorrecto
    - modificación del esquema
    """

    contaminated = batch.copy()

    # 1. Missing value
    contaminated.loc[
        contaminated.index[0],
        "lag_24h",
    ] = pd.NA

    # 2. Fila duplicada
    duplicated_row = contaminated.iloc[[1]].copy()

    contaminated = pd.concat(
        [contaminated, duplicated_row]
    )

    # 3. Outlier extremo
    contaminated.loc[
        contaminated.index[2],
        "lag_168h",
    ] = 9999.0

    # 4. Datatype incorrecto
    contaminated["rollmean_24h"] = (
        contaminated["rollmean_24h"].astype("object")
    )

    contaminated.loc[
        contaminated.index[3],
        "rollmean_24h",
    ] = "INVALID_VALUE"

    # 5. Modificación del esquema
    contaminated["unexpected_column"] = "unexpected"

    return contaminated


def check_schema(reference, batch):
    """Detecta columnas faltantes o inesperadas."""

    expected = set(reference.columns)
    received = set(batch.columns)

    missing_columns = sorted(expected - received)
    unexpected_columns = sorted(received - expected)

    if missing_columns or unexpected_columns:
        return {
            "check": "schema",
            "status": FAIL,
            "action": "BLOCK",
            "detail": (
                f"Columnas faltantes: {missing_columns}; "
                f"columnas inesperadas: {unexpected_columns}"
            ),
        }

    return {
        "check": "schema",
        "status": PASS,
        "action": "CONTINUE",
        "detail": "El esquema coincide con el esperado.",
    }


def check_missing(batch):
    """Detecta valores faltantes."""

    missing = batch.isna().sum()
    affected = missing[missing > 0].to_dict()

    if affected:
        return {
            "check": "missing_values",
            "status": FAIL,
            "action": "BLOCK",
            "detail": f"Valores faltantes detectados: {affected}",
        }

    return {
        "check": "missing_values",
        "status": PASS,
        "action": "CONTINUE",
        "detail": "No se detectaron valores faltantes.",
    }


def check_duplicates(batch):
    """Detecta filas o timestamps duplicados."""

    duplicated_rows = int(batch.duplicated().sum())
    duplicated_index = int(batch.index.duplicated().sum())

    if duplicated_rows > 0 or duplicated_index > 0:
        return {
            "check": "duplicates",
            "status": WARNING,
            "action": "WARN",
            "detail": (
                f"Filas duplicadas: {duplicated_rows}; "
                f"timestamps duplicados: {duplicated_index}"
            ),
        }

    return {
        "check": "duplicates",
        "status": PASS,
        "action": "CONTINUE",
        "detail": "No se detectaron duplicados.",
    }


def check_dtypes(reference, batch):
    """Compara los tipos de datos contra el batch de referencia."""

    mismatches = {}

    for column in reference.columns:
        if column in batch.columns:
            expected_dtype = str(reference[column].dtype)
            current_dtype = str(batch[column].dtype)

            if expected_dtype != current_dtype:
                mismatches[column] = {
                    "expected": expected_dtype,
                    "received": current_dtype,
                }

    if mismatches:
        return {
            "check": "data_types",
            "status": FAIL,
            "action": "BLOCK",
            "detail": f"Tipos incompatibles: {mismatches}",
        }

    return {
        "check": "data_types",
        "status": PASS,
        "action": "CONTINUE",
        "detail": "Los tipos de datos coinciden con los esperados.",
    }


def check_extreme_outliers(reference, batch):
    """
    Detecta valores fuera de límites robustos calculados
    a partir de la referencia mediante IQR.
    """

    outliers = {}

    numeric_columns = reference.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        if column not in batch.columns:
            continue

        current_values = pd.to_numeric(
            batch[column],
            errors="coerce",
        )

        q1 = reference[column].quantile(0.25)
        q3 = reference[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (5 * iqr)
        upper_bound = q3 + (5 * iqr)

        count = int(
            (
                (current_values < lower_bound)
                | (current_values > upper_bound)
            ).sum()
        )

        if count > 0:
            outliers[column] = count

    if outliers:
        return {
            "check": "extreme_outliers",
            "status": WARNING,
            "action": "WARN",
            "detail": (
                "Valores extremos detectados "
                f"con límites de 5*IQR: {outliers}"
            ),
        }

    return {
        "check": "extreme_outliers",
        "status": PASS,
        "action": "CONTINUE",
        "detail": "No se detectaron outliers extremos.",
    }


def validate_production_batch(reference, batch):
    """Ejecuta todas las validaciones del batch."""

    return [
        check_schema(reference, batch),
        check_missing(batch),
        check_duplicates(batch),
        check_dtypes(reference, batch),
        check_extreme_outliers(reference, batch),
    ]


def determine_overall_action(results):
    """
    FAIL    -> BLOCK
    WARNING -> WARN
    PASS    -> CONTINUE
    """

    if any(result["status"] == FAIL for result in results):
        return "BLOCK"

    if any(
        result["status"] == WARNING
        for result in results
    ):
        return "WARN"

    return "CONTINUE"


def save_quality_report(results, overall_action):
    """Guarda evidencia auditable del monitoreo."""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "overall_action": overall_action,
        "results": results,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return REPORT_PATH


def main():
    df = load_monitoring_data()

    reference, _, _, batch_3 = (
        split_reference_and_production(df)
    )

    contaminated_batch = contaminate_batch(
        batch_3
    )

    print("=== QUALITY MONITORING ===")
    print()
    print(
        f"Batch original: {len(batch_3):,} filas"
    )
    print(
        "Batch contaminado: "
        f"{len(contaminated_batch):,} filas"
    )
    print()

    results = validate_production_batch(
        reference,
        contaminated_batch,
    )

    for result in results:
        print(
            f"[{result['check']}] "
            f"{result['status']} -> "
            f"{result['action']}"
        )
        print(f"  {result['detail']}")

    overall_action = determine_overall_action(
        results
    )

    report_path = save_quality_report(
        results,
        overall_action,
    )

    print()
    print(
        f"DECISIÓN FINAL: {overall_action}"
    )
    print(
        f"Reporte guardado en: {report_path}"
    )


if __name__ == "__main__":
    main()