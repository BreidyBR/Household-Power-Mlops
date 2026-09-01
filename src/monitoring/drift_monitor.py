"""
Monitoreo de Data Drift mediante Population Stability Index (PSI).

Compara la distribución de las variables del conjunto REFERENCE
contra los batches de producción.
"""

import numpy as np
import pandas as pd

from src.monitoring.production_batches import (
    TARGET,
    load_monitoring_data,
    split_reference_and_production,
)


PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

N_BINS = 10


def calculate_psi(reference, current, n_bins=N_BINS):
    """
    Calcula Population Stability Index (PSI) entre dos distribuciones.

    Los bins se construyen con cuantiles de la distribución de referencia.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return np.nan

    quantiles = np.linspace(0, 1, n_bins + 1)

    bin_edges = np.unique(
        reference.quantile(quantiles).to_numpy()
    )

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts = pd.cut(
        reference,
        bins=bin_edges,
        include_lowest=True,
    ).value_counts(sort=False)

    current_counts = pd.cut(
        current,
        bins=bin_edges,
        include_lowest=True,
    ).value_counts(sort=False)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-6

    reference_pct = reference_pct.clip(lower=epsilon)
    current_pct = current_pct.clip(lower=epsilon)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def classify_psi(psi_value):
    """Clasifica el nivel de drift según los umbrales del proyecto."""

    if np.isnan(psi_value):
        return "UNAVAILABLE"

    if psi_value >= PSI_ALERT_THRESHOLD:
        return "ALERT"

    if psi_value >= PSI_WARNING_THRESHOLD:
        return "WARNING"

    return "OK"


def evaluate_batch_drift(reference, batch):
    """
    Calcula PSI para todas las features numéricas del batch.

    La variable target no se utiliza para detectar P(X) drift.
    """

    feature_columns = [
        column
        for column in reference.columns
        if column != TARGET
    ]

    results = []

    for column in feature_columns:
        psi_value = calculate_psi(
            reference[column],
            batch[column],
        )

        results.append(
            {
                "feature": column,
                "psi": psi_value,
                "status": classify_psi(psi_value),
            }
        )

    return pd.DataFrame(results)


def print_drift_summary(batch_name, results):
    """Muestra un resumen del drift detectado en un batch."""

    print()
    print(f"=== {batch_name} ===")

    status_counts = (
        results["status"]
        .value_counts()
        .to_dict()
    )

    print("Estados:", status_counts)

    top_drift = (
        results
        .sort_values("psi", ascending=False)
        .head(5)
    )

    print()
    print("Top 5 features por PSI:")

    for _, row in top_drift.iterrows():
        print(
            f"{row['feature']}: "
            f"PSI={row['psi']:.4f} | "
            f"{row['status']}"
        )


def simulate_drift(batch):
    """
    Simula un cambio controlado en P(X) sobre una copia del batch.

    Se alteran variables relacionadas con el consumo eléctrico para
    representar un escenario futuro con mayor nivel de consumo.
    El batch original permanece intacto.
    """

    drifted_batch = batch.copy()

    features_to_shift = [
        "lag_24h",
        "lag_168h",
        "rollmean_24h",
        "rollmean_168h",
    ]

    drifted_batch[features_to_shift] = (
        drifted_batch[features_to_shift] * 1.35
    )

    return drifted_batch


def main():
    df = load_monitoring_data()

    reference, batch_1, batch_2, batch_3 = (
        split_reference_and_production(df)
    )

    batches = {
        "PRODUCTION_BATCH_1": batch_1,
        "PRODUCTION_BATCH_2": batch_2,
        "PRODUCTION_BATCH_3": batch_3,
    }

    print("=== DATA DRIFT MONITORING - PSI ===")
    print(
        f"Umbrales: "
        f"WARNING >= {PSI_WARNING_THRESHOLD}, "
        f"ALERT >= {PSI_ALERT_THRESHOLD}"
    )

    for batch_name, batch in batches.items():
        results = evaluate_batch_drift(
            reference,
            batch,
        )

        print_drift_summary(
            batch_name,
            results,
        )

    drifted_batch_3 = simulate_drift(batch_3)

    drifted_results = evaluate_batch_drift(
        reference,
        drifted_batch_3,
    )

    print()
    print("=== SIMULATED DRIFT - PRODUCTION_BATCH_3 ===")
    print(
        "Se aplicó un incremento controlado del 35% "
        "a variables históricas de consumo."
    )

    print_drift_summary(
        "PRODUCTION_BATCH_3_DRIFTED",
        drifted_results,
    )


if __name__ == "__main__":
    main()