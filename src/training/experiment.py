"""
Día 8 — Experiment Tracking (MLflow), sección J del enunciado.

Este script NO vuelve a decidir qué modelo usar: esa investigación y
justificación ya se hizo en notebooks/04_training.ipynb (Día 7), comparando
baseline, Regresión Lineal, Random Forest, Gradient Boosting,
HistGradientBoosting y una versión regularizada de Random Forest. Aquí se
toma exactamente esa misma comparación y se instrumenta con MLflow, para que
cada modelo probado quede registrado como un run auditable: qué algoritmo,
con qué hiperparámetros, sobre qué datos, con qué resultado.

Cada run registra como mínimo (sección J):
    Parameters: algorithm, hyperparameters, feature_set, random_seed, data_version
    Metrics:    MAE, RMSE, sMAPE
    Artifacts:  modelo, gráfico de residuales, configuración

Uso:
    python src/training/experiment.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlruns
"""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.training.train import (  # noqa: E402
    RANDOM_STATE,
    data_version,
    ensure_experiment,
    load_data,
    smape,
    temporal_split,
)


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(smape(y_true, y_pred)),
    }


def plot_residuals(y_true, y_pred, title, out_path):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50)
    ax.set_title(f"Residuales — {title}")
    ax.set_xlabel("Error (kW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def log_run(run_name, algorithm, params, metrics, y_valid, y_pred, model, feature_columns, dv, tmp_dir):
    """
    Loggea un run completo a MLflow: parámetros mínimos exigidos, métricas,
    y artifacts (modelo si existe, gráfico de residuales, configuración).
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "algorithm": algorithm,
            "feature_set": f"{len(feature_columns)} features (calendario+lags+rolling)",
            "random_seed": RANDOM_STATE,
            "data_version": dv,
            **{f"hp_{k}": v for k, v in params.items()},
        })
        mlflow.log_metrics(metrics)

        residual_path = tmp_dir / f"{run_name}_residuals.png"
        plot_residuals(y_valid, y_pred, run_name, residual_path)
        mlflow.log_artifact(str(residual_path))

        config_path = tmp_dir / f"{run_name}_config.json"
        config_path.write_text(json.dumps({
            "algorithm": algorithm,
            "hyperparameters": params,
            "feature_columns": feature_columns,
            "random_seed": RANDOM_STATE,
            "data_version": dv,
        }, indent=2))
        mlflow.log_artifact(str(config_path))

        if model is not None:
            mlflow.sklearn.log_model(model, name="model")

        print(f"[{run_name}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  sMAPE={metrics['smape']:.2f}%")
        return run.info.run_id, metrics


def main():
    print("=== INICIO DE EXPERIMENT TRACKING (Dia 8) ===")

    ensure_experiment()

    df = load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = temporal_split(df)
    feature_columns = X_train.columns.tolist()
    dv = data_version()

    print(f"Dataset: {len(df):,} filas, data_version={dv}")
    print(f"Train={len(X_train):,} Valid={len(X_valid):,} Test={len(X_test):,}")
    print()

    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ---- Run 0: baseline estacional (lag_24h) ----
        # Justificado en notebooks/04_training.ipynb, sección 3: el EDA
        # mostró autocorrelación fuerte en el ciclo diario, por lo que el
        # baseline naive usa el consumo observado como referencia mínima.
        y_pred = X_valid["lag_24h"]
        metrics = evaluate(y_valid, y_pred)
        _, results["baseline"] = log_run(
            "baseline_lag24h", "baseline_seasonal_naive", {}, metrics,
            y_valid, y_pred, None, feature_columns, dv, tmp_dir,
        )

        # ---- Run 1: Regresión Lineal ----
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)
        _, results["linear_regression"] = log_run(
            "linear_regression", "LinearRegression", {}, metrics,
            y_valid, y_pred, model, feature_columns, dv, tmp_dir,
        )

        # ---- Run 2: Random Forest (sin regularizar) ----
        params = {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)
        _, results["random_forest"] = log_run(
            "random_forest", "RandomForestRegressor", params, metrics,
            y_valid, y_pred, model, feature_columns, dv, tmp_dir,
        )

        # ---- Run 3: Gradient Boosting ----
        params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_STATE}
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)
        _, results["gradient_boosting"] = log_run(
            "gradient_boosting", "GradientBoostingRegressor", params, metrics,
            y_valid, y_pred, model, feature_columns, dv, tmp_dir,
        )

        # ---- Run 4: Random Forest regularizado (candidato final) ----
        params = {
            "n_estimators": 100, "max_depth": 15, "min_samples_split": 10,
            "min_samples_leaf": 5, "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)
        run_id, results["random_forest_regularized"] = log_run(
            "random_forest_regularized", "RandomForestRegressor", params, metrics,
            y_valid, y_pred, model, feature_columns, dv, tmp_dir,
        )

        # ---- Run 5: HistGradientBoosting ----
        params = {"learning_rate": 0.1, "max_iter": 100, "max_depth": 10, "random_state": RANDOM_STATE}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)
        _, results["hist_gradient_boosting"] = log_run(
            "hist_gradient_boosting", "HistGradientBoostingRegressor", params, metrics,
            y_valid, y_pred, model, feature_columns, dv, tmp_dir,
        )

    print()
    print("=== RESUMEN (validacion) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["mae"]):
        print(f"{name:28s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  sMAPE={m['smape']:.2f}%")

    best_name = min(results, key=lambda k: results[k]["mae"])
    print()
    print(f"Mejor candidato por MAE en validacion: {best_name}")
    print(
        "Criterio de seleccion completo (documentado en notebooks/04_training.ipynb, "
        "seccion 10): confirmado ademas con TimeSeriesSplit de 5 particiones antes de "
        "elegir el modelo final. El registro en el Model Registry se hace en "
        "src/training/train.py, sobre el modelo ya seleccionado."
    )
    print("=== EXPERIMENT TRACKING COMPLETADO ===")


if __name__ == "__main__":
    main()
