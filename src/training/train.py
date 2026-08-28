import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_hourly.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"

TARGET = "target"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

RANDOM_STATE = 42

# --- MLflow (Día 8) ---
# MLflow >= 3.0 puso en modo mantenimiento el backend de tracking de solo
# archivos; se usa SQLite para el tracking store (metadata de runs/params/
# metrics/registry) y se mantienen los artefactos (modelos, gráficos) en
# disco local bajo mlruns/, igual que antes.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = f"file:{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "household-power-forecasting"
REGISTERED_MODEL_NAME = "household-power-forecaster"

MODEL_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# FUNCIONES
# ============================================================

def load_data():
    """
    Carga el dataset procesado generado durante
    la etapa de Feature Engineering.
    """

    df = pd.read_parquet(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{TARGET}'."
        )

    return df


def temporal_split(df):
    """
    Divide los datos respetando el orden temporal.

    Train:      70%
    Validation: 15%
    Test:       15%
    """

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]

    X_test = X.iloc[valid_end:]
    y_test = y.iloc[valid_end:]

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )


def build_model():
    """
    Construye el Random Forest regularizado seleccionado
    durante la etapa de experimentación.
    """

    return RandomForestRegressor(**MODEL_HYPERPARAMS)


def data_version():
    """
    Hash SHA-256 (primeros 12 caracteres) del dataset usado en esta corrida,
    para poder responder con certeza "¿con qué datos exactos se entrenó este
    modelo?" (ver src/training/experiment.py, misma función).
    """

    h = hashlib.sha256()
    with open(DATA_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def ensure_experiment():
    """
    Crea el experimento de MLflow con `artifact_location` explícito la
    primera vez, y lo reutiliza en corridas posteriores. Ver
    src/training/experiment.py para el detalle de por qué es necesario.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT_NAME)


def plot_residuals(y_true, y_pred, out_path):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50)
    ax.set_title("Residuales — modelo final (test)")
    ax.set_xlabel("Error (kW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_predictions(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true.index, y_true.values, label="real", linewidth=0.8)
    ax.plot(y_true.index, y_pred, label="predicción", linewidth=0.8, alpha=0.8)
    ax.set_title("Predicciones vs. valores reales — test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def log_and_register_model(model, metrics, feature_columns, X_test, y_test, y_pred):
    """
    Sección J + K del enunciado, sobre el modelo YA seleccionado (el
    criterio de selección se documentó y ejecutó en
    notebooks/04_training.ipynb, sección 10-11, y se reprodujo como
    comparación de candidatos en src/training/experiment.py):

    1. Loggea el run final a MLflow: parámetros, métricas de test,
       artifacts (modelo, gráficos, configuración).
    2. Registra el modelo en el Model Registry y lo transiciona por el
       ciclo Experiment -> Candidate -> Validation -> Production:
       - "Experiment": las 6 corridas de src/training/experiment.py.
       - "Candidate": este run (reentrenado con train+validation).
       - "Validation": ya superada antes de llegar aquí (TimeSeriesSplit
         de 5 particiones, ver notebook sección 10) — por eso se
         transiciona directo a Staging.
       - "Production": promoción final, explícita, tras confirmar el
         desempeño sobre el test set separado (metrics de esta función).
    """

    ensure_experiment()
    dv = data_version()

    with mlflow.start_run(run_name="random_forest_regularized_final") as run:
        mlflow.log_params({
            "algorithm": "RandomForestRegressor",
            "feature_set": f"{len(feature_columns)} features (calendario+lags+rolling)",
            "random_seed": RANDOM_STATE,
            "data_version": dv,
            **{f"hp_{k}": v for k, v in MODEL_HYPERPARAMS.items()},
        })
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            residual_path = tmp_dir / "residuals.png"
            plot_residuals(y_test, y_pred, residual_path)
            mlflow.log_artifact(str(residual_path))

            predictions_path = tmp_dir / "predictions.png"
            plot_predictions(y_test, y_pred, predictions_path)
            mlflow.log_artifact(str(predictions_path))

            config_path = tmp_dir / "config.json"
            config_path.write_text(json.dumps({
                "algorithm": "RandomForestRegressor",
                "hyperparameters": MODEL_HYPERPARAMS,
                "feature_columns": feature_columns,
                "random_seed": RANDOM_STATE,
                "data_version": dv,
            }, indent=2))
            mlflow.log_artifact(str(config_path))

        mlflow.sklearn.log_model(model, name="model")

        # register_model crea el modelo registrado si es la primera vez, o
        # agrega una nueva versión si ya existe -- no requiere distinguir
        # ambos casos a mano.
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=mv.version, stage="Staging",
        )
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=mv.version, stage="Production",
        )

        print(
            f"Modelo '{REGISTERED_MODEL_NAME}' v{mv.version} registrado y "
            "promovido a Production (Experiment -> Candidate -> Validation -> Production)."
        )


def smape(y_true, y_pred):
    """
    Calcula Symmetric Mean Absolute Percentage Error.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    return 200 * np.mean(
        np.abs(y_true[mask] - y_pred[mask])
        / denominator[mask]
    )


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo sobre el conjunto de test.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    smape_value = smape(y_test, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape_value,
    }


def save_model(model):
    """
    Guarda el modelo entrenado para su posterior uso.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)


# ============================================================
# PIPELINE DE ENTRENAMIENTO
# ============================================================

def main():

    print("=== HOUSEHOLD POWER MLOPS - TRAINING ===")

    # 1. Carga
    df = load_data()

    print()
    print("Dataset cargado correctamente")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {df.shape[1]}")

    # 2. Split temporal
    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = temporal_split(df)

    print()
    print("=== SPLIT TEMPORAL ===")
    print(f"Train:      {len(X_train):,}")
    print(f"Validation: {len(X_valid):,}")
    print(f"Test:       {len(X_test):,}")

    # 3. Combinar train + validation
    # El modelo ya fue seleccionado durante experimentación.
    X_dev = pd.concat([X_train, X_valid])
    y_dev = pd.concat([y_train, y_valid])

    # 4. Crear y entrenar modelo final
    model = build_model()

    print()
    print("Entrenando Random Forest regularizado...")

    model.fit(X_dev, y_dev)

    print("Entrenamiento completado.")

    # 5. Evaluación final
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print()
    print("=== EVALUACIÓN FINAL ===")
    print(f"MAE:   {metrics['mae']:.4f} kW")
    print(f"RMSE:  {metrics['rmse']:.4f} kW")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    # 6. Guardar modelo (artefacto local, además del Model Registry)
    save_model(model)

    print()
    print("Modelo guardado en:")
    print(MODEL_PATH)

    # 7. MLflow: loggear el run final y registrar el modelo (Dia 8, secciones J y K)
    print()
    print("=== MLFLOW TRACKING + MODEL REGISTRY ===")
    y_pred_test = model.predict(X_test)
    log_and_register_model(
        model, metrics, X_dev.columns.tolist(), X_test, y_test, y_pred_test,
    )


if __name__ == "__main__":
    main()