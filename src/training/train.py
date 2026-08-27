from pathlib import Path

import joblib
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

    return RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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

    # 6. Guardar modelo
    save_model(model)

    print()
    print("Modelo guardado en:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()