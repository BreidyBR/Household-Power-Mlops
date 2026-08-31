"""
Retraining Trigger
------------------

Define la lógica que determina cuándo recomendar el reentrenamiento
del modelo en producción.

La decisión combina dos señales:

1. Data Drift significativo.
2. Degradación del desempeño del modelo.

Un cambio en la distribución de los datos NO implica necesariamente
que el modelo haya dejado de funcionar correctamente, por lo que
el drift por sí solo no dispara el reentrenamiento.
"""

# ---------------------------------------------------------------------
# Métricas de referencia del modelo en el conjunto de evaluación final
# ---------------------------------------------------------------------

BASELINE_MAE = 0.4331
BASELINE_RMSE = 0.5907

# Se considera degradación cuando una métrica empeora más de un 20 %
# respecto al desempeño de referencia.
PERFORMANCE_TOLERANCE = 0.20


def calculate_performance_thresholds():
    """
    Calcula los límites máximos aceptables de MAE y RMSE.

    Returns
    -------
    dict
        Thresholds de desempeño permitidos.
    """

    return {
        "mae_threshold": BASELINE_MAE * (1 + PERFORMANCE_TOLERANCE),
        "rmse_threshold": BASELINE_RMSE * (1 + PERFORMANCE_TOLERANCE),
    }


def evaluate_retraining_trigger(
    significant_drift: bool,
    mae_t: float,
    rmse_t: float,
):
    """
    Evalúa si el modelo debería ser reentrenado.

    La regla utilizada es:

        significant_drift
        AND
        performance_degradation

    La degradación ocurre si MAE o RMSE superan su threshold.

    Parameters
    ----------
    significant_drift : bool
        Indica si el monitoreo de datos detectó drift significativo.

    mae_t : float
        MAE observado en el batch de producción.

    rmse_t : float
        RMSE observado en el batch de producción.

    Returns
    -------
    dict
        Resultado completo de la decisión.
    """

    thresholds = calculate_performance_thresholds()

    mae_degraded = mae_t > thresholds["mae_threshold"]
    rmse_degraded = rmse_t > thresholds["rmse_threshold"]

    performance_degradation = mae_degraded or rmse_degraded

    retrain = significant_drift and performance_degradation

    if retrain:
        reason = (
            "Drift significativo y degradación del modelo detectados. "
            "Se recomienda activar el pipeline de reentrenamiento."
        )

    elif significant_drift and not performance_degradation:
        reason = (
            "Existe drift significativo, pero el desempeño del modelo "
            "permanece dentro de los límites aceptables. No se reentrena."
        )

    elif not significant_drift and performance_degradation:
        reason = (
            "El desempeño del modelo se degradó, pero no existe drift "
            "significativo. Se recomienda investigar antes de reentrenar."
        )

    else:
        reason = (
            "No existe drift significativo ni degradación relevante "
            "del modelo. No se requiere reentrenamiento."
        )

    return {
        "significant_drift": significant_drift,
        "mae_t": round(mae_t, 4),
        "rmse_t": round(rmse_t, 4),
        "mae_threshold": round(thresholds["mae_threshold"], 4),
        "rmse_threshold": round(thresholds["rmse_threshold"], 4),
        "mae_degraded": mae_degraded,
        "rmse_degraded": rmse_degraded,
        "performance_degradation": performance_degradation,
        "retrain": retrain,
        "reason": reason,
    }


def print_retraining_decision(result):
    """
    Muestra de forma legible la decisión del Retraining Trigger.
    """

    print("\nRETRAINING TRIGGER")
    print("-" * 50)

    print(f"Drift significativo:       {result['significant_drift']}")
    print(
        f"MAE actual:                {result['mae_t']} "
        f"(threshold: {result['mae_threshold']})"
    )
    print(
        f"RMSE actual:               {result['rmse_t']} "
        f"(threshold: {result['rmse_threshold']})"
    )

    print(
        f"Degradación del modelo:    "
        f"{result['performance_degradation']}"
    )

    print(f"RETRAIN:                   {result['retrain']}")
    print(f"Razón:                     {result['reason']}")


if __name__ == "__main__":

    # Caso de prueba:
    # existe drift, pero el modelo todavía mantiene un desempeño aceptable.

    result = evaluate_retraining_trigger(
        significant_drift=True,
        mae_t=0.4708,
        rmse_t=0.6571,
    )

    print_retraining_decision(result)