import os
import math
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Household Power Forecaster",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Household Power Forecaster")
st.write("Pronóstico de consumo eléctrico residencial a 24 horas.")


# ============================================================
# ESTADO DE LA API
# ============================================================

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    health_data = health_response.json()

    if health_data.get("status") == "ok" and health_data.get("model_loaded"):
        st.success("API disponible y modelo cargado")
    else:
        st.warning("La API responde, pero el modelo podría no estar disponible.")

except requests.RequestException:
    st.error("No se pudo conectar con la API.")
    st.info(f"Verifica que la API esté disponible en: {API_URL}")
    st.stop()


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

try:
    model_response = requests.get(f"{API_URL}/model-info", timeout=5)
    model_response.raise_for_status()
    model_data = model_response.json()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Modelo",
        model_data.get("model_name", "N/A"),
    )

    col2.metric(
        "Versión",
        model_data.get("version", "N/A"),
    )

    col3.metric(
        "Stage",
        model_data.get("stage", "N/A"),
    )

except requests.RequestException:
    model_data = {}
    st.warning("No fue posible consultar la información del modelo.")


st.divider()

st.header("Nuevo pronóstico")

st.write(
    "Ingresa las condiciones actuales e históricas del consumo. "
    "El sistema generará un pronóstico para las próximas 24 horas."
)


# ============================================================
# FORMULARIO
# ============================================================

with st.form("prediction_form"):

    st.subheader("📅 Información temporal")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour = st.number_input(
            "Hora del día",
            min_value=0,
            max_value=23,
            value=20,
            step=1,
        )

    with col2:
        day_of_week = st.selectbox(
            "Día de la semana",
            options=[
                ("Lunes", 0),
                ("Martes", 1),
                ("Miércoles", 2),
                ("Jueves", 3),
                ("Viernes", 4),
                ("Sábado", 5),
                ("Domingo", 6),
            ],
            format_func=lambda x: x[0],
        )[1]

    with col3:
        month = st.selectbox(
            "Mes",
            options=list(range(1, 13)),
            index=11,
        )

    st.subheader("⚡ Consumo histórico")

    col1, col2, col3 = st.columns(3)

    with col1:
        lag_1h = st.number_input(
            "Consumo hace 1 hora (kW)",
            min_value=0.0,
            value=1.50,
            step=0.01,
        )

        lag_24h = st.number_input(
            "Consumo hace 24 horas (kW)",
            min_value=0.0,
            value=1.60,
            step=0.01,
        )

    with col2:
        lag_2h = st.number_input(
            "Consumo hace 2 horas (kW)",
            min_value=0.0,
            value=1.40,
            step=0.01,
        )

        lag_48h = st.number_input(
            "Consumo hace 48 horas (kW)",
            min_value=0.0,
            value=1.50,
            step=0.01,
        )

    with col3:
        lag_3h = st.number_input(
            "Consumo hace 3 horas (kW)",
            min_value=0.0,
            value=1.30,
            step=0.01,
        )

        lag_168h = st.number_input(
            "Consumo hace 168 horas (kW)",
            min_value=0.0,
            value=1.40,
            step=0.01,
        )

    st.subheader("📊 Estadísticas recientes")

    col1, col2 = st.columns(2)

    with col1:
        rollmean_3h = st.number_input(
            "Promedio últimas 3 horas",
            min_value=0.0,
            value=1.40,
            step=0.01,
        )

        rollmean_6h = st.number_input(
            "Promedio últimas 6 horas",
            min_value=0.0,
            value=1.30,
            step=0.01,
        )

        rollmean_24h = st.number_input(
            "Promedio últimas 24 horas",
            min_value=0.0,
            value=1.10,
            step=0.01,
        )

        rollmean_168h = st.number_input(
            "Promedio últimas 168 horas",
            min_value=0.0,
            value=1.00,
            step=0.01,
        )

    with col2:
        rollstd_3h = st.number_input(
            "Desviación estándar últimas 3 horas",
            min_value=0.0,
            value=0.10,
            step=0.01,
        )

        rollstd_6h = st.number_input(
            "Desviación estándar últimas 6 horas",
            min_value=0.0,
            value=0.20,
            step=0.01,
        )

        rollstd_24h = st.number_input(
            "Desviación estándar últimas 24 horas",
            min_value=0.0,
            value=0.30,
            step=0.01,
        )

        rollstd_168h = st.number_input(
            "Desviación estándar últimas 168 horas",
            min_value=0.0,
            value=0.30,
            step=0.01,
        )

    submitted = st.form_submit_button(
        "⚡ Generar pronóstico",
        use_container_width=True,
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if submitted:

    is_weekend = 1 if day_of_week >= 5 else 0

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    payload = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "lag_1h": lag_1h,
        "lag_2h": lag_2h,
        "lag_3h": lag_3h,
        "lag_24h": lag_24h,
        "lag_48h": lag_48h,
        "lag_168h": lag_168h,
        "rollmean_3h": rollmean_3h,
        "rollstd_3h": rollstd_3h,
        "rollmean_6h": rollmean_6h,
        "rollstd_6h": rollstd_6h,
        "rollmean_24h": rollmean_24h,
        "rollstd_24h": rollstd_24h,
        "rollmean_168h": rollmean_168h,
        "rollstd_168h": rollstd_168h,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        prediction = response.json()

        st.divider()

        st.success("Pronóstico generado correctamente")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "⚡ Consumo pronosticado",
            f"{prediction['forecast']:.3f} kW",
        )

        col2.metric(
            "Horizonte",
            prediction.get("horizon", "24h"),
        )

        col3.metric(
            "Versión del modelo",
            prediction.get("model_version", "N/A"),
        )

        with st.expander("Ver datos enviados al modelo"):
            st.json(payload)

    except requests.RequestException as error:
        st.error("No fue posible generar la predicción.")
        st.exception(error)