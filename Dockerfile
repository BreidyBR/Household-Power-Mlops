FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/

# Modelo final seleccionado y registrado como Production
COPY models/random_forest_model.joblib ./models/random_forest_model.joblib

# La API detecta esta ruta y carga el modelo empaquetado
ENV MODEL_PATH=/app/models/random_forest_model.joblib
ENV MODEL_VERSION=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]