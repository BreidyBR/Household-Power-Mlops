# Household Power MLOps

Proyecto MLOps End-to-End para el pronóstico del consumo eléctrico residencial utilizando el dataset **Individual Household Electric Power Consumption**.

## Objetivo

Diseñar, implementar y documentar una solución completa de Machine Learning bajo principios de MLOps, desde la ingesta de datos crudos hasta el despliegue, monitoreo y propuesta de reentrenamiento del modelo.

El sistema busca ser reproducible, versionado, automatizable, desplegable, observable, mantenible y auditable.

## Problema de Machine Learning

El proyecto corresponde a un problema de **series de tiempo** cuyo objetivo es pronosticar el consumo eléctrico futuro a partir del historial de consumo doméstico.

Durante el análisis de los datos se determinará y justificará la frecuencia de trabajo y el horizonte de predicción utilizados.

## Dataset

**Individual Household Electric Power Consumption**

El proyecto utiliza el dataset Individual Household Electric Power Consumption, publicado en el UCI Machine Learning Repository.

El conjunto de datos contiene 2,075,259 registros de mediciones de consumo eléctrico residencial.

Los datos son obtenidos mediante un proceso de ingesta reproducible desde la fuente oficial y almacenados localmente en `data/raw/`.

Los archivos de datos crudos no son versionados en Git debido a su tamaño y se encuentran excluidos mediante `.gitignore`.

## Data Ingestion

La ingesta de datos se encuentra implementada en:

```text
src/ingestion/ingest.py
```

El proceso realiza automáticamente:

- descarga del dataset desde UCI Machine Learning Repository;
- extracción del archivo original;
- almacenamiento del dataset en `data/raw/`;
- validación básica de existencia y tamaño;
- conteo de registros;
- cálculo del hash SHA-256;
- generación de metadata de la ingesta.

Para ejecutar la ingesta desde la raíz del proyecto:

```powershell
python src/ingestion/ingest.py
```

La metadata generada registra la fuente, fecha de ingesta en UTC, cantidad de registros, tamaño del archivo y hash SHA-256.

Si el dataset ya se encuentra disponible en `data/raw/`, el proceso evita descargarlo nuevamente.

## Arquitectura MLOps

El proyecto seguirá el siguiente flujo:

```text
Fuente de datos
      ↓
Data Ingestion
      ↓
Raw / Bronze
      ↓
Data Validation
   ↙       ↘
 Fail      Pass
  ↓          ↓
Alert    Data Cleaning
             ↓
      Feature Pipeline
             ↓
          Training
             ↓
         Evaluation
             ↓
           MLflow
 Tracking + Artifacts
   + Model Registry
             ↓
       Best Candidate
             ↓
         Dockerize
             ↓
          Model API
             ↓
         Production
             ↓
         Monitoring
      ↙       ↓       ↘
Data Drift   Model    System
             Performance
             Metrics
      ↘       ↓       ↙
       Retrain Trigger
```

## Estructura del repositorio

```text
Household-Power-Mlops/
│
├── api/                    # API de inferencia (FastAPI)
├── ui/                     # Interfaz Streamlit
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/                 # Modelo final versionado (random_forest_model.joblib)
├── notebooks/
├── reports/
│   ├── data_quality/
│   └── monitoring/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── cleaning/
│   ├── features/
│   ├── training/
│   └── monitoring/
├── tests/
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── requirements-api.txt
└── requirements-ui.txt
```

> Nota: `configs/`, `data/production/`, `src/evaluation/` y `src/utils/` existieron como carpetas vacías desde la estructura inicial del proyecto, pero nunca se usaron — cada script del pipeline define sus propias constantes y rutas de forma independiente (ver, por ejemplo, `src/training/train.py` o `src/monitoring/production_batches.py`, que hace la división reference/producción en memoria sin escribir a `data/production/`). Se eliminaron para que la estructura del repositorio refleje únicamente lo que realmente existe y se usa.

## Tecnologías

- Python
- Git y GitHub
- pandas
- NumPy
- scikit-learn
- MLflow
- FastAPI
- Docker
- pytest
- Jupyter

## Entorno de desarrollo

Actualmente el proyecto utiliza **Python 3.14.3**.

Crear el entorno virtual:

```powershell
py -3.14 -m venv .venv
```

Activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements-dev.txt
```

## Estrategia de ramas

El proyecto utiliza el siguiente flujo de trabajo:

```text
main
  ↓
develop
  ↓
feature/...
```

- `main`: versión estable.
- `develop`: integración del desarrollo.
- `feature/...`: desarrollo de funcionalidades específicas.

Cada integrante realizará cambios mediante ramas y commits descriptivos para mantener la trazabilidad del proyecto.

## Data Validation

Las Data Quality Gates del proyecto se encuentran implementadas en:

```text
src/validation/validate.py
```

El script ejecuta ocho reglas automáticas sobre el dataset crudo, cada una con un veredicto `PASS`, `WARNING` o `FAIL`:

- columnas obligatorias presentes;
- número mínimo de filas;
- timestamps válidos;
- duplicados bajo umbral;
- missing bajo umbral;
- `Global_active_power` >= 0;
- frecuencia temporal correcta (1 minuto);
- gaps temporales controlados.

Los umbrales de cada regla están justificados con el comportamiento real del dataset (2,075,259 registros), documentado en `notebooks/01_data_quality_cleaning.ipynb`. Un `FAIL` en cualquier regla marca el pipeline como bloqueado; un `WARNING` se registra pero no detiene el proceso.

Para ejecutar la validación desde la raíz del proyecto:

```powershell
python src/validation/validate.py
```

Se puede agregar `--strict` para que el proceso termine con código de salida 1 si alguna regla resulta en `FAIL` (pensado para uso en automatización/CI):

```powershell
python src/validation/validate.py --strict
```

Cada corrida genera `reports/data_quality/data_quality_report.json`, con el veredicto global y el detalle de las ocho reglas, como evidencia auditable de la validación.

## Data Cleaning

La limpieza reproducible del dataset se encuentra implementada en:

```text
src/cleaning/clean.py
```

Las decisiones de limpieza se investigaron y justificaron previamente en `notebooks/01_data_quality_cleaning.ipynb` (Parte II), y este script convierte esa lógica en un pipeline reproducible que no depende de Jupyter. El proceso realiza:

- construcción del índice temporal a partir de `Date` y `Time`;
- conversión de las siete variables eléctricas a tipo numérico, tratando el símbolo `?` como valor faltante;
- análisis de la distribución temporal de los valores faltantes (rachas cortas vs. rachas largas);
- interpolación temporal lineal únicamente sobre rachas de valores faltantes ≤ 60 minutos;
- validación de que la limpieza no elimina filas ni introduce inconsistencias (índice creciente, sin duplicados, sin valores negativos).

Las rachas de valores faltantes mayores a 60 minutos (9 rachas, ~25,691 minutos, hasta ~5 días de duración) se dejan intencionalmente como `NaN`: imputarlas linealmente fabricaría una tendencia artificial. Su tratamiento definitivo se difiere a la etapa de Feature Engineering, en conjunto con la frecuencia de resampling que se elija ahí.

Para ejecutar la limpieza desde la raíz del proyecto:

```powershell
python src/cleaning/clean.py
```

El resultado se guarda en `data/interim/household_power_cleaned.parquet`, preservando las 2,075,259 filas originales.

## EDA temporal

El análisis exploratorio de la serie temporal se encuentra documentado en:

```text
notebooks/02_eda.ipynb
```

El EDA se realiza sobre el dataset limpio generado por el pipeline de Data Cleaning y tiene como objetivo comprender la estructura temporal del consumo eléctrico antes de definir la estrategia de Feature Engineering y modelado.

Durante el análisis se estudiaron:

- estadísticos descriptivos y distribución de las variables eléctricas;
- evolución temporal de `Global_active_power`;
- patrones de consumo por hora del día, día de la semana y mes;
- tendencia mediante medias móviles;
- estacionalidad mediante descomposición de la serie;
- autocorrelación horaria;
- valores extremos y su distribución temporal;
- gaps temporales y duración de los períodos sin información;
- relaciones entre las variables eléctricas.

Los resultados muestran patrones temporales claros. El consumo presenta variaciones según la hora del día, con mayor actividad durante la mañana y especialmente durante la tarde-noche. También se observan diferencias semanales, con mayores niveles promedio durante el fin de semana, y un patrón anual caracterizado por mayor consumo alrededor del inicio y final del año.

La autocorrelación evidencia dependencia entre observaciones consecutivas y ciclos asociados aproximadamente con períodos de 24 horas y sus múltiplos. Estos resultados justifican evaluar posteriormente características temporales y rezagos durante Feature Engineering.

El análisis mediante IQR identificó 94,925 observaciones extremas en `Global_active_power` (4.63 % de las observaciones válidas). Estos valores no se eliminan automáticamente, ya que su concentración en períodos habituales de mayor consumo sugiere que una parte importante corresponde a comportamiento real y no necesariamente a errores.

Después del resampling horario se identificaron 421 horas completamente sin información agrupadas en 8 gaps. La mayoría corresponde a interrupciones prolongadas y el gap máximo alcanza 119 horas. Por esta razón, estos períodos no se imputan de forma general durante el EDA.

Finalmente, `Global_intensity` presenta una correlación prácticamente perfecta con `Global_active_power` (~0.999), mientras que los submedidores aportan relaciones de distinta magnitud. Esta posible redundancia deberá evaluarse durante la selección de características.

Los resultados del EDA proporcionan la evidencia necesaria para la siguiente etapa del proyecto, donde se definirán las características temporales, rezagos, estadísticas móviles, horizonte de predicción y estrategia definitiva de preparación de la serie, evitando el uso de información futura.

## Feature Engineering

La construcción del dataset supervisado se investiga y justifica en:

```text
notebooks/03_feature_engineering.ipynb
```

y se traslada a un pipeline reproducible en:

```text
src/features/build_features.py
```

Partiendo del dataset limpio (`data/interim/household_power_cleaned.parquet`), el proceso define:

- **Frecuencia horaria**, decisión que el EDA había dejado pendiente: conserva el patrón intradía identificado y coincide con las unidades ya definidas para lags y rolling windows;
- **Imputación estacional causal en cascada** (168h → 24h), utilizando exclusivamente información pasada. Las horas faltantes se intentan completar primero con el valor observado 168 horas antes y, si este no está disponible, con el valor de 24 horas antes. Si el valor continúa ausente, se conserva como `NaN` y la fila se excluye posteriormente si afecta una feature necesaria. En este dataset, las 421 horas faltantes fueron resueltas utilizando el valor de 168 horas antes, por lo que el respaldo de 24 horas no fue necesario;
- **Variable objetivo** `Global_active_power` con **horizonte de 24 horas**, alineado con el ciclo diario dominante encontrado en el EDA;
- **Features de calendario**: `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`;
- **Lags**: 1, 2, 3, 24, 48 y 168 horas, cada uno correspondiente a un pico real de autocorrelación observado en el EDA;
- **Rolling mean/std**: ventanas de 3, 6, 24 y 168 horas, calculadas sobre la serie desplazada un paso (`shift(1)`) para no incluir el valor del instante actual;
- **Verificación explícita de ausencia de leakage**: se confirma que cada lag corresponde exactamente a su desplazamiento temporal, que el target efectivamente mira 24 horas hacia adelante, y que ninguna feature coincide sospechosamente con el valor contemporáneo de la serie.

Para ejecutar la etapa de Feature Engineering desde la raíz del proyecto:

```powershell
python src/features/build_features.py
```

El resultado se guarda en `data/processed/features_hourly.parquet`: 33,976 filas × 20 features + `target` (21 columnas en total). Se eliminan 613 filas: 168 filas iniciales sin historia suficiente para construir las features de 168 horas, 24 filas finales sin horizonte disponible a t+24h y 421 filas cuyo target corresponde a una observación originalmente faltante. De esta manera, las features pueden aprovechar la imputación causal basada en información pasada, pero el modelo se entrenará únicamente con targets correspondientes a observaciones reales.

## Training y Evaluation

La etapa de experimentación, comparación y selección de modelos se encuentra documentada en:

```text
notebooks/04_training.ipynb
```

El entrenamiento parte del dataset supervisado generado durante Feature Engineering. Los datos se dividen respetando estrictamente el orden temporal, sin utilizar `shuffle`, en conjuntos de entrenamiento, validación y test.

Como punto de referencia se utiliza un baseline estacional basado en el consumo observado 24 horas antes. Posteriormente se evalúan diferentes algoritmos de Machine Learning:

- Regresión Lineal;
- Random Forest;
- Gradient Boosting;
- HistGradientBoosting.

Durante la experimentación se identificó sobreajuste en la configuración inicial de Random Forest. Por esta razón, se evaluó una versión regularizada mediante restricciones en la profundidad de los árboles y en el número mínimo de observaciones requeridas para divisiones y hojas.

Los modelos candidatos finales se evaluaron mediante `TimeSeriesSplit` con cinco particiones temporales, permitiendo comprobar su comportamiento en diferentes períodos sin alterar el orden cronológico de la serie.

Random Forest regularizado presentó el mejor desempeño promedio durante esta validación temporal:

- MAE promedio: **0.5227 kW**;
- RMSE promedio: **0.7106 kW**.

Después de seleccionar el modelo, Random Forest regularizado se reentrenó utilizando conjuntamente los conjuntos de entrenamiento y validación. El conjunto de test permaneció separado durante todo el proceso de selección y se utilizó únicamente para realizar la evaluación final.

El modelo final obtuvo aproximadamente:

- MAE: **0.43 kW**;
- RMSE: **0.59 kW**;
- sMAPE: **46.53 %**.

En comparación con el baseline estacional, el modelo final reduce considerablemente el error de predicción. El análisis de los resultados muestra que Random Forest logra representar adecuadamente el comportamiento general del consumo eléctrico, aunque presenta mayor dificultad para reproducir algunos picos elevados y cambios bruscos.

La lógica necesaria para reproducir el entrenamiento final se encuentra implementada en:

```text
src/training/train.py
```

Para ejecutar el entrenamiento desde la raíz del proyecto:

```powershell
python src/training/train.py
```

El script carga `data/processed/features_hourly.parquet`, realiza la separación temporal, reentrena el Random Forest regularizado utilizando los datos de entrenamiento y validación, calcula las métricas finales sobre el conjunto de test y guarda el modelo entrenado en:

```text
models/random_forest_model.joblib
```

De esta manera, el entrenamiento final puede reproducirse independientemente del notebook utilizado durante la experimentación.

## MLflow Tracking y Model Registry

El seguimiento de experimentos y el registro del modelo se implementan en:

```text
src/training/experiment.py
src/training/train.py
```

`experiment.py` no vuelve a decidir qué modelo usar — toma exactamente la misma comparación ya investigada y justificada en `04_training.ipynb` (baseline, Regresión Lineal, Random Forest, Gradient Boosting, Random Forest regularizado, HistGradientBoosting) y la instrumenta con MLflow: cada modelo queda registrado como un run independiente, con:

- **Parameters**: `algorithm`, hiperparámetros, `feature_set`, `random_seed`, `data_version` (hash SHA-256 del dataset usado);
- **Metrics**: MAE, RMSE, sMAPE;
- **Artifacts**: el modelo entrenado, un gráfico de residuales y la configuración completa en JSON.

`train.py` (la etapa de entrenamiento final, ya documentada arriba) loggea su propio run sobre el modelo reentrenado con train+validation y evaluado en test, y además **registra el modelo en el Model Registry** de MLflow, representando el ciclo pedido por el proyecto:

```text
Experiment (6 runs de experiment.py)
      ↓
Candidate (run final de train.py, reentrenado con train+validation)
      ↓
Validation (ya superada antes de llegar aquí: TimeSeriesSplit de 5 particiones, ver sección Training y Evaluation)
      ↓
Production (promoción explícita, tras confirmar el desempeño sobre el test set separado)
```

El criterio de selección es explícito y reproducible: menor MAE promedio en `TimeSeriesSplit`, confirmado sobre el conjunto de test. Random Forest regularizado es, de forma consistente, el mejor candidato tanto en la comparación de validación como en la validación temporal.

Para ejecutar ambos scripts desde la raíz del proyecto:

```powershell
python src/training/experiment.py
python src/training/train.py
```

Para explorar los resultados en el navegador:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlruns
```

y abrir `http://localhost:5000`. `mlflow.db` y `mlruns/` son artefactos locales regenerables y no se versionan en Git; cada integrante del equipo genera los suyos ejecutando los scripts anteriores. El modelo final `models/random_forest_model.joblib`, en cambio, se mantiene versionado en el repositorio para permitir la construcción reproducible de la imagen Docker.

> Nota: al igual que en `src/validation/validate.py` (Día 4), MLflow >= 3.0 puso en modo mantenimiento el backend de tracking de solo archivos. El tracking store usa SQLite (`mlflow.db`) y los artefactos (modelos, gráficos) se mantienen en `mlruns/` local.

## API de Inferencia y Testing

La API de inferencia se encuentra implementada en:

```text
api/main.py
```

La API sirve el modelo final seleccionado durante la etapa de entrenamiento y registrado en MLflow como `household-power-forecaster`. El mecanismo de carga depende del entorno de ejecución: durante el desarrollo local, la API obtiene el modelo marcado como `Production` en el Model Registry de MLflow; dentro del contenedor Docker, carga el artefacto `models/random_forest_model.joblib` empaquetado en la imagen mediante la variable de entorno `MODEL_PATH`. De esta manera, el contenedor puede ejecutarse de forma independiente del servidor local de MLflow, manteniendo el mismo modelo seleccionado para producción. Expone tres endpoints:

- **`GET /health`** — estado del servicio y si el modelo cargó correctamente;
- **`GET /model-info`** — nombre, versión y stage del modelo que está sirviendo en ese momento;
- **`POST /predict`** — recibe las 20 features del modelo y devuelve `{"forecast": ..., "horizon": "24h", "model_version": "..."}`.

Para levantarla localmente desde la raíz del proyecto:

```powershell
uvicorn api.main:app --reload --port 8000
```

Ejemplo de uso:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "hour": 20, "day_of_week": 5, "month": 12, "is_weekend": 1,
  "hour_sin": -1.0, "hour_cos": 0.0,
  "lag_1h": 1.5, "lag_2h": 1.4, "lag_3h": 1.3, "lag_24h": 1.6, "lag_48h": 1.5, "lag_168h": 1.4,
  "rollmean_3h": 1.4, "rollstd_3h": 0.1, "rollmean_6h": 1.3, "rollstd_6h": 0.2,
  "rollmean_24h": 1.1, "rollstd_24h": 0.3, "rollmean_168h": 1.0, "rollstd_168h": 0.3
}'
```

Las pruebas se encuentran en `tests/`:

- **`test_data.py`** — esquema, tipos, rangos y ausencia de missing sobre `data/processed/features_hourly.parquet`;
- **`test_model.py`** — carga el modelo real desde el Model Registry y verifica que un input válido produzca un pronóstico numérico y físicamente plausible, y que un input inválido (columna faltante, tipo incorrecto) genere un error, no un resultado silencioso;
- **`test_api.py`** — usa `TestClient` de FastAPI para confirmar que una petición válida responde `200` con el schema esperado, y que peticiones inválidas (campo faltante, tipo incorrecto, cuerpo vacío) responden `422`.

Para ejecutar todas las pruebas desde la raíz del proyecto:

```powershell
pytest tests/ -v
```

## Docker

La API y el modelo final se encuentran contenerizados mediante `Dockerfile`. La imagen utiliza `python:3.14-slim` como base y un archivo de dependencias específico para producción, `requirements-api.txt`, que contiene únicamente las librerías necesarias para ejecutar el servicio de inferencia.

Durante la construcción se incorpora el modelo final generado por `src/training/train.py`:

```text
models/random_forest_model.joblib
```

Dentro del contenedor, la variable de entorno `MODEL_PATH` apunta a este artefacto, permitiendo que la API cargue el modelo mediante `joblib` sin depender de MLflow durante la inferencia. En el entorno local se mantiene la integración con MLflow Model Registry para cargar el modelo registrado en `Production`.

Antes de construir la imagen, el modelo debe haber sido generado ejecutando:

```powershell
python src/training/train.py
```

Para construir la imagen Docker desde la raíz del proyecto:

```powershell
docker build -t grupo7-mlops .
```

Para ejecutar el contenedor:

```powershell
docker run --rm -p 8000:8000 --name grupo7-mlops-api grupo7-mlops
```

La API queda disponible en:

```text
http://localhost:8000
```

La ejecución del contenedor fue validada mediante los endpoints `/health`, `/model-info` y `/predict`. El endpoint de predicción respondió correctamente con un pronóstico a 24 horas utilizando el modelo empaquetado en la imagen.

Para reducir el tamaño de la imagen de producción, Docker utiliza `requirements-api.txt` en lugar del archivo general `requirements.txt`. Esto evita instalar dependencias utilizadas únicamente durante análisis, entrenamiento y experimentación. Con esta separación, la imagen pasó de aproximadamente 1.37 GB a 707 MB, manteniendo el mismo comportamiento de la API y del modelo.

El archivo `.dockerignore` evita incorporar al contexto de construcción datasets, notebooks, artefactos locales de MLflow, entornos virtuales y otros archivos que no son necesarios para la inferencia.

## Monitoring

El sistema incorpora monitoreo en producción en tres niveles: métricas operativas del servicio, drift de los datos y desempeño del modelo. Adicionalmente, se implementa monitoreo de calidad sobre los batches de producción para detectar entradas anómalas antes de utilizarlas.

### System Monitoring

El monitoreo operativo se encuentra implementado en:

```text
src/monitoring/system_monitor.py
```

La API registra automáticamente:

- **latency**: tiempo promedio de respuesta en milisegundos;
- **throughput**: solicitudes procesadas por segundo;
- **error rate**: proporción de solicitudes que producen errores internos del servicio (5xx);
- **availability**: proporción de solicitudes atendidas sin errores internos.

Las métricas acumuladas pueden consultarse mediante:

```text
GET /metrics
```

El propio endpoint `/metrics` se excluye del conteo para evitar que la consulta de observabilidad altere las métricas que está midiendo.

### Reference y Production Batches

La construcción de los períodos utilizados para monitoring se encuentra en:

```text
src/monitoring/production_batches.py
```

Los datos se dividen respetando estrictamente el orden temporal. El 85 % histórico utilizado durante el desarrollo del modelo (train + validation) se utiliza como **REFERENCE**, mientras que el 15 % final, no utilizado para entrenar el modelo, representa producción y se divide cronológicamente en tres batches:

- REFERENCE: 28,879 filas;
- PRODUCTION_BATCH_1: 1,699 filas;
- PRODUCTION_BATCH_2: 1,699 filas;
- PRODUCTION_BATCH_3: 1,699 filas.

La división puede reproducirse mediante:

```powershell
python -m src.monitoring.production_batches
```

### Data Drift

La detección de cambios en la distribución de las features se implementa en:

```text
src/monitoring/drift_monitor.py
```

Se utiliza **Population Stability Index (PSI)** comparando cada batch de producción contra REFERENCE. Como criterio operativo del proyecto:

- PSI < 0.10 → `OK`;
- 0.10 ≤ PSI < 0.25 → `WARNING`;
- PSI ≥ 0.25 → `ALERT`.

Estos valores funcionan como umbrales operativos de monitoreo y no se interpretan como límites universales. Una alerta de drift indica un cambio en la distribución de los datos, pero no implica por sí sola degradación del modelo.

Además de evaluar los batches naturales, el módulo incluye una simulación controlada de drift sobre una copia de `PRODUCTION_BATCH_3`, incrementando un 35 % algunas variables históricas de consumo. La simulación no modifica el dataset original.

Para ejecutar el análisis:

```powershell
python -m src.monitoring.drift_monitor
```

### Model Monitoring

El desempeño del modelo sobre los batches de producción se evalúa en:

```text
src/monitoring/model_monitor.py
```

Cuando el ground truth está disponible, se calculan **MAE_t** y **RMSE_t** para cada período:

| Batch | MAE_t | RMSE_t |
|---|---:|---:|
| PRODUCTION_BATCH_1 | 0.4317 kW | 0.5820 kW |
| PRODUCTION_BATCH_2 | 0.3968 kW | 0.5256 kW |
| PRODUCTION_BATCH_3 | 0.4708 kW | 0.6571 kW |

Esto permite distinguir entre un cambio en la distribución de los datos y una degradación real del desempeño predictivo.

Para reproducir la evaluación:

```powershell
python -m src.monitoring.model_monitor
```

### Production Data Quality Monitoring

La validación de calidad de los batches de producción se encuentra en:

```text
src/monitoring/quality_monitor.py
```

Para comprobar el comportamiento del sistema ante datos problemáticos se genera una copia de `PRODUCTION_BATCH_3` y se introducen de forma controlada:

- un valor faltante;
- una fila duplicada;
- un outlier extremo;
- un datatype incorrecto;
- una modificación del esquema mediante una columna inesperada.

El dataset original no se modifica durante esta simulación.

El monitor aplica el flujo:

```text
Detect → Block / Warn → Log
```

Los problemas incompatibles con una inferencia segura, como schema incorrecto, missing values o tipos incompatibles, producen `FAIL → BLOCK`. Los duplicados y valores extremos generan `WARNING → WARN`, ya que requieren revisión pero no necesariamente representan datos inválidos.

La evidencia de cada ejecución se registra en:

```text
reports/monitoring/quality_report.json
```

Para ejecutar la simulación y validación:

```powershell
python -m src.monitoring.quality_monitor
```

### Retraining Trigger

La estrategia de reentrenamiento se encuentra implementada en:

```text
src/monitoring/retraining_trigger.py
```

La decisión de recomendar un reentrenamiento combina **dos señales**, no una sola:

```text
significant_drift  AND  performance_degradation  ->  RETRAIN
```

Un cambio en la distribución de los datos (drift) no implica por sí solo que el modelo haya dejado de funcionar correctamente — por eso el drift nunca dispara el reentrenamiento en solitario. La degradación de desempeño se calcula comparando `MAE_t`/`RMSE_t` (de `model_monitor.py`) contra el desempeño de referencia del modelo en test, permitiendo hasta un 20 % de tolerancia antes de considerarlo degradado.

Ejemplo real con `PRODUCTION_BATCH_3` (drift significativo detectado, pero desempeño estable):

```text
Drift significativo:       True
MAE actual:                0.4708 (threshold: 0.5197)
RETRAIN:                   False
Razón: Existe drift significativo, pero el desempeño del modelo
permanece dentro de los límites aceptables. No se reentrena.
```

Para ejecutar el caso de prueba:

```powershell
python -m src.monitoring.retraining_trigger
```

> **Nota sobre artefactos regenerables:** algunos scripts del pipeline vuelven a generar archivos que se mantienen versionados como evidencia o artefactos del proyecto. Por ejemplo, `src/training/train.py` actualiza `models/random_forest_model.joblib` y `src/monitoring/quality_monitor.py` actualiza `reports/monitoring/quality_report.json`. Por esta razón, después de ejecutar estos procesos Git puede mostrar dichos archivos como modificados. Este comportamiento es esperado y no indica un error en la ejecución.
>
> Si los comandos se ejecutaron únicamente para comprobar la reproducibilidad del proyecto y no se desea conservar los artefactos regenerados, pueden restaurarse a la versión registrada en Git mediante:
>
> ```powershell
> git restore models/random_forest_model.joblib reports/monitoring/quality_report.json
> ```
>
> Este paso es opcional y no forma parte del pipeline MLOps; únicamente permite devolver el repositorio a un estado limpio después de realizar pruebas locales.

## Interfaz y Despliegue

Como complemento al proyecto (no forma parte de los entregables obligatorios del enunciado), se construyó una interfaz visual y su despliegue público, para facilitar la demo.

### Interfaz Streamlit

```text
ui/app.py
```

Consume la API mediante peticiones HTTP (`GET /health`, `GET /model-info`, `POST /predict`) — no importa el modelo directamente. Muestra el estado de la API y del modelo cargado, y un formulario para generar un pronóstico a 24 horas sin necesidad de construir el JSON manualmente.

Para ejecutarla localmente (requiere la API corriendo en paralelo):

```powershell
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
pip install -r requirements-ui.txt
streamlit run ui/app.py
```

Por defecto apunta a `http://127.0.0.1:8000`; se puede sobreescribir con la variable de entorno `API_URL`.

### Despliegue en Render

El archivo `render.yaml` (Blueprint de Render) define ambos servicios — la API (Docker, reutilizando el `Dockerfile` ya validado) y la interfaz Streamlit — para desplegarlos con un solo Blueprint:

1. Crear una cuenta en [render.com](https://render.com) (tiene plan gratuito) y conectarla a GitHub.
2. **New +** → **Blueprint** → seleccionar este repositorio → Render detecta `render.yaml` automáticamente y muestra los dos servicios.
3. Aplicar el Blueprint. El primer deploy de la API tarda unos minutos (construye la imagen Docker).
4. Una vez que `household-power-api` esté desplegado, copiar su URL pública (ej. `https://household-power-api-xxxx.onrender.com`).
5. Ir al servicio `household-power-ui` → pestaña **Environment** → completar la variable `API_URL` con esa URL (con `https://`) → guardar (esto redeploya la UI automáticamente).
6. Abrir la URL pública de `household-power-ui` — debería funcionar igual que en local.

> El plan gratuito de Render "duerme" un servicio tras un período de inactividad; la primera solicitud después de eso puede tardar 30-60 segundos en responder mientras el servicio despierta. Se recomienda abrir el link unos minutos antes de la demo.

## Estado del proyecto

- [x] Repositorio Git creado
- [x] Git configurado
- [x] Ramas `main` y `develop` creadas
- [x] Rama inicial de configuración creada
- [x] `.gitignore` configurado
- [x] Entorno virtual creado
- [x] Dependencias iniciales definidas
- [x] Estructura inicial del proyecto creada
- [x] Data Ingestion
- [x] Data Validation
- [x] Data Cleaning
- [x] EDA temporal
- [x] Feature Pipeline
- [x] Training
- [x] Evaluation
- [x] MLflow Tracking
- [x] Model Registry
- [x] Docker
- [x] Model API
- [x] Testing
- [x] Monitoring
- [x] Retraining Trigger

## Equipo

Proyecto desarrollado por el **Grupo 7**.

| Integrante | GitHub |
|---|---|
| Breidy Bonilla | [@BreidyBR](https://github.com/BreidyBR) |
| Oscar Eduardo Sánchez Barahona | [@oscar2903](https://github.com/oscar2903) |

Ambos integrantes participaron a lo largo de todas las etapas del proyecto (ingesta, calidad de datos, EDA, feature engineering, modelado, MLflow, API, Docker, monitoreo y despliegue), trabajando mediante ramas independientes y Pull Requests revisados antes de fusionarse a `develop`, y de `develop` a `main` una vez verificado el proyecto completo.
