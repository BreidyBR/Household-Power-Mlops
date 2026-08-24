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
├── api/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── production/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── data_quality/
│   └── monitoring/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── cleaning/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── monitoring/
│   └── utils/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

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
- [ ] Data Validation
- [ ] Data Cleaning
- [ ] Feature Pipeline
- [ ] Training
- [ ] Evaluation
- [ ] MLflow Tracking
- [ ] Model Registry
- [ ] Docker
- [ ] Model API
- [ ] Monitoring
- [ ] Retraining Trigger

## Proyecto desarrollado por el **Grupo 7**.
