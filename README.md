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

El dataset contiene mediciones de consumo eléctrico residencial registradas a lo largo del tiempo.

Los datos crudos no serán versionados directamente en Git. El proyecto contará con un proceso reproducible de ingesta para obtenerlos.

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
- [ ] Data Ingestion
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
