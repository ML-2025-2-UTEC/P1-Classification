# Proyecto de Clasificación de Riesgo Crediticio (Implementación Manual)

Este proyecto tiene como objetivo construir un modelo de Machine Learning para predecir el riesgo crediticio de clientes, implementando todos los componentes principales (algoritmos, transformaciones, métricas) de forma manual en Python, utilizando `numpy` y `pandas` como base.

## Descripción

El objetivo es seguir una metodología de ciencia de datos rigurosa y reproducible, desde el análisis exploratorio hasta la evaluación e interpretabilidad del modelo, con la restricción de no utilizar librerías de alto nivel como `scikit-learn` para las implementaciones centrales (a excepción de KNN).

## Estructura del Repositorio

```
project_name/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── eda/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── models/
├── experiments/
├── reports/
├── Makefile
├── requirements.txt
└── README.md
```

-   `src/`: Contiene todo el código fuente modular en español.
-   `data/`: Almacena los datos crudos y procesados.
-   `tests/`: Pruebas unitarias para garantizar la calidad del código.
-   `notebooks/`: Jupyter notebooks para exploración y prototipado.
-   `reports/`: Informes generados, como análisis EDA y resultados de modelos.
-   `models/`: Modelos entrenados y guardados.
-   `experiments/`: Registros de experimentos de entrenamiento y tuning.

## Cómo Empezar

### 1. Prerrequisitos

-   Python 3.8 o superior
-   `pip`

### 2. Instalación

Clona el repositorio y crea un entorno virtual:

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
python -m venv venv
source venv/bin/activate  # En Windows: venv\\Scripts\\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 3. Uso del Makefile

El `Makefile` proporciona comandos para automatizar las tareas más comunes:

-   `make install`: Instala las dependencias.
-   `make test`: Ejecuta todas las pruebas unitarias.
-   `make eda`: Genera el reporte de análisis exploratorio.
-   `make train`: Entrena el modelo por defecto (Regresión Logística).
- - `make clean`: Limpia todos los artefactos generados (datos procesados, modelos, reportes).
-   `make help`: Muestra todos los comandos disponibles.

### 4. Ejecución del Pipeline

Para ejecutar el pipeline completo (EDA, preprocesamiento, entrenamiento y evaluación), puedes usar:

```bash
make run_pipeline
```

## Restricciones del Proyecto

-   **Implementación Manual:** Todos los algoritmos, transformaciones y métricas deben ser implementados manualmente.
-   **Librerías Permitidas:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`. `scikit-learn` solo está permitido para la implementación de `KNeighborsClassifier`.
-   **Idioma:** Todo el código (variables, funciones, comentarios) debe estar en español.