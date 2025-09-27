<meta charset="UTF-8">
<!-- TÍTULO ANIMADO -->
<meta charset="UTF-8">
<h1 align="center">
    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&center=true&vCenter=true&width=800&height=70&duration=4000&lines=Clasificación:+Predicción+de+Riesgo+Crediticio" />
</h1>

<h3 align="center">📚 Curso: Machine Learning 📚</h3>

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

## 📑 Tabla de Contenidos
1. [Información del Proyecto](#informacion-del-proyecto)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Metodología Implementada](#metodologia-implementada)
4. [Resultados y Experimentos](#resultados-y-experimentos)
5. [Ejecución del Proyecto](#ejecucion-del-proyecto)
6. [Características del Sistema](#caracteristicas-del-sistema)

---

## 📝 Información del Proyecto
<a name="informacion-del-proyecto"></a>

Este proyecto implementa un sistema de **clasificación multiclase** para predecir el nivel de riesgo de impago en préstamos comerciales. El objetivo es clasificar clientes en 3 categorías de riesgo:
- **0**: Riesgo Bajo
- **1**: Riesgo Medio  
- **2**: Riesgo Alto

### Objetivos del Proyecto
- **Implementación de Algoritmos ML Propios**: Desarrollo desde cero de 3 algoritmos de machine learning para clasificación multiclase
- **Análisis Exploratorio Exhaustivo**: Comprensión profunda del dataset y sus patrones subyacentes
- **Optimización y Evaluación**: Sistema robusto de métricas y optimización de hiperparámetros
- **Consideraciones de Negocio**: Enfoque en minimizar costos de clasificación errónea en contexto financiero

### Características Clave
- **Dataset**: 20,000 instancias con 35 características financieras y demográficas
- **Implementaciones Propias**: 3 algoritmos ML desarrollados desde cero (Logistic Regression, SVM, Random Forest)
- **Algoritmo de Referencia**: KNN usando Sklearn como baseline
- **Técnicas Avanzadas**: Feature engineering, selección de características, validación cruzada
- **Visualizaciones**: Plots generados con asistencia de IA para análisis exploratorio

### Tecnologías Utilizadas
- **Lenguaje**: Python 3.11+
- **Librerías Principales**: NumPy, Pandas, Matplotlib, Seaborn
- **Notebooks**: Jupyter para desarrollo interactivo
- **Optimización**: Operaciones vectorizadas para eficiencia computacional

---

<details>
  <summary><strong>🏗️ Estructura del Proyecto</strong></summary>
  <a name="estructura-del-proyecto"></a>

```
P1-Classification/
├── data/
│   ├── raw/                          # Datos originales
│   │   ├── datos_entrenamiento_riesgo.csv
│   │   └── datos_prueba_riesgo.csv
│   └── processed/                    # Datos procesados y transformados
│       ├── feature_names.txt
│       ├── preprocessing_metadata.json
│       └── *.csv, *.npy             # Datos entrenamiento/test procesados
├── notebooks/                        # 📊 NOTEBOOKS PRINCIPALES
│   ├── 01_eda.ipynb                 # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb        # Pipeline de Preprocesamiento
│   └── 03_modeling.ipynb            # Entrenamiento y Evaluación
├── src/                             # Código fuente modular
│   ├── data/loader.py              # Utilidades de carga de datos
│   ├── models/                     # 🤖 IMPLEMENTACIONES PROPIAS
│   │   ├── logistic_regression.py  # Regresión Logística Multinomial
│   │   ├── svm.py                  # Support Vector Machine
│   │   ├── random_forest.py        # Random Forest
│   │   └── base_classifier.py      # Clase base para modelos
│   ├── evaluation/metrics.py       # Sistema de métricas
│   └── utils/                      # Utilidades de optimización
├── experiments/                     # 📈 RESULTADOS Y EXPERIMENTOS
│   ├── best_hyperparameters.json
│   ├── optimization_summary.csv
│   └── feature_importance_*.csv
└── tests/                          # Pruebas unitarias
```

### Descripción de Directorios:
- **📁 data/**: Contiene todos los datasets tanto originales como procesados
- **📁 notebooks/**: Notebooks principales con todo el análisis y experimentación
- **📁 src/**: Código fuente modular y reutilizable del proyecto
- **📁 experiments/**: Resultados de experimentos, métricas y configuraciones óptimas
- **📁 tests/**: Pruebas unitarias para validar el correcto funcionamiento

</details>

---

<details>
  <summary><strong>🔬 Metodología Implementada</strong></summary>
  <a name="metodologia-implementada"></a>

### 1. Análisis Exploratorio de Datos (EDA)
**📍 Ver**: `notebooks/01_eda.ipynb`

- **Análisis estadístico** completo de las 35 variables
- **Visualizaciones** de distribuciones y correlaciones (*generadas con asistencia de IA*)
- **Detección de outliers** y patrones en los datos
- **Análisis de balanceamiento** de clases objetivo

### 2. Preprocesamiento de Datos
**📍 Ver**: `notebooks/02_preprocessing.ipynb`

- **Limpieza de datos**: Tratamiento de valores faltantes
- **Transformaciones**:
  - Normalización Z-score para variables numéricas
  - Encoding de variables categóricas
- **Feature Engineering**: Creación de ratios financieros y scores compuestos
- **Selección de características**: Eliminación de features redundantes
- **Resultado**: 37 características finales para modelado

### 3. Implementación de Modelos
**📍 Ver**: `notebooks/03_modeling.ipynb` y `src/models/`

#### 🤖 Algoritmos Implementados Desde Cero:

1. **Regresión Logística Multinomial** (`src/models/logistic_regression.py`)
   - Implementación con regularización L1/L2
   - Optimización con gradiente descendente
   - Softmax para clasificación multiclase
   - **Complejidad**: O(n × m × iterations)

2. **Support Vector Machine** (`src/models/svm.py`)
   - Enfoque One-vs-Rest para multiclase
   - Kernels: Lineal y RBF (Radial Basis Function)
   - Optimización del margen máximo
   - **Complejidad**: O(n³) para entrenamiento

3. **Random Forest** (`src/models/random_forest.py`)
   - Ensemble de árboles de decisión
   - Bagging y votación por mayoría
   - Selección aleatoria de características
   - **Complejidad**: O(n × log(n) × trees)

#### 📚 Algoritmo de Referencia (Sklearn):
4. **K-Nearest Neighbors (KNN)** - Usado como baseline de comparación

### 4. Evaluación y Optimización
**📍 Ver resultados en**: `experiments/`

#### Métricas de Evaluación:
- **Accuracy**: Precisión global del modelo
- **Precision, Recall, F1-Score**: Por clase y promediadas
- **Matriz de Confusión**: Análisis detallado de errores
- **Validación Cruzada**: 5-fold estratificada

#### Optimización de Hiperparámetros:
- **Grid Search** para exploración exhaustiva
- **Selección automática** de mejores parámetros
- **Guardado de configuraciones** óptimas en `experiments/best_hyperparameters.json`

</details>

---

<details>
  <summary><strong>📊 Resultados y Experimentos</strong></summary>
  <a name="resultados-y-experimentos"></a>

### 🎯 Notebooks Principales (Ejecutar en orden):
1. **`notebooks/01_eda.ipynb`** 
   - Análisis estadístico completo del dataset
   - Visualizaciones de distribuciones y correlaciones
   - Detección de patrones y outliers
   
2. **`notebooks/02_preprocessing.ipynb`** 
   - Pipeline completo de limpieza de datos
   - Transformaciones y feature engineering
   - Validación de calidad de datos
   
3. **`notebooks/03_modeling.ipynb`** 
   - Entrenamiento de todos los modelos
   - Evaluación comparativa y métricas
   - Optimización de hiperparámetros

### 📈 Archivos de Resultados:
- **`experiments/optimization_summary.csv`** 
  - Comparación de rendimiento de todos los modelos
  - Métricas de evaluación (Accuracy, F1-Score, Precision, Recall)
  
- **`experiments/best_hyperparameters.json`** 
  - Configuraciones óptimas encontradas para cada modelo
  - Parámetros de regularización y arquitectura
  
- **`experiments/feature_importance_*.csv`** 
  - Análisis de importancia de características
  - Rankings de features más relevantes para la predicción

### 🔍 Datos Procesados:
- **`data/processed/`** 
  - Datasets limpios y transformados
  - Archivos en formatos CSV y NumPy (.npy)
  - Metadata de preprocesamiento

### 🎯 Resultados Esperados:
- **Modelos Entrenados**: 4 algoritmos diferentes comparados
- **Métricas de Evaluación**: Precision, Recall, F1-Score, Accuracy
- **Visualizaciones**: Matrices de confusión y curvas de aprendizaje
- **Interpretabilidad**: Feature importance y análisis de errores

</details>

---

<details>
  <summary><strong>🚀 Ejecución del Proyecto</strong></summary>
  <a name="ejecucion-del-proyecto"></a>

### Requisitos del Sistema:
```bash
# Instalar dependencias
pip install -r requirements.txt

# Versiones recomendadas
Python >= 3.11
NumPy >= 1.24.0
Pandas >= 2.0.0
Matplotlib >= 3.7.0
Seaborn >= 0.12.0
Jupyter >= 1.0.0
```

### Pasos de Ejecución:

1. **Configuración del Entorno**:
   ```bash
   # Clonar el repositorio
   git clone https://github.com/ML-2025-2-UTEC/P1-Classification.git
   cd P1-Classification
   
   # Crear entorno virtual (recomendado)
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```

2. **Análisis Exploratorio**: 
   - Ejecutar `notebooks/01_eda.ipynb`
   - Tiempo estimado: 10-15 minutos
   - Genera visualizaciones y estadísticas descriptivas

3. **Preprocesamiento**: 
   - Ejecutar `notebooks/02_preprocessing.ipynb`
   - Tiempo estimado: 5-10 minutos
   - Procesa y guarda datos limpios en `data/processed/`

4. **Modelado Completo**: 
   - Ejecutar `notebooks/03_modeling.ipynb`
   - Tiempo estimado: 20-30 minutos
   - Entrena todos los modelos y genera resultados

### Resultados Esperados:
- **Archivos generados** en `experiments/` con métricas y configuraciones
- **Modelos entrenados** guardados para reutilización
- **Visualizaciones** de rendimiento y análisis comparativo
- **Datasets procesados** en `data/processed/` para futuras ejecuciones

</details>

---

<details>
  <summary><strong>📁 Características del Sistema</strong></summary>
  <a name="caracteristicas-del-sistema"></a>

### Modelos Implementados:
- ✅ **3 algoritmos propios**: Logistic Regression, SVM, Random Forest
- ✅ **1 algoritmo sklearn**: KNN (baseline)
- ✅ **Sistema completo** de evaluación y optimización

### Arquitectura del Código:
- 🔧 **Diseño modular**: Separación clara entre carga de datos, modelos, evaluación
- 📊 **Visualizaciones comprehensivas**: Plots generados con asistencia de IA
- ⚡ **Operaciones vectorizadas**: NumPy para máxima eficiencia computacional
- 💾 **Persistencia inteligente**: Guardado automático de modelos y configuraciones
- 🧪 **Pruebas unitarias**: Validación automática de funcionalidades críticas

### Consideraciones de Negocio:
- 🎯 **Minimización de riesgo**: Foco en reducir falsos negativos (Alto→Bajo riesgo)
- 📈 **Análisis de costos**: Evaluación del impacto financiero de errores de clasificación
- 🔍 **Interpretabilidad**: Feature importance y análisis de decisiones del modelo
- � **Compliance**: Documentación exhaustiva para auditoría y regulación

### Optimizaciones Técnicas:
- **Paralelización**: Operaciones matriciales optimizadas
- **Memory Management**: Uso eficiente de memoria para datasets grandes
- **Scalabilidad**: Arquitectura preparada para datasets más grandes
- **Reproducibilidad**: Seeds fijos y logging de experimentos

### Métricas de Calidad:
- **Cobertura de código**: Pruebas unitarias para funciones críticas
- **Documentación**: Docstrings completos y comentarios explicativos
- **Estándares**: PEP 8 y mejores prácticas de Python
- **Versionado**: Control de versiones con Git para seguimiento de cambios

</details>

---

## 🔗 Referencias y Créditos

### Tecnologías Utilizadas:
- **Python**: Lenguaje principal del proyecto
- **NumPy**: Operaciones matriciales y vectorizadas
- **Pandas**: Manipulación y análisis de datos
- **Matplotlib/Seaborn**: Visualización de datos
- **Jupyter**: Desarrollo interactivo y documentación

### Agradecimientos:
- **Visualizaciones**: Los plots y gráficos fueron generados con asistencia de IA para optimizar la presentación de datos
- **Documentación**: Inspirado en mejores prácticas de documentación de proyectos ML

### Dataset:
- Datos sintéticos de riesgo crediticio con 20,000 instancias
- 35 características financieras y demográficas
- 3 clases de riesgo: Bajo (0), Medio (1), Alto (2)

<div align="center">
    <img src="https://skillicons.dev/icons?i=python,jupyter,numpy,matplotlib" />
</div>

---

<div align="center">
    <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
    <img src="https://img.shields.io/badge/ML-Classification-blue?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge" />
</div>