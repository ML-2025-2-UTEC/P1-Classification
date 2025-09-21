# Classification Project: Predicting Default Risk in Commercial Loans

## Descripción del Proyecto
Este proyecto tiene como objetivo construir un modelo de clasificación multiclase para predecir el nivel de riesgo de impago de clientes que solicitan un préstamo comercial. El dataset contiene 20,000 instancias con 35 características, clasificadas en 3 niveles: 0 (Bajo), 1 (Medio), 2 (Alto).

## Estructura del Proyecto
```
PROYECTO 1/
├── data/
│   ├── raw/                     # Datos originales
│   │   ├── datos_entrenamiento_riesgo.csv
│   │   └── datos_prueba_riesgo.csv
│   └── processed/               # Datos procesados
├── src/
│   ├── data/                    # Funciones de carga y preprocesamiento
│   ├── models/                  # Implementaciones de algoritmos ML
│   ├── evaluation/              # Métricas y evaluación
│   └── utils/                   # Utilidades generales
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   ├── 02_preprocessing.ipynb   # Preprocesamiento de datos
│   ├── 03_modeling.ipynb       # Entrenamiento de modelos
│   └── 04_evaluation.ipynb     # Evaluación y comparación
├── experiments/                 # Resultados de experimentos
├── reports/                     # Informe final IEEE LaTeX
└── tests/                       # Pruebas unitarias
```

## Plan de Desarrollo

### Fase 1: Análisis y Preprocesamiento de Datos (Semana 1-2)
#### 🔄 Estado Actual: En Progreso

**Objetivos:**
- Comprensión profunda del dataset
- Análisis exploratorio exhaustivo
- Preprocesamiento robusto

**Tareas Completadas:**
- ✅ Estructura del proyecto creada
- ✅ Plan de desarrollo establecido

**Tareas en Progreso:**
- 🔄 Exploración inicial del dataset
- 🔄 Configuración de utilidades de carga de datos

**Próximas Tareas:**
1. **Exploración de Datos (EDA)**
   - Análisis estadístico descriptivo
   - Visualización de distribuciones
   - Detección de valores faltantes y outliers
   - Análisis de correlaciones entre features
   - Distribución de clases objetivo

2. **Preprocesamiento**
   - Limpieza de datos
   - Normalización/estandarización
   - Tratamiento de valores faltantes
   - Feature engineering básico

### Fase 2: Selección y Reducción de Características (Semana 3)

**Objetivos:**
- Manejar la alta dimensionalidad (35 features)
- Identificar features más relevantes
- Aplicar técnicas de reducción dimensional

**Tareas Planificadas:**
1. **Análisis de Importancia**
   - Feature importance con árboles de decisión
   - Análisis de correlación
   - Pruebas estadísticas (ANOVA, Chi-cuadrado)

2. **Técnicas de Reducción**
   - PCA (Principal Component Analysis)
   - Selección univariada
   - Selección recursiva de features

3. **Comparación de Enfoques**
   - Evaluación del impacto en el rendimiento
   - Justificación de la dimensionalidad final

### Fase 3: Implementación de Modelos (Semana 4-5)

**Objetivos:**
- Implementar algoritmos desde cero (sin sklearn para ML)
- Aplicar al menos 3 algoritmos diferentes
- Optimizar hiperparámetros

**Algoritmos a Implementar:**
1. **Regresión Logística Multinomial**
   - Gradiente descendente
   - Regularización L1/L2
   - Función softmax para multiclase

2. **Support Vector Machine (SVM)**
   - Implementación con kernel lineal
   - Enfoque One-vs-Rest para multiclase
   - Optimización con método simplex

3. **Random Forest**
   - Árboles de decisión desde cero
   - Bootstrap aggregating
   - Votación por mayoría

**Librerías Permitidas:**
- ✅ pandas, numpy, matplotlib, seaborn
- ✅ StandardScaler (sklearn.preprocessing)
- ❌ Algoritmos ML de sklearn, xgboost, etc.

### Fase 4: Evaluación y Optimización (Semana 6)

**Objetivos:**
- Evaluación exhaustiva con múltiples métricas
- Optimización de hiperparámetros
- Análisis de matriz de confusión

**Métricas a Implementar:**
1. **Métricas por Clase**
   - Precision, Recall, F1-Score
   - Especificidad y Sensibilidad
   - AUC-ROC para cada clase

2. **Métricas Globales**
   - Accuracy
   - Macro y Micro averaging
   - Matriz de confusión detallada

3. **Validación Cruzada**
   - K-fold cross-validation
   - Stratified sampling
   - Análisis de sesgo-varianza

**Consideraciones de Costos:**
- Análisis del costo de clasificación errónea
- Peso mayor a errores High→Low vs Low→High
- Optimización de umbral de decisión

### Fase 5: Interpretabilidad y Conclusiones (Semana 7)

**Objetivos:**
- Interpretación del modelo final
- Identificación de features más importantes
- Estrategias de negocio

**Análisis de Interpretabilidad:**
1. **Feature Importance**
   - Coeficientes de regresión logística
   - Importancia en Random Forest
   - Análisis SHAP values (implementación propia)

2. **Análisis de Decisiones**
   - Casos de estudio específicos
   - Perfiles de riesgo por segmento
   - Umbrales de decisión óptimos

3. **Recomendaciones de Negocio**
   - Estrategias de aprobación de préstamos
   - Segmentación de clientes
   - Políticas de pricing diferenciado

### Fase 6: Reporte IEEE LaTeX (Semana 8)

**Objetivos:**
- Documento profesional en formato IEEE
- Presentación clara de metodología y resultados
- Justificación de decisiones técnicas

**Estructura del Reporte:**
1. **Abstract y Introduction**
2. **Literature Review** (opcional)
3. **Methodology**
4. **Results and Analysis**
5. **Discussion**
6. **Conclusions and Future Work**

## Entregables por Fase

### Entregables Técnicos
- [ ] Código modular y reutilizable en `src/`
- [ ] Notebooks documentados en `notebooks/`
- [ ] Resultados de experimentos en `experiments/`
- [ ] Pruebas unitarias en `tests/`
- [ ] Reporte final IEEE LaTeX en `reports/`

### Criterios de Evaluación (20 pts)
1. **Data Understanding & Preprocessing (5 pts)**
   - EDA exhaustivo (2 pts)
   - Preprocesamiento robusto (3 pts)

2. **Modeling & Training (6 pts)**
   - Selección y aplicación de modelos (3 pts)
   - Manejo de dimensionalidad (3 pts)

3. **Evaluation & Optimization (5 pts)**
   - Métricas de evaluación (2 pts)
   - Optimización de modelos (3 pts)

4. **Report & Conclusions (4 pts)**
   - Claridad y estructura (2 pts)
   - Interpretación de negocio (2 pts)

## Próximos Pasos Inmediatos

### Para la Próxima Sesión:
1. **Explorar los datasets** (`datos_entrenamiento_riesgo.csv`, `datos_prueba_riesgo.csv`)
2. **Completar EDA inicial** en `notebooks/01_eda.ipynb`
3. **Implementar utilidades de carga** en `src/data/loader.py`
4. **Configurar preprocesamiento básico** en `src/data/preprocessing.py`

### Preguntas para Resolver:
- ¿Hay valores faltantes en el dataset?
- ¿Cuál es la distribución de las clases objetivo?
- ¿Qué features tienen mayor variabilidad?
- ¿Existen outliers significativos?

### Decisiones Técnicas Pendientes:
- Estrategia para manejar valores faltantes
- Método de normalización (StandardScaler vs MinMaxScaler)
- Técnica de reducción dimensional a utilizar
- Estrategia de validación cruzada

## Recursos y Referencias
- [IEEE LaTeX Template](https://www.ieee.org/conferences/publishing/templates.html)
- Datasets: `data/raw/datos_entrenamiento_riesgo.csv`, `data/raw/datos_prueba_riesgo.csv`
- Documentación de características en el PDF del proyecto

---

**Última actualización:** Septiembre 21, 2025  
**Estado del proyecto:** Fase 1 - Análisis y Preprocesamiento (En Progreso)