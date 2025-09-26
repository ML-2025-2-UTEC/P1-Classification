# -*- coding: utf-8 -*-
"""
Pruebas unitarias para los algoritmos de ML implementados manualmente.
"""

import pytest
import numpy as np
from src.models.algorithms_manual import RegresionLogisticaManual, SVMManual, RandomForestManual, ArbolDecisionManual
from sklearn.datasets import make_classification, make_moons

@pytest.fixture
def datos_clasificacion_binaria():
    """Genera un conjunto de datos de juguete para clasificación binaria."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def datos_clasificacion_multiclase():
    """Genera un conjunto de datos de juguete para clasificación multiclase."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def test_regresion_logistica_manual_multiclase(datos_clasificacion_multiclase):
    """
    Prueba que la Regresión Logística Manual se entrena y predice en un
    problema multiclase.
    """
    X, y = datos_clasificacion_multiclase
    modelo = RegresionLogisticaManual(tasa_aprendizaje=0.1, epocas=100, semilla_aleatoria=0)

    modelo.fit(X, y)
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)

    assert predicciones.shape == (X.shape[0],), "La forma de las predicciones no es correcta."
    assert probabilidades.shape == (X.shape[0], len(np.unique(y))), "La forma de las probabilidades no es correcta."
    assert np.all(np.isclose(probabilidades.sum(axis=1), 1.0)), "Las probabilidades no suman 1."

    precision = np.mean(predicciones == y)
    assert precision > 1.0 / len(np.unique(y)), "El modelo no parece aprender (precisión no es mejor que aleatoria)."

def test_svm_manual_multiclase(datos_clasificacion_multiclase):
    """Prueba que el SVM manual con One-vs-Rest funciona en un problema multiclase."""
    X, y = datos_clasificacion_multiclase
    modelo = SVMManual(tasa_aprendizaje=0.01, epocas=50, C=1.0, semilla_aleatoria=1)

    modelo.fit(X, y)
    predicciones = modelo.predict(X)

    assert predicciones.shape == (X.shape[0],)
    precision = np.mean(predicciones == y)
    assert precision > 1.0 / len(np.unique(y)), "El SVM no parece aprender."

def test_arbol_decision_manual(datos_clasificacion_binaria):
    """Prueba que el Árbol de Decisión manual puede ajustarse y predecir."""
    X, y = datos_clasificacion_binaria
    arbol = ArbolDecisionManual(max_profundidad=5)

    arbol.fit(X, y)
    predicciones = arbol.predict(X)

    assert predicciones.shape == (X.shape[0],)
    precision = np.mean(predicciones == y)
    assert precision > 0.7, "El árbol de decisión no está aprendiendo correctamente."

def test_random_forest_manual(datos_clasificacion_multiclase):
    """Prueba que el Random Forest manual funciona en un problema multiclase."""
    X, y = datos_clasificacion_multiclase
    modelo = RandomForestManual(n_estimadores=10, max_profundidad=5, semilla_aleatoria=42)

    modelo.fit(X, y)
    predicciones = modelo.predict(X)

    assert predicciones.shape == (X.shape[0],)
    precision = np.mean(predicciones == y)
    assert precision > 1.0 / len(np.unique(y)), "El Random Forest no parece aprender."