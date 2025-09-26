# -*- coding: utf-8 -*-
"""
Pruebas unitarias para las métricas de evaluación manuales.
"""

import pytest
import numpy as np
from src.evaluation.metrics_manual import (
    matriz_confusion_manual,
    precision_recall_f1_por_clase,
    balanced_accuracy_manual
)

@pytest.fixture
def datos_prueba_metricas():
    """Datos de prueba para las métricas."""
    y_verdadero = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_predicho  = np.array([0, 1, 1, 0, 1, 2, 2, 1, 2])
    # Matriz de confusión esperada:
    # Pred: 0, 1, 2
    # Ver:0 [[2, 0, 1],
    # Ver:1  [0, 3, 0],
    # Ver:2  [0, 1, 2]]
    return y_verdadero, y_predicho

def test_matriz_confusion_manual(datos_prueba_metricas):
    """Prueba que la matriz de confusión se calcula correctamente."""
    y_verdadero, y_predicho = datos_prueba_metricas
    matriz_esperada = np.array([[2, 0, 1], [0, 3, 0], [0, 1, 2]])

    matriz_calculada, _ = matriz_confusion_manual(y_verdadero, y_predicho)

    np.testing.assert_array_equal(matriz_calculada, matriz_esperada)

def test_precision_recall_f1_por_clase(datos_prueba_metricas):
    """Prueba que las métricas por clase se calculan correctamente."""
    y_verdadero, y_predicho = datos_prueba_metricas
    matriz, _ = matriz_confusion_manual(y_verdadero, y_predicho)
    metricas = precision_recall_f1_por_clase(matriz)

    # Clase 0: TP=2, FP=0, FN=1 -> P=1.0, R=0.666, F1=0.8
    assert np.isclose(metricas["clase_0"]["precision"], 1.0)
    assert np.isclose(metricas["clase_0"]["recall"], 2/3)
    assert np.isclose(metricas["clase_0"]["f1_score"], 0.8)

    # Clase 1: TP=3, FP=1, FN=0 -> P=0.75, R=1.0, F1=0.857
    assert np.isclose(metricas["clase_1"]["precision"], 0.75)
    assert np.isclose(metricas["clase_1"]["recall"], 1.0)
    assert np.isclose(metricas["clase_1"]["f1_score"], 0.857, atol=1e-3)

    # Clase 2: TP=2, FP=1, FN=1 -> P=0.666, R=0.666, F1=0.666
    assert np.isclose(metricas["clase_2"]["precision"], 2/3)
    assert np.isclose(metricas["clase_2"]["recall"], 2/3)
    assert np.isclose(metricas["clase_2"]["f1_score"], 2/3)

def test_balanced_accuracy_manual(datos_prueba_metricas):
    """Prueba que la exactitud balanceada se calcula correctamente."""
    y_verdadero, y_predicho = datos_prueba_metricas
    matriz, _ = matriz_confusion_manual(y_verdadero, y_predicho)

    # Recall_0 = 2/3, Recall_1 = 3/3, Recall_2 = 2/3
    # BA = (0.666 + 1.0 + 0.666) / 3 = 0.777
    ba_calculada = balanced_accuracy_manual(matriz)
    assert np.isclose(ba_calculada, (2/3 + 1.0 + 2/3) / 3)