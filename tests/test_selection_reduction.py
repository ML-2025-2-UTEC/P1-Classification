# -*- coding: utf-8 -*-
"""
Pruebas unitarias para los módulos de selección y reducción de dimensionalidad.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from src.features.selection_manual import SelectKBestManual, anova_f_score_manual, RFEManual, vif_manual
from src.features.reduction_manual import PCAManual
from src.models.algorithms_manual import RegresionLogisticaManual

@pytest.fixture
def datos_prueba_seleccion():
    """
    Genera un conjunto de datos de prueba determinista para selección de características.
    - feature_0 y feature_2 están fuertemente correlacionadas con y.
    - El resto de features son ruido.
    """
    np.random.seed(42)
    y = np.array([0] * 50 + [1] * 50)
    # Crear características informativas
    feature_0 = y * 10 + np.random.normal(0, 1, 100)
    feature_2 = y * -5 + np.random.normal(0, 1, 100)
    # Crear características de ruido
    noise_features = np.random.rand(100, 8)

    X = pd.DataFrame({
        'feature_0': feature_0,
        'feature_1': noise_features[:, 0], # Ruido
        'feature_2': feature_2,
        'feature_3': noise_features[:, 1],
        'feature_4': noise_features[:, 2],
        'feature_5': noise_features[:, 3],
        'feature_6': noise_features[:, 4],
        'feature_7': noise_features[:, 5],
        'feature_8': noise_features[:, 6],
        'feature_9': noise_features[:, 7],
    })
    return X, pd.Series(y)


def test_select_k_best_manual(datos_prueba_seleccion):
    """Prueba que SelectKBestManual selecciona las k características correctas."""
    X, y = datos_prueba_seleccion
    # Seleccionar las 2 mejores características
    selector = SelectKBestManual(score_func=anova_f_score_manual, k=2)
    selector.fit(X, y)

    # Con el dataset determinista, las características seleccionadas DEBEN ser feature_0 y feature_2.
    # Usamos un set para ignorar el orden.
    assert set(selector.feature_names_) == {'feature_0', 'feature_2'}

    X_transformado = selector.transform(X)
    assert X_transformado.shape == (100, 2)

def test_rfe_manual(datos_prueba_seleccion):
    """Prueba que la Eliminación Recursiva de Características funciona."""
    X, y = datos_prueba_seleccion
    # Usamos solo las 5 primeras features para acelerar la prueba
    X_reducido = X.iloc[:, :5]
    estimador = RegresionLogisticaManual(epocas=50) # Usar menos épocas para que la prueba sea rápida

    rfe = RFEManual(estimador=estimador, n_features_to_select=2)
    rfe.fit(X_reducido, y)

    assert len(rfe.feature_names_) == 2
    assert sum(rfe.support_) == 2
    # Las características seleccionadas deberían ser las informativas
    assert 'feature_0' in rfe.feature_names_
    assert 'feature_2' in rfe.feature_names_

    X_transformado = rfe.transform(X_reducido)
    assert X_transformado.shape == (100, 2)

def test_vif_manual():
    """Prueba que el cálculo de VIF identifica la multicolinealidad."""
    # Crear un DataFrame con alta colinealidad
    data = {'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10], # B = 2*A
            'C': [5, 3, 1, 4, 2]}  # No colineal
    df = pd.DataFrame(data)

    vif_df = vif_manual(df)

    # El VIF para A y B debe ser infinito (o muy alto)
    assert vif_df.loc[vif_df['feature'] == 'A', 'VIF'].iloc[0] > 1e6
    assert vif_df.loc[vif_df['feature'] == 'B', 'VIF'].iloc[0] > 1e6
    # El VIF para C debe ser bajo
    assert vif_df.loc[vif_df['feature'] == 'C', 'VIF'].iloc[0] < 2

def test_pca_manual(datos_prueba_seleccion):
    """Prueba que PCAManual reduce la dimensionalidad correctamente."""
    X, _ = datos_prueba_seleccion
    pca = PCAManual(n_componentes=2)

    pca.fit(X.values)
    X_transformado = pca.transform(X.values)

    assert X_transformado.shape == (100, 2)
    assert pca.componentes_.shape == (2, 10)
    assert len(pca.varianza_explicada_ratio_) == 10
    assert np.isclose(sum(pca.varianza_explicada_ratio_), 1.0)
    # La primera componente debe explicar más varianza que la segunda
    assert pca.varianza_explicada_ratio_[0] > pca.varianza_explicada_ratio_[1]