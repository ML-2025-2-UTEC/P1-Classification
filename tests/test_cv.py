# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo de validación cruzada y búsqueda de hiperparámetros.
"""

import pytest
import numpy as np
from src.models.algorithms_manual import RegresionLogisticaManual
from src.evaluation.cv_manual import StratifiedKFoldManual, grid_search_manual
from sklearn.datasets import make_classification

@pytest.fixture
def datos_prueba_cv():
    """Genera un conjunto de datos de prueba para validación cruzada."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )
    return X, y

def test_stratified_k_fold_manual(datos_prueba_cv):
    """Prueba que StratifiedKFoldManual genera los pliegues correctamente."""
    X, y = datos_prueba_cv
    n_splits = 3
    cv = StratifiedKFoldManual(n_splits=n_splits, shuffle=True, random_state=42)

    pliegues = list(cv.split(X, y))

    assert len(pliegues) == n_splits

    # Verificar que los conjuntos de entrenamiento y prueba son disjuntos
    # y que juntos forman el conjunto de datos completo.
    for indices_ent, indices_test in pliegues:
        assert len(np.intersect1d(indices_ent, indices_test)) == 0
        assert len(np.union1d(indices_ent, indices_test)) == len(X)

        # Verificar estratificación (proporción de clases similar)
        proporcion_original = np.mean(y)
        proporcion_test = np.mean(y[indices_test])
        assert np.isclose(proporcion_original, proporcion_test, atol=0.15)


def test_grid_search_manual(datos_prueba_cv):
    """Prueba que la búsqueda en rejilla manual encuentra los mejores parámetros."""
    X, y = datos_prueba_cv

    # Rejilla de parámetros simple para la prueba
    rejilla = {
        'tasa_aprendizaje': [0.1, 0.01],
        'C': [1.0, 10.0]
    }

    cv = StratifiedKFoldManual(n_splits=2, shuffle=True, random_state=42)

    mejores_params, mejor_score = grid_search_manual(
        estimador_clase=RegresionLogisticaManual,
        rejilla_parametros=rejilla,
        X=X,
        y=y,
        cv=cv
    )

    assert mejores_params is not None
    assert 'tasa_aprendizaje' in mejores_params
    assert 'C' in mejores_params
    assert mejor_score > 0.5 # Debería ser mejor que aleatorio
    assert isinstance(mejor_score, float)