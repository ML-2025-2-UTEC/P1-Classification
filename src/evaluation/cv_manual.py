# -*- coding: utf-8 -*-
"""
Este módulo contiene implementaciones manuales de estrategias de validación cruzada,
búsqueda de hiperparámetros y evaluación de modelos.
"""

import numpy as np
import pandas as pd
import time
import itertools
from .metrics_manual import balanced_accuracy_manual, matriz_confusion_manual

class StratifiedKFoldManual:
    """
    Implementación manual de la validación cruzada estratificada (Stratified K-Fold).
    (El código de esta clase se mantiene como en el paso anterior)
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = np.asarray(y)
        indices = np.arange(len(y))

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        indices_por_clase = [indices[y == clase] for clase in np.unique(y)]
        pliegues_por_clase = [np.array_split(idx, self.n_splits) for idx in indices_por_clase]

        for i in range(self.n_splits):
            indices_prueba = np.concatenate([pliegue[i] for pliegue in pliegues_por_clase if len(pliegue[i]) > 0])
            indices_entrenamiento = np.setdiff1d(indices, indices_prueba)
            yield indices_entrenamiento, indices_prueba

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# --- Búsqueda de Hiperparámetros Manual ---

def grid_search_manual(estimador_clase, rejilla_parametros, X, y, cv, metrica_evaluacion=balanced_accuracy_manual):
    """
    Realiza una búsqueda de hiperparámetros en rejilla (Grid Search) manual.

    Args:
        estimador_clase: La clase del estimador a utilizar (ej. RegresionLogisticaManual).
        rejilla_parametros (dict): Diccionario con los nombres de los parámetros y sus valores a probar.
        X (np.array): Datos de características.
        y (np.array): Etiquetas de destino.
        cv (object): Objeto de validación cruzada con un método `split` (ej. StratifiedKFoldManual).
        metrica_evaluacion (function): La función para evaluar el rendimiento del modelo.

    Returns:
        (dict, float): Un tupla con el mejor conjunto de parámetros y el mejor score promedio.
    """
    print("Iniciando Grid Search manual...")

    # Generar todas las combinaciones de parámetros
    claves_parametros = rejilla_parametros.keys()
    valores_parametros = rejilla_parametros.values()
    combinaciones = list(itertools.product(*valores_parametros))

    mejor_score = -1
    mejores_parametros = None

    for i, params_tupla in enumerate(combinaciones):
        params_actuales = dict(zip(claves_parametros, params_tupla))
        print(f"Probando combinación {i+1}/{len(combinaciones)}: {params_actuales}")

        scores_pliegue = []
        for indices_ent, indices_val in cv.split(X, y):
            X_ent, X_val = X[indices_ent], X[indices_val]
            y_ent, y_val = y[indices_ent], y[indices_val]

            estimador = estimador_clase(**params_actuales)
            estimador.fit(X_ent, y_ent)
            predicciones = estimador.predict(X_val)

            matriz_conf, _ = matriz_confusion_manual(y_val, predicciones)
            score = metrica_evaluacion(matriz_conf)
            scores_pliegue.append(score)

        score_promedio = np.mean(scores_pliegue)
        print(f"  Score promedio de CV: {score_promedio:.4f}")

        if score_promedio > mejor_score:
            mejor_score = score_promedio
            mejores_parametros = params_actuales

    print(f"\nMejores parámetros encontrados: {mejores_parametros}")
    print(f"Mejor score de CV: {mejor_score:.4f}")

    return mejores_parametros, mejor_score

# --- Validación Cruzada Anidada Manual ---

def nested_cv_manual(estimador_clase, rejilla_parametros, X, y, cv_externo, cv_interno, metrica_evaluacion=balanced_accuracy_manual):
    """
    Realiza una validación cruzada anidada (Nested CV) manual.

    Args:
        estimador_clase: La clase del estimador.
        rejilla_parametros (dict): La rejilla de parámetros para la búsqueda interna.
        X, y: Datos y etiquetas.
        cv_externo: CV para el bucle externo (evaluación).
        cv_interno: CV para el bucle interno (selección de hiperparámetros).
        metrica_evaluacion: La función de métrica para la evaluación final.

    Returns:
        list: Una lista con los scores de rendimiento para cada pliegue del bucle externo.
    """
    print("Iniciando Nested Cross-Validation manual...")
    scores_externos = []

    for i, (indices_ent, indices_test) in enumerate(cv_externo.split(X, y)):
        print(f"\n--- Pliegue Externo {i+1}/{cv_externo.get_n_splits()} ---")
        X_ent, X_test = X[indices_ent], X[indices_test]
        y_ent, y_test = y[indices_ent], y[indices_test]

        # Bucle interno: encontrar los mejores hiperparámetros en el conjunto de entrenamiento
        mejores_params, _ = grid_search_manual(
            estimador_clase, rejilla_parametros, X_ent, y_ent, cv_interno, metrica_evaluacion
        )

        # Entrenar el modelo final con los mejores parámetros en todo el conjunto de entrenamiento del pliegue externo
        print(f"Entrenando modelo final del pliegue con: {mejores_params}")
        modelo_final = estimador_clase(**mejores_params)
        modelo_final.fit(X_ent, y_ent)

        # Evaluar en el conjunto de prueba del pliegue externo
        predicciones_test = modelo_final.predict(X_test)
        matriz_conf_test, _ = matriz_confusion_manual(y_test, predicciones_test)
        score_externo = metrica_evaluacion(matriz_conf_test)

        print(f"Score en el conjunto de prueba del pliegue externo: {score_externo:.4f}")
        scores_externos.append(score_externo)

    print("\n--- Resultados de Nested CV ---")
    print(f"Scores por pliegue: {scores_externos}")
    print(f"Score promedio: {np.mean(scores_externos):.4f} +/- {np.std(scores_externos):.4f}")

    return scores_externos