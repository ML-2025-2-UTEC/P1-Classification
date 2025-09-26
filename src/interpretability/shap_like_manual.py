# -*- coding: utf-8 -*-
"""
Módulo para la interpretabilidad de modelos de Machine Learning.

Contiene implementaciones manuales de técnicas de interpretabilidad como:
- Importancia de características global (basada en coeficientes o impureza).
- Importancia por permutación.
- Gráficos de Dependencia Parcial (PDP).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .evaluation.metrics_manual import balanced_accuracy_manual, matriz_confusion_manual

# --- 1. Importancia de Características Global ---

def importancia_features_global(modelo, feature_names):
    """
    Calcula la importancia global de las características para un modelo entrenado.

    Args:
        modelo: Un modelo entrenado (ej. RegresionLogisticaManual, RandomForestManual).
        feature_names (list): Lista con los nombres de las características.

    Returns:
        pd.DataFrame: Un DataFrame con las características y sus scores de importancia.
    """
    print("Calculando importancia de características global...")

    if hasattr(modelo, 'pesos_'): # Modelo lineal como Regresión Logística
        # Usar el valor absoluto promedio de los coeficientes a través de las clases
        importancias = np.mean(np.abs(modelo.pesos_), axis=0)

    elif hasattr(modelo, 'feature_importances_'): # Modelos de árbol (requiere implementación)
        # Esta parte asume que el RandomForestManual ha sido modificado para calcular
        # y almacenar la importancia de características (ej. Mean Decrease Impurity).
        importancias = modelo.feature_importances_

    else:
        raise TypeError("El tipo de modelo no es compatible para el cálculo de importancia directa. "
                        "Use importancia_permutacion_manual en su lugar.")

    df_importancia = pd.DataFrame({'feature': feature_names, 'importancia': importancias})
    df_importancia = df_importancia.sort_values(by='importancia', ascending=False)

    return df_importancia

# --- 2. Importancia por Permutación Manual ---

def importancia_permutacion_manual(modelo, X_val, y_val, metrica, n_repeticiones=10):
    """
    Calcula la importancia de características mediante permutación.

    Args:
        modelo: Un modelo entrenado.
        X_val (np.array): Datos de validación para las características.
        y_val (np.array): Etiquetas de validación.
        metrica (function): Función de métrica para evaluar el rendimiento (ej. balanced_accuracy_manual).
        n_repeticiones (int): Número de veces que se permuta cada característica para promediar.

    Returns:
        pd.DataFrame: Un DataFrame con las características y la caída en el rendimiento.
    """
    print("Calculando importancia por permutación...")

    # 1. Calcular el score base del modelo
    predicciones_base = modelo.predict(X_val)
    matriz_conf_base, _ = matriz_confusion_manual(y_val, predicciones_base)
    score_base = metrica(matriz_conf_base)

    importancias = {}

    for i in range(X_val.shape[1]):
        scores_permutados = []
        for _ in range(n_repeticiones):
            X_permutado = X_val.copy()
            # Permutar la columna i
            np.random.shuffle(X_permutado[:, i])

            predicciones_permutadas = modelo.predict(X_permutado)
            matriz_conf_perm, _ = matriz_confusion_manual(y_val, predicciones_permutadas)
            score_permutado = metrica(matriz_conf_perm)
            scores_permutados.append(score_permutado)

        caida_score = score_base - np.mean(scores_permutados)
        importancias[f'feature_{i}'] = caida_score

    df_importancia = pd.DataFrame(list(importancias.items()), columns=['feature', 'caida_score_promedio'])
    df_importancia = df_importancia.sort_values(by='caida_score_promedio', ascending=False)

    return df_importancia

# --- 3. Gráfico de Dependencia Parcial (PDP) Manual ---

def graficar_pdp_manual(modelo, X, feature_idx, feature_name, ruta_guardado):
    """
    Calcula y grafica un Gráfico de Dependencia Parcial (PDP) para una característica.

    Args:
        modelo: Un modelo entrenado que tenga un método `predict_proba`.
        X (pd.DataFrame): El conjunto de datos (preferiblemente de entrenamiento o validación).
        feature_idx (int): El índice de la columna de la característica a analizar.
        feature_name (str): El nombre de la característica para el título del gráfico.
        ruta_guardado (str): Ruta para guardar el gráfico PNG.
    """
    print(f"Generando PDP para la característica: {feature_name}...")

    # Crear una rejilla de valores para la característica de interés
    valores_rejilla = np.linspace(X.iloc[:, feature_idx].min(), X.iloc[:, feature_idx].max(), num=50)

    predicciones_promedio = []

    for valor in valores_rejilla:
        X_modificado = X.copy()
        # Fijar la característica de interés al valor de la rejilla
        X_modificado.iloc[:, feature_idx] = valor

        # Obtener las probabilidades predichas
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X_modificado.values)
            # Promediar las probabilidades a través de todas las muestras para cada clase
            # Aquí promediamos la probabilidad de la clase positiva (clase 1)
            # Para multiclase, se necesitaría un gráfico por clase.
            prediccion_promedio = np.mean(probabilidades[:, 1]) # Asumiendo que la clase 1 es la de interés
        else:
            # Si no hay predict_proba, usar predicciones y promediar (menos informativo)
            predicciones = modelo.predict(X_modificado.values)
            prediccion_promedio = np.mean(predicciones)

        predicciones_promedio.append(prediccion_promedio)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(valores_rejilla, predicciones_promedio, marker='o', linestyle='-')
    plt.title(f'Gráfico de Dependencia Parcial (PDP) para {feature_name}')
    plt.xlabel(f'Valores de {feature_name}')
    plt.ylabel('Predicción Promedio (Probabilidad de Clase 1)')
    plt.grid(True)
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"PDP para '{feature_name}' guardado en: {ruta_guardado}")