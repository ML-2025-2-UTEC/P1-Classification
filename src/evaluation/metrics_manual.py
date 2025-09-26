# -*- coding: utf-8 -*-
"""
Este módulo contiene funciones para calcular métricas de evaluación de modelos
de Machine Learning de forma manual, utilizando principalmente NumPy.
"""

import numpy as np
import pandas as pd

def matriz_confusion_manual(y_verdadero, y_predicho, etiquetas=None):
    """Calcula la matriz de confusión de forma manual."""
    if etiquetas is None:
        etiquetas = sorted(list(np.unique(np.concatenate((y_verdadero, y_predicho)))))

    mapeo_etiquetas = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
    num_etiquetas = len(etiquetas)
    matriz = np.zeros((num_etiquetas, num_etiquetas), dtype=int)

    for i in range(len(y_verdadero)):
        idx_verdadero = mapeo_etiquetas.get(y_verdadero[i])
        idx_predicho = mapeo_etiquetas.get(y_predicho[i])
        if idx_verdadero is not None and idx_predicho is not None:
            matriz[idx_verdadero, idx_predicho] += 1

    return matriz, etiquetas

def precision_recall_f1_por_clase(matriz_confusion):
    """Calcula la precisión, el recall y el F1-score para cada clase."""
    num_clases = matriz_confusion.shape[0]
    metricas = {}

    for i in range(num_clases):
        TP = matriz_confusion[i, i]
        FP = matriz_confusion[:, i].sum() - TP
        FN = matriz_confusion[i, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metricas[f"clase_{i}"] = {"precision": precision, "recall": recall, "f1_score": f1_score}

    return metricas

def macro_f1_score(metricas_por_clase):
    """Calcula el F1-score promedio (macro)."""
    return np.mean([d["f1_score"] for d in metricas_por_clase.values()])

def weighted_f1_score(metricas_por_clase, matriz_confusion):
    """Calcula el F1-score ponderado por el soporte de cada clase."""
    f1s_ponderados = []
    total_muestras = matriz_confusion.sum()
    for i, (clase, metricas) in enumerate(metricas_por_clase.items()):
        soporte_clase = matriz_confusion[i, :].sum()
        f1s_ponderados.append(metricas["f1_score"] * soporte_clase)
    return sum(f1s_ponderados) / total_muestras if total_muestras > 0 else 0.0

def balanced_accuracy_manual(matriz_confusion):
    """Calcula la exactitud balanceada (promedio del recall por clase)."""
    recalls = []
    num_clases = matriz_confusion.shape[0]
    for i in range(num_clases):
        TP = matriz_confusion[i, i]
        FN = matriz_confusion[i, :].sum() - TP
        recall_clase = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recalls.append(recall_clase)
    return np.mean(recalls)

# --- Nuevas Métricas de Curva ---

def roc_auc_manual(y_verdadero, y_pred_proba, promediado='macro'):
    """
    Calcula el área bajo la curva ROC (AUC-ROC) de forma manual.
    Para problemas multiclase, usa la estrategia One-vs-Rest (OvR).
    """
    clases = np.unique(y_verdadero)
    n_clases = len(clases)
    aucs_por_clase = []

    for i, clase in enumerate(clases):
        # Binarizar las etiquetas para la estrategia OvR
        y_verdadero_binario = (y_verdadero == clase).astype(int)
        scores_clase = y_pred_proba[:, i]

        # Ordenar scores y etiquetas
        desc_score_indices = np.argsort(scores_clase, kind="mergesort")[::-1]
        y_verdadero_ordenado = y_verdadero_binario[desc_score_indices]
        scores_ordenados = scores_clase[desc_score_indices]

        # Calcular TP y FP acumulados
        tps = np.cumsum(y_verdadero_ordenado)
        fps = np.cumsum(1 - y_verdadero_ordenado)

        # Calcular TPR y FPR
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)

        # Añadir puntos (0,0) y (1,1)
        tpr = np.r_[0, tpr, 1]
        fpr = np.r_[0, fpr, 1]

        # Calcular área usando la regla del trapecio
        auc = np.trapz(tpr, fpr)
        aucs_por_clase.append(auc)

    if promediado == 'macro':
        return np.mean(aucs_por_clase)
    else:
        # Podrían implementarse otros promedios (weighted, etc.)
        return aucs_por_clase

def pr_auc_manual(y_verdadero, y_pred_proba, promediado='macro'):
    """
    Calcula el área bajo la curva Precisión-Recall (PR-AUC) de forma manual.
    Para problemas multiclase, usa la estrategia One-vs-Rest (OvR).
    """
    clases = np.unique(y_verdadero)
    n_clases = len(clases)
    aucs_por_clase = []

    for i, clase in enumerate(clases):
        y_verdadero_binario = (y_verdadero == clase).astype(int)
        scores_clase = y_pred_proba[:, i]

        # Ordenar por scores
        desc_score_indices = np.argsort(scores_clase, kind="mergesort")[::-1]
        y_verdadero_ordenado = y_verdadero_binario[desc_score_indices]

        # Calcular precisión y recall en cada umbral
        tps_acumulado = np.cumsum(y_verdadero_ordenado)
        puntos_datos = np.arange(1, len(y_verdadero_ordenado) + 1)

        precision = tps_acumulado / puntos_datos
        recall = tps_acumulado / np.sum(y_verdadero_binario) if np.sum(y_verdadero_binario) > 0 else np.zeros_like(tps_acumulado)

        # Asegurar que el recall es monótonamente creciente
        recall = np.r_[0, recall]
        precision = np.r_[1, precision] # Iniciar con precisión 1 en recall 0

        # Calcular área usando la regla del trapecio
        auc = np.trapz(precision, recall)
        aucs_por_clase.append(auc)

    if promediado == 'macro':
        return np.mean(aucs_por_clase)
    else:
        return aucs_por_clase