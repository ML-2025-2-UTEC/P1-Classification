# -*- coding: utf-8 -*-
"""
Módulo para la selección manual de características.

Contiene implementaciones manuales de métodos de selección de características como
SelectKBest (usando ANOVA F-test y Chi-cuadrado), RFE (Recursive Feature Elimination)
y cálculo de VIF (Variance Inflation Factor).
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
from src.models.algorithms_manual import RegresionLogisticaManual

# --- 1. SelectKBest Manual ---

def anova_f_score_manual(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calcula el F-score de ANOVA para cada característica numérica en X contra y."""
    scores = {}
    for col in X.columns:
        # Agrupar la característica por cada clase en y
        grupos = [X[col][y == c] for c in y.unique()]
        # Eliminar grupos con menos de 2 muestras para evitar errores en f_oneway
        grupos_validos = [g for g in grupos if len(g) >= 2]
        if len(grupos_validos) < 2:
            f_val, p_val = 0, 1 # No se puede calcular, asignar score bajo
        else:
            f_val, p_val = f_oneway(*grupos_validos)
        scores[col] = f_val
    return pd.Series(scores)

def chi2_score_manual(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calcula el estadístico Chi-cuadrado para cada característica categórica en X contra y."""
    scores = {}
    for col in X.columns:
        tabla_contingencia = pd.crosstab(X[col], y)
        chi2, p, _, _ = chi2_contingency(tabla_contingencia)
        scores[col] = chi2
    return pd.Series(scores)

class SelectKBestManual:
    """Selecciona las k mejores características basadas en una función de puntuación."""
    def __init__(self, score_func, k=10):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        self.scores_ = self.score_func(X, y)
        self.feature_names_ = self.scores_.nlargest(self.k).index.tolist()
        return self

    def transform(self, X):
        return X[self.feature_names_]

# --- 2. RFE (Recursive Feature Elimination) Manual ---

class RFEManual:
    """Eliminación Recursiva de Características."""
    def __init__(self, estimador, n_features_to_select=None):
        self.estimador = estimador
        self.n_features_to_select = n_features_to_select
        self.ranking_ = None
        self.support_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            self.n_features_to_select = n_features // 2

        features_restantes = list(X.columns)
        self.ranking_ = {}

        for i in range(n_features - self.n_features_to_select):
            self.estimador.fit(X[features_restantes].values, y.values)
            # Usar la magnitud de los coeficientes como importancia
            # Promediar la importancia a través de las clases para multiclase
            importancias = np.mean(np.abs(self.estimador.pesos_), axis=0)

            peor_feature_idx = np.argmin(importancias)
            peor_feature_nombre = features_restantes.pop(peor_feature_idx)
            self.ranking_[peor_feature_nombre] = n_features - i

        for feat in features_restantes:
            self.ranking_[feat] = 1

        self.support_ = [feat in features_restantes for feat in X.columns]
        self.feature_names_ = features_restantes
        return self

    def transform(self, X):
        return X[self.feature_names_]

# --- 3. VIF (Variance Inflation Factor) Manual ---

def _resolver_ols_para_vif(X: np.ndarray, y: np.ndarray) -> float:
    """Resuelve una regresión lineal y devuelve R^2 para el cálculo de VIF."""
    # Añadir intercepto a X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    try:
        # Ecuación normal: (X^T * X)^-1 * X^T * y
        coeficientes = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    except np.linalg.LinAlgError:
        # Usar pseudoinversa si la matriz es singular
        coeficientes = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    y_pred = X_b.dot(coeficientes)
    ss_total = np.sum((y - y.mean())**2)
    ss_residual = np.sum((y - y_pred)**2)

    # Evitar división por cero si ss_total es cero
    if ss_total == 0:
        return 1.0 # R^2 es 1 si y no tiene varianza

    r_cuadrado = 1 - (ss_residual / ss_total)
    return r_cuadrado

def vif_manual(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de la Varianza (VIF) para cada característica.
    """
    print("Calculando VIF para cada característica...")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [0.0] * len(X.columns)

    for i, col in enumerate(X.columns):
        y_vif = X[col].values
        X_vif = X.drop(columns=[col]).values

        r_cuadrado = _resolver_ols_para_vif(X_vif, y_vif)

        # VIF = 1 / (1 - R^2)
        # Evitar división por cero si R^2 es 1
        if np.isclose(r_cuadrado, 1.0):
            vif = np.inf
        else:
            vif = 1 / (1 - r_cuadrado)

        vif_data.loc[i, "VIF"] = vif

    return vif_data.sort_values(by="VIF", ascending=False)