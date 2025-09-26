# -*- coding: utf-8 -*-
"""
Este módulo contiene las implementaciones manuales de algoritmos de Machine Learning.
El objetivo es replicar la funcionalidad de librerías como scikit-learn desde cero,
utilizando únicamente numpy para las operaciones numéricas.
"""

import numpy as np
from collections import Counter

class RegresionLogisticaManual:
    """
    Implementación de Regresión Logística Multinomial (Softmax) desde cero.
    (El código de esta clase se mantiene como en el paso anterior)
    """

    def __init__(self, tasa_aprendizaje=0.1, epocas=1000, C=1.0, penalizacion='l2', tolerancia=1e-6, semilla_aleatoria=0):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.C = C
        self.penalizacion = penalizacion
        self.tolerancia = tolerancia
        self.generador_aleatorio = np.random.RandomState(semilla_aleatoria)
        self.pesos_ = None
        self.sesgo_ = None
        self.clases_ = None

    def _one_hot(self, y, num_clases):
        Y_one_hot = np.zeros((len(y), num_clases))
        for i, val in enumerate(y):
            Y_one_hot[i, val] = 1
        return Y_one_hot

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_muestras, n_caracteristicas = X.shape
        self.clases_ = np.unique(y)
        K = len(self.clases_)
        mapeo_clases = {clase: i for i, clase in enumerate(self.clases_)}
        y_mapeado = np.array([mapeo_clases[clase] for clase in y])
        self.pesos_ = self.generador_aleatorio.normal(0, 0.01, size=(K, n_caracteristicas))
        self.sesgo_ = np.zeros(K)
        Y_one_hot = self._one_hot(y_mapeado, K)
        for _ in range(self.epocas):
            logits = X.dot(self.pesos_.T) + self.sesgo_
            probabilidades = self._softmax(logits)
            gradiente_pesos = -( (Y_one_hot - probabilidades).T.dot(X) ) / n_muestras
            gradiente_sesgo = -(Y_one_hot - probabilidades).sum(axis=0) / n_muestras
            if self.penalizacion == 'l2':
                gradiente_pesos += (1.0 / self.C) * self.pesos_
            self.pesos_ -= self.tasa_aprendizaje * gradiente_pesos
            self.sesgo_ -= self.tasa_aprendizaje * gradiente_sesgo

    def predict_proba(self, X):
        if self.pesos_ is None or self.sesgo_ is None:
            raise RuntimeError("El modelo debe ser entrenado antes de realizar predicciones.")
        logits = X.dot(self.pesos_.T) + self.sesgo_
        return self._softmax(logits)

    def predict(self, X):
        probabilidades = self.predict_proba(X)
        indices_predichos = np.argmax(probabilidades, axis=1)
        return np.array([self.clases_[i] for i in indices_predichos])

# --- Implementación MEJORADA de Árbol de Decisión y Random Forest ---

class Nodo:
    """Clase auxiliar para representar un nodo en el árbol de decisión."""
    def __init__(self, indice_feature=None, umbral=None, hijo_izquierdo=None, hijo_derecho=None, *, valor=None):
        self.indice_feature = indice_feature
        self.umbral = umbral
        self.hijo_izquierdo = hijo_izquierdo
        self.hijo_derecho = hijo_derecho
        self.valor = valor # Para hojas: distribución de clases

    def es_hoja(self):
        return self.valor is not None

class ArbolDecisionManual:
    """Implementación manual de un Árbol de Decisión con predict_proba e importancia de características."""
    def __init__(self, min_muestras_division=2, max_profundidad=100, n_total_features=None):
        self.min_muestras_division = min_muestras_division
        self.max_profundidad = max_profundidad
        self.raiz = None
        self.clases_ = None
        self.n_total_features = n_total_features
        self.feature_importances_ = np.zeros(self.n_total_features) if self.n_total_features else None

    def _impureza_gini(self, y):
        _, conteos = np.unique(y, return_counts=True)
        probabilidades = conteos / len(y) if len(y) > 0 else 0
        return 1 - np.sum(probabilidades**2)

    def _mejor_division(self, X, y, indices_features_arbol):
        mejor_ganancia = -1
        mejor_feature, mejor_umbral = None, None
        impureza_actual = self._impureza_gini(y)
        n_muestras_actual = len(y)

        for idx_col, idx_feature_original in enumerate(indices_features_arbol):
            umbrales = np.unique(X[:, idx_col])
            for umbral in umbrales:
                y_izquierdo = y[X[:, idx_col] <= umbral]
                y_derecho = y[X[:, idx_col] > umbral]
                if len(y_izquierdo) == 0 or len(y_derecho) == 0:
                    continue

                p_izquierdo = len(y_izquierdo) / n_muestras_actual
                impureza_ponderada = p_izquierdo * self._impureza_gini(y_izquierdo) + (1 - p_izquierdo) * self._impureza_gini(y_derecho)
                ganancia_info = impureza_actual - impureza_ponderada

                if ganancia_info > mejor_ganancia:
                    mejor_ganancia = ganancia_info
                    mejor_feature_idx_local = idx_col
                    mejor_umbral = umbral

        if mejor_ganancia > 0 and self.feature_importances_ is not None:
            idx_feature_original = indices_features_arbol[mejor_feature_idx_local]
            self.feature_importances_[idx_feature_original] += (n_muestras_actual / self.n_muestras_total) * mejor_ganancia

        return mejor_feature_idx_local if mejor_ganancia > 0 else None, mejor_umbral

    def _construir_arbol(self, X, y, indices_features_arbol, profundidad=0):
        n_muestras, _ = X.shape
        if (profundidad >= self.max_profundidad or len(np.unique(y)) == 1 or n_muestras < self.min_muestras_division):
            conteos_hoja = Counter(y)
            distribucion = np.array([conteos_hoja.get(c, 0) for c in self.clases_])
            return Nodo(valor=distribucion)

        idx_feature, umbral = self._mejor_division(X, y, indices_features_arbol)
        if idx_feature is None:
            conteos_hoja = Counter(y)
            distribucion = np.array([conteos_hoja.get(c, 0) for c in self.clases_])
            return Nodo(valor=distribucion)

        indices_izquierdos = X[:, idx_feature] <= umbral
        X_izq, y_izq = X[indices_izquierdos], y[indices_izquierdos]
        X_der, y_der = X[~indices_izquierdos], y[~indices_izquierdos]

        hijo_izquierdo = self._construir_arbol(X_izq, y_izq, indices_features_arbol, profundidad + 1)
        hijo_derecho = self._construir_arbol(X_der, y_der, indices_features_arbol, profundidad + 1)
        return Nodo(idx_feature, umbral, hijo_izquierdo, hijo_derecho)

    def fit(self, X, y, indices_features_arbol=None):
        self.clases_ = np.unique(y)
        self.n_muestras_total = len(y)
        if indices_features_arbol is None:
            indices_features_arbol = np.arange(X.shape[1])
        if self.n_total_features is None:
            self.n_total_features = X.shape[1]
            self.feature_importances_ = np.zeros(self.n_total_features)
        self.raiz = self._construir_arbol(X, y, indices_features_arbol)

    def _predecir_proba_muestra(self, x, nodo):
        if nodo.es_hoja():
            return nodo.valor
        if x[nodo.indice_feature] <= nodo.umbral:
            return self._predecir_proba_muestra(x, nodo.hijo_izquierdo)
        else:
            return self._predecir_proba_muestra(x, nodo.hijo_derecho)

    def predict_proba(self, X):
        distribuciones = np.array([self._predecir_proba_muestra(x, self.raiz) for x in X])
        suma_dist = np.sum(distribuciones, axis=1, keepdims=True)
        return distribuciones / suma_dist

    def predict(self, X):
        probabilidades = self.predict_proba(X)
        return np.array([self.clases_[i] for i in np.argmax(probabilidades, axis=1)])

class RandomForestManual:
    """Implementación manual de RF con predict_proba e importancia de características."""
    def __init__(self, n_estimadores=100, max_profundidad=10, min_muestras_division=2, max_features='sqrt', semilla_aleatoria=0):
        self.n_estimadores = n_estimadores
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.max_features = max_features
        self.semilla_aleatoria = semilla_aleatoria
        self.arboles = []
        self.clases_ = None
        self.feature_importances_ = None

    def _bootstrap_sample(self, X, y, seed):
        n_muestras = X.shape[0]
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_muestras, size=n_muestras, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.clases_ = np.unique(y)
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)

        if self.max_features == 'sqrt':
            n_features_sub = int(np.sqrt(n_features))
        else:
            n_features_sub = n_features

        rng = np.random.RandomState(self.semilla_aleatoria)

        for i in range(self.n_estimadores):
            arbol = ArbolDecisionManual(
                max_profundidad=self.max_profundidad,
                min_muestras_division=self.min_muestras_division,
                n_total_features=n_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y, self.semilla_aleatoria + i)
            indices_features = rng.choice(n_features, n_features_sub, replace=False)

            arbol.fit(X_sample[:, indices_features], y_sample, indices_features_arbol=indices_features)
            self.arboles.append((arbol, indices_features))
            self.feature_importances_ += arbol.feature_importances_

        self.feature_importances_ /= self.n_estimadores

    def predict_proba(self, X):
        predicciones_proba = np.zeros((X.shape[0], len(self.clases_)))
        for arbol, indices_features in self.arboles:
            predicciones_proba += arbol.predict_proba(X[:, indices_features])
        return predicciones_proba / self.n_estimadores

    def predict(self, X):
        probabilidades = self.predict_proba(X)
        return np.array([self.clases_[i] for i in np.argmax(probabilidades, axis=1)])

class SVMManual:
    """Implementación de SVM lineal con estrategia One-vs-Rest."""
    def __init__(self, tasa_aprendizaje=0.01, epocas=100, C=1.0, semilla_aleatoria=0):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.C = C
        self.generador_aleatorio = np.random.RandomState(semilla_aleatoria)
        self.clasificadores_ = {}
        self.clases_ = None

    def fit(self, X, y):
        self.clases_ = np.unique(y)
        for clase in self.clases_:
            y_binario = np.where(y == clase, 1, -1)
            pesos = self.generador_aleatorio.normal(0, 0.01, size=X.shape[1])
            sesgo = 0
            for _ in range(self.epocas):
                for i, x_i in enumerate(X):
                    condicion = y_binario[i] * (np.dot(x_i, pesos) + sesgo) >= 1
                    if condicion:
                        grad_w = (2 / self.epocas) * pesos
                        pesos -= self.tasa_aprendizaje * grad_w
                    else:
                        grad_w = (2 / self.epocas) * pesos - self.C * y_binario[i] * x_i
                        grad_b = -self.C * y_binario[i]
                        pesos -= self.tasa_aprendizaje * grad_w
                        sesgo -= self.tasa_aprendizaje * grad_b
            self.clasificadores_[clase] = {'pesos': pesos, 'sesgo': sesgo}

    def predict_scores(self, X):
        """Devuelve los scores de la función de decisión para la calibración."""
        scores = np.zeros((X.shape[0], len(self.clases_)))
        for i, clase in enumerate(self.clases_):
            pesos = self.clasificadores_[clase]['pesos']
            sesgo = self.clasificadores_[clase]['sesgo']
            scores[:, i] = np.dot(X, pesos) + sesgo
        return scores

    def predict(self, X):
        scores = self.predict_scores(X)
        indices_predichos = np.argmax(scores, axis=1)
        return np.array([self.clases_[i] for i in indices_predichos])