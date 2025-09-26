# -*- coding: utf-8 -*-
"""
Módulo para la reducción de dimensionalidad manual.

Contiene una implementación manual del Análisis de Componentes Principales (PCA).
"""

import numpy as np
import pandas as pd

class PCAManual:
    """
    Implementación manual del Análisis de Componentes Principales (PCA).

    La reducción de dimensionalidad se logra proyectando los datos en un
    subespacio de menor dimensión definido por los componentes principales.

    Parámetros:
    -----------
    n_componentes : int
        El número de componentes principales a conservar.
    """

    def __init__(self, n_componentes: int):
        if n_componentes < 1:
            raise ValueError("n_componentes debe ser al menos 1.")
        self.n_componentes = n_componentes
        self.componentes_ = None
        self.media_ = None
        self.varianza_explicada_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Ajusta el modelo PCA a los datos X.

        Calcula los componentes principales de X.

        Args:
            X (np.ndarray): Los datos de entrenamiento, de forma (n_muestras, n_características).
        """
        # 1. Centrar los datos (restar la media)
        self.media_ = np.mean(X, axis=0)
        X_centrado = X - self.media_

        # 2. Calcular la SVD (Singular Value Decomposition)
        # U: Vectores singulares izquierdos
        # S: Valores singulares (en un vector 1D)
        # Vt: Vectores singulares derechos (transpuestos)
        U, S, Vt = np.linalg.svd(X_centrado, full_matrices=False)

        # Los componentes principales son las filas de Vt
        self.componentes_ = Vt[:self.n_componentes]

        # 3. Calcular la varianza explicada
        varianza_explicada = (S ** 2) / (len(X) - 1)
        varianza_total = np.sum(varianza_explicada)
        self.varianza_explicada_ratio_ = varianza_explicada / varianza_total

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la reducción de dimensionalidad a X.

        Proyecta X en el espacio de los componentes principales.

        Args:
            X (np.ndarray): Datos a transformar, de forma (n_muestras, n_características).

        Returns:
            np.ndarray: Datos transformados, de forma (n_muestras, n_componentes).
        """
        if self.componentes_ is None:
            raise RuntimeError("El modelo PCA debe ser ajustado (fit) antes de transformar los datos.")

        # Centrar los datos con la media del conjunto de entrenamiento
        X_centrado = X - self.media_

        # Proyectar los datos en los componentes principales
        X_transformado = X_centrado.dot(self.componentes_.T)

        return X_transformado

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo a los datos y luego los transforma.

        Args:
            X (np.ndarray): Datos de entrenamiento.

        Returns:
            np.ndarray: Datos transformados.
        """
        self.fit(X)
        return self.transform(X)