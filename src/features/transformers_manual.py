# -*- coding: utf-8 -*-
"""
Este módulo contiene clases para transformaciones de datos manuales,
emulando la funcionalidad de scikit-learn pero con implementaciones propias.
"""

import pandas as pd
import numpy as np


class ImputadorNumericoMediana:
    """
    Imputa valores faltantes en columnas numéricas utilizando la mediana.
    """

    def __init__(self, variables=None):
        if not isinstance(variables, (list, tuple)) and variables is not None:
            raise ValueError("El parámetro 'variables' debe ser una lista o tupla.")
        self.variables = variables
        self.imputadores_ = {}

    def fit(self, X: pd.DataFrame):
        """
        Aprende la mediana de las columnas especificadas.

        Args:
            X (pd.DataFrame): El DataFrame de entrenamiento.
        """
        columnas_a_imputar = self.variables if self.variables else X.select_dtypes(include=np.number).columns

        for col in columnas_a_imputar:
            if col not in X.columns:
                raise ValueError(f"La columna '{col}' no se encuentra en el DataFrame.")
            mediana = X[col].median()
            self.imputadores_[col] = mediana
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la imputación por mediana al DataFrame.

        Args:
            X (pd.DataFrame): El DataFrame a transformar.

        Returns:
            pd.DataFrame: El DataFrame con los valores imputados.
        """
        X_transformado = X.copy()
        for col, valor_imputacion in self.imputadores_.items():
            X_transformado[col] = X_transformado[col].fillna(valor_imputacion)
        return X_transformado


class ImputadorCategoricoModa:
    """
    Imputa valores faltantes en columnas categóricas utilizando la moda.
    """

    def __init__(self, variables=None):
        if not isinstance(variables, (list, tuple)) and variables is not None:
            raise ValueError("El parámetro 'variables' debe ser una lista o tupla.")
        self.variables = variables
        self.imputadores_ = {}

    def fit(self, X: pd.DataFrame):
        """
        Aprende la moda de las columnas especificadas.

        Args:
            X (pd.DataFrame): El DataFrame de entrenamiento.
        """
        columnas_a_imputar = self.variables if self.variables else X.select_dtypes(include='object').columns

        for col in columnas_a_imputar:
            if col not in X.columns:
                raise ValueError(f"La columna '{col}' no se encuentra en el DataFrame.")
            # La moda puede devolver una serie, tomamos el primer elemento.
            moda = X[col].mode()[0]
            self.imputadores_[col] = moda
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la imputación por moda al DataFrame.

        Args:
            X (pd.DataFrame): El DataFrame a transformar.

        Returns:
            pd.DataFrame: El DataFrame con los valores imputados.
        """
        X_transformado = X.copy()
        for col, valor_imputacion in self.imputadores_.items():
            X_transformado[col] = X_transformado[col].fillna(valor_imputacion)
        return X_transformado


class EscaladorEstandarManual:
    """
    Escala las variables numéricas para que tengan media 0 y desviación estándar 1.
    """

    def __init__(self):
        self.media_ = {}
        self.desviacion_estandar_ = {}

    def fit(self, X: pd.DataFrame):
        """
        Calcula la media y la desviación estándar de las columnas numéricas.

        Args:
            X (pd.DataFrame): El DataFrame de entrenamiento.
        """
        for col in X.select_dtypes(include=np.number).columns:
            self.media_[col] = X[col].mean()
            self.desviacion_estandar_[col] = X[col].std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el escalado estándar al DataFrame.

        Args:
            X (pd.DataFrame): El DataFrame a transformar.

        Returns:
            pd.DataFrame: El DataFrame con las variables escaladas.
        """
        X_transformado = X.copy()
        for col in self.media_:
            # Evitar división por cero si la desviación estándar es 0
            if self.desviacion_estandar_[col] > 0:
                X_transformado[col] = (X[col] - self.media_[col]) / self.desviacion_estandar_[col]
            else:
                X_transformado[col] = 0 # O dejarlo como está, dependiendo del caso de uso.
        return X_transformado


class OneHotManual:
    """
    Realiza codificación one-hot para variables categóricas.
    """

    def __init__(self, variables=None, drop_last=False):
        self.variables = variables
        self.drop_last = drop_last
        self.categorias_ = {}

    def fit(self, X: pd.DataFrame):
        """
        Aprende las categorías únicas de las columnas especificadas.

        Args:
            X (pd.DataFrame): El DataFrame de entrenamiento.
        """
        columnas_a_codificar = self.variables if self.variables else X.select_dtypes(include='object').columns

        for col in columnas_a_codificar:
            categorias = X[col].unique().tolist()
            self.categorias_[col] = categorias
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la codificación one-hot al DataFrame.

        Args:
            X (pd.DataFrame): El DataFrame a transformar.

        Returns:
            pd.DataFrame: El DataFrame con las variables codificadas.
        """
        X_transformado = X.copy()
        for col, categorias in self.categorias_.items():
            categorias_a_usar = categorias[:-1] if self.drop_last else categorias
            for categoria in categorias_a_usar:
                nombre_nueva_col = f"{col}_{categoria}"
                X_transformado[nombre_nueva_col] = (X_transformado[col] == categoria).astype(int)
            X_transformado = X_transformado.drop(col, axis=1)
        return X_transformado