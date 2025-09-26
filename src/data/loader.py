# -*- coding: utf-8 -*-
"""
Este módulo contiene funciones para cargar datos crudos y guardar datos procesados.
"""

import os
import pandas as pd

# Carpeta raíz del proyecto = un nivel arriba de src/
RUTA_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_DATOS_CRUDOS = os.path.join(RUTA_PROYECTO, "data", "raw")
RUTA_DATOS_PROCESADOS = os.path.join(RUTA_PROYECTO, "data", "processed")


def cargar_csv_crudo(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo CSV desde la carpeta de datos crudos (data/raw/).

    Args:
        nombre_archivo (str): El nombre del archivo CSV que se encuentra en data/raw/.

    Returns:
        pd.DataFrame: Un DataFrame de pandas con los datos del archivo.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra en la ruta especificada.
    """
    ruta_completa = os.path.join(RUTA_DATOS_CRUDOS, nombre_archivo)
    print(f"Cargando datos desde: {ruta_completa}")
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {ruta_completa}")
    return pd.read_csv(ruta_completa)


def guardar_df_procesado(df: pd.DataFrame, nombre_archivo: str):
    """
    Guarda un DataFrame en formato CSV en la carpeta de datos procesados (data/processed/).

    Args:
        df (pd.DataFrame): El DataFrame que se va a guardar.
        nombre_archivo (str): El nombre con el que se guardará el archivo CSV.
    """
    if not os.path.exists(RUTA_DATOS_PROCESADOS):
        os.makedirs(RUTA_DATOS_PROCESADOS)
    
    ruta_completa = os.path.join(RUTA_DATOS_PROCESADOS, nombre_archivo)
    print(f"Guardando datos procesados en: {ruta_completa}")
    df.to_csv(ruta_completa, index=False)


def describir_esquema(df: pd.DataFrame) -> dict:
    """
    Analiza un DataFrame y devuelve un diccionario con su esquema.

    El esquema incluye el número de columnas, sus nombres, tipos de datos y cantidad de nulos.

    Args:
        df (pd.DataFrame): El DataFrame a analizar.

    Returns:
        dict: Un diccionario que contiene información del esquema del DataFrame.
    """
    esquema = {
        "total_filas": int(df.shape[0]),
        "total_columnas": int(df.shape[1]),
        "info_columnas": [
            {
                "nombre_columna": col,
                "tipo_dato": str(df[col].dtype),
                "valores_nulos": int(df[col].isnull().sum())
            }
            for col in df.columns
        ],
    }
    return esquema