# -*- coding: utf-8 -*-
"""
Módulo para la ingeniería de características (Feature Engineering).

Este script contiene funciones para crear nuevas características a partir de las existentes
en el conjunto de datos, como la creación de ratios, la binarización de variables
y el agrupamiento por cuantiles (binning).
"""

import argparse
import pandas as pd
import numpy as np
import os

# Suponemos que loader.py está en un directorio hermano `data`
from src.data.loader import cargar_csv_crudo, guardar_df_procesado

# --- Funciones de Ingeniería de Características ---

def crear_ratios_financieros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevos ratios financieros a partir de columnas existentes.

    Ejemplos (estos nombres de columna son hipotéticos y deben adaptarse al dataset real):
    - 'proporcion_deuda_ingresos'
    - 'proporcion_uso_credito'
    """
    print("Creando ratios financieros...")
    df_transformado = df.copy()

    # Ejemplo 1: Proporción de deuda total sobre ingresos anuales
    # Se añade un valor pequeño al denominador para evitar división por cero.
    if 'deuda_total' in df.columns and 'ingresos_anuales' in df.columns:
        df_transformado['ratio_deuda_ingresos'] = df['deuda_total'] / (df['ingresos_anuales'] + 1e-6)

    # Ejemplo 2: Proporción del monto del préstamo sobre el patrimonio neto
    if 'monto_solicitado' in df.columns and 'patrimonio_neto' in df.columns:
        df_transformado['ratio_prestamo_patrimonio'] = df['monto_solicitado'] / (df['patrimonio_neto'] + 1e-6)

    return df_transformado

def binarizar_banderas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas categóricas con dos valores (ej. 'Si'/'No') a formato binario (1/0).
    """
    print("Binarizando banderas...")
    df_transformado = df.copy()

    # Ejemplo: Binarizar una columna 'propiedad_vivienda'
    if 'propiedad_vivienda' in df.columns:
        # Asumimos que los valores son 'propia' y 'alquilada' o similar
        mapeo = {val: i for i, val in enumerate(df['propiedad_vivienda'].unique())}
        if len(mapeo) == 2: # Solo binarizar si hay exactamente dos categorías
            df_transformado['propiedad_vivienda_bin'] = df['propiedad_vivienda'].map(mapeo)

    return df_transformado

def agrupar_por_cuantiles(df: pd.DataFrame, columna: str, n_bins: int) -> pd.DataFrame:
    """
    Agrupa una columna numérica en `n_bins` contenedores basados en cuantiles (qcut).

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columna (str): Nombre de la columna numérica a agrupar.
        n_bins (int): Número de contenedores (bins).

    Returns:
        pd.DataFrame: DataFrame con la nueva columna agrupada.
    """
    print(f"Agrupando '{columna}' en {n_bins} cuantiles...")
    df_transformado = df.copy()

    if columna in df.columns:
        try:
            etiquetas = [f'cuantil_{i+1}' for i in range(n_bins)]
            df_transformado[f'{columna}_binned'] = pd.qcut(
                df[columna],
                q=n_bins,
                labels=etiquetas,
                duplicates='drop' # Ignorar errores si los cuantiles no son únicos
            )
        except ValueError as e:
            print(f"No se pudo agrupar la columna '{columna}': {e}")

    return df_transformado

# --- Script Principal ---

def construir_caracteristicas(ruta_entrada: str, ruta_salida: str):
    """
    Orquesta todo el proceso de ingeniería de características.
    """
    print("--- Iniciando el proceso de Ingeniería de Características ---")

    # Cargar datos crudos
    nombre_archivo_entrada = os.path.basename(ruta_entrada)
    df = cargar_csv_crudo(nombre_archivo_entrada)

    # Aplicar transformaciones
    df = crear_ratios_financieros(df)
    df = binarizar_banderas(df)

    # Ejemplo de binning, adaptar la columna según el dataset
    if 'edad' in df.columns:
        df = agrupar_por_cuantiles(df, 'edad', n_bins=5)
    if 'ingresos_anuales' in df.columns:
        df = agrupar_por_cuantiles(df, 'ingresos_anuales', n_bins=10)

    # Guardar el dataset con las nuevas características
    nombre_archivo_salida = os.path.basename(ruta_salida)
    guardar_df_procesado(df, nombre_archivo_salida)

    print("\n--- Ingeniería de Características Completada ---")
    print(f"Dataset con nuevas características guardado en: {ruta_salida}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Ingeniería de Características.")
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta al archivo CSV de datos crudos de entrada (ej. data/raw/datos_entrenamiento_riesgo.csv).")
    parser.add_argument("--output", type=str, required=True,
                        help="Ruta para guardar el archivo CSV procesado con las nuevas características (ej. data/processed/X_train_features.csv).")

    args = parser.parse_args()

    # Asegurarse de que el directorio de salida exista
    directorio_salida = os.path.dirname(args.output)
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    construir_caracteristicas(ruta_entrada=args.input, ruta_salida=args.output)