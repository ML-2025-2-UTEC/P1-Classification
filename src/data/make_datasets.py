# -*- coding: utf-8 -*-
"""
Módulo para crear los conjuntos de datos procesados para el modelado.

Este script orquesta un pipeline de preprocesamiento completo:
1. Carga los datos crudos de entrenamiento y prueba.
2. Separa características y variable objetivo.
3. Aplica un pipeline de transformaciones (imputación, one-hot, escalado).
   - El pipeline se ajusta (fit) ÚNICAMENTE con los datos de entrenamiento.
   - Luego, se aplica la transformación a los datos de entrenamiento y prueba.
4. Guarda los conjuntos de datos procesados (X_train, y_train, X_test, y_test)
   en `data/processed/` en formatos .csv y .npy.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib

# Añadir la raíz del proyecto al path para importar módulos de src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import cargar_csv_crudo
from src.features.transformers_manual import ImputadorNumericoMediana, ImputadorCategoricoModa, EscaladorEstandarManual, OneHotManual

def procesar_datos(ruta_entrenamiento: str, ruta_prueba: str, ruta_salida: str):
    """
    Ejecuta el pipeline completo de preprocesamiento de datos.
    """
    print("--- Iniciando el pipeline de preprocesamiento de datos ---")

    # --- 1. Carga de Datos ---
    df_entrenamiento = cargar_csv_crudo(os.path.basename(ruta_entrenamiento))
    df_prueba = cargar_csv_crudo(os.path.basename(ruta_prueba))
    print(f"Datos de entrenamiento cargados. Forma: {df_entrenamiento.shape}")
    print(f"Datos de prueba cargados. Forma: {df_prueba.shape}")

    # --- 2. Separación de Características y Objetivo ---
    # Asumimos que la columna objetivo se llama 'nivel_riesgo'
    columna_objetivo = 'nivel_riesgo'

    X_train = df_entrenamiento.drop(columna_objetivo, axis=1)
    y_train_raw = df_entrenamiento[columna_objetivo]

    X_test = df_prueba.drop(columna_objetivo, axis=1)
    y_test_raw = df_prueba[columna_objetivo]

    # Codificar etiquetas del objetivo (Bajo: 0, Medio: 1, Alto: 2)
    mapeo_objetivo = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
    y_train = y_train_raw.map(mapeo_objetivo)
    y_test = y_test_raw.map(mapeo_objetivo)

    # --- 3. Pipeline de Transformación ---
    # Identificar tipos de columnas desde el conjunto de entrenamiento
    columnas_numericas = X_train.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = X_train.select_dtypes(include='object').columns.tolist()

    # a) Imputación
    imputador_num = ImputadorNumericoMediana(variables=columnas_numericas)
    imputador_cat = ImputadorCategoricoModa(variables=columnas_categoricas)

    print("Ajustando imputadores con datos de entrenamiento...")
    imputador_num.fit(X_train)
    imputador_cat.fit(X_train)

    print("Aplicando imputación a entrenamiento y prueba...")
    X_train_imputado = imputador_num.transform(X_train)
    X_train_imputado = imputador_cat.transform(X_train_imputado)

    X_test_imputado = imputador_num.transform(X_test)
    X_test_imputado = imputador_cat.transform(X_test_imputado)

    # b) One-Hot Encoding
    codificador_onehot = OneHotManual(variables=columnas_categoricas, drop_last=True)

    print("Ajustando codificador OneHot con datos de entrenamiento...")
    codificador_onehot.fit(X_train_imputado)

    print("Aplicando OneHot a entrenamiento y prueba...")
    X_train_codificado = codificador_onehot.transform(X_train_imputado)
    X_test_codificado = codificador_onehot.transform(X_test_imputado)

    # Asegurarse de que ambos dataframes tengan las mismas columnas
    train_cols = set(X_train_codificado.columns)
    test_cols = set(X_test_codificado.columns)

    columnas_faltantes_en_test = list(train_cols - test_cols)
    if columnas_faltantes_en_test:
        print(f"Añadiendo columnas faltantes a test: {columnas_faltantes_en_test}")
        for col in columnas_faltantes_en_test:
            X_test_codificado[col] = 0

    columnas_extra_en_test = list(test_cols - train_cols)
    if columnas_extra_en_test:
        print(f"Eliminando columnas extra de test: {columnas_extra_en_test}")
        X_test_codificado = X_test_codificado.drop(columns=columnas_extra_en_test)

    X_test_codificado = X_test_codificado[X_train_codificado.columns]

    # c) Escalado
    columnas_a_escalar = X_train_codificado.select_dtypes(include=np.number).columns.tolist()
    escalador = EscaladorEstandarManual()

    print("Ajustando escalador con datos de entrenamiento...")
    escalador.fit(X_train_codificado[columnas_a_escalar])

    print("Aplicando escalado a entrenamiento y prueba...")
    X_train_procesado = escalador.transform(X_train_codificado[columnas_a_escalar])
    X_test_procesado = escalador.transform(X_test_codificado[columnas_a_escalar])

    # --- 4. Guardado de Datos Procesados ---
    print(f"Forma final de X_train: {X_train_procesado.shape}")
    print(f"Forma final de X_test: {X_test_procesado.shape}")

    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    # Guardar en formato .csv
    X_train_procesado.to_csv(os.path.join(ruta_salida, 'X_train_processed.csv'), index=False)
    y_train.to_csv(os.path.join(ruta_salida, 'y_train_processed.csv'), index=False)
    X_test_procesado.to_csv(os.path.join(ruta_salida, 'X_test_processed.csv'), index=False)
    y_test.to_csv(os.path.join(ruta_salida, 'y_test_processed.csv'), index=False)
    print("Archivos .csv guardados.")

    # Guardar en formato .npy
    np.save(os.path.join(ruta_salida, 'X_train_processed.npy'), X_train_procesado.values)
    np.save(os.path.join(ruta_salida, 'y_train_processed.npy'), y_train.values)
    np.save(os.path.join(ruta_salida, 'X_test_processed.npy'), X_test_procesado.values)
    np.save(os.path.join(ruta_salida, 'y_test_processed.npy'), y_test.values)
    print("Archivos .npy guardados.")

    print("\n--- Pipeline de preprocesamiento completado ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para procesar datos crudos y crear datasets de modelado.")
    parser.add_argument("--train_input", type=str, default="data/raw/datos_entrenamiento_riesgo.csv",
                        help="Ruta al archivo CSV de entrenamiento crudo.")
    parser.add_argument("--test_input", type=str, default="data/raw/datos_prueba_riesgo.csv",
                        help="Ruta al archivo CSV de prueba crudo.")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directorio donde se guardarán los datasets procesados.")

    args = parser.parse_args()

    procesar_datos(
        ruta_entrenamiento=args.train_input,
        ruta_prueba=args.test_input,
        ruta_salida=args.output_dir
    )