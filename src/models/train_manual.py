# -*- coding: utf-8 -*-
"""
Script para entrenar los modelos de Machine Learning manuales.

Este script carga los datos procesados, entrena un modelo especificado
y guarda el artefacto del modelo entrenado para su uso posterior.
"""

import argparse
import pandas as pd
import joblib
from src.data.loader import cargar_csv_crudo
from src.models.algorithms_manual import RegresionLogisticaManual
# Importar otros modelos manuales aquí a medida que se implementen
# from src.models.algorithms_manual import RandomForestManual, SVMManual
from src.models.knn_wrapper import crear_pipeline_knn

# Mapeo de nombres de modelos a sus clases/funciones constructoras
MODELOS_DISPONIBLES = {
    "regresion_logistica": RegresionLogisticaManual,
    # "random_forest": RandomForestManual,
    # "svm": SVMManual,
    "knn": crear_pipeline_knn,
}

def entrenar_modelo(nombre_modelo: str, ruta_datos_entrenamiento: str, ruta_guardado_modelo: str):
    """
    Función principal para orquestar el entrenamiento del modelo.

    Args:
        nombre_modelo (str): El nombre del modelo a entrenar (debe estar en MODELOS_DISPONIBLES).
        ruta_datos_entrenamiento (str): Ruta al archivo CSV con los datos de entrenamiento procesados.
        ruta_guardado_modelo (str): Ruta donde se guardará el modelo entrenado (archivo .joblib).
    """
    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise ValueError(f"Modelo '{nombre_modelo}' no reconocido. Disponibles: {list(MODELOS_DISPONIBLES.keys())}")

    print(f"Cargando datos de entrenamiento desde '{ruta_datos_entrenamiento}'...")
    df_entrenamiento = pd.read_csv(ruta_datos_entrenamiento)

    # Separar características (X) y objetivo (y)
    # Asumimos que la última columna es el objetivo
    X_train = df_entrenamiento.iloc[:, :-1].values
    y_train = df_entrenamiento.iloc[:, -1].values

    print(f"Entrenando el modelo: {nombre_modelo}...")
    constructor_modelo = MODELOS_DISPONIBLES[nombre_modelo]
    modelo = constructor_modelo() # Se pueden añadir hiperparámetros aquí

    modelo.fit(X_train, y_train)

    print(f"Guardando el modelo entrenado en '{ruta_guardado_modelo}'...")
    joblib.dump(modelo, ruta_guardado_modelo)
    print("Entrenamiento completado y modelo guardado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar un modelo de ML manualmente.")
    parser.add_argument("--modelo", type=str, required=True, choices=MODELOS_DISPONIBLES.keys(),
                        help="Nombre del modelo a entrenar.")
    parser.add_argument("--datos", type=str, required=True,
                        help="Ruta al archivo CSV de datos de entrenamiento procesados.")
    parser.add_argument("--guardar_en", type=str, required=True,
                        help="Ruta para guardar el archivo del modelo entrenado (.joblib).")

    args = parser.parse_args()

    entrenar_modelo(
        nombre_modelo=args.modelo,
        ruta_datos_entrenamiento=args.datos,
        ruta_guardado_modelo=args.guardar_en
    )