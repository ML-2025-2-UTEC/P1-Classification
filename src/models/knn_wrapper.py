# -*- coding: utf-8 -*-
"""
Este módulo contiene un wrapper para el clasificador KNeighborsClassifier
de scikit-learn, para cumplir con la restricción de que solo este algoritmo
puede ser utilizado desde una librería externa.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def crear_pipeline_knn(n_neighbors=5, metric='euclidean', weights='uniform'):
    """
    Crea un pipeline de scikit-learn que primero escala los datos y luego
    aplica el clasificador KNN.

    Args:
        **kwargs: Argumentos para pasar al constructor de KNeighborsClassifier,
                  como n_neighbors, weights, metric, etc.

    Returns:
        sklearn.pipeline.Pipeline: Un pipeline de scikit-learn listo para usar.
    """
    pipeline = Pipeline([
        ('escalador', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric))
    ])
    return pipeline

# Ejemplo de cómo se podría usar:
#
# from src.models.knn_wrapper import crear_pipeline_knn
#
# # Crear el pipeline con 5 vecinos
# pipeline_knn = crear_pipeline_knn(n_neighbors=5, weights='distance')
#
# # Entrenar el modelo
# pipeline_knn.fit(X_train, y_train)
#
# # Realizar predicciones
# predicciones = pipeline_knn.predict(X_test)