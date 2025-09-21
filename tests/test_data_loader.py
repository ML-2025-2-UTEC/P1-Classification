"""
Casos de prueba para funciones de carga y preprocesamiento de datos.
"""

import sys
import os
import pandas as pd
import numpy as np
from src.data.loader import (
    load_training_data, 
    load_test_data,
    get_feature_info,
    separate_features_target,
    encode_target_labels,
    DataPreprocessor
)

def test_load_data():
    """Probar funciones de carga de datos."""
    # Esto necesitaría los archivos de datos reales para ejecutarse
    try:
        train_data = load_training_data('../data/raw/datos_entrenamiento_riesgo.csv')
        test_data = load_test_data('../data/raw/datos_prueba_riesgo.csv')
        
        assert len(train_data) > 0, "Los datos de entrenamiento no deberían estar vacíos"
        assert len(test_data) > 0, "Los datos de prueba no deberían estar vacíos"
        assert 'nivel_riesgo' in train_data.columns, "La columna objetivo debería existir"
        
        print("✓ Pruebas de carga de datos pasaron")
        return True
    except FileNotFoundError:
        print("⚠ Archivos de datos no encontrados - saltando pruebas de carga de datos")
        return False


def test_feature_info():
    """Probar función de información de características."""
    feature_info = get_feature_info()
    
    assert 'financial' in feature_info, "Debería tener características financieras"
    assert 'payment_history' in feature_info, "Debería tener características de historial de pagos"
    assert 'demographic' in feature_info, "Debería tener características demográficas"
    assert 'target' in feature_info, "Debería tener variable objetivo"
    
    print("✓ Pruebas de información de características pasaron")


def test_target_encoding():
    """Probar codificación de etiquetas objetivo."""
    # Crear datos de muestra
    sample_target = pd.Series(['Bajo', 'Medio', 'Alto', 'Bajo', 'Alto'])
    encoded = encode_target_labels(sample_target)
    
    expected = pd.Series([0, 1, 2, 0, 2])
    assert encoded.equals(expected), "La codificación del objetivo debería funcionar correctamente"
    
    print("✓ Pruebas de codificación de objetivo pasaron")


def test_data_preprocessor():
    """Probar clase de preprocesador de datos."""
    # Crear datos de muestra
    np.random.seed(42)
    X_sample = pd.DataFrame({
        'feature1': np.random.normal(100, 20, 100),
        'feature2': np.random.normal(50, 10, 100),
        'feature3': np.random.normal(0, 1, 100)
    })
    
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X_sample)
    
    # Verificar que las medias estén cerca de 0 y las desviaciones cerca de 1
    means = X_processed.mean()
    stds = X_processed.std()
    
    assert all(abs(means) < 0.1), "Las medias deberían estar cerca de 0 después de estandarización"
    assert all(abs(stds - 1) < 0.1), "Las desviaciones deberían estar cerca de 1 después de estandarización"
    
    print("✓ Pruebas del preprocesador de datos pasaron")


if __name__ == "__main__":
    print("Ejecutando pruebas de carga y preprocesamiento de datos...")
    print("=" * 50)
    
    test_feature_info()
    test_target_encoding()
    test_data_preprocessor()
    test_load_data()
    
    print("=" * 50)
    print("¡Todas las pruebas disponibles completadas!")