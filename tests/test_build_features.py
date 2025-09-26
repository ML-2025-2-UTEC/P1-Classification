# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo de construcción de características (build_features.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    crear_ratios_financieros,
    binarizar_banderas,
    agrupar_por_cuantiles
)

@pytest.fixture
def df_prueba_features():
    """DataFrame de prueba para la ingeniería de características. Corregido para tener la misma longitud."""
    return pd.DataFrame({
        'deuda_total': [10000, 20000, 5000, 15000, 30000],
        'ingresos_anuales': [50000, 80000, 20000, 60000, 100000],
        'propiedad_vivienda': ['propia', 'alquilada', 'propia', 'alquilada', 'propia'],
        'edad': [25, 45, 65, 35, 55]
    })

def test_crear_ratios_financieros(df_prueba_features):
    """Prueba que los ratios financieros se crean correctamente."""
    df = df_prueba_features.copy()
    df_transformado = crear_ratios_financieros(df)

    assert 'ratio_deuda_ingresos' in df_transformado.columns
    assert np.isclose(df_transformado['ratio_deuda_ingresos'].iloc[0], 10000 / 50000)
    assert np.isclose(df_transformado['ratio_deuda_ingresos'].iloc[1], 20000 / 80000)

def test_binarizar_banderas(df_prueba_features):
    """Prueba que la binarización de variables funciona correctamente."""
    df = df_prueba_features.copy()
    df_transformado = binarizar_banderas(df)

    assert 'propiedad_vivienda_bin' in df_transformado.columns
    assert df_transformado['propiedad_vivienda_bin'].dtype == 'int64'

    # Comprobación robusta: los conteos de las categorías deben coincidir
    conteos_originales = df['propiedad_vivienda'].value_counts().sort_values().values
    conteos_binarios = df_transformado['propiedad_vivienda_bin'].value_counts().sort_values().values

    np.testing.assert_array_equal(conteos_originales, conteos_binarios)


def test_agrupar_por_cuantiles(df_prueba_features):
    """Prueba que el agrupamiento por cuantiles se realiza correctamente."""
    df_transformado = agrupar_por_cuantiles(df_prueba_features, 'edad', n_bins=2)

    assert 'edad_binned' in df_transformado.columns
    # La columna 'edad' es [25, 45, 65, 35, 55]. El cuantil 0.5 (mediana) es 45.
    # Se espera que 25 y 35 estén en el primer cuantil y 55 y 65 en el segundo.
    # 45 puede caer en cualquiera de los dos dependiendo de la implementación de pd.qcut.
    assert df_transformado['edad_binned'].cat.categories.tolist() == ['cuantil_1', 'cuantil_2']
    assert df_transformado['edad_binned'].value_counts()['cuantil_1'] >= 2
    assert df_transformado['edad_binned'].value_counts()['cuantil_2'] >= 2