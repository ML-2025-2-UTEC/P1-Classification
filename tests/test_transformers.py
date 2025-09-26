# -*- coding: utf-8 -*-
"""
Pruebas unitarias para los transformadores de datos manuales.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.transformers_manual import (
    ImputadorNumericoMediana,
    ImputadorCategoricoModa,
    EscaladorEstandarManual,
    OneHotManual
)

@pytest.fixture
def df_prueba():
    """DataFrame de prueba para los transformadores."""
    return pd.DataFrame({
        "numerica_con_nulos": [1, 2, np.nan, 4, 5],
        "categorica_con_nulos": ["A", "B", "A", np.nan, "B"],
        "numerica_sin_nulos": [10, 20, 30, 40, 50],
        "categorica_sin_nulos": ["X", "Y", "X", "Y", "X"]
    })

def test_imputador_numerico_mediana(df_prueba):
    """Prueba que el imputador de mediana funciona correctamente."""
    imputador = ImputadorNumericoMediana(variables=["numerica_con_nulos"])
    imputador.fit(df_prueba)
    df_transformado = imputador.transform(df_prueba)

    assert df_transformado["numerica_con_nulos"].isnull().sum() == 0
    assert df_transformado["numerica_con_nulos"].iloc[2] == df_prueba["numerica_con_nulos"].median()

def test_imputador_categorico_moda(df_prueba):
    """Prueba que el imputador de moda funciona correctamente."""
    imputador = ImputadorCategoricoModa(variables=["categorica_con_nulos"])
    imputador.fit(df_prueba)
    df_transformado = imputador.transform(df_prueba)

    assert df_transformado["categorica_con_nulos"].isnull().sum() == 0
    assert df_transformado["categorica_con_nulos"].iloc[3] == df_prueba["categorica_con_nulos"].mode()[0]

def test_escalador_estandar_manual(df_prueba):
    """Prueba que el escalador estándar manual centra y escala los datos."""
    escalador = EscaladorEstandarManual()
    df_numerico = df_prueba[["numerica_sin_nulos"]].copy()
    escalador.fit(df_numerico)
    df_transformado = escalador.transform(df_numerico)

    assert np.isclose(df_transformado["numerica_sin_nulos"].mean(), 0)
    assert np.isclose(df_transformado["numerica_sin_nulos"].std(), 1, atol=0.01) # La std de la muestra (ddof=1) es 1

def test_one_hot_manual(df_prueba):
    """Prueba que el codificador One-Hot manual crea las columnas correctas."""
    codificador = OneHotManual(variables=["categorica_sin_nulos"])
    codificador.fit(df_prueba)
    df_transformado = codificador.transform(df_prueba)

    assert "categorica_sin_nulos_X" in df_transformado.columns
    assert "categorica_sin_nulos_Y" in df_transformado.columns
    assert "categorica_sin_nulos" not in df_transformado.columns
    assert df_transformado["categorica_sin_nulos_X"].sum() == 3
    assert df_transformado["categorica_sin_nulos_Y"].sum() == 2