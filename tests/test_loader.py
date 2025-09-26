# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo de carga de datos (loader.py).
"""

import pytest
import pandas as pd
from src.data.loader import cargar_csv_crudo, describir_esquema

def test_cargar_csv_crudo_existente(tmp_path, monkeypatch):
    """Prueba que se carga un CSV existente correctamente."""
    # Crear un directorio temporal para simular 'data/raw'
    directorio_datos_crudos = tmp_path
    ruta_archivo = directorio_datos_crudos / "test.csv"
    df_esperado = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_esperado.to_csv(ruta_archivo, index=False)

    # Usar monkeypatch para que la función bajo prueba use el directorio temporal
    monkeypatch.setattr("src.data.loader.RUTA_DATOS_CRUDOS", str(directorio_datos_crudos))

    # Ejecutar la función a probar
    df_cargado = cargar_csv_crudo("test.csv")

    # Verificar que el DataFrame cargado es igual al esperado
    pd.testing.assert_frame_equal(df_cargado, df_esperado)


def test_cargar_csv_crudo_no_existente():
    """Prueba que se lanza un FileNotFoundError si el archivo no existe."""
    with pytest.raises(FileNotFoundError):
        cargar_csv_crudo("archivo_que_definitivamente_no_existe.csv")


def test_describir_esquema():
    """Prueba que el esquema del DataFrame se describe correctamente."""
    df = pd.DataFrame({
        "numerica": [1, 2, None],
        "categorica": ["A", "B", "A"]
    })
    esquema = describir_esquema(df)

    assert esquema["total_filas"] == 3
    assert esquema["total_columnas"] == 2
    assert esquema["info_columnas"][0]["nombre_columna"] == "numerica"
    assert esquema["info_columnas"][0]["valores_nulos"] == 1
    assert esquema["info_columnas"][1]["tipo_dato"] == "object"