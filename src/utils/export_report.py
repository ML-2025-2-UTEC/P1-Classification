# -*- coding: utf-8 -*-
"""
Módulo para exportar un reporte consolidado en formato Markdown.

Este script recopila los artefactos generados (figuras y tablas CSV)
desde el directorio 'reports/' y los compila en un único archivo
informe_final.md.
"""

import os
import argparse
import pandas as pd
import base64

def codificar_imagen_a_base64(ruta_imagen: str) -> str:
    """Codifica una imagen a formato base64 para incrustarla en Markdown."""
    try:
        with open(ruta_imagen, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except IOError:
        return ""

def generar_reporte_markdown(ruta_reportes: str, ruta_salida: str):
    """
    Genera un archivo Markdown consolidando los resultados del EDA y modelos.

    Args:
        ruta_reportes (str): La ruta al directorio 'reports/'.
        ruta_salida (str): La ruta completa donde se guardará el archivo .md final.
    """
    print(f"Iniciando la generación del reporte en '{ruta_salida}'...")

    informe = []

    # --- Título ---
    informe.append("# Reporte Consolidado del Proyecto de Riesgo Crediticio\n")

    # --- Sección de EDA ---
    informe.append("## 1. Análisis Exploratorio de Datos (EDA)\n")

    # Incrustar resumen básico
    ruta_resumen = os.path.join(ruta_reportes, "resumen_basico.txt")
    if os.path.exists(ruta_resumen):
        informe.append("### 1.1. Resumen Estadístico y de Tipos de Datos\n")
        with open(ruta_resumen, 'r') as f:
            informe.append(f"```\n{f.read()}\n```\n")

    # Incrustar tabla de nulos
    ruta_nulos = os.path.join(ruta_reportes, "tabla_nulos.csv")
    if os.path.exists(ruta_nulos):
        informe.append("### 1.2. Análisis de Valores Nulos\n")
        df_nulos = pd.read_csv(ruta_nulos)
        informe.append(df_nulos.to_markdown(index=False))
        informe.append("\n")

    # Incrustar figuras
    informe.append("### 1.3. Visualizaciones\n")
    ruta_figuras = os.path.join(ruta_reportes, "figs")
    if os.path.exists(ruta_figuras):
        for nombre_figura in sorted(os.listdir(ruta_figuras)):
            if nombre_figura.endswith(".png"):
                ruta_completa_fig = os.path.join(ruta_figuras, nombre_figura)
                titulo_figura = os.path.splitext(nombre_figura)[0].replace('_', ' ').title()

                informe.append(f"#### {titulo_figura}\n")
                base64_img = codificar_imagen_a_base64(ruta_completa_fig)
                if base64_img:
                    informe.append(f"![{titulo_figura}](data:image/png;base64,{base64_img})\n")

    # --- Sección de Modelado (Placeholder) ---
    # En un pipeline real, aquí se leerían los resultados de los experimentos.
    informe.append("\n## 2. Resultados de Modelado\n")
    informe.append("Esta sección contendría las métricas de evaluación de los modelos, "
                   "las matrices de confusión y los resultados de la validación cruzada anidada.\n")

    # Escribir el archivo final
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write("\n".join(informe))

    print(f"Reporte final guardado exitosamente en '{ruta_salida}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generador de Reportes Consolidados.")
    parser.add_argument("--dir_reportes", type=str, default="reports",
                        help="Directorio donde se encuentran los artefactos (CSVs, figuras).")
    parser.add_argument("--salida", type=str, default="reporte_final.md",
                        help="Nombre del archivo Markdown de salida.")

    args = parser.parse_args()

    generar_reporte_markdown(ruta_reportes=args.dir_reportes, ruta_salida=args.salida)