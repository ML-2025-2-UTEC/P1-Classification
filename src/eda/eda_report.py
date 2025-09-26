# -*- coding: utf-8 -*-
"""
Módulo para generar un reporte de Análisis Exploratorio de Datos (EDA) de forma programática.

Este script carga un conjunto de datos, realiza varios análisis como resúmenes estadísticos,
análisis de valores nulos, visualizaciones y cálculos de correlación, y guarda los
resultados en formato CSV y PNG en el directorio 'reports/'.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# --- Constantes de Rutas ---
RUTA_REPORTES = "reports"
RUTA_FIGURAS = os.path.join(RUTA_REPORTES, "figs")

# --- Funciones de Análisis ---

def resumen_basico(df: pd.DataFrame, ruta_guardado: str):
    """Genera y guarda un resumen básico del DataFrame (info y describe)."""
    print("Generando resumen básico...")
    with open(ruta_guardado, 'w') as f:
        f.write("--- Resumen General (df.info()) ---\n")
        # Redirigir la salida de info() a un buffer de string
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())

        f.write("\n\n--- Resumen Estadístico (df.describe().T) ---\n")
        resumen_estadistico = df.describe().T
        f.write(resumen_estadistico.to_string())
    print(f"Resumen básico guardado en: {ruta_guardado}")

def tabla_nulos(df: pd.DataFrame, ruta_guardado: str):
    """Calcula el porcentaje de valores nulos por columna y lo guarda en un CSV."""
    print("Calculando tabla de valores nulos...")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100
    tabla = pd.DataFrame({
        "conteo_nulos": nulos,
        "porcentaje_nulos": porcentaje_nulos
    }).sort_values(by="porcentaje_nulos", ascending=False)
    tabla.to_csv(ruta_guardado)
    print(f"Tabla de nulos guardada en: {ruta_guardado}")

def graficar_univariado_numerico(df: pd.DataFrame, columna: str, ruta_guardado: str):
    """Crea y guarda un histograma con una curva de densidad (KDE) para una columna numérica."""
    print(f"Graficando distribución para '{columna}'...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], kde=True, bins=30)
    plt.title(f"Distribución de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(ruta_guardado)
    plt.close()

def graficar_boxplot_por_clase(df: pd.DataFrame, col_numerica: str, col_clase: str, ruta_guardado: str):
    """Crea y guarda un boxplot de una variable numérica agrupada por una variable de clase."""
    print(f"Graficando boxplot para '{col_numerica}' por '{col_clase}'...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=col_clase, y=col_numerica, data=df)
    plt.title(f"Boxplot de {col_numerica} por {col_clase}")
    plt.xlabel(col_clase)
    plt.ylabel(col_numerica)
    plt.grid(True)
    plt.savefig(ruta_guardado)
    plt.close()

def matriz_correlacion_pearson(df_numerico: pd.DataFrame, ruta_guardado: str):
    """Calcula y guarda una matriz de correlación de Pearson como un heatmap."""
    print("Calculando matriz de correlación de Pearson...")
    correlaciones = df_numerico.corr(method='pearson')
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title("Matriz de Correlación de Pearson")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Heatmap de correlación guardado en: {ruta_guardado}")

def cramers_v_manual(col_a, col_b):
    """Calcula la V de Cramer entre dos columnas categóricas."""
    tabla_contingencia = pd.crosstab(col_a, col_b)
    chi2, _, _, _ = chi2_contingency(tabla_contingencia)
    n = tabla_contingencia.sum().sum()
    phi2 = chi2 / n
    r, k = tabla_contingencia.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    # Evitar división por cero
    if min((k_corr-1), (r_corr-1)) == 0:
        return 0
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

def deteccion_outliers_iqr(df: pd.DataFrame, columna: str) -> pd.Index:
    """Detecta outliers en una columna numérica usando el Rango Intercuartílico (IQR)."""
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers.index

def pca_2d_manual(X_numerico: pd.DataFrame, ruta_guardado: str, y_clase=None):
    """
    Realiza un PCA manual a 2 componentes y grafica los resultados.
    Colorea los puntos si se proporciona una variable de clase.
    """
    print("Realizando PCA manual a 2 dimensiones...")
    # 1. Centrar los datos
    X_centrado = X_numerico - X_numerico.mean()
    # 2. Calcular la matriz de covarianza
    matriz_cov = np.cov(X_centrado, rowvar=False)
    # 3. Calcular autovalores y autovectores
    autovalores, autovectores = np.linalg.eigh(matriz_cov)
    # 4. Ordenar autovectores por autovalores descendentes
    indices_ordenados = np.argsort(autovalores)[::-1]
    autovectores_ordenados = autovectores[:, indices_ordenados]
    # 5. Seleccionar los 2 componentes principales
    componentes_principales = autovectores_ordenados[:, :2]
    # 6. Proyectar los datos
    X_pca = X_centrado.dot(componentes_principales)
    X_pca.columns = ["Componente Principal 1", "Componente Principal 2"]

    # Graficar
    plt.figure(figsize=(12, 8))
    if y_clase is not None:
        sns.scatterplot(x=X_pca.iloc[:, 0], y=X_pca.iloc[:, 1], hue=y_clase, palette="viridis", s=50, alpha=0.7)
        plt.title("PCA Manual a 2 Componentes por Clase")
    else:
        sns.scatterplot(x=X_pca.iloc[:, 0], y=X_pca.iloc[:, 1], s=50, alpha=0.7)
        plt.title("PCA Manual a 2 Componentes")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfico de PCA guardado en: {ruta_guardado}")

def balance_clases(df: pd.DataFrame, col_clase: str, ruta_guardado: str):
    """Calcula y grafica el balance de la clase objetivo."""
    print(f"Graficando balance para la clase '{col_clase}'...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col_clase, data=df, palette="viridis")
    plt.title(f"Balance de Clases para '{col_clase}'")
    plt.xlabel("Clase")
    plt.ylabel("Conteo")
    plt.savefig(ruta_guardado)
    plt.close()

# --- Script Principal ---

def ejecutar_eda(ruta_archivo: str, col_objetivo: str):
    """
    Orquesta la ejecución de todas las funciones de EDA.
    """
    # Crear directorios si no existen
    if not os.path.exists(RUTA_REPORTES):
        os.makedirs(RUTA_REPORTES)
    if not os.path.exists(RUTA_FIGURAS):
        os.makedirs(RUTA_FIGURAS)

    # Cargar datos
    print(f"Cargando datos desde: {ruta_archivo}")
    df = pd.read_csv(ruta_archivo)

    # Separar columnas numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = df.select_dtypes(include='object').columns.tolist()
    if col_objetivo in columnas_numericas:
        columnas_numericas.remove(col_objetivo)
    if col_objetivo in columnas_categoricas:
        columnas_categoricas.remove(col_objetivo)

    # 1. Resumen básico
    resumen_basico(df, os.path.join(RUTA_REPORTES, "resumen_basico.txt"))

    # 2. Tabla de nulos
    tabla_nulos(df, os.path.join(RUTA_REPORTES, "tabla_nulos.csv"))

    # 3. Gráficos univariados para todas las variables numéricas
    for col in columnas_numericas:
        graficar_univariado_numerico(df, col, os.path.join(RUTA_FIGURAS, f"dist_{col}.png"))

    # 4. Boxplots por clase para todas las variables numéricas
    for col in columnas_numericas:
        graficar_boxplot_por_clase(df, col, col_objetivo, os.path.join(RUTA_FIGURAS, f"boxplot_{col}_por_{col_objetivo}.png"))

    # 5. Matriz de correlación de Pearson
    matriz_correlacion_pearson(df[columnas_numericas], os.path.join(RUTA_FIGURAS, "heatmap_pearson.png"))

    # 6. PCA 2D
    if len(columnas_numericas) >= 2:
        pca_2d_manual(df[columnas_numericas], os.path.join(RUTA_FIGURAS, "pca_2d_manual.png"), df[col_objetivo])

    # 7. Balance de clases
    balance_clases(df, col_objetivo, os.path.join(RUTA_FIGURAS, f"balance_{col_objetivo}.png"))

    print("\n--- EDA Completado ---")
    print(f"Reportes y figuras guardados en el directorio '{RUTA_REPORTES}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Reportes EDA.")
    parser.add_argument("--input", type=str, required=True, help="Ruta al archivo CSV de entrada.")
    parser.add_argument("--target", type=str, required=True, help="Nombre de la columna objetivo.")

    args = parser.parse_args()

    ejecutar_eda(ruta_archivo=args.input, col_objetivo=args.target)