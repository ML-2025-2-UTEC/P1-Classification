# -*- coding: utf-8 -*-
"""
Módulo para la interpretabilidad de modelos de Machine Learning.

Contiene implementaciones manuales de técnicas de interpretabilidad como:
- Importancia de características global (basada en coeficientes o impureza).
- Importancia por permutación.
- Gráficos de Dependencia Parcial (PDP).
"""
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.metrics_manual import balanced_accuracy_manual, matriz_confusion_manual

# --- 1. Importancia de Características Global ---

def analizar_importancia_inteligente(modelo, feature_names, X_val, y_val, n_repeticiones=5):
    """
    Función inteligente que detecta automáticamente el tipo de modelo y usa
    la función apropiada para calcular importancia de características.
    
    Args:
        modelo: Modelo entrenado (puede ser pipeline)
        feature_names: Lista con nombres de características
        X_val: Datos de validación (DataFrame o array)
        y_val: Etiquetas de validación
        n_repeticiones: Número de repeticiones para permutación (si es necesario)
    
    Returns:
        pd.DataFrame: DataFrame con importancia de características
    """
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    
    print("Detectando tipo de modelo...")
    
    # Extraer el modelo real si es un pipeline
    modelo_real = modelo
    if isinstance(modelo, Pipeline):
        modelo_real = modelo.named_steps[list(modelo.named_steps.keys())[-1]]
        print(f"Pipeline detectado, extrayendo: {type(modelo_real).__name__}")
    
    modelo_nombre = type(modelo_real).__name__
    print(f"Tipo de modelo: {modelo_nombre}")
    
    # Detectar si es KNN (múltiples formas de identificarlo)
    es_knn = any([
        'KNN' in modelo_nombre.upper(),
        'NEIGHBOR' in modelo_nombre.upper(),
        'kNN' in str(type(modelo_real)),
        hasattr(modelo_real, 'n_neighbors'),
        'KNeighbors' in modelo_nombre
    ])
    
    if es_knn:
        print("Modelo KNN detectado - usando importancia por permutación")
        return _importancia_knn_optimizada(modelo, feature_names, X_val, y_val, n_repeticiones)
    
    else:
        print("Modelo con posible importancia global - intentando...")
        try:
            # Intentar usar tu función existente
            df_importancia = importancia_features_global(modelo, feature_names)
            print("Importancia global calculada exitosamente")
            return df_importancia
        
        except TypeError as e:
            print(f"Importancia global no disponible: {str(e).split('.')[0]}")
            print("Usando importancia por permutación como alternativa...")
            return _importancia_permutacion_optimizada(modelo, feature_names, X_val, y_val, n_repeticiones)


def _importancia_knn_optimizada(modelo, feature_names, X_val, y_val, n_repeticiones=5):
    """
    Función optimizada específicamente para KNN usando importancia por permutación.
    """
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np
    
    print(f"Calculando importancia por permutación para KNN ({n_repeticiones} repeticiones)...")
    
    # Convertir datos si es necesario
    if hasattr(X_val, 'values'):
        X_array = X_val.values
    else:
        X_array = np.array(X_val)
    
    # Score base
    pred_base = modelo.predict(X_array)
    score_base = accuracy_score(y_val, pred_base)
    print(f"Accuracy base: {score_base:.4f}")
    
    importancias = {}
    
    # Calcular importancia para cada característica
    for i, feature in enumerate(feature_names):
        if i % 10 == 0 or i == len(feature_names) - 1:
            print(f"   Procesando {i+1}/{len(feature_names)}: {feature}")
        
        scores_perm = []
        for _ in range(n_repeticiones):
            X_perm = X_array.copy()
            # Permutar solo esta característica
            np.random.shuffle(X_perm[:, i])
            pred_perm = modelo.predict(X_perm)
            score_perm = accuracy_score(y_val, pred_perm)
            scores_perm.append(score_perm)
        
        # Importancia = caída en rendimiento
        importancia = score_base - np.mean(scores_perm)
        importancias[feature] = importancia
    
    # Crear DataFrame ordenado
    df_importancia = pd.DataFrame(
        list(importancias.items()), 
        columns=['feature', 'importancia']
    ).sort_values('importancia', ascending=False)
    
    print("Importancia KNN calculada exitosamente")
    return df_importancia


def _importancia_permutacion_optimizada(modelo, feature_names, X_val, y_val, n_repeticiones=5):
    """
    Función optimizada de importancia por permutación para modelos no-KNN.
    """
    import pandas as pd
    import numpy as np
    
    print(f"Calculando importancia por permutación genérica ({n_repeticiones} repeticiones)...")
    
    # Usar tu función existente pero optimizada
    try:
        # Intentar usar balanced_accuracy_manual si está disponible
        df_importancia = calcular_importancia_modelo(
            modelo, X_val, y_val, feature_names, 
            metodo='permutacion', n_repeticiones=n_repeticiones
        )
        print("Importancia por permutación calculada exitosamente")
        return df_importancia
        
    except Exception as e:
        print(f"Error con función existente: {e}")
        # Fallback simple usando accuracy
        return _importancia_knn_optimizada(modelo, feature_names, X_val, y_val, n_repeticiones)



def importancia_features_global(modelo, feature_names):
    if isinstance(modelo, Pipeline):
        # Tomar el último estimador del pipeline
        modelo = modelo.named_steps[list(modelo.named_steps.keys())[-1]]

    if hasattr(modelo, 'pesos_'):
        importancias = np.mean(np.abs(modelo.pesos_), axis=0)
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
    else:
        raise TypeError(
            "El tipo de modelo no es compatible para el cálculo de importancia directa. "
            "Use importancia_permutacion_manual en su lugar."
        )

    df_importancia = pd.DataFrame({'feature': feature_names, 'importancia': importancias})
    df_importancia = df_importancia.sort_values(by='importancia', ascending=False)
    return df_importancia

# --- 2. Importancia por Permutación Manual ---
def calcular_importancia_modelo(modelo, X_val, y_val, feature_names, metodo='auto', n_repeticiones=10):
    """
    Calcula la importancia de características de un modelo.

    Args:
        modelo: Modelo entrenado o pipeline de sklearn.
        X_val: pd.DataFrame de validación.
        y_val: np.array de etiquetas de validación.
        feature_names: lista de nombres de features.
        metodo: 'auto', 'global', 'permutacion'
        n_repeticiones: número de repeticiones para permutación.

    Returns:
        pd.DataFrame con columnas ['feature', 'importancia']
    """
    # Si es pipeline, extraer el modelo final
    if isinstance(modelo, Pipeline):
        modelo_real = modelo.named_steps[list(modelo.named_steps.keys())[-1]]
    else:
        modelo_real = modelo

    # Intentar importancia global
    if metodo in ['auto', 'global']:
        if hasattr(modelo_real, 'pesos_'):
            importancias = np.mean(np.abs(modelo_real.pesos_), axis=0)
            df_importancia = pd.DataFrame({'feature': feature_names, 'importancia': importancias})
            return df_importancia.sort_values(by='importancia', ascending=False)
        elif hasattr(modelo_real, 'feature_importances_'):
            importancias = modelo_real.feature_importances_
            df_importancia = pd.DataFrame({'feature': feature_names, 'importancia': importancias})
            return df_importancia.sort_values(by='importancia', ascending=False)
        elif metodo == 'global':
            raise TypeError("El modelo no tiene importancia global disponible.")

    # Si no hay importancia global o se fuerza permutación
    print("Usando importancia por permutación...")
    importancias = {}
    pred_base = modelo.predict(X_val.values)
    score_base = balanced_accuracy_manual(pd.crosstab(y_val, pred_base))  # ejemplo simple

    for i, feat in enumerate(feature_names):
        scores_perm = []
        for _ in range(n_repeticiones):
            X_perm = X_val.copy()
            X_perm.iloc[:, i] = np.random.permutation(X_perm.iloc[:, i])
            pred_perm = modelo.predict(X_perm.values)
            score_perm = balanced_accuracy_manual(pd.crosstab(y_val, pred_perm))
            scores_perm.append(score_perm)
        importancias[feat] = score_base - np.mean(scores_perm)

    df_importancia = pd.DataFrame(list(importancias.items()), columns=['feature', 'importancia'])
    return df_importancia.sort_values(by='importancia', ascending=False)

def importancia_permutacion_manual(modelo, X_val, y_val, metrica, n_repeticiones=10):
    """
    Calcula la importancia de características mediante permutación.

    Args:
        modelo: Un modelo entrenado.
        X_val (np.array): Datos de validación para las características.
        y_val (np.array): Etiquetas de validación.
        metrica (function): Función de métrica para evaluar el rendimiento (ej. balanced_accuracy_manual).
        n_repeticiones (int): Número de veces que se permuta cada característica para promediar.

    Returns:
        pd.DataFrame: Un DataFrame con las características y la caída en el rendimiento.
    """
    print("Calculando importancia por permutación...")

    # 1. Calcular el score base del modelo
    predicciones_base = modelo.predict(X_val)
    matriz_conf_base, _ = matriz_confusion_manual(y_val, predicciones_base)
    score_base = metrica(matriz_conf_base)

    importancias = {}

    for i in range(X_val.shape[1]):
        scores_permutados = []
        for _ in range(n_repeticiones):
            X_permutado = X_val.copy()
            # Permutar la columna i
            np.random.shuffle(X_permutado[:, i])

            predicciones_permutadas = modelo.predict(X_permutado)
            matriz_conf_perm, _ = matriz_confusion_manual(y_val, predicciones_permutadas)
            score_permutado = metrica(matriz_conf_perm)
            scores_permutados.append(score_permutado)

        caida_score = score_base - np.mean(scores_permutados)
        importancias[f'feature_{i}'] = caida_score

    df_importancia = pd.DataFrame(list(importancias.items()), columns=['feature', 'caida_score_promedio'])
    df_importancia = df_importancia.sort_values(by='caida_score_promedio', ascending=False)

    return df_importancia

# --- 3. Gráfico de Dependencia Parcial (PDP) Manual ---

def graficar_pdp_manual(modelo, X, feature_idx, feature_name, ruta_guardado):
    """
    Calcula y grafica un Gráfico de Dependencia Parcial (PDP) para una característica.

    Args:
        modelo: Un modelo entrenado que tenga un método `predict_proba`.
        X (pd.DataFrame): El conjunto de datos (preferiblemente de entrenamiento o validación).
        feature_idx (int): El índice de la columna de la característica a analizar.
        feature_name (str): El nombre de la característica para el título del gráfico.
        ruta_guardado (str): Ruta para guardar el gráfico PNG.
    """
    print(f"Generando PDP para la característica: {feature_name}...")

    # Crear una rejilla de valores para la característica de interés
    valores_rejilla = np.linspace(X.iloc[:, feature_idx].min(), X.iloc[:, feature_idx].max(), num=50)

    predicciones_promedio = []

    for valor in valores_rejilla:
        X_modificado = X.copy()
        # Fijar la característica de interés al valor de la rejilla
        X_modificado.iloc[:, feature_idx] = valor

        # Obtener las probabilidades predichas
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X_modificado.values)
            # Promediar las probabilidades a través de todas las muestras para cada clase
            # Aquí promediamos la probabilidad de la clase positiva (clase 1)
            # Para multiclase, se necesitaría un gráfico por clase.
            prediccion_promedio = np.mean(probabilidades[:, 1]) # Asumiendo que la clase 1 es la de interés
        else:
            # Si no hay predict_proba, usar predicciones y promediar (menos informativo)
            predicciones = modelo.predict(X_modificado.values)
            prediccion_promedio = np.mean(predicciones)

        predicciones_promedio.append(prediccion_promedio)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(valores_rejilla, predicciones_promedio, marker='o', linestyle='-')
    plt.title(f'Gráfico de Dependencia Parcial (PDP) para {feature_name}')
    plt.xlabel(f'Valores de {feature_name}')
    plt.ylabel('Predicción Promedio (Probabilidad de Clase 1)')
    plt.grid(True)
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"PDP para '{feature_name}' guardado en: {ruta_guardado}")