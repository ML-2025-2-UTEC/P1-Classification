"""
Utilidades de carga y preprocesamiento de datos para el proyecto de clasificación de riesgo crediticio.
"""

import pandas as pd
import numpy as np
import sys
import os

# Agregar src al path para importar módulos personalizados
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import StandardScaler


def load_training_data(file_path):
    """Cargar dataset de entrenamiento desde archivo CSV."""
    return pd.read_csv(file_path)


def load_test_data(file_path):
    """Cargar dataset de prueba desde archivo CSV."""
    return pd.read_csv(file_path)


def get_feature_info():
    """Retornar información sobre las características del dataset organizadas por categoría."""
    features_info = {
        'financial': [
            'deuda_total', 'proporcion_ingreso_deuda', 'monto_solicitado', 
            'tasa_interes', 'lineas_credito_abiertas', 'saldo_promedio_bancario',
            'ingresos_familiares_per_capita', 'puntuacion_credito_bureau', 
            'antiguedad_laboral_meses', 'patrimonio_neto', 'gastos_mensuales_fijos',
            'capital_circulante', 'capacidad_ahorro_mensual', 'ingresos_inversion'
        ],
        'payment_history': [
            'retrasos_pago_ultimos_6_meses', 'mora_historica_dias', 
            'porcentaje_utilizacion_credito', 'pagos_puntuales_ultimos_12_meses',
            'deudas_canceladas_historicas', 'maximo_retraso_pago_dias',
            'numero_cuentas_cerradas', 'proporcion_pagos_a_tiempo',
            'consultas_credito_recientes', 'cambios_en_habitos_pago'
        ],
        'demographic': [
            'edad', 'nivel_educativo', 'estado_civil', 'numero_dependientes',
            'propiedad_vivienda', 'tipo_vivienda', 'residencia_antiguedad_meses',
            'sector_laboral', 'numero_empleos_ultimos_5_anos', 'frecuencia_transacciones_mensuales'
        ],
        'target': 'nivel_riesgo'
    }
    return features_info


def separate_features_target(df):
    """Separar características y variable objetivo."""
    X = df.drop('nivel_riesgo', axis=1)
    y = df['nivel_riesgo']
    return X, y


def encode_target_labels(y):
    """Codificar etiquetas objetivo a valores numéricos: Bajo=0, Medio=1, Alto=2."""
    mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
    return y.map(mapping)


def get_numerical_categorical_features(df):
    """Identificar características numéricas y categóricas."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remover target si está presente
    if 'nivel_riesgo' in numerical_features:
        numerical_features.remove('nivel_riesgo')
    if 'nivel_riesgo' in categorical_features:
        categorical_features.remove('nivel_riesgo')
    
    return numerical_features, categorical_features


class DataPreprocessor:
    """Manejar operaciones de preprocesamiento de datos."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X_train):
        """Ajustar preprocesador en datos de entrenamiento y transformarlos."""
        numerical_features, _ = get_numerical_categorical_features(X_train)
        
        X_processed = X_train.copy()
        X_processed[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        self.is_fitted = True
        return X_processed
    
    def transform(self, X):
        """Transformar datos usando preprocesador ajustado."""
        if not self.is_fitted:
            raise ValueError("Preprocesador debe estar ajustado antes de transformar datos")
        
        numerical_features, _ = get_numerical_categorical_features(X)
        
        X_processed = X.copy()
        X_processed[numerical_features] = self.scaler.transform(X[numerical_features])
        
        return X_processed


def check_missing_values(df):
    """Verificar valores faltantes en el dataset."""
    missing_info = df.isnull().sum()
    missing_percentage = (missing_info / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_info.index,
        'Missing_Count': missing_info.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def detect_outliers_iqr(df, column):
    """Detectar outliers usando método IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound