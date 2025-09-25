"""
Funciones de utilidad para visualización de datos, selección de características y funciones auxiliares generales.
"""

import numpy as np
import pandas as pd


def feature_importance_trees(model, feature_names=None):
    """Calcular importancia de características desde modelos basados en árboles."""
    if hasattr(model, 'trees'):  # Random Forest
        importances = np.zeros(len(feature_names) if feature_names else model.trees[0].tree['feature'])
        
        # This is a simplified version - in practice you'd traverse the trees
        # and calculate importance based on impurity reduction
        return importances
    
    else:
        raise ValueError("Model doesn't support feature importance calculation")


def correlation_matrix(df, target_column=None):
    """Calcular matriz de correlación para características numéricas."""
    numerical_df = df.select_dtypes(include=[np.number])
    
    if target_column and target_column in numerical_df.columns:
        # Calcular correlaciones con el target
        target_correlations = numerical_df.corr()[target_column].sort_values(ascending=False)
        return target_correlations
    else:
        return numerical_df.corr()


class FeatureSelector:
    """Utilidades de selección de características."""
    
    def __init__(self):
        self.selected_features = None
    
    def select_k_best_correlation(self, X, y, k=10):
        """Seleccionar k características con mayor correlación al target."""
        if isinstance(X, pd.DataFrame):
            correlations = abs(X.corrwith(pd.Series(y)))
            top_features = correlations.nlargest(k).index.tolist()
            self.selected_features = top_features
            return X[top_features]
        else:
            raise ValueError("X debe ser un pandas DataFrame para selección basada en correlación")
    
    def select_variance_threshold(self, X, threshold=0.0):
        """Remover características con baja varianza."""
        if isinstance(X, pd.DataFrame):
            variances = X.var()
            selected_features = variances[variances > threshold].index.tolist()
            self.selected_features = selected_features
            return X[selected_features]
        else:
            variances = np.var(X, axis=0)
            selected_indices = np.where(variances > threshold)[0]
            self.selected_features = selected_indices
            return X[:, selected_indices]


class PCA:
    """Implementación de Análisis de Componentes Principales."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
        self.mean = None
    
    def fit(self, X):
        """Ajustar PCA en los datos."""
        # Centrar los datos
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calcular matriz de covarianza
        cov_matrix = np.cov(X_centered.T)
        
        # Calcular eigenvalues y eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Ordenar por eigenvalues (descendente)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Seleccionar número de componentes
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        
        # Calcular ratio de varianza explicada
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
    
    def transform(self, X):
        """Transform data to principal components."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """Fit PCA and transform data."""
        self.fit(X)
        return self.transform(X)
    
    def get_cumulative_variance_ratio(self):
        """Get cumulative explained variance ratio."""
        return np.cumsum(self.explained_variance_ratio)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and testing sets."""
    if random_state:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class StandardScaler:
    """Standard Scaler implementation (alternative to sklearn)."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, X):
        """Fit scaler on training data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        # Avoid division by zero
        std_safe = np.where(self.std == 0, 1, self.std)
        return (X - self.mean) / std_safe
    
    def fit_transform(self, X):
        """Fit scaler and transform data."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform (scale back to original)."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        std_safe = np.where(self.std == 0, 1, self.std)
        return X * std_safe + self.mean


def get_categorical_encodings(df, categorical_columns):
    """Get encoding mappings for categorical variables."""
    encodings = {}
    
    for col in categorical_columns:
        if col in df.columns:
            unique_values = df[col].unique()
            # Remove NaN values if present
            unique_values = unique_values[pd.notna(unique_values)]
            encoding = {value: idx for idx, value in enumerate(unique_values)}
            encodings[col] = encoding
    
    return encodings


def apply_categorical_encodings(df, encodings):
    """Apply categorical encodings to dataframe."""
    df_encoded = df.copy()
    
    for col, encoding in encodings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(encoding)
    
    return df_encoded