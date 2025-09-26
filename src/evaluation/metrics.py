"""
Métricas de evaluación optimizadas para clasificación multiclase.
Implementación vectorizada usando operaciones de NumPy para máxima eficiencia.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import os


class ModelEvaluator:
    """Evaluador completo de modelos de clasificación multiclase."""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, classes=None):
        """Calcula matriz de confusión optimizada."""
        if classes is None:
            classes = np.unique(np.concatenate([y_true, y_pred]))
        
        n_classes = len(classes)
        # Mapear clases a índices
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        y_true_idx = np.array([class_to_idx[cls] for cls in y_true])
        y_pred_idx = np.array([class_to_idx[cls] for cls in y_pred])
        
        # Usar broadcasting para calcular CM de forma vectorizada
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = np.sum((y_true_idx == i) & (y_pred_idx == j))
        
        return cm, classes
    
    @staticmethod
    def classification_report(y_true, y_pred, classes=None):
        """Genera reporte completo de clasificación."""
        cm, classes = ModelEvaluator.confusion_matrix(y_true, y_pred, classes)
        n_classes = len(classes)
        
        # Calcular métricas por clase
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        support = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            support[i] = tp + fn
            
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1_score[i] = 0.0
        
        # Métricas globales
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        
        # Macro average
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall) 
        macro_f1 = np.mean(f1_score)
        
        # Weighted average
        weights = support / np.sum(support)
        weighted_precision = np.average(precision, weights=weights)
        weighted_recall = np.average(recall, weights=weights)
        weighted_f1 = np.average(f1_score, weights=weights)
        
        return {
            'confusion_matrix': cm,
            'classes': classes,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support.astype(int)
            },
            'accuracy': accuracy,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
        }
    
    @staticmethod
    def cross_validate(model, X, y, cv=5, random_state=None):
        """Realiza validación cruzada estratificada."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        scores = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Crear copia del modelo
            model_params = {}
            for attr in dir(model):
                if not attr.startswith('_') and not callable(getattr(model, attr)):
                    model_params[attr] = getattr(model, attr)
            
            model_copy = type(model)(**{k: v for k, v in model_params.items() 
                                      if k not in ['is_fitted', 'weights', 'bias', 'trees', 
                                                  'classes', 'n_classes', 'feature_importances',
                                                  'cost_history', 'binary_classifiers', 'root',
                                                  'n_features', 'components', 'explained_variance',
                                                  'explained_variance_ratio', 'mean', 
                                                  'n_components_selected']})
            
            # Entrenar modelo
            model_copy.fit(X_train, y_train)
            
            # Predecir
            y_pred = model_copy.predict(X_val)
            
            # Calcular métricas
            report = ModelEvaluator.classification_report(y_val, y_pred)
            
            scores['accuracy'].append(report['accuracy'])
            scores['precision_macro'].append(report['macro_avg']['precision'])
            scores['recall_macro'].append(report['macro_avg']['recall'])
            scores['f1_macro'].append(report['macro_avg']['f1_score'])
        
        # Calcular estadísticas
        cv_results = {}
        for metric, values in scores.items():
            cv_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'scores': values
            }
        
        return cv_results


class FeatureSelector:
    """Selector de características usando múltiples técnicas."""
    
    @staticmethod
    def correlation_filter(X, threshold=0.95):
        """Remueve features altamente correlacionadas."""
        correlation_matrix = np.corrcoef(X.T)
        correlation_matrix = np.abs(correlation_matrix)
        
        # Encontrar pares altamente correlacionados
        high_corr_pairs = np.where((correlation_matrix > threshold) & 
                                  (correlation_matrix < 1.0))
        
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i < j:  # Evitar duplicados
                features_to_remove.add(j)
        
        features_to_keep = [i for i in range(X.shape[1]) if i not in features_to_remove]
        
        return features_to_keep, list(features_to_remove)
    
    @staticmethod
    def variance_filter(X, threshold=0.01):
        """Remueve features con baja varianza."""
        variances = np.var(X, axis=0)
        features_to_keep = np.where(variances > threshold)[0]
        features_to_remove = np.where(variances <= threshold)[0]
        
        return features_to_keep.tolist(), features_to_remove.tolist()
    
    @staticmethod
    def univariate_selection(X, y, k_best=20):
        """Selecciona k mejores features usando ANOVA F-test."""
        classes = np.unique(y)
        f_scores = []
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            groups = [feature_data[y == cls] for cls in classes]
            
            # Calcular F-statistic manualmente
            k = len(groups)  # número de grupos
            n = len(feature_data)  # número total de muestras
            
            # Medias de grupos y global
            group_means = [np.mean(group) for group in groups]
            overall_mean = np.mean(feature_data)
            
            # Suma de cuadrados entre grupos (SSB)
            ssb = sum(len(group) * (mean - overall_mean)**2 
                     for group, mean in zip(groups, group_means))
            
            # Suma de cuadrados dentro de grupos (SSW)
            ssw = sum(sum((x - group_mean)**2 for x in group)
                     for group, group_mean in zip(groups, group_means))
            
            # Grados de libertad
            df_between = k - 1
            df_within = n - k
            
            # F-statistic
            if df_within > 0 and ssw > 0:
                f_stat = (ssb / df_between) / (ssw / df_within)
                f_scores.append(f_stat if not np.isnan(f_stat) else 0)
            else:
                f_scores.append(0)
        
        f_scores = np.array(f_scores)
        selected_features = np.argsort(f_scores)[-k_best:]
        
        return selected_features.tolist(), f_scores


class DimensionalityReducer:
    """Implementación de PCA desde cero."""
    
    def __init__(self, n_components=None, explained_variance_threshold=0.95):
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None
        self.n_components_selected = None
    
    def fit(self, X):
        """Ajusta PCA a los datos."""
        # Centrar datos
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calcular matriz de covarianza
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Ordenar por eigenvalues descendente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determinar número de componentes
        if self.n_components is None:
            cumsum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_components_selected = np.argmax(
                cumsum_variance >= self.explained_variance_threshold
            ) + 1
        else:
            self.n_components_selected = min(self.n_components, len(eigenvalues))
        
        # Guardar componentes principales
        self.components = eigenvectors[:, :self.n_components_selected].T
        self.explained_variance = eigenvalues[:self.n_components_selected]
        self.explained_variance_ratio = (
            self.explained_variance / np.sum(eigenvalues)
        )
        
        return self
    
    def transform(self, X):
        """Transforma datos al espacio reducido."""
        if self.components is None:
            raise ValueError("PCA debe ser ajustado antes de transformar")
        
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def fit_transform(self, X):
        """Ajusta y transforma en un solo paso."""
        return self.fit(X).transform(X)


class ModelPersistence:
    """Utilidades para guardar y cargar modelos entrenados."""
    
    @staticmethod
    def save_model(model, filepath, metadata=None):
        """Guarda modelo y metadatos usando pickle."""
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'model_type': type(model).__name__
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(filepath):
        """Carga modelo desde archivo pickle."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data.get('metadata', {})
    
    @staticmethod
    def save_results(results, filepath):
        """Guarda resultados de experimentos."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    @staticmethod
    def load_results(filepath):
        """Carga resultados de experimentos."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Funciones auxiliares para compatibilidad
def accuracy_score(y_true, y_pred):
    """Calcula accuracy de forma vectorizada."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro', classes=None):
    """Calcula precision con diferentes tipos de promedio."""
    report = ModelEvaluator.classification_report(y_true, y_pred, classes)
    
    if average == 'macro':
        return report['macro_avg']['precision']
    elif average == 'weighted':
        return report['weighted_avg']['precision']
    else:  # per class
        return report['per_class']['precision']


def recall_score(y_true, y_pred, average='macro', classes=None):
    """Calcula recall con diferentes tipos de promedio."""
    report = ModelEvaluator.classification_report(y_true, y_pred, classes)
    
    if average == 'macro':
        return report['macro_avg']['recall']
    elif average == 'weighted':
        return report['weighted_avg']['recall']
    else:  # per class
        return report['per_class']['recall']


def f1_score(y_true, y_pred, average='macro', classes=None):
    """Calcula F1-score con diferentes tipos de promedio."""
    report = ModelEvaluator.classification_report(y_true, y_pred, classes)
    
    if average == 'macro':
        return report['macro_avg']['f1_score']
    elif average == 'weighted':
        return report['weighted_avg']['f1_score']
    else:  # per class
        return report['per_class']['f1_score']


def confusion_matrix(y_true, y_pred, classes=None):
    """Wrapper para la función de matriz de confusión."""
    cm, classes_used = ModelEvaluator.confusion_matrix(y_true, y_pred, classes)
    return cm
