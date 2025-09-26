import numpy as np
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """
    Clase base abstracta para todos los clasificadores implementados.
    Proporciona la interfaz común y validaciones básicas.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.n_classes = None
        self.classes = None
        
    @abstractmethod
    def fit(self, X, y):
        """Entrena el modelo con los datos proporcionados."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Realiza predicciones sobre nuevos datos."""
        pass
    
    def predict_proba(self, X):
        """Retorna probabilidades de clase. Implementación opcional."""
        raise NotImplementedError("predict_proba no implementado para este clasificador")
    
    def _validate_input(self, X, y=None):
        """Valida que los inputs tengan formato correcto."""
        X = np.array(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")
        
        if y is not None:
            y = np.array(y)
            if len(X) != len(y):
                raise ValueError("X e y deben tener el mismo número de muestras")
            
            return X, y
        
        return X
    
    def _setup_classes(self, y):
        """Configura información sobre las clases del problema."""
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        if self.n_classes < 2:
            raise ValueError("Necesita al menos 2 clases diferentes")
    
    def score(self, X, y):
        """Calcula accuracy del modelo."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        predictions = self.predict(X)
        return np.mean(predictions == y)


class ClassificationMetrics:
    """Utilidades para calcular métricas de clasificación."""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, classes=None):
        """Calcula matriz de confusión usando operaciones vectoriales."""
        if classes is None:
            classes = np.unique(np.concatenate([y_true, y_pred]))
        
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm, classes
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred, classes=None, average='macro'):
        """Calcula precision, recall y F1-score por clase y promediado."""
        cm, classes = ClassificationMetrics.confusion_matrix(y_true, y_pred, classes)
        n_classes = len(classes)
        
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0.0
        
        if average == 'macro':
            return np.mean(precision), np.mean(recall), np.mean(f1), precision, recall, f1
        elif average == 'weighted':
            weights = np.bincount(y_true) / len(y_true)
            return (np.average(precision, weights=weights), 
                   np.average(recall, weights=weights),
                   np.average(f1, weights=weights), 
                   precision, recall, f1)
        else:
            return precision, recall, f1
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calcula accuracy usando operaciones vectoriales."""
        return np.mean(y_true == y_pred)
