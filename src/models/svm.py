import numpy as np
from .base_classifier import BaseClassifier


class SVMMulticlass(BaseClassifier):
    """
    Implementación de SVM Multiclase usando estrategia One-vs-Rest.
    Cada clasificador binario resuelve el problema de optimización cuadrática
    usando gradiente descendente con hinge loss.
    """
    
    def __init__(self, C=1.0, kernel='linear', learning_rate=0.001, 
                 max_iterations=1000, tolerance=1e-6, random_state=None):
        super().__init__()
        self.C = C  # Parámetro de regularización
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        self.weights = None
        self.bias = None
        self.binary_classifiers = []
    
    def _linear_kernel(self, X1, X2):
        """Kernel lineal: K(x1, x2) = x1 · x2"""
        return X1 @ X2.T
    
    def _rbf_kernel(self, X1, X2, gamma=0.1):
        """Kernel RBF (Gaussian): K(x1, x2) = exp(-gamma * ||x1 - x2||²)"""
        dist_matrix = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                     np.sum(X2**2, axis=1).reshape(1, -1) - \
                     2 * X1 @ X2.T
        return np.exp(-gamma * dist_matrix)
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Calcula matriz de kernel entre X1 y X2."""
        if X2 is None:
            X2 = X1
        
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Kernel '{self.kernel}' no soportado")
    
    def fit(self, X, y):
        """Entrena el modelo usando estrategia One-vs-Rest."""
        X, y = self._validate_input(X, y)
        self._setup_classes(y)
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.binary_classifiers = []
        
        # Entrenar un clasificador binario para cada clase
        for class_idx, target_class in enumerate(self.classes):
            # Crear etiquetas binarias: +1 para clase actual, -1 para el resto
            y_binary = np.where(y == target_class, 1, -1)
            
            # Entrenar clasificador binario
            classifier = self._train_binary_svm(X, y_binary)
            self.binary_classifiers.append(classifier)
        
        self.is_fitted = True
        return self
    
    def _train_binary_svm(self, X, y):
        """Entrena un clasificador SVM binario."""
        n_samples, n_features = X.shape
        
        # Inicializar pesos y bias
        weights = np.random.normal(0, 0.01, n_features)
        bias = 0.0
        
        for iteration in range(self.max_iterations):
            # Calcular decisiones
            decisions = X @ weights + bias
            
            # Hinge loss gradient
            margins = y * decisions
            
            # Identificar ejemplos que violan el margen
            support_vectors = margins < 1
            
            if not np.any(support_vectors):
                break
            
            # Calcular gradientes
            dw = weights / self.C  # Término de regularización L2
            db = 0.0
            
            # Agregar gradientes del hinge loss
            for i in range(n_samples):
                if support_vectors[i]:
                    dw -= y[i] * X[i]
                    db -= y[i]
            
            # Actualizar parámetros
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        
        return {'weights': weights, 'bias': bias}
    
    def _predict_binary(self, X, classifier):
        """Realiza predicción con un clasificador binario."""
        weights = classifier['weights']
        bias = classifier['bias']
        return X @ weights + bias
    
    def predict(self, X):
        """Realiza predicciones multiclase usando One-vs-Rest."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Obtener puntuaciones de cada clasificador binario
        scores = np.zeros((n_samples, self.n_classes))
        
        for class_idx, classifier in enumerate(self.binary_classifiers):
            scores[:, class_idx] = self._predict_binary(X, classifier)
        
        # Predecir clase con mayor puntuación
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]
    
    def predict_proba(self, X):
        """Aproxima probabilidades usando función sigmoide en las puntuaciones."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Obtener puntuaciones
        scores = np.zeros((n_samples, self.n_classes))
        for class_idx, classifier in enumerate(self.binary_classifiers):
            scores[:, class_idx] = self._predict_binary(X, classifier)
        
        # Convertir a probabilidades usando softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities
    
    def get_feature_importance(self):
        """Calcula importancia promedio de features across clasificadores."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancia")
        
        importance = np.zeros(len(self.binary_classifiers[0]['weights']))
        
        for classifier in self.binary_classifiers:
            importance += np.abs(classifier['weights'])
        
        return importance / len(self.binary_classifiers)
    
    def get_support_vector_info(self):
        """Retorna información sobre vectores de soporte por clase."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        info = {}
        for i, class_name in enumerate(self.classes):
            classifier = self.binary_classifiers[i]
            info[f'Class_{class_name}'] = {
                'weights_norm': np.linalg.norm(classifier['weights']),
                'bias': classifier['bias'],
                'weights': classifier['weights']
            }
        
        return info
