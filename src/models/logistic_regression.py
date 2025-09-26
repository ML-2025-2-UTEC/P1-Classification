import numpy as np
from .base_classifier import BaseClassifier


class LogisticRegressionMulticlass(BaseClassifier):
    """
    Implementación de Regresión Logística Multinomial usando Softmax.
    Utiliza gradiente descendente con regularización L1/L2.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization='l2', 
                 lambda_reg=0.01, tolerance=1e-6, random_state=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.random_state = random_state
        
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _softmax(self, z):
        """Función softmax estable numéricamente."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """Codifica etiquetas en formato one-hot."""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def _compute_cost(self, X, y_one_hot):
        """Calcula el costo con regularización."""
        z = X @ self.weights + self.bias
        probabilities = self._softmax(z)
        
        # Cross-entropy loss
        epsilon = 1e-15  # Evitar log(0)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        cross_entropy = -np.mean(np.sum(y_one_hot * np.log(probabilities), axis=1))
        
        # Regularización
        reg_term = 0
        if self.regularization == 'l1':
            reg_term = self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_term = self.lambda_reg * np.sum(self.weights ** 2)
        elif self.regularization == 'elastic_net':
            reg_term = self.lambda_reg * (0.5 * np.sum(self.weights ** 2) + 
                                        0.5 * np.sum(np.abs(self.weights)))
        
        return cross_entropy + reg_term
    
    def _compute_gradients(self, X, y_one_hot, probabilities):
        """Calcula gradientes con regularización."""
        n_samples = X.shape[0]
        
        # Gradientes base
        dw = (X.T @ (probabilities - y_one_hot)) / n_samples
        db = np.mean(probabilities - y_one_hot, axis=0)
        
        # Regularización de pesos
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
        elif self.regularization == 'elastic_net':
            dw += self.lambda_reg * (self.weights + np.sign(self.weights))
        
        return dw, db
    
    def fit(self, X, y):
        """Entrena el modelo usando gradiente descendente."""
        X, y = self._validate_input(X, y)
        self._setup_classes(y)
        
        # Mapear clases a índices 0, 1, 2, ...
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        y_mapped = np.array([class_to_idx[cls] for cls in y])
        
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.weights = np.random.normal(0, 0.01, (n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        # Codificar etiquetas
        y_one_hot = self._one_hot_encode(y_mapped)
        
        # Gradiente descendente
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            probabilities = self._softmax(z)
            
            # Calcular costo
            cost = self._compute_cost(X, y_one_hot)
            self.cost_history.append(cost)
            
            # Criterio de convergencia
            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost
            
            # Backward pass
            dw, db = self._compute_gradients(X, y_one_hot, probabilities)
            
            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predice probabilidades para cada clase."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X = self._validate_input(X)
        z = X @ self.weights + self.bias
        return self._softmax(z)
    
    def predict(self, X):
        """Realiza predicciones de clase."""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]
    
    def get_feature_importance(self):
        """Retorna importancia de features basada en magnitud de pesos."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancia")
        
        # Promedio de magnitudes absolutas across clases
        return np.mean(np.abs(self.weights), axis=1)
