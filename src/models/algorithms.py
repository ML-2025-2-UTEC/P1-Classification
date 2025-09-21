"""
Implementaciones de algoritmos de Machine Learning desde cero.
Todos los algoritmos implementados sin usar librerías externas de ML.
"""

import numpy as np


class LogisticRegressionMulticlass:
    """Implementación de Regresión Logística Multiclase usando softmax y gradiente descendente."""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization  # 'l1', 'l2', or None
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.classes = None
        self.n_classes = None
    
    def _softmax(self, z):
        """Calcular función de activación softmax."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """Convertir etiquetas a codificación one-hot."""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def fit(self, X, y):
        """Entrenar el modelo de regresión logística multiclase."""
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Inicializar pesos y bias
        self.weights = np.random.normal(0, 0.01, (n_features, self.n_classes))
        self.bias = np.zeros((1, self.n_classes))
        
        # Convertir etiquetas a codificación one-hot
        y_one_hot = self._one_hot_encode(y)
        
        # Gradiente descendente
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(z)
            
            # Calcular costo
            cost = self._compute_cost(y_one_hot, y_pred)
            
            # Backward pass
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1/n_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)
            
            # Agregar regularización
            if self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += self.lambda_reg * self.weights
            
            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _compute_cost(self, y_true, y_pred):
        """Calcular costo de entropía cruzada."""
        n_samples = y_true.shape[0]
        cost = -np.sum(y_true * np.log(y_pred + 1e-15)) / n_samples
        
        # Agregar costo de regularización
        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2)
        
        return cost
    
    def predict_proba(self, X):
        """Predecir probabilidades de clase."""
        z = np.dot(X, self.weights) + self.bias
        return self._softmax(z)
    
    def predict(self, X):
        """Realizar predicciones."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class SVMMulticlass:
    """Support Vector Machine con estrategia One-vs-Rest para clasificación multiclase."""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.C = C  # Regularization parameter
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X, y):
        """Train SVM classifiers using One-vs-Rest strategy."""
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        for class_label in self.classes:
            # Create binary labels (1 for current class, -1 for others)
            y_binary = np.where(y == class_label, 1, -1)
            
            # Train binary SVM classifier
            w = np.zeros(n_features)
            b = 0
            
            for _ in range(self.max_iterations):
                for i, x_i in enumerate(X):
                    condition = y_binary[i] * (np.dot(x_i, w) - b) >= 1
                    
                    if condition:
                        w -= self.learning_rate * (2 * (1/self.max_iterations) * w)
                    else:
                        w -= self.learning_rate * (2 * (1/self.max_iterations) * w - np.dot(self.C, y_binary[i] * x_i))
                        b -= self.learning_rate * self.C * y_binary[i]
            
            self.classifiers[class_label] = {'weights': w, 'bias': b}
    
    def predict(self, X):
        """Make predictions using trained classifiers."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        for i, class_label in enumerate(self.classes):
            w = self.classifiers[class_label]['weights']
            b = self.classifiers[class_label]['bias']
            scores[:, i] = np.dot(X, w) - b
        
        return np.argmax(scores, axis=1)


class DecisionTree:
    """Decision Tree implementation for classification."""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        gini_parent = self._gini_impurity(y)
        gini_left = self._gini_impurity(y_left)
        gini_right = self._gini_impurity(y_right)
        
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return gini_parent - weighted_gini
    
    def _best_split(self, X, y):
        """Find the best split for the current node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_class(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            leaf_value = self._most_common_class(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _most_common_class(self, y):
        """Return the most common class in y."""
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self._build_tree(X, y)
    
    def _predict_single(self, x, tree):
        """Predict a single sample."""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_single(x, self.tree) for x in X])


class RandomForest:
    """Random Forest implementation using multiple decision trees."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
    
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_random_features(self, n_features):
        """Select random features for each tree."""
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
        
        return np.random.choice(n_features, max_features, replace=False)
    
    def fit(self, X, y):
        """Train the random forest."""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Select random features
            feature_indices = self._get_random_features(n_features)
            
            # Train decision tree on bootstrap sample with selected features
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_bootstrap[:, feature_indices], y_bootstrap)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
    
    def predict(self, X):
        """Make predictions using majority voting."""
        tree_predictions = []
        
        for i, tree in enumerate(self.trees):
            feature_indices = self.feature_indices[i]
            predictions = tree.predict(X[:, feature_indices])
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions).T
        
        # Majority voting
        final_predictions = []
        for row in tree_predictions:
            classes, counts = np.unique(row, return_counts=True)
            final_predictions.append(classes[np.argmax(counts)])
        
        return np.array(final_predictions)