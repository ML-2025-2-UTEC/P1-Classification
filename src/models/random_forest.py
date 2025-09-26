import numpy as np
from .base_classifier import BaseClassifier


class DecisionTreeNode:
    """Nodo de un árbol de decisión."""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, 
                 value=None, samples=None, impurity=None):
        self.feature = feature      # Índice de feature para split
        self.threshold = threshold  # Valor umbral para split
        self.left = left           # Nodo izquierdo
        self.right = right         # Nodo derecho
        self.value = value         # Clase predicha (nodo hoja)
        self.samples = samples     # Número de muestras en nodo
        self.impurity = impurity   # Impureza del nodo


class DecisionTree:
    """Implementación de Árbol de Decisión para clasificación."""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features = None
        self.n_classes = None
        self.classes = None
    
    def _gini_impurity(self, y):
        """Calcula impureza de Gini."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """Calcula entropía."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]  # Evitar log(0)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, y, left_y, right_y):
        """Calcula ganancia de información usando Gini."""
        n = len(y)
        if n == 0:
            return 0
        
        parent_impurity = self._gini_impurity(y)
        
        n_left, n_right = len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0
        
        weighted_impurity = (n_left / n) * self._gini_impurity(left_y) + \
                           (n_right / n) * self._gini_impurity(right_y)
        
        return parent_impurity - weighted_impurity
    
    def _best_split(self, X, y, feature_indices):
        """Encuentra el mejor split para los features dados."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Construye el árbol recursivamente."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Condiciones de parada
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            
            most_common_class = np.bincount(y).argmax()
            return DecisionTreeNode(
                value=most_common_class,
                samples=n_samples,
                impurity=self._gini_impurity(y)
            )
        
        # Seleccionar features aleatorios
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            n_features_to_use = min(self.max_features, n_features)
            feature_indices = np.random.choice(
                n_features, n_features_to_use, replace=False
            )
        
        # Encontrar mejor split
        best_feature, best_threshold, best_gain = self._best_split(
            X, y, feature_indices
        )
        
        if best_feature is None:
            most_common_class = np.bincount(y).argmax()
            return DecisionTreeNode(
                value=most_common_class,
                samples=n_samples,
                impurity=self._gini_impurity(y)
            )
        
        # Dividir datos
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Construir subárboles
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            samples=n_samples,
            impurity=self._gini_impurity(y)
        )
    
    def fit(self, X, y):
        """Entrena el árbol de decisión."""
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Mapear clases a índices
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        y_mapped = np.array([class_to_idx[cls] for cls in y])
        
        self.root = self._build_tree(X, y_mapped)
        return self
    
    def _predict_single(self, x, node):
        """Predice para una muestra individual."""
        if node.value is not None:  # Nodo hoja
            return self.classes[node.value]
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """Realiza predicciones."""
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def _get_feature_importance_recursive(self, node, importances, total_samples):
        """Calcula importancia de features recursivamente."""
        if node.value is not None:  # Nodo hoja
            return
        
        if node.feature is not None:
            # Calcular importancia basada en ganancia de información
            left_samples = node.left.samples if node.left else 0
            right_samples = node.right.samples if node.right else 0
            
            importance = (node.samples / total_samples) * node.impurity
            if node.left:
                importance -= (left_samples / total_samples) * node.left.impurity
            if node.right:
                importance -= (right_samples / total_samples) * node.right.impurity
            
            importances[node.feature] += importance
        
        # Recursión en subárboles
        if node.left:
            self._get_feature_importance_recursive(node.left, importances, total_samples)
        if node.right:
            self._get_feature_importance_recursive(node.right, importances, total_samples)
    
    def get_feature_importance(self):
        """Calcula importancia de features."""
        importances = np.zeros(self.n_features)
        if self.root:
            self._get_feature_importance_recursive(
                self.root, importances, self.root.samples
            )
        
        # Normalizar
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance
        
        return importances


class RandomForestMulticlass(BaseClassifier):
    """
    Implementación de Random Forest para clasificación multiclase.
    Utiliza bootstrap aggregating (bagging) y votación por mayoría.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None, n_jobs=1):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.trees = []
        self.feature_importances = None
    
    def _get_max_features(self, n_features):
        """Calcula número de features a usar por árbol."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y, random_state):
        """Crea muestra bootstrap."""
        n_samples = X.shape[0]
        np.random.seed(random_state)
        
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            indices = np.arange(n_samples)
        
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Entrena el Random Forest."""
        X, y = self._validate_input(X, y)
        self._setup_classes(y)
        
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Entrenar cada árbol
        for i in range(self.n_estimators):
            tree_random_state = None if self.random_state is None else self.random_state + i
            
            # Crear muestra bootstrap
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y, tree_random_state)
            
            # Entrenar árbol
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=tree_random_state
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        # Calcular importancia de features
        self._calculate_feature_importances()
        
        self.is_fitted = True
        return self
    
    def _calculate_feature_importances(self):
        """Calcula importancia promedio de features across todos los árboles."""
        if not self.trees:
            return
        
        n_features = self.trees[0].n_features
        total_importance = np.zeros(n_features)
        
        for tree in self.trees:
            total_importance += tree.get_feature_importance()
        
        self.feature_importances = total_importance / len(self.trees)
    
    def predict(self, X):
        """Realiza predicciones usando votación por mayoría."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Obtener predicciones de todos los árboles
        predictions = np.zeros((n_samples, len(self.trees)), dtype=object)
        
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Votación por mayoría
        final_predictions = []
        for i in range(n_samples):
            unique_preds, counts = np.unique(predictions[i], return_counts=True)
            most_common = unique_preds[np.argmax(counts)]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Aproxima probabilidades basadas en proporción de votos."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Inicializar matriz de probabilidades
        probabilities = np.zeros((n_samples, self.n_classes))
        
        # Obtener predicciones de todos los árboles
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            
            for i, pred in enumerate(tree_predictions):
                class_idx = np.where(self.classes == pred)[0][0]
                probabilities[i, class_idx] += 1
        
        # Normalizar por número de árboles
        probabilities = probabilities / len(self.trees)
        
        return probabilities
    
    def get_feature_importance(self):
        """Retorna importancia de features."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancia")
        
        return self.feature_importances
