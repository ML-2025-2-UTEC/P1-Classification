"""
Utilidades para optimización de hiperparámetros y validación de modelos.
Implementación de Grid Search y Random Search desde cero.
"""

import numpy as np
import itertools
from typing import Dict, List, Any, Tuple
import time


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros usando Grid Search y Random Search."""
    
    def __init__(self, scoring='f1_macro', cv=5, random_state=None):
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = {}
        
    def grid_search(self, model_class, param_grid, X, y, n_jobs=1):
        """Búsqueda exhaustiva en grilla de hiperparámetros."""
        from evaluation.metrics import ModelEvaluator
        
        # Generar todas las combinaciones de parámetros
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        results = []
        best_score = -np.inf
        best_params = None
        
        print(f"Grid Search: {len(param_combinations)} combinaciones a evaluar")
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            # Crear modelo con parámetros actuales
            model = model_class(**params, random_state=self.random_state)
            
            try:
                # Validación cruzada
                cv_results = ModelEvaluator.cross_validate(
                    model, X, y, cv=self.cv, random_state=self.random_state
                )
                
                score = cv_results[f'{self.scoring}']['mean']
                std_score = cv_results[f'{self.scoring}']['std']
                
                result = {
                    'params': params,
                    'mean_score': score,
                    'std_score': std_score,
                    'cv_results': cv_results
                }
                results.append(result)
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"  [{i+1}/{len(param_combinations)}] Score: {score:.4f} ± {std_score:.4f} | {params}")
                
            except Exception as e:
                print(f"  [{i+1}/{len(param_combinations)}] ERROR: {str(e)} | {params}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        
        return self
    
    def random_search(self, model_class, param_distributions, X, y, n_iter=50):
        """Búsqueda aleatoria de hiperparámetros."""
        from evaluation.metrics import ModelEvaluator
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        results = []
        best_score = -np.inf
        best_params = None
        
        print(f"Random Search: {n_iter} iteraciones")
        
        for i in range(n_iter):
            # Generar parámetros aleatorios
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, tuple):
                    if len(distribution) == 2:  # (min, max) for uniform
                        params[param_name] = np.random.uniform(distribution[0], distribution[1])
                    elif len(distribution) == 3:  # (min, max, 'int') for integers
                        params[param_name] = np.random.randint(distribution[0], distribution[1])
            
            # Crear modelo con parámetros aleatorios
            model = model_class(**params, random_state=self.random_state)
            
            try:
                # Validación cruzada
                cv_results = ModelEvaluator.cross_validate(
                    model, X, y, cv=self.cv, random_state=self.random_state
                )
                
                score = cv_results[f'{self.scoring}']['mean']
                std_score = cv_results[f'{self.scoring}']['std']
                
                result = {
                    'params': params,
                    'mean_score': score,
                    'std_score': std_score,
                    'cv_results': cv_results
                }
                results.append(result)
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"  [{i+1}/{n_iter}] Score: {score:.4f} ± {std_score:.4f} | {params}")
                
            except Exception as e:
                print(f"  [{i+1}/{n_iter}] ERROR: {str(e)} | {params}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        
        return self


class EarlyStoppingCallback:
    """Callback para detener entrenamiento temprano cuando no hay mejora."""
    
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, epoch, current_score, model=None):
        """Evalúa si debe detener el entrenamiento."""
        if self.best_score is None:
            self.best_score = current_score
            if model and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            if model and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True  # Stop training
        
        return False  # Continue training
    
    def _get_model_weights(self, model):
        """Obtiene pesos del modelo para restaurar después."""
        if hasattr(model, 'weights') and hasattr(model, 'bias'):
            return {'weights': model.weights.copy(), 'bias': model.bias.copy()}
        return None
    
    def restore_weights(self, model):
        """Restaura los mejores pesos al modelo."""
        if self.best_weights and hasattr(model, 'weights'):
            model.weights = self.best_weights['weights'].copy()
            model.bias = self.best_weights['bias'].copy()


class LearningRateScheduler:
    """Programador de tasa de aprendizaje."""
    
    def __init__(self, schedule_type='step', **kwargs):
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.initial_lr = None
        
    def __call__(self, epoch, current_lr):
        """Calcula nueva tasa de aprendizaje."""
        if self.initial_lr is None:
            self.initial_lr = current_lr
            
        if self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 100)
            gamma = self.kwargs.get('gamma', 0.1)
            return self.initial_lr * (gamma ** (epoch // step_size))
            
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            return self.initial_lr * (gamma ** epoch)
            
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 1000)
            eta_min = self.kwargs.get('eta_min', 0.0)
            return eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
            
        return current_lr


class ModelValidator:
    """Validador de modelos con múltiples métricas."""
    
    @staticmethod
    def learning_curves(model_class, X, y, train_sizes=None, cv=5, **model_params):
        """Genera curvas de aprendizaje para evaluar overfitting."""
        from evaluation.metrics import ModelEvaluator
        from sklearn.model_selection import train_test_split
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        n_samples = len(X)
        results = {
            'train_sizes': [],
            'train_scores': [],
            'val_scores': []
        }
        
        for train_size in train_sizes:
            n_train = int(n_samples * train_size)
            if n_train < 10:  # Mínimo de muestras
                continue
                
            train_scores_fold = []
            val_scores_fold = []
            
            for fold in range(cv):
                # División estratificada
                X_subset, _, y_subset, _ = train_test_split(
                    X, y, train_size=train_size, stratify=y, random_state=fold
                )
                
                X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                    X_subset, y_subset, test_size=0.2, stratify=y_subset, random_state=fold
                )
                
                # Entrenar modelo
                model = model_class(**model_params, random_state=fold)
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluar
                train_pred = model.predict(X_train_fold)
                val_pred = model.predict(X_val_fold)
                
                train_report = ModelEvaluator.classification_report(y_train_fold, train_pred)
                val_report = ModelEvaluator.classification_report(y_val_fold, val_pred)
                
                train_scores_fold.append(train_report['macro_avg']['f1_score'])
                val_scores_fold.append(val_report['macro_avg']['f1_score'])
            
            results['train_sizes'].append(n_train)
            results['train_scores'].append(np.mean(train_scores_fold))
            results['val_scores'].append(np.mean(val_scores_fold))
        
        return results
    
    @staticmethod
    def bias_variance_analysis(model_class, X, y, n_bootstraps=50, test_size=0.2, **model_params):
        """Análisis de sesgo-varianza usando bootstrap."""
        from sklearn.model_selection import train_test_split
        from evaluation.metrics import ModelEvaluator
        
        predictions_list = []
        
        for i in range(n_bootstraps):
            # División bootstrap
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i, stratify=y
            )
            
            # Entrenar modelo
            model = model_class(**model_params, random_state=i)
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            predictions_list.append(y_pred)
        
        # Calcular métricas de sesgo-varianza
        predictions_array = np.array(predictions_list)
        
        # Varianza: variabilidad en las predicciones
        variance = np.var(predictions_array, axis=0).mean()
        
        # Sesgo: diferencia entre predicción promedio y valor real
        # (Simplificado para clasificación)
        mean_predictions = np.mean(predictions_array, axis=0)
        
        return {
            'variance': variance,
            'mean_predictions': mean_predictions,
            'n_bootstraps': n_bootstraps,
            'predictions': predictions_list
        }


def plot_learning_curves(learning_curves_results, title="Learning Curves"):
    """Visualiza curvas de aprendizaje."""
    import matplotlib.pyplot as plt
    
    train_sizes = learning_curves_results['train_sizes']
    train_scores = learning_curves_results['train_scores']
    val_scores = learning_curves_results['val_scores']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1-Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def optimize_threshold(y_true, y_proba, metric='f1_macro'):
    """Optimiza umbral de decisión para clasificación binaria."""
    from evaluation.metrics import ModelEvaluator
    
    if y_proba.shape[1] != 2:
        raise ValueError("Optimización de umbral solo para clasificación binaria")
    
    thresholds = np.linspace(0.1, 0.9, 50)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        report = ModelEvaluator.classification_report(y_true, y_pred)
        
        if metric == 'f1_macro':
            score = report['macro_avg']['f1_score']
        elif metric == 'accuracy':
            score = report['accuracy']
        elif metric == 'precision_macro':
            score = report['macro_avg']['precision']
        elif metric == 'recall_macro':
            score = report['macro_avg']['recall']
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, thresholds, scores
