"""
Métricas de evaluación y utilidades de validación.
Todas las métricas implementadas desde cero sin usar librerías externas de ML.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes=None):
    """Calcular matriz de confusión."""
    if n_classes is None:
        n_classes = max(max(y_true), max(y_pred)) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    
    return cm


def accuracy_score(y_true, y_pred):
    """Calcular precisión (accuracy)."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro', n_classes=None):
    """Calcular puntuación de precisión."""
    if n_classes is None:
        n_classes = max(max(y_true), max(y_pred)) + 1
    
    cm = confusion_matrix(y_true, y_pred, n_classes)
    
    if average == 'macro':
        precisions = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
    
    elif average == 'micro':
        tp_total = np.sum([cm[i, i] for i in range(n_classes)])
        fp_total = np.sum(cm) - tp_total
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    
    else:  # per class
        precisions = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.array(precisions)


def recall_score(y_true, y_pred, average='macro', n_classes=None):
    """Compute recall score."""
    if n_classes is None:
        n_classes = max(max(y_true), max(y_pred)) + 1
    
    cm = confusion_matrix(y_true, y_pred, n_classes)
    
    if average == 'macro':
        recalls = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls)
    
    elif average == 'micro':
        tp_total = np.sum([cm[i, i] for i in range(n_classes)])
        fn_total = np.sum(cm) - tp_total
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    else:  # per class
        recalls = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        return np.array(recalls)


def f1_score(y_true, y_pred, average='macro', n_classes=None):
    """Calcular puntuación F1."""
    precision = precision_score(y_true, y_pred, average='none', n_classes=n_classes)
    recall = recall_score(y_true, y_pred, average='none', n_classes=n_classes)
    
    f1_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (p * r) / (p + r))
    
    f1_scores = np.array(f1_scores)
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'micro':
        # For micro-averaging, F1 = precision = recall = accuracy
        return accuracy_score(y_true, y_pred)
    else:
        return f1_scores


def classification_report(y_true, y_pred, class_names=None, n_classes=None):
    """Generate a classification report."""
    if n_classes is None:
        n_classes = max(max(y_true), max(y_pred)) + 1
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    precision_per_class = precision_score(y_true, y_pred, average='none', n_classes=n_classes)
    recall_per_class = recall_score(y_true, y_pred, average='none', n_classes=n_classes)
    f1_per_class = f1_score(y_true, y_pred, average='none', n_classes=n_classes)
    
    # Support (number of samples per class)
    support = []
    for i in range(n_classes):
        support.append(np.sum(y_true == i))
    
    # Calculate averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    micro_precision = precision_score(y_true, y_pred, average='micro', n_classes=n_classes)
    micro_recall = recall_score(y_true, y_pred, average='micro', n_classes=n_classes)
    micro_f1 = f1_score(y_true, y_pred, average='micro', n_classes=n_classes)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    report = {
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1_score': f1_per_class,
            'support': support,
            'class_names': class_names
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1
        },
        'accuracy': accuracy,
        'total_samples': len(y_true)
    }
    
    return report


class KFoldCrossValidator:
    """Implementación de Validación Cruzada K-Fold."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """Generar índices para divisiones de entrenamiento/prueba."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, test_indices
    
    def cross_validate(self, model, X, y, scoring='accuracy'):
        """Perform cross-validation and return scores."""
        scores = []
        
        for train_indices, test_indices in self.split(X, y):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # Create a copy of the model for each fold
            model_copy = type(model)(**model.__dict__)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)
            
            if scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif scoring == 'precision':
                score = precision_score(y_test, y_pred, average='macro')
            elif scoring == 'recall':
                score = recall_score(y_test, y_pred, average='macro')
            elif scoring == 'f1':
                score = f1_score(y_test, y_pred, average='macro')
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")
            
            scores.append(score)
        
        return np.array(scores)


def cost_sensitive_evaluation(y_true, y_pred, cost_matrix=None):
    """
    Evaluate model with cost-sensitive metrics.
    
    For loan default prediction:
    - Classifying High risk as Low risk (false negative) is more costly
    - Classifying Low risk as High risk (false positive) is less costly
    """
    if cost_matrix is None:
        # Default cost matrix for loan risk (Low=0, Medium=1, High=2)
        # Rows: true class, Columns: predicted class
        cost_matrix = np.array([
            [0, 1, 2],    # True Low:  correctly classified=0, predicted Medium=1, predicted High=2
            [2, 0, 1],    # True Medium: predicted Low=2, correctly classified=0, predicted High=1
            [10, 5, 0]    # True High: predicted Low=10 (very costly), predicted Medium=5, correctly classified=0
        ])
    
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    total_cost = np.sum(cm * cost_matrix)
    
    # Normalize by number of samples
    average_cost = total_cost / len(y_true)
    
    return {
        'total_cost': total_cost,
        'average_cost': average_cost,
        'confusion_matrix': cm,
        'cost_matrix': cost_matrix
    }


def plot_confusion_matrix_data(y_true, y_pred, class_names=None, normalize=False):
    """Prepare data for confusion matrix plotting."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    return {
        'confusion_matrix': cm,
        'class_names': class_names,
        'normalized': normalize
    }