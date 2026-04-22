# -*- coding: utf-8 -*-
"""
metrics.py

Shared classification metrics for subject-level disorder classification
(narcolepsy, OSA, depression). Computes accuracy, balanced accuracy,
Cohen's kappa, macro/weighted F1, per-class F1, precision, recall, and the
confusion matrix in one call so downstream trainers and evaluators can log
all quantities from a single dictionary.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the standard set of subject-level classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels as a 1-D array of class indices.
        y_pred (np.ndarray): Predicted labels as a 1-D array of class indices.

    Returns:
        dict: Dictionary containing accuracy, balanced accuracy, Cohen's kappa,
              macro F1, precision (macro), recall (macro), weighted F1, per-class
              F1, and the confusion matrix.
    """
    # Overall accuracy of predictions
    acc = accuracy_score(y_true, y_pred)
    # Balanced accuracy (mean per-class recall, robust to class imbalance)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    # Cohen's kappa score measuring agreement normalized by chance
    kappa = cohen_kappa_score(y_true, y_pred)
    # Macro-averaged F1 score treating all classes equally
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    # F1 score for each class individually
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    # Macro-averaged precision, ignoring divisions by zero
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    # Macro-averaged recall, ignoring divisions by zero
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    # Weighted F1 score accounting for class imbalance
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    # Confusion matrix as a 2-D array
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'kappa': float(kappa),
        'macro_f1': float(macro_f1),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(weighted_f1),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
    }
