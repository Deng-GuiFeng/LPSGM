# -*- coding: utf-8 -*-
"""
metrics.py

This module provides evaluation metrics for disorder classification tasks within the LPSGM project.
It computes a comprehensive set of performance indicators including accuracy, Cohen's kappa, 
macro-averaged and weighted F1 scores, precision, recall, per-class F1 scores, and the confusion matrix.
These metrics facilitate robust assessment of model predictions against ground truth labels in sleep staging 
and mental disorder diagnosis applications.
"""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate multiple classification metrics to evaluate prediction performance.

    Args:
        y_true (np.ndarray): Ground truth labels as a 1D array.
        y_pred (np.ndarray): Predicted labels as a 1D array.

    Returns:
        dict: Dictionary containing accuracy, Cohen's kappa, macro F1 score, precision, recall,
              weighted F1 score, per-class F1 scores, and confusion matrix.
    """
    # Overall accuracy of predictions
    acc = accuracy_score(y_true, y_pred)
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
    # Confusion matrix as a 2D array
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(acc),
        'kappa': float(kappa),
        'macro_f1': float(macro_f1),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(weighted_f1),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
    }
