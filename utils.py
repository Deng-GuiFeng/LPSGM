# -*- coding: utf-8 -*-
"""
utils.py

This utility module provides helper functions and classes used throughout the LPSGM project.
It includes tools for metrics calculation, model parameter summarization, random seed setting,
string parsing, and file searching. These utilities facilitate consistent evaluation of model
performance, reproducibility, and general-purpose operations needed across different components
of the Large Polysomnography Model (LPSGM) for sleep staging and mental disorder diagnosis.
"""

import numpy as np
import torch
import os
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score


def info(*tensors):
    """
    Prints shape, data type, minimum and maximum values for each tensor provided.

    Args:
        *tensors (torch.Tensor): One or more tensors to inspect.
    """
    for t in tensors:
        print(t.shape, t.dtype, t.min(),  t.max())


def str_to_set(s: str) -> set:
    """
    Converts a comma-separated string into a set of stripped substrings.

    Args:
        s (str): Input string containing comma-separated items.

    Returns:
        set: A set of unique, trimmed string items. Returns empty set if input is empty or None.
    """
    if not s:
        return set()
    return {item.strip() for item in s.split(',') if item.strip()}


class AverageMeter(object):
    """
    Computes and stores the average, sum, and current value of a metric.
    Useful for tracking metrics during training or evaluation.

    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all statistics to initial state.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with new value(s).

        Args:
            val (float): New value to incorporate.
            n (int, optional): Number of occurrences of val. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, force=False):
    """
    Sets random seed for reproducibility across random, numpy, and torch libraries.

    Args:
        seed (int): The seed value to set.
        force (bool, optional): If True, forces deterministic behavior in CUDA backend.
                                Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if force:
        # Enforce deterministic algorithms for reproducibility at the cost of performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_metric_legacy(y_true, y_pred):
    """
    Computes standard classification metrics for 5-class sleep staging.

    Args:
        y_true (list or array-like): Ground truth labels (0-4).
        y_pred (list or array-like): Predicted labels (0-4).

    Returns:
        tuple: Contains overall accuracy, macro F1 score, confusion matrix,
               F1 scores for each sleep stage (wake, N1, N2, N3, REM),
               and Cohen's Kappa score.
    """
    if type(y_true) == list:
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    # Calculate F1 score for each individual class by treating it as binary classification
    wake_f1 = f1_score(y_true==0, y_pred==0)
    n1_f1 = f1_score(y_true==1, y_pred==1)
    n2_f1 = f1_score(y_true==2, y_pred==2)
    n3_f1 = f1_score(y_true==3, y_pred==3)
    rem_f1 = f1_score(y_true==4, y_pred==4)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa


def get_metric(y_true, y_pred, num_classes=5):
    """
    Calculates classification metrics supporting 5, 4, or 3 class evaluations for sleep staging.

    Args:
        y_true (array-like): Ground truth labels with values in range 0–4.
        y_pred (array-like): Predicted labels with values in range 0–4.
        num_classes (int): Number of classes to evaluate (5, 4, or 3).

    Returns:
        tuple: Contains overall accuracy (float), macro F1 score (float),
               confusion matrix (ndarray with shape (num_classes, num_classes)),
               per-class F1 scores (tuple), and Cohen's Kappa score (float).

               Per-class F1 order depends on num_classes:
               - 5 classes: (wake, N1, N2, N3, REM)
               - 4 classes: (wake, N1+N2, N3, REM)
               - 3 classes: (wake, NREM, REM)
    """
    # Convert inputs to numpy arrays of int64 type
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    # Define label mapping based on desired number of classes
    if num_classes == 5:
        # Original five classes: 0=Wake,1=N1,2=N2,3=N3,4=REM
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        labels = [0, 1, 2, 3, 4]
    elif num_classes == 4:
        # Merge N1(1) and N2(2) into new class 1; W=0, N1+N2=1, N3=2, REM=3
        mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
        labels = [0, 1, 2, 3]
    elif num_classes == 3:
        # Merge N1/N2/N3 into new class 1; W=0, NREM=1, REM=2
        mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
        labels = [0, 1, 2]
    else:
        raise ValueError("num_classes must be one of 5, 4, or 3")

    # Apply mapping to true and predicted labels
    y_true_m = np.vectorize(mapping.get)(y_true)
    y_pred_m = np.vectorize(mapping.get)(y_pred)

    # Calculate overall accuracy and macro-averaged F1 score
    acc = accuracy_score(y_true_m, y_pred_m)
    f1_macro = f1_score(y_true_m, y_pred_m, average='macro')
    # Compute confusion matrix for mapped labels
    cm = confusion_matrix(y_true_m, y_pred_m, labels=labels)

    # Calculate F1 score for each class individually
    per_class_f1 = tuple(
        f1_score(y_true_m == lbl, y_pred_m == lbl)
        for lbl in labels
    )

    # Calculate Cohen's Kappa score for agreement measure
    kappa = cohen_kappa_score(y_true_m, y_pred_m)

    return acc, f1_macro, cm, *per_class_f1, kappa


def model_summary(model):
    """
    Prints a summary of the model including total parameters, trainable parameters,
    untrainable parameters, model architecture, and parameter gradient requirements.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
    """
    print("Total Param:", end='')
    print(sum(p.numel() for p in model.parameters()))
    print("Trainable Param:", end='')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad == True))
    print("Untrainable Param:", end='')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad == False))

    print('\n'*5, '*'*30, "MODEL ARCHITECTURE", '*'*30, '\n')
    print(model)

    print('\n'*5, '*'*30, "PARAM GRADIENT", '*'*30, '\n')
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def find_files_with_suffix(root_dir, suffix):
    """
    Recursively searches for files with a specific suffix within a directory tree.

    Args:
        root_dir (str): Root directory path to start searching from.
        suffix (str): File suffix to match (e.g., '.txt', '.pth').

    Returns:
        list: List of full file paths matching the suffix.
    """
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files
