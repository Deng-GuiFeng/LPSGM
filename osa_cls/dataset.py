# -*- coding: utf-8 -*-
"""
dataset.py

APPLES OSA-severity data adapter for the shared classification pipeline.

This module supplies only the task-specific bindings (default CSV path,
default NPZ data root, CSV column names) and delegates all heavy lifting to
``cls_core.dataset.load_subjects_from_csv``. The returned tuple
``(subject_dirs, labels, subject_ids)`` is consumed by
``cls_core.trainer.PooledTrainer`` as well as the ``simple_eval`` /
``hierarchical_eval`` evaluators.

Labels: ``0 = Non-severe`` (Mild + Moderate), ``1 = Severe``. See
``preprocess/apples_osa_labels.csv``.
"""

import os
from typing import List, Tuple

from cls_core.dataset import load_subjects_from_csv


DEFAULT_LABEL_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'preprocess', 'apples_osa_labels.csv',
)
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'APPLES',
)

LABEL_NAMES = {0: 'Non-severe', 1: 'Severe'}


def load_subjects(
    csv_path: str = DEFAULT_LABEL_CSV,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[List[str], List[int], List[str]]:
    """Load APPLES OSA subjects as ``(subject_dirs, labels, subject_ids)``."""
    return load_subjects_from_csv(csv_path, data_root, id_col='fileid', label_col='label')
