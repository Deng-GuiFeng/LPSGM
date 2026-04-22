# -*- coding: utf-8 -*-
"""
dataset.py

APPLES depression data adapter for the shared classification pipeline.

This module supplies only the task-specific bindings (default CSV path,
default NPZ data root, CSV column names) and delegates all heavy lifting to
``cls_core.dataset.load_subjects_from_csv``. The returned tuple
``(subject_dirs, labels, subject_ids)`` is consumed by
``cls_core.trainer.PooledTrainer`` and the ``simple_eval`` evaluator.

Labels: ``0 = Non-depressed``, ``1 = Depressed``. See
``preprocess/apples_dep_labels.csv`` (460 subjects: 327 Non-depressed,
133 Depressed). The label-derivation rules combine the self-reported
``depressionmedhxhp`` field with HAMD and BDI clinical scale scores; see
the manuscript's Supplementary Methods for the exact criteria.
"""

import os
from typing import List, Tuple

from cls_core.dataset import load_subjects_from_csv


DEFAULT_LABEL_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'preprocess', 'apples_dep_labels.csv',
)
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'APPLES',
)

LABEL_NAMES = {0: 'Non-depressed', 1: 'Depressed'}


def load_subjects(
    csv_path: str = DEFAULT_LABEL_CSV,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[List[str], List[int], List[str]]:
    """Load APPLES depression subjects as ``(subject_dirs, labels, subject_ids)``."""
    return load_subjects_from_csv(csv_path, data_root, id_col='fileid', label_col='label')
