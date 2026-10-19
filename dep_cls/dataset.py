# -*- coding: utf-8 -*-
"""
dataset.py

APPLES depression data adapter for the shared classification pipeline.

Labels are computed on the fly from the NSRR APPLES dataset table
(``apples-dataset-0.1.0.csv``) by ``preprocess/apples_labels.py`` from the raw
``depressionmedhxhp``, ``hamdtotalscore`` and ``bditotalscore`` fields; no
per-subject label file is stored in the repository. The returned tuple
``(subject_dirs, labels, subject_ids)`` is consumed by
``cls_core.trainer.PooledTrainer`` and the ``simple_eval`` evaluator.

Labels: ``0 = Non-depressed``, ``1 = Depressed`` (surrogate label; 460
subjects: 327 Non-depressed, 133 Depressed). See ``preprocess/apples_labels.py``
and the Supplementary Methods for the exact criteria.
"""

import os
import sys
from typing import List, Tuple

from cls_core.dataset import load_subjects_from_labels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocess'))
from apples_labels import DEFAULT_NSRR_CSV, depression_labels  # noqa: E402


DEFAULT_LABEL_CSV = DEFAULT_NSRR_CSV
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'APPLES',
)

LABEL_NAMES = {0: 'Non-depressed', 1: 'Depressed'}


def load_subjects(
    csv_path: str = DEFAULT_LABEL_CSV,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load APPLES depression subjects as ``(subject_dirs, labels, subject_ids)``.

    Args:
        csv_path: Path to the NSRR APPLES dataset table ``apples-dataset-0.1.0.csv``.
        data_root: Preprocessed APPLES NPZ root.
    """
    return load_subjects_from_labels(depression_labels(csv_path), data_root)
