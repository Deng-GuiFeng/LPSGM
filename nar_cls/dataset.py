# -*- coding: utf-8 -*-
"""
dataset.py

MNC 3-class narcolepsy data adapter for the shared classification pipeline.

The MNC dataset is split into six cohorts (CNC, DHC, FHC, IHC, KHC, SSC),
each preprocessed by its own script under ``preprocess/MNC/``. The
preprocessing writes one subdirectory per subject under
``data/MNC-<cohort>/<subject_id>/``, containing a single ``.npz`` file with
the usual PSG channels plus a ``Diagnosis`` integer field (0 = Non-narcolepsy
Control, 1 = Type 1 Narcolepsy, 2 = Other Hypersomnia).

This module only consumes that on-disk schema: it scans the six cohort
directories, reads the ``Diagnosis`` field from each subject's NPZ, and
returns the ``(subject_dirs, labels, subject_ids)`` triple expected by
``cls_core.trainer.PooledTrainer`` and the ``simple_eval`` evaluator.
Subject identifiers produced by the MNC preprocessing are already globally
unique across cohorts (CHC / DHC / FHC / IHC / KHC / SSC prefixes are
disjoint), so they are used as-is.
"""

import os
from typing import List, Tuple

import numpy as np


# One entry per MNC cohort preprocessed by preprocess/MNC/{cohort}.py.
# Matches the dst_root values ``./data/MNC-<cohort>/`` written by those scripts.
MNC_COHORTS = ['MNC-CNC', 'MNC-DHC', 'MNC-FHC', 'MNC-IHC', 'MNC-KHC', 'MNC-SSC']

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

LABEL_NAMES = {
    0: 'Non-narcolepsy Control',
    1: 'Type 1 Narcolepsy',
    2: 'Other Hypersomnia',
}


def load_subjects(data_root: str = DEFAULT_DATA_ROOT) -> Tuple[List[str], List[int], List[str]]:
    """
    Enumerate MNC subjects across all six cohorts.

    For each cohort directory, every immediate subdirectory is treated as one
    subject. The first ``.npz`` file inside the subject directory is opened
    to read the ``Diagnosis`` integer label. Subjects whose NPZ is missing
    the ``Diagnosis`` field or is unreadable are skipped.

    Args:
        data_root: Root containing ``MNC-<cohort>/`` subdirectories. Defaults
            to the repository's ``data/`` directory.

    Returns:
        Tuple ``(subject_dirs, labels, subject_ids)``:
            - ``subject_dirs``: absolute path to each subject directory.
            - ``labels``: integer class label (0 / 1 / 2) aligned with ``subject_dirs``.
            - ``subject_ids``: MNC subject identifier (e.g. ``"CHC001"``),
              unique across cohorts.
    """
    data_root = os.path.realpath(data_root)
    subject_dirs: List[str] = []
    labels: List[int] = []
    subject_ids: List[str] = []

    for cohort in MNC_COHORTS:
        cohort_dir = os.path.join(data_root, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for sid in sorted(os.listdir(cohort_dir)):
            subj_dir = os.path.join(cohort_dir, sid)
            if not os.path.isdir(subj_dir):
                continue
            npzs = [f for f in os.listdir(subj_dir) if f.endswith('.npz')]
            if not npzs:
                continue
            try:
                with np.load(os.path.join(subj_dir, sorted(npzs)[0])) as npz:
                    if 'Diagnosis' not in npz:
                        continue
                    label = int(npz['Diagnosis'])
            except Exception:
                continue
            subject_dirs.append(subj_dir)
            labels.append(label)
            subject_ids.append(sid)

    return subject_dirs, labels, subject_ids
