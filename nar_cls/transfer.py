# -*- coding: utf-8 -*-
"""
transfer.py

Cross-dataset evaluation of the MNC Normal vs Abnormal models on an external
cohort (HANG7 in the paper), using the five cross-validation fold models as
one ensemble (see ``cls_core.transfer_eval``).

HANG7 is a private clinical dataset and is not distributed with this
repository. The target data are read from preprocessed NPZ files under
``<data_root>/<subject_id>/`` (written by ``preprocess/HANG7.py``) and a label
CSV with ``subject_id`` and ``label`` columns (0 = Normal, 1 = Abnormal). To
evaluate the models on another cohort, preprocess it into the same format and
point ``--data-root`` and ``--label-csv`` to it, or adapt
``load_target_subjects``.

Usage:
    python -m nar_cls.transfer --model-root weights/screening/mnc_binary \
        --data-root data/HANG7 --label-csv data/hang7_binary_labels.csv \
        --out-dir run_transfer/mnc_to_hang7
"""

from cls_core.dataset import load_subjects_from_csv
from cls_core.transfer_eval import cli_main


def load_target_subjects(csv_path: str, data_root: str):
    """Load the target cohort as ``(subject_dirs, labels, subject_ids)``."""
    return load_subjects_from_csv(csv_path, data_root, id_col='subject_id', label_col='label')


if __name__ == '__main__':
    cli_main(
        load_target_subjects,
        default_model_root='./run_nar_binary',
        default_data_root='./data/HANG7',
        default_label_csv='./data/hang7_binary_labels.csv',
        default_out_dir='./run_transfer/mnc_to_hang7',
    )
