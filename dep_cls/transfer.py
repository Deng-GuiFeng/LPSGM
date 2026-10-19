# -*- coding: utf-8 -*-
"""
transfer.py

Cross-dataset evaluation of the APPLES depression models on an external
cohort (SYSU in the paper), using the five cross-validation fold models as
one ensemble (see ``cls_core.transfer_eval``).

SYSU is a private clinical dataset and is not distributed with this
repository. The target data are read from preprocessed NPZ files under
``<data_root>/<subject_dir_name>/`` (written by ``preprocess/SYSU.py``) and a
label CSV with ``subject_dir_name`` and ``label`` columns (0 = healthy control,
1 = major depressive disorder). To evaluate the models on another cohort,
preprocess it into the same format and point ``--data-root`` and
``--label-csv`` to it, or adapt ``load_target_subjects``.

Usage:
    python -m dep_cls.transfer --model-root weights/screening/apples_dep \
        --data-root data/SYSU --label-csv data/sysu_dep_labels.csv \
        --out-dir run_transfer/apples_to_sysu
"""

from cls_core.dataset import load_subjects_from_csv
from cls_core.transfer_eval import cli_main


def load_target_subjects(csv_path: str, data_root: str):
    """Load the target cohort as ``(subject_dirs, labels, subject_ids)``."""
    return load_subjects_from_csv(csv_path, data_root, id_col='subject_dir_name', label_col='label')


if __name__ == '__main__':
    cli_main(
        load_target_subjects,
        default_model_root='./run_dep',
        default_data_root='./data/SYSU',
        default_label_csv='./data/sysu_dep_labels.csv',
        default_out_dir='./run_transfer/apples_to_sysu',
    )
