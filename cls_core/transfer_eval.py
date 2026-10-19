# -*- coding: utf-8 -*-
"""
transfer_eval.py

Cross-dataset evaluation of a screening task: the cross-validation fold models
of a source task are applied to an external target dataset as one ensemble.

Each fold of the source task provides a fine-tuned LPSGM backbone
(``best.pth``) and a screening head (``screening_head.npz``, written by
``simple_eval``). For every target subject:

1. Each fold model extracts the subject-level feature (mean over the
   subject's ``seq_len``-epoch windows, with the window stride stored in the
   head) and applies its own head to obtain class probabilities.
2. The ensemble probability is the average of the fold probabilities (soft
   voting); the predicted class is the one with the highest ensemble
   probability (no threshold tuning).

No target data are used for training or fitting. Metrics are computed once
over all target subjects (AUC / accuracy / balanced accuracy / Cohen's kappa /
macro F1 / per-class F1 / confusion matrix).

Both model layouts are accepted under ``model_root``:

    fold{N}/best.pth + fold{N}/screening_head.npz             (released models)
    fold{N}/model_dir/best.pth + fold{N}/screening_head.npz   (trainer output)
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .dataset import SubjectWindowedDataset, collate_fn
from .simple_eval import _extract_subject_features, _load_backbone, load_screening_head


def _fold_files(model_root: str, fold_id: int):
    """Return ``(checkpoint_path, head_path)`` of one fold model."""
    fold_dir = os.path.join(model_root, f'fold{fold_id}')
    ckpt_path = os.path.join(fold_dir, 'best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(fold_dir, 'model_dir', 'best.pth')
    return ckpt_path, os.path.join(fold_dir, 'screening_head.npz')


def run_transfer_eval(
    subjects,
    model_root: str,
    out_dir: str,
    num_folds: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """
    Apply the ``num_folds`` fold models under ``model_root`` to the target subjects.

    Args:
        subjects: Tuple ``(subject_dirs, labels, subject_ids)`` of the target dataset.
        model_root: Directory containing ``fold0 .. fold{num_folds-1}`` of the source task.
        out_dir: Output directory for ``transfer_eval.json`` and ``transfer_preds.npz``.
        num_folds: Number of fold models in the ensemble.
        batch_size, num_workers: Feature-extraction knobs.

    Returns:
        Result dictionary; also written to ``{out_dir}/transfer_eval.json``.
    """
    subject_dirs, labels, _subject_ids = subjects
    print(f'Target: {len(subject_dirs)} subjects, dist={np.bincount(np.array(labels)).tolist()}')

    head = np.load(_fold_files(model_root, 0)[1])
    seq_len, stride = int(head['seq_len']), int(head['stride'])
    target_ds = SubjectWindowedDataset(
        subject_dirs, labels, seq_len=seq_len, stride=stride,
        n_load_workers=16, verbose=False,
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    fold_probs, classes, sids, y_true = [], None, None, None
    for fold_id in range(num_folds):
        ckpt_path, head_path = _fold_files(model_root, fold_id)
        scaler, clf = load_screening_head(head_path)
        model = _load_backbone(ckpt_path, num_classes=len(clf.classes_))
        X, y, fold_sids = _extract_subject_features(model, target_loader, f'fold{fold_id} model')
        if sids is None:
            classes, sids, y_true = clf.classes_, fold_sids, y
        fold_probs.append(clf.predict_proba(scaler.transform(X)))
        del model
        torch.cuda.empty_cache()

    # Soft voting: average the class probabilities of the fold models.
    fold_probs = np.stack(fold_probs)
    probs = fold_probs.mean(axis=0)
    preds = classes[np.argmax(probs, axis=1)]

    try:
        if len(classes) == 2:
            auc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, probs, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0
    mf1 = float(f1_score(y_true, preds, average='macro'))
    bal = float(balanced_accuracy_score(y_true, preds))
    acc = float(accuracy_score(y_true, preds))
    kappa = float(cohen_kappa_score(y_true, preds))
    cm = confusion_matrix(y_true, preds, labels=classes).tolist()
    pcf1 = f1_score(y_true, preds, labels=classes, average=None).tolist()

    result = {
        'model_root': model_root,
        'method': 'fold_model_ensemble_soft_voting',
        'num_folds': num_folds,
        'feature_windows': f'seq_len={seq_len}, stride={stride}',
        'test_metrics': {
            'macro_f1': mf1,
            'balanced_accuracy': bal,
            'auc': auc,
            'accuracy': acc,
            'kappa': kappa,
            'per_class_f1': pcf1,
            'confusion_matrix': cm,
        },
        'n_test_subjects': int(len(y_true)),
    }

    print(f'\n=== Transfer evaluation ({num_folds}-fold ensemble) ===')
    print(f'  MF1={mf1:.4f} bal={bal:.4f} AUC={auc:.4f} acc={acc:.4f} kappa={kappa:.4f}')
    print(f'  CM={cm}')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'transfer_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')

    preds_path = os.path.join(out_dir, 'transfer_preds.npz')
    np.savez(
        preds_path,
        subject_ids=np.array(sids),
        y_true=np.array(y_true, dtype=np.int64),
        y_prob=(probs[:, 1] if len(classes) == 2 else probs).astype(np.float64),
        fold_probs=fold_probs.astype(np.float64),
    )
    print(f'Saved: {preds_path}')
    return result


def cli_main(
    load_subjects_fn,
    default_model_root: str,
    default_data_root: str,
    default_label_csv: str,
    default_out_dir: str,
):
    """
    Shared ``python -m`` entry point body. Task CLIs call this function, passing
    their target ``load_subjects`` implementation and defaults.
    """
    p = argparse.ArgumentParser(description='Cross-dataset evaluation with the fold-model ensemble')
    p.add_argument('--model-root', type=str, default=default_model_root,
                   help='Directory containing fold0..fold{N-1} of the source task')
    p.add_argument('--data-root', type=str, default=default_data_root,
                   help='Preprocessed NPZ root of the target dataset')
    p.add_argument('--label-csv', type=str, default=default_label_csv,
                   help='Label CSV of the target dataset')
    p.add_argument('--out-dir', type=str, default=default_out_dir)
    p.add_argument('--num-folds', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()

    subjects = load_subjects_fn(args.label_csv, args.data_root)
    run_transfer_eval(
        subjects=subjects,
        model_root=args.model_root,
        out_dir=args.out_dir,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
