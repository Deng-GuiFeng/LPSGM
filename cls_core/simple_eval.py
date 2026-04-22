# -*- coding: utf-8 -*-
"""
simple_eval.py

Subject-level linear probing on top of a fine-tuned LPSGM backbone.

Per fold:

1. Load the fold's fine-tuned LPSGM backbone checkpoint.
2. For each subject, extract the mean-pooled ``feat_dim``-dim sequence feature
   averaged over every sliding window of the subject.
3. Fit a single class-weighted logistic regression on the per-subject mean
   features from the outer fold's trainval split (80 % of subjects).
4. Apply the logistic regression directly to the outer test split (20 %) and
   report AUC / accuracy / balanced accuracy / macro F1 / per-class F1 /
   confusion matrix.

The evaluator reuses exactly the same ``StratifiedKFold`` outer split as
``trainer.PooledTrainer`` (same ``n_splits``, ``shuffle=True`` and
``random_state``), so the test split here matches the one used to report
backbone metrics.
"""

import argparse
import json
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SubjectWindowedDataset, collate_fn
from .model import LPSGMPooledClassifier


DEFAULT_ARCH = dict(
    architecture='cat_cls',
    epoch_encoder_dropout=0.0,
    transformer_num_heads=8,
    transformer_dropout=0.0,
    transformer_attn_dropout=0.0,
    ch_num=9,
    seq_len=20,
    ch_emb_dim=32,
    seq_emb_dim=64,
    num_transformer_blocks=4,
    clamp_value=10.0,
)


def _extract_subject_features(model: LPSGMPooledClassifier, loader, desc: str):
    """
    Extract subject-level mean-pooled features.

    For each subject, average the pooled feature vector across all windows.

    Returns:
        X: ``(num_subjects, feat_dim)`` float32 array.
        y: ``(num_subjects,)`` int array of labels.
        sids: Sorted list of subject IDs corresponding to rows of X / y.
    """
    subj_sum, subj_count, subj_labels = {}, {}, {}
    with torch.no_grad():
        for x, y, seq_idx, ch_idx, mask, ori_len, names in tqdm(loader, desc=desc):
            x = x.cuda(non_blocking=True)
            seq_idx = seq_idx.cuda(non_blocking=True)
            ch_idx = ch_idx.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True).bool()
            feats = model.extract_features(x, mask, ch_idx, seq_idx, ori_len.tolist())
            feats_np = feats.cpu().numpy()
            labs = y.numpy()
            for i, sid in enumerate(names):
                if sid not in subj_sum:
                    subj_sum[sid] = feats_np[i].astype(np.float64)
                    subj_count[sid] = 1
                else:
                    subj_sum[sid] += feats_np[i].astype(np.float64)
                    subj_count[sid] += 1
                subj_labels[sid] = int(labs[i][0]) if labs.ndim == 2 else int(labs[i])
    sids = sorted(subj_sum.keys())
    X = np.stack([subj_sum[sid] / subj_count[sid] for sid in sids]).astype(np.float32)
    y = np.array([subj_labels[sid] for sid in sids])
    return X, y, sids


def _load_backbone(ckpt_path: str, num_classes: int, arch_overrides: Optional[dict] = None) -> LPSGMPooledClassifier:
    """Instantiate ``LPSGMPooledClassifier`` and load a checkpoint's backbone weights."""
    arch = dict(DEFAULT_ARCH)
    if arch_overrides:
        arch.update(arch_overrides)
    model_args = SimpleNamespace(**arch)
    model = LPSGMPooledClassifier(model_args, num_classes=num_classes)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.cuda().eval()


def run_simple_eval(
    fold_id: int,
    subjects,
    run_root: str,
    num_classes: int = 2,
    kfolds: int = 5,
    seed: int = 42,
    seq_len: int = 20,
    stride: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    C: float = 1.0,
    arch_overrides: Optional[dict] = None,
    ckpt_path: Optional[str] = None,
) -> dict:
    """
    Run ``simple_eval`` for a single fold.

    Args:
        fold_id: Outer fold index (``0 .. kfolds - 1``).
        subjects: Tuple ``(subject_dirs, labels, subject_ids)`` from the task's
            ``dataset.load_subjects``.
        run_root: Directory containing the trainer's output
            ``{run_root}/fold{fold_id}/model_dir/best.pth``.
        num_classes: Number of classes in the task.
        kfolds: Outer k-fold count. Must match the trainer's configuration.
        seed: Random seed. Must match the trainer's configuration.
        seq_len, stride, batch_size, num_workers: Feature-extraction knobs.
        C: Inverse regularization strength of the logistic regression.
        arch_overrides: Optional dict of backbone architecture overrides when
            the checkpoint was trained with non-default values.
        ckpt_path: Optional explicit checkpoint path. Defaults to
            ``{run_root}/fold{fold_id}/model_dir/best.pth``.

    Returns:
        Result dictionary; also written to ``{run_root}/fold{fold_id}/simple_eval.json``.
    """
    subject_dirs, labels, _subject_ids = subjects
    labels_arr = np.array(labels)

    # Outer k-fold split, matched to the trainer.
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    splits = list(skf.split(np.arange(len(labels_arr)), labels_arr))
    trainval_idx, test_idx = splits[fold_id]

    train_dirs = [subject_dirs[i] for i in trainval_idx]
    train_labels = labels_arr[trainval_idx].tolist()
    test_dirs = [subject_dirs[i] for i in test_idx]
    test_labels = labels_arr[test_idx].tolist()
    print(f'Fold {fold_id}: trainval={len(train_dirs)}, test={len(test_dirs)}')

    # Load the fold's fine-tuned backbone.
    if ckpt_path is None:
        ckpt_path = os.path.join(run_root, f'fold{fold_id}', 'model_dir', 'best.pth')
    model = _load_backbone(ckpt_path, num_classes=num_classes, arch_overrides=arch_overrides)

    # Build datasets with a shared preload cache.
    train_ds = SubjectWindowedDataset(
        train_dirs, train_labels, seq_len=seq_len, stride=stride,
        n_load_workers=16, verbose=False,
    )
    test_ds = SubjectWindowedDataset(
        test_dirs, test_labels, seq_len=seq_len, stride=stride,
        n_load_workers=16, verbose=False, preload_data=train_ds._data,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # Extract subject-level mean-pooled features.
    X_train, y_train, _ = _extract_subject_features(model, train_loader, f'train fold{fold_id}')
    X_test, y_test, test_sids = _extract_subject_features(model, test_loader, f'test fold{fold_id}')
    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

    # Standardize features using train-only statistics.
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)

    # Fit a single class-weighted logistic regression on the trainval split
    # and apply it directly to the test split (no threshold tuning).
    clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_train_n, y_train)
    test_probs = clf.predict_proba(X_test_n)
    test_preds = clf.predict(X_test_n)

    if num_classes == 2:
        try:
            auc = float(roc_auc_score(y_test, test_probs[:, 1]))
        except Exception:
            auc = 0.0
    else:
        try:
            auc = float(roc_auc_score(y_test, test_probs, multi_class='ovr', average='macro'))
        except Exception:
            auc = 0.0

    mf1 = float(f1_score(y_test, test_preds, average='macro'))
    bal = float(balanced_accuracy_score(y_test, test_preds))
    acc = float(accuracy_score(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds).tolist()
    pcf1 = f1_score(y_test, test_preds, average=None).tolist()

    result = {
        'fold': fold_id,
        'ckpt': ckpt_path,
        'method': 'frozen_backbone_linear_probing',
        'feature_dim': int(X_train.shape[1]),
        'classifier': f'LogisticRegression(C={C}, class_weight=balanced)',
        'test_metrics': {
            'macro_f1': mf1,
            'balanced_accuracy': bal,
            'auc': auc,
            'accuracy': acc,
            'per_class_f1': pcf1,
            'confusion_matrix': cm,
        },
        'n_train_subjects': int(len(y_train)),
        'n_test_subjects': int(len(y_test)),
    }

    print(f'\n=== Fold {fold_id} simple_eval ===')
    print(f'  MF1={mf1:.4f} bal={bal:.4f} AUC={auc:.4f} acc={acc:.4f}')
    print(f'  CM={cm}')

    out_dir = os.path.join(run_root, f'fold{fold_id}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'simple_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')

    preds_path = os.path.join(out_dir, 'simple_preds.npz')
    np.savez(
        preds_path,
        subject_ids=np.array(test_sids),
        y_true=np.array(y_test, dtype=np.int64),
        y_prob=(test_probs[:, 1] if num_classes == 2 else test_probs).astype(np.float64),
    )
    print(f'Saved: {preds_path}')

    return result


def cli_main(load_subjects_fn, default_run_root: str = './run', default_num_classes: int = 2):
    """
    Shared ``python -m`` entry point body. Task CLIs call this function, passing
    their own ``load_subjects`` implementation and defaults.
    """
    p = argparse.ArgumentParser(description='Subject-level linear probing evaluator')
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--run-root', type=str, default=default_run_root)
    p.add_argument('--num-classes', type=int, default=default_num_classes)
    p.add_argument('--kfolds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--ckpt', type=str, default=None,
                   help='Explicit checkpoint path; defaults to '
                        '{run-root}/fold{fold}/model_dir/best.pth')
    p.add_argument('--label-csv', type=str, default=None)
    p.add_argument('--data-root', type=str, default=None)
    args = p.parse_args()

    subjects = load_subjects_fn(
        **({'csv_path': args.label_csv} if args.label_csv else {}),
        **({'data_root': args.data_root} if args.data_root else {}),
    )
    run_simple_eval(
        fold_id=args.fold,
        subjects=subjects,
        run_root=args.run_root,
        num_classes=args.num_classes,
        kfolds=args.kfolds,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        C=args.C,
        ckpt_path=args.ckpt,
    )
