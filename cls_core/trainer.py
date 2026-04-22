# -*- coding: utf-8 -*-
"""
trainer.py

Shared trainer for subject-level disorder classification.

The trainer implements a uniform k-fold protocol used by every downstream
disorder-classification task (narcolepsy / OSA / depression):

- Outer split: ``StratifiedKFold(n_splits=kfolds)`` → each fold produces
  (trainval_idx, test_idx) with ratio 80 % / 20 % by default.
- Inner split: ``StratifiedShuffleSplit(test_size=val_fraction)`` on each
  fold's trainval portion → (train_idx, val_idx) with ratio 75 % / 25 % of
  the trainval portion by default, i.e. 60 % / 20 % / 20 % of the full
  dataset.

Per fold the backbone is fine-tuned end-to-end on the training split with a
warmup-cosine learning-rate schedule, early-stopped by balanced accuracy on
the validation split, and finally evaluated on the test split. Outputs are
written under ``{run_root}/fold{N}/`` with the following layout:

    fold{N}/
        model_dir/{best.pth, last.pth}
        metrics/{val_best_metrics.json, test_metrics.json}
        predicts/test_preds.csv
        splits/subjects.json
        logs/                # TensorBoard event files

All behavior is configured through ``TrainConfig``; downstream modules only
supply a task-specific ``label_names`` dict and the subjects list returned
by their own dataset loader.
"""

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from .dataset import build_dataloaders
from .metrics import compute_metrics
from .model import LPSGMPooledClassifier


@dataclass
class TrainConfig:
    """Training configuration shared by every disorder-classification task."""

    # -- Backbone architecture (match the pretrained LPSGM weights) --
    pretrained_path: str = ''
    architecture: str = 'cat_cls'
    epoch_encoder_dropout: float = 0.0
    transformer_num_heads: int = 8
    transformer_dropout: float = 0.0
    transformer_attn_dropout: float = 0.0
    ch_num: int = 9
    seq_len: int = 20
    ch_emb_dim: int = 32
    seq_emb_dim: int = 64
    num_transformer_blocks: int = 4
    clamp_value: float = 10.0

    # -- Task --
    num_classes: int = 2

    # -- Data --
    label_csv: str = ''
    data_root: str = ''
    batch_size: int = 32
    num_workers: int = 16
    train_stride: int = 5
    val_stride: int = 1
    test_stride: int = 1

    # -- Cross-validation --
    kfolds: int = 5
    val_fraction: float = 0.25   # proportion of trainval to hold out as val
    seed: int = 42
    fold_ids: str = ''           # optional "0,1,2" to run a subset

    # -- Optimization --
    epochs: int = 5
    warmup_epochs: int = 1
    lr_backbone: float = 1e-6
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-8
    grad_clip: float = 1.0
    eval_every: int = 1
    class_weight: str = 'auto'   # 'none' | 'auto' | 'w0,w1,...'
    amp: bool = True
    freeze_backbone: bool = False

    # -- Output --
    run_root: str = './run'
    save_preds: bool = True


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    """Cosine-annealing scheduler with a linear warmup prefix."""

    def __init__(self, optimizer, T_max, warmup_epoch, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [base_lr * (self.last_epoch + 1) / max(1, self.warmup_epoch) for base_lr in self.base_lrs]
        return super().get_lr()


class AverageMeter:
    """Running average of a scalar value across a training epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    @property
    def avg(self):
        return self.sum / self.cnt if self.cnt > 0 else 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n


def _json_safe(obj):
    """Recursively convert numpy / tuple containers into JSON-serializable forms."""
    if isinstance(obj, dict):
        return {_json_safe(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_safe(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows, header=None) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow([ri if not hasattr(ri, 'item') else ri.item() for ri in r])


class PooledTrainer:
    """
    Driver for the uniform k-fold fine-tuning protocol.

    Args:
        cfg: ``TrainConfig`` instance.
        subjects: Tuple ``(subject_dirs, subject_labels, subject_ids)`` as
            returned by each task's ``dataset.load_subjects``.
        label_names: Optional ``{int: str}`` mapping from class index to
            human-readable name. Used only for logging.
    """

    def __init__(self, cfg: TrainConfig, subjects, label_names: Optional[Dict[int, str]] = None):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.label_names = label_names or {i: str(i) for i in range(cfg.num_classes)}

        os.makedirs(cfg.run_root, exist_ok=True)

        self.subject_dirs, self.labels, self.subject_ids = subjects
        assert len(self.subject_dirs) > 0, "No valid subjects found."

        dist = Counter(self.labels)
        print(f"Loaded {len(self.subject_dirs)} subjects (num_classes={self.num_classes}):")
        for lab in sorted(dist):
            print(f"  Class {lab} ({self.label_names.get(lab, '?')}): {dist[lab]}")

        try:
            manifest = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'args': _json_safe(vars(cfg)),
                'total_subjects': len(self.subject_dirs),
                'distribution': {self.label_names.get(k, '?'): v for k, v in dist.items()},
            }
            _write_json(os.path.join(cfg.run_root, 'run_manifest.json'), manifest)
        except Exception as exc:
            print(f"[WARN] manifest write failed: {exc}")

    # ---------- model / optimizer ----------

    def build_model(self):
        """Construct the model, load pretrained backbone, return (model, optimizer, scheduler)."""
        class _A:
            pass
        a = _A()
        for attr in ['architecture', 'epoch_encoder_dropout', 'transformer_num_heads',
                     'transformer_dropout', 'transformer_attn_dropout', 'ch_num',
                     'seq_len', 'ch_emb_dim', 'seq_emb_dim', 'num_transformer_blocks',
                     'clamp_value']:
            setattr(a, attr, getattr(self.cfg, attr))

        model = LPSGMPooledClassifier(a, num_classes=self.num_classes)

        if self.cfg.pretrained_path and os.path.exists(self.cfg.pretrained_path):
            report = model.load_from_pretrained(self.cfg.pretrained_path)
            print(f"Loaded pretrained backbone: {report}")
        else:
            print("No pretrained weights loaded")

        if self.cfg.freeze_backbone:
            for p in model.epoch_encoder.parameters():
                p.requires_grad = False
            for p in model.seq_encoder.parameters():
                p.requires_grad = False
            print("Backbone frozen.")

        model = nn.DataParallel(model).cuda()

        m = model.module if isinstance(model, nn.DataParallel) else model
        backbone_params = list(p for p in m.epoch_encoder.parameters() if p.requires_grad) + \
                          list(p for p in m.seq_encoder.parameters() if p.requires_grad)
        head_params = list(p for p in m.classifier.parameters() if p.requires_grad)

        param_groups = [{'params': head_params, 'lr': self.cfg.lr_head}]
        if not self.cfg.freeze_backbone:
            param_groups.insert(0, {'params': backbone_params, 'lr': self.cfg.lr_backbone})

        optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        scheduler = WarmupCosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs,
            warmup_epoch=self.cfg.warmup_epochs,
            eta_min=self.cfg.eta_min,
        )
        return model, optimizer, scheduler

    def make_class_weight(self, y_train):
        """Build a class-weight tensor according to ``cfg.class_weight``."""
        if self.cfg.class_weight == 'none':
            return None
        if self.cfg.class_weight == 'auto':
            y = np.array(y_train)
            counts = np.bincount(y, minlength=self.num_classes)
            if 0 in counts:
                return None
            total = len(y)
            weights = total / (self.num_classes * counts)
            print(f"Class weights: {', '.join(f'{w:.3f}' for w in weights)}")
            return torch.tensor(weights, dtype=torch.float32).cuda()
        parts = [float(x) for x in self.cfg.class_weight.split(',')]
        return torch.tensor(parts, dtype=torch.float32).cuda()

    # ---------- splits ----------

    def split_kfolds(self):
        """
        Produce ``[(train_idx, val_idx, test_idx), ...]`` of length ``kfolds``.

        Outer: ``StratifiedKFold(n_splits=kfolds)`` yields (trainval, test).
        Inner: ``StratifiedShuffleSplit(test_size=val_fraction)`` splits
               trainval into (train, val). With the default
               ``kfolds=5, val_fraction=0.25`` this produces 60 % / 20 % / 20 %.
        """
        labs = np.array(self.labels)
        subs = np.arange(len(self.labels))
        skf = StratifiedKFold(n_splits=self.cfg.kfolds, shuffle=True, random_state=self.cfg.seed)
        folds = []
        for trainval_idx, test_idx in skf.split(subs, labs):
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.cfg.val_fraction,
                random_state=self.cfg.seed,
            )
            tr_local, val_local = next(sss.split(trainval_idx, labs[trainval_idx]))
            tr_idx = trainval_idx[tr_local]
            val_idx = trainval_idx[val_local]
            folds.append((tr_idx, val_idx, test_idx))
        return folds

    # ---------- evaluation ----------

    def evaluate_subject_level(self, model, loader, desc: str = 'Eval'):
        """
        Evaluate the subject-level pooled classifier.

        Because the pooled model outputs one set of logits per window and each
        window carries the subject's label, we aggregate windows to subjects
        by averaging per-window softmax probabilities, then argmax for the
        subject-level prediction.
        """
        model.eval()
        subj_logits: Dict[str, list] = {}
        subj_labels: Dict[str, int] = {}

        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for x, y, seq_idx, ch_idx, mask, ori_len, names in pbar:
                x = x.cuda(non_blocking=True)
                seq_idx = seq_idx.cuda(non_blocking=True)
                ch_idx = ch_idx.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True).bool()
                ori_len_list = ori_len.tolist()

                out = model(x, mask, ch_idx, seq_idx, ori_len_list)
                logits = out.softmax(dim=-1).cpu().numpy()
                labs = y.numpy()

                for i, sid in enumerate(names):
                    subj_logits.setdefault(sid, []).append(logits[i])
                    subj_labels[sid] = int(labs[i][0]) if labs.ndim == 2 else int(labs[i])

        sids = sorted(subj_logits.keys())
        logits_agg = np.stack([np.mean(subj_logits[sid], axis=0) for sid in sids], axis=0)
        labs_agg = np.array([subj_labels[sid] for sid in sids])
        preds_agg = logits_agg.argmax(axis=1)

        # Window-level (sample-level) metrics alongside the subject-level ones.
        all_logits = np.concatenate([np.stack(subj_logits[sid]) for sid in sids])
        all_labs = np.concatenate([[subj_labels[sid]] * len(subj_logits[sid]) for sid in sids])
        all_preds = all_logits.argmax(axis=1)

        sample_metrics = compute_metrics(all_labs, all_preds)
        subject_metrics = compute_metrics(labs_agg, preds_agg)

        # AUC: binary uses the positive-class probability; multi-class uses macro OVR.
        try:
            n_present = len(set(labs_agg))
            if self.num_classes == 2 and n_present == 2:
                subject_metrics['auc'] = float(roc_auc_score(labs_agg, logits_agg[:, 1]))
            elif self.num_classes > 2 and n_present == self.num_classes:
                subject_metrics['auc'] = float(
                    roc_auc_score(labs_agg, logits_agg, multi_class='ovr', average='macro')
                )
            else:
                subject_metrics['auc'] = 0.0
        except Exception as exc:
            print(f'[WARN] AUC computation failed: {exc}')
            subject_metrics['auc'] = 0.0

        return {
            'cls_sample': sample_metrics,
            'cls_subject': subject_metrics,
            'subject_logits': logits_agg,
            'subject_labels': labs_agg,
            'subject_preds': preds_agg,
            'subject_ids': sids,
        }

    # ---------- fold training ----------

    def run_fold(self, fold_id, tr_idx, val_idx, test_idx):
        """Train and evaluate a single fold, writing all outputs under ``run_root/fold{N}``."""
        run_dir = os.path.join(self.cfg.run_root, f'fold{fold_id}')
        for sub in ['model_dir', 'predicts', 'metrics', 'splits']:
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        logger = SummaryWriter(os.path.join(run_dir, 'logs'))

        train_dirs = [self.subject_dirs[i] for i in tr_idx]
        train_labels = [self.labels[i] for i in tr_idx]
        val_dirs = [self.subject_dirs[i] for i in val_idx]
        val_labels = [self.labels[i] for i in val_idx]
        test_dirs = [self.subject_dirs[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]

        loaders = build_dataloaders(
            train_dirs, train_labels, val_dirs, val_labels, test_dirs, test_labels,
            seq_len=self.cfg.seq_len, batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            train_stride=self.cfg.train_stride, val_stride=self.cfg.val_stride,
            test_stride=self.cfg.test_stride,
        )

        try:
            split = {
                'fold': fold_id,
                'train': [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in tr_idx],
                'val': [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in val_idx],
                'test': [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in test_idx],
            }
            _write_json(os.path.join(run_dir, 'splits', 'subjects.json'), split)
        except Exception as exc:
            print(f"[WARN] split write failed: {exc}")

        print(f"Split stats | train: {len(loaders['train'].dataset)} windows, "
              f"val: {len(loaders['val'].dataset)}, test: {len(loaders['test'].dataset)}")

        model, optimizer, scheduler = self.build_model()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        class_weight = self.make_class_weight(train_labels)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        best_val_score = -1.0
        best_path = os.path.join(run_dir, 'model_dir', 'best.pth')

        for epoch in range(self.cfg.epochs):
            model.train()
            loss_meter = AverageMeter()
            pbar = tqdm(loaders['train'], desc=f"Fold{fold_id} Epoch {epoch}")
            for x, y, seq_idx, ch_idx, mask, ori_len, names in pbar:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                seq_idx = seq_idx.cuda(non_blocking=True)
                ch_idx = ch_idx.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True).bool()

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    out = model(x, mask, ch_idx, seq_idx, ori_len.tolist())
                    # Pooled model: one logits vector per subject; take the shared
                    # subject-level label from the first epoch of the window.
                    loss = criterion(out, y[:, 0])

                scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), n=y.size(0))
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

            scheduler.step()
            logger.add_scalar('train/loss', loss_meter.avg, epoch)

            if epoch % max(1, self.cfg.eval_every) == 0:
                val_metrics = self.evaluate_subject_level(model, loaders['val'], desc='Eval-val')
                sm = val_metrics['cls_subject']
                logger.add_scalar('val/subject_acc', sm['accuracy'], epoch)
                logger.add_scalar('val/subject_bal_acc', sm['balanced_accuracy'], epoch)
                logger.add_scalar('val/subject_macro_f1', sm['macro_f1'], epoch)

                print(f"  Val: acc={sm['accuracy']:.4f}, bal_acc={sm['balanced_accuracy']:.4f}, "
                      f"auc={sm.get('auc', 0):.4f}, mf1={sm['macro_f1']:.4f}, "
                      f"kappa={sm['kappa']:.4f}, per_class_f1={[round(x, 3) for x in sm['per_class_f1']]}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                }, os.path.join(run_dir, 'model_dir', 'last.pth'))

                # Select by balanced accuracy (robust to class imbalance).
                select_score = sm['balanced_accuracy']
                if select_score > best_val_score:
                    best_val_score = select_score
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                    }, best_path)
                    _write_json(os.path.join(run_dir, 'metrics', 'val_best_metrics.json'), val_metrics)

        logger.close()

        # Fall back to the last epoch if val never improved (e.g. 1-epoch smoke test).
        if not os.path.exists(best_path):
            torch.save({'model_state_dict': model.state_dict(), 'epoch': self.cfg.epochs - 1}, best_path)
        best_ckpt = torch.load(best_path)
        model.load_state_dict(best_ckpt['model_state_dict'])
        test_metrics = self.evaluate_subject_level(model, loaders['test'], desc='Test')
        _write_json(os.path.join(run_dir, 'metrics', 'test_metrics.json'), test_metrics)

        if self.cfg.save_preds:
            _write_csv(
                os.path.join(run_dir, 'predicts', 'test_preds.csv'),
                [[sid, int(pred), int(lab)] for sid, pred, lab in
                 zip(test_metrics['subject_ids'],
                     test_metrics['subject_preds'],
                     test_metrics['subject_labels'])],
                header=['subject_id', 'pred', 'label'],
            )

        tm = test_metrics['cls_subject']
        print(f"Fold {fold_id} done. Best val bal_acc: {best_val_score:.4f}, "
              f"AUC={tm.get('auc', 0):.4f}, "
              f"Test: acc={tm['accuracy']:.4f}, bal_acc={tm['balanced_accuracy']:.4f}, "
              f"mf1={tm['macro_f1']:.4f}, pcf1={[round(x, 3) for x in tm['per_class_f1']]}")

    def run(self):
        """Iterate over every fold (or a subset specified by ``cfg.fold_ids``)."""
        folds = self.split_kfolds()
        if self.cfg.fold_ids.strip():
            selected = {int(x) for x in self.cfg.fold_ids.split(',') if x.strip()}
            print(f"Restricting to folds {sorted(selected)}")
        else:
            selected = None
        for fold_id, (tr_idx, val_idx, test_idx) in enumerate(folds):
            if selected is not None and fold_id not in selected:
                continue
            print(f"\n=== Fold {fold_id} ===")
            self.run_fold(fold_id, tr_idx, val_idx, test_idx)
        print("\n=== All requested folds completed ===")
