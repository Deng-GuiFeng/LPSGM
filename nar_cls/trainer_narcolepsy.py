# -*- coding: utf-8 -*-
"""
trainer_narcolepsy.py

Trainer module for narcolepsy classification within the LPSGM (Large Polysomnography Model) project.
This file defines the training pipeline, including data loading, model building, training loop,
evaluation, and result saving for narcolepsy diagnosis using polysomnography data.

Key functionalities:
- Configuration management via TrainConfig dataclass
- Model initialization with optional pretrained weights and parameter freezing
- Stratified k-fold splitting of subjects for cross-validation
- Data loader construction for train/val/test splits
- Training with mixed precision and gradient clipping
- Subject-level and sample-level evaluation metrics computation
- Logging with TensorBoard and CSV files
- Saving best and last model checkpoints and predictions

This trainer supports 2-class (merged NT1) or 3-class classification depending on configuration.
"""

import os
import csv
import math
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from .model import LPSGMNar
from .dataset import load_subjects
from .online_dataloader import build_dataloaders_nar
from .metrics import compute_metrics


@dataclass
class TrainConfig:
    """
    Configuration dataclass for training parameters and hyperparameters.

    Attributes:
        pretrained_path (str): Path to pretrained model weights.
        architecture (str): Model architecture type.
        epoch_encoder_dropout (float): Dropout rate in epoch encoder.
        transformer_num_heads (int): Number of attention heads in transformer.
        transformer_dropout (float): Dropout rate in transformer blocks.
        transformer_attn_dropout (float): Attention dropout rate.
        ch_num (int): Number of input channels.
        seq_len (int): Sequence length for transformer input.
        ch_emb_dim (int): Channel embedding dimension.
        seq_emb_dim (int): Sequence embedding dimension.
        num_transformer_blocks (int): Number of transformer blocks.
        clamp_value (float): Clamp value for activations.

        batch_size (int): Batch size for training.
        num_workers (int): Number of data loader workers.
        kfolds (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

        epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of warmup epochs for LR scheduler.
        lr_backbone (float): Learning rate for backbone parameters.
        lr_head (float): Learning rate for classifier head.
        weight_decay (float): Weight decay for optimizer.
        eta_min (float): Minimum learning rate for cosine annealing.
        grad_clip (float): Gradient clipping norm value.
        train_stride (int): Stride for training data windowing.
        val_stride (int): Stride for validation data windowing.
        test_stride (int): Stride for test data windowing.
        eval_every (int): Frequency (in epochs) to run evaluation.
        enable_train_eval (bool): Whether to evaluate on training set during training.
        class_weight (str): Class weighting strategy ('none', 'auto', or custom weights).
        amp (bool): Enable automatic mixed precision training.
        freeze_backbone (bool): Freeze backbone parameters during training.

        cls_loss_w (float): Weight for classification loss.

        run_root (str): Root directory for saving run outputs.
        save_preds (bool): Whether to save prediction CSV files.

        merge_NT1 (bool): Whether to merge NT1 classes (2-class classification).
    """
    pretrained_path: str
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

    batch_size: int = 64
    num_workers: int = 32
    kfolds: int = 5
    seed: int = 42

    epochs: int = 20
    warmup_epochs: int = 3
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-8
    grad_clip: float = 1.0
    train_stride: int = 1
    val_stride: int = 1
    test_stride: int = 1
    eval_every: int = 1
    enable_train_eval: bool = True
    class_weight: str = 'none'  # 'none' | 'auto' | 'w0,w1,w2'
    amp: bool = True
    freeze_backbone: bool = False

    # loss weights
    cls_loss_w: float = 1.0

    run_root: str = './run_nar'
    save_preds: bool = True

    merge_NT1: bool = False


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    """
    Learning rate scheduler combining warmup and cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        warmup_epoch (int): Number of warmup epochs.
        eta_min (float): Minimum learning rate.
        last_epoch (int): The index of last epoch.
        verbose (bool): If True, prints a message to stdout on LR update.
    """
    def __init__(self, optimizer, T_max, warmup_epoch, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        # Linear warmup for initial epochs
        if self.last_epoch < self.warmup_epoch:
            return [base_lr * (self.last_epoch + 1) / max(1, self.warmup_epoch) for base_lr in self.base_lrs]
        # Cosine annealing after warmup
        return super().get_lr()


class AverageMeter:
    """
    Utility class for tracking and averaging metrics during training or evaluation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the meter to initial state."""
        self.sum = 0.0
        self.cnt = 0

    @property
    def avg(self):
        """Compute current average."""
        return self.sum / self.cnt if self.cnt > 0 else 0.0

    def update(self, val, n=1):
        """
        Update meter with new value.

        Args:
            val (float): Value to add.
            n (int): Number of occurrences of val.
        """
        self.sum += float(val) * n
        self.cnt += n


def subject_id_from_name(name: str) -> str:
    """
    Extract subject ID from file or folder name.

    Args:
        name (str): File or folder name.

    Returns:
        str: Subject ID (currently returns input name unchanged).
    """
    return name


# -------- helpers for run/fold subject recording --------
def json_safe(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object potentially containing numpy types.

    Returns:
        Object with numpy types converted to native Python types.
    """
    import numpy as _np
    if isinstance(obj, dict):
        return {json_safe(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(json_safe(v) for v in obj)
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def _ensure_dir(path: str):
    """
    Create directory if it does not exist.

    Args:
        path (str): Directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj):
    """
    Write JSON object to file with pretty formatting.

    Args:
        path (str): File path to write JSON.
        obj: Object to serialize as JSON.
    """
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[List], header: List[str] | None = None):
    """
    Write rows of data to CSV file with optional header.

    Args:
        path (str): File path to write CSV.
        rows (List[List]): Rows of data.
        header (List[str] | None): Optional header row.
    """
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            # Convert tensor scalars to native Python types if needed
            w.writerow([r_i if not hasattr(r_i, 'item') else r_i.item() for r_i in r])


class NarcolepsyTrainer:
    """
    Trainer class for narcolepsy classification using LPSGM model.

    Handles data loading, model building, training, evaluation, and checkpointing.

    Args:
        cfg (TrainConfig): Configuration object with training parameters.
    """
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.run_root, exist_ok=True)

        # Load and index subjects with labels
        self.subjects = load_subjects(cfg.merge_NT1)
        self.subject_ids = [s.subject_id for s in self.subjects]
        self.labels = [s.diagnosis for s in self.subjects]
        assert len(self.subjects) == len(self.labels) and len(self.subjects) > 0, "No valid subjects found."

        # Set number of classes based on whether NT1 classes are merged
        self.num_classes = 2 if cfg.merge_NT1 else 3

        # Write run-level manifest file and subject list CSV
        try:
            run_manifest = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'args': json_safe(vars(cfg)),
                'total_subjects': len(self.subjects),
                'subjects': [
                    {'id': sid, 'label': int(lb)} for sid, lb in zip(self.subject_ids, self.labels)
                ],
            }
            _write_json(os.path.join(cfg.run_root, 'run_manifest.json'), run_manifest)
            _write_csv(os.path.join(cfg.run_root, 'run_subjects.csv'),
                      [[sid, int(lb)] for sid, lb in zip(self.subject_ids, self.labels)],
                      header=['subject_id', 'label'])
        except Exception as e:
            print(f"[WARN] failed to write run manifest: {e}")

    def build_model(self):
        """
        Build the LPSGM narcolepsy classification model, load pretrained weights if available,
        optionally freeze backbone parameters, and prepare optimizer and scheduler.

        Returns:
            tuple: (model, optimizer, scheduler)
        """
        # Build a simple args-like object to pass to model constructor
        class A:
            pass
        a = A()
        a.architecture = self.cfg.architecture
        a.epoch_encoder_dropout = self.cfg.epoch_encoder_dropout
        a.transformer_num_heads = self.cfg.transformer_num_heads
        a.transformer_dropout = self.cfg.transformer_dropout
        a.transformer_attn_dropout = self.cfg.transformer_attn_dropout
        a.ch_num = self.cfg.ch_num
        a.seq_len = self.cfg.seq_len
        a.ch_emb_dim = self.cfg.ch_emb_dim
        a.seq_emb_dim = self.cfg.seq_emb_dim
        a.num_transformer_blocks = self.cfg.num_transformer_blocks
        a.clamp_value = self.cfg.clamp_value

        model = LPSGMNar(a, self.num_classes)
        
        # Load pretrained backbone weights if path exists
        if self.cfg.pretrained_path and os.path.exists(self.cfg.pretrained_path):
            ckpt = torch.load(self.cfg.pretrained_path, map_location='cpu')
            pretrained_sd = ckpt.get('model_state_dict', ckpt)
            
            # Filter out classifier weights and remove 'module.' prefix if present
            backbone_sd = {}
            for key, value in pretrained_sd.items():
                clean_key = key.replace('module.', '')
                if not clean_key.startswith('classifier.'):
                    backbone_sd[clean_key] = value
            
            # Load backbone weights with non-strict mode to allow missing keys
            missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
            print(f"Loaded pretrained backbone: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        else:
            print("No pretrained weights loaded")

        if self.cfg.freeze_backbone:
            # Freeze epoch_encoder and seq_encoder parameters to prevent updates during training
            for p in model.backbone.epoch_encoder.parameters():
                p.requires_grad = False
            for p in model.backbone.seq_encoder.parameters():
                p.requires_grad = False
            print("Backbone parameters frozen.")

        # Wrap model with DataParallel and move to GPU
        model = nn.DataParallel(model).cuda()

        # Prepare optimizer parameter groups: separate backbone and classifier head
        if isinstance(model, nn.DataParallel):
            backbone_params = []
            backbone_params.extend([p for p in model.module.backbone.epoch_encoder.parameters() if p.requires_grad])
            backbone_params.extend([p for p in model.module.backbone.seq_encoder.parameters() if p.requires_grad])
            head_params = [p for p in model.module.backbone.classifier.parameters() if p.requires_grad]
        else:
            backbone_params = []
            backbone_params.extend([p for p in model.backbone.epoch_encoder.parameters() if p.requires_grad])
            backbone_params.extend([p for p in model.backbone.seq_encoder.parameters() if p.requires_grad])
            head_params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]

        # Construct optimizer with different learning rates for backbone and head
        if self.cfg.freeze_backbone:
            optimizer = AdamW([
                {'params': head_params, 'lr': self.cfg.lr_head},
            ], weight_decay=self.cfg.weight_decay)
        else:
            optimizer = AdamW([
                {'params': backbone_params, 'lr': self.cfg.lr_backbone},
                {'params': head_params, 'lr': self.cfg.lr_head},
            ], weight_decay=self.cfg.weight_decay)

        # Learning rate scheduler with warmup and cosine annealing
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs,
            warmup_epoch=self.cfg.warmup_epochs,
            eta_min=self.cfg.eta_min,
            verbose=True,
        )

        return model, optimizer, scheduler

    def make_class_weight(self, y_train: List[int]):
        """
        Compute class weights for CrossEntropyLoss based on training label distribution or custom input.

        Args:
            y_train (List[int]): List of training labels.

        Returns:
            torch.Tensor or None: Tensor of class weights on CUDA or None if no weighting.
        """
        if self.cfg.class_weight == 'none':
            return None
        if self.cfg.class_weight == 'auto':
            if self.cfg.merge_NT1:
                # Compute weights for 2-class classification based on subject-level label counts
                y = np.array(y_train)
                n0 = np.sum(y == 0)
                n1 = np.sum(y == 1)
                if n0 == 0 or n1 == 0:
                    return None
                total = n0 + n1
                w0 = total / (2 * n0)
                w1 = total / (2 * n1)
                print(f"Class weights: {w0:.3f}, {w1:.3f}")
                return torch.tensor([w0, w1], dtype=torch.float32).cuda()
            else:
                # Compute weights for 3-class classification based on subject-level label counts
                y = np.array(y_train)
                n0 = np.sum(y == 0)
                n1 = np.sum(y == 1)
                n2 = np.sum(y == 2)
                if n0 == 0 or n1 == 0 or n2 == 0:
                    return None
                total = n0 + n1 + n2
                w0 = total / (3 * n0)
                w1 = total / (3 * n1)
                w2 = total / (3 * n2)
                print(f"Class weights: {w0:.3f}, {w1:.3f}, {w2:.3f}")
                return torch.tensor([w0, w1, w2], dtype=torch.float32).cuda()
        
        # Parse custom class weights string like "0.4,0.3,0.3"
        parts = [float(x) for x in self.cfg.class_weight.split(',')]
        return torch.tensor(parts, dtype=torch.float32).cuda()

    def split_kfolds(self):
        """
        Perform stratified k-fold splitting of subjects into train, validation, and test sets.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: List of (train_idx, val_idx, test_idx) tuples.
        """
        skf = StratifiedKFold(n_splits=self.cfg.kfolds, shuffle=True, random_state=self.cfg.seed)
        folds = []
        subs = np.array(self.subject_ids)
        labs = np.array(self.labels)
        for trval_idx, test_idx in skf.split(subs, labs):
            # Further split training+validation set into train and val with stratification
            # Validation size is 10% of total, so proportion in trval is 0.1 / 0.8 = 0.125
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=self.cfg.seed)
            tr_idx, val_idx = next(sss.split(trval_idx, labs[trval_idx]))
            tr_idx = trval_idx[tr_idx]
            val_idx = trval_idx[val_idx]
            folds.append((tr_idx, val_idx, test_idx))
        return folds

    def get_subject_pairs(self, indices):
        """
        Convert subject indices to lists of subject directories and corresponding labels.

        Args:
            indices (Iterable[int]): Indices of subjects.

        Returns:
            Tuple[List[str], List[int]]: Lists of subject directories and labels.
        """
        pairs = []
        labels = []
        for idx in indices:
            sub = self.subjects[idx]
            if len(sub.npz_paths) == 0:
                continue
            # Use directory of first npz file as subject directory
            subject_dir = os.path.dirname(sub.npz_paths[0])
            if subject_dir and os.path.isdir(subject_dir):
                pairs.append(subject_dir)
                labels.append(sub.diagnosis)
        return pairs, labels

    def evaluate_subject_level(self, model, loader, desc='Eval'):
        """
        Evaluate model predictions aggregated at subject level.

        Args:
            model (nn.Module): Trained model.
            loader (DataLoader): DataLoader for evaluation data.
            desc (str): Description for progress bar.

        Returns:
            dict: Dictionary containing sample-level and subject-level metrics,
                  aggregated logits, labels, predictions, and subject IDs.
        """
        model.eval()
        subj_logits, subj_labels = {}, {}
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for x, y, seq_idx, ch_idx, mask, ori_len, names in pbar:
                x = x.cuda(non_blocking=True)
                seq_idx = seq_idx.cuda(non_blocking=True)
                ch_idx = ch_idx.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True).bool()
                ori_len_list = ori_len.tolist()
                
                # Forward pass to obtain logits
                out = model(x, mask, ch_idx, seq_idx, ori_len_list)
                # Average logits over sequence dimension and apply softmax
                logits = out.mean(dim=1).softmax(dim=-1).cpu().numpy()
                labs = y.numpy()
                
                # Accumulate logits and labels per subject
                for i, sid in enumerate(names):
                    subj_logits.setdefault(sid, []).append(logits[i])
                    subj_labels[sid] = int(labs[i][0]) if labs.ndim == 2 else int(labs[i])

        # Aggregate logits per subject by averaging
        sids = sorted(subj_logits.keys())
        logits_agg = np.stack([np.mean(subj_logits[sid], axis=0) for sid in sids], axis=0)
        labs_agg = np.array([subj_labels[sid] for sid in sids])
        preds_agg = logits_agg.argmax(axis=1)
        
        # Compute sample-level metrics treating each window independently
        all_logits = np.concatenate([np.stack(subj_logits[sid]) for sid in sids])
        all_labs = np.concatenate([[subj_labels[sid]] * len(subj_logits[sid]) for sid in sids])
        all_preds = all_logits.argmax(axis=1)
        
        sample_metrics = compute_metrics(all_labs, all_preds)
        subject_metrics = compute_metrics(labs_agg, preds_agg)
        
        return {
            'cls_sample': sample_metrics,
            'cls_subject': subject_metrics,
            'subject_logits': logits_agg,
            'subject_labels': labs_agg,
            'subject_preds': preds_agg,
            'subject_ids': sids
        }

    def run_fold(self, fold_id: int, tr_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        """
        Run training and evaluation for a single fold.

        Args:
            fold_id (int): Fold index.
            tr_idx (np.ndarray): Training subject indices.
            val_idx (np.ndarray): Validation subject indices.
            test_idx (np.ndarray): Test subject indices.
        """
        run_dir = os.path.join(self.cfg.run_root, f'fold{fold_id}')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'model_dir'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'predicts'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'splits'), exist_ok=True)
        logger = SummaryWriter(os.path.join(run_dir, 'logs'))

        # Get subject directories and labels for each split
        train_dirs, train_labels = self.get_subject_pairs(tr_idx)
        val_dirs, val_labels = self.get_subject_pairs(val_idx)
        test_dirs, test_labels = self.get_subject_pairs(test_idx)

        # Build dataloaders for train, val, and test sets
        loaders = build_dataloaders_nar(
            train_dirs, train_labels,
            val_dirs, val_labels,
            test_dirs, test_labels,
            seq_len=self.cfg.seq_len,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            train_stride=self.cfg.train_stride,
            val_stride=self.cfg.val_stride,
            test_stride=self.cfg.test_stride,
        )

        # Save split subject information to JSON and CSV files
        try:
            split_payload = {
                'fold': fold_id,
                'train': [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in tr_idx],
                'val':   [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in val_idx],
                'test':  [{'id': self.subject_ids[i], 'label': int(self.labels[i])} for i in test_idx],
            }
            _write_json(os.path.join(run_dir, 'splits', 'subjects.json'), split_payload)
            _write_csv(os.path.join(run_dir, 'splits', 'train_subjects.csv'),
                       [[self.subject_ids[i], int(self.labels[i])] for i in tr_idx], header=['subject_id', 'label'])
            _write_csv(os.path.join(run_dir, 'splits', 'val_subjects.csv'),
                       [[self.subject_ids[i], int(self.labels[i])] for i in val_idx], header=['subject_id', 'label'])
            _write_csv(os.path.join(run_dir, 'splits', 'test_subjects.csv'),
                       [[self.subject_ids[i], int(self.labels[i])] for i in test_idx], header=['subject_id', 'label'])
        except Exception as e:
            print(f"[WARN] failed to write split subjects: {e}")

        # Compute and print split statistics for windows and batches
        train_windows = len(loaders['train'].dataset)
        train_batches = int(np.ceil(train_windows / self.cfg.batch_size))
        val_windows = len(loaders['val'].dataset)
        val_batches = int(np.ceil(val_windows / self.cfg.batch_size))
        test_windows = len(loaders['test'].dataset)
        test_batches = int(np.ceil(test_windows / self.cfg.batch_size))
        print(
            f"Split stats | train: windows={train_windows}, batches={train_batches}; "
            f"val: windows={val_windows}, batches={val_batches}; "
            f"test: windows={test_windows}, batches={test_batches}"
        )
        try:
            stats_payload = {
                'train': {'windows': int(train_windows), 'batches': int(train_batches)},
                'val':   {'windows': int(val_windows),   'batches': int(val_batches)},
                'test':  {'windows': int(test_windows),  'batches': int(test_batches)},
            }
            _write_json(os.path.join(run_dir, 'splits', 'split_stats.json'), stats_payload)
        except Exception as e:
            print(f"[WARN] failed to write split stats: {e}")

        # Optionally create a train evaluation loader with stride=1 for reporting training metrics
        train_eval_loader = None
        if self.cfg.enable_train_eval:
            from torch.utils.data import DataLoader
            from .online_dataloader import WindowedDatasetNar, collate_fn_nar
            train_eval_dataset = WindowedDatasetNar(train_dirs, train_labels, self.cfg.seq_len, stride=1)
            train_eval_loader = DataLoader(
                train_eval_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn_nar,
            )

        # Build model, optimizer, and scheduler
        model, optimizer, scheduler = self.build_model()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        
        # Prepare classification loss with optional class weights
        class_weight = self.make_class_weight(train_labels)
        criterion_cls = nn.CrossEntropyLoss(weight=class_weight)

        best_val_subject_f1 = -1.0  # Track best validation macro F1 score
        best_path = os.path.join(run_dir, 'model_dir', 'best.pth')

        # Open CSV file for logging training and validation metrics
        metrics_csv = open(os.path.join(run_dir, 'metrics', 'train_val_metrics.csv'), 'w', newline='')
        csv_writer = csv.writer(metrics_csv)
        csv_writer.writerow([
            'epoch', 'split',
            'sample_acc', 'sample_precision', 'sample_recall', 'sample_f1', 'sample_macro_f1',
            'subject_acc', 'subject_precision', 'subject_recall', 'subject_f1', 'subject_macro_f1'
        ])

        for epoch in range(self.cfg.epochs):
            model.train()
            loss_meter = AverageMeter()
            pbar = tqdm(loaders['train'], desc=f"Fold{fold_id} Train Epoch {epoch}")
            for x, y, seq_idx, ch_idx, mask, ori_len, names in pbar:
                # Move inputs and targets to GPU asynchronously
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                seq_idx = seq_idx.cuda(non_blocking=True)
                ch_idx = ch_idx.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True).bool()
                ori_len_list = ori_len.tolist()

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    # Forward pass
                    out = model(x, mask, ch_idx, seq_idx, ori_len_list)
                    # Compute classification loss
                    loss_cls = criterion_cls(out.view(-1, self.num_classes), y.view(-1))
                    loss = self.cfg.cls_loss_w * loss_cls
                
                # Backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    # Clip gradients to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), n=y.size(0))
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

            # Step learning rate scheduler after each epoch
            scheduler.step()
            logger.add_scalar('train/loss', loss_meter.avg, epoch)

            # Evaluate model at specified intervals
            if (epoch % max(1, self.cfg.eval_every)) == 0:
                # Evaluate on training set if enabled
                if self.cfg.enable_train_eval and train_eval_loader is not None:
                    train_metrics = self.evaluate_subject_level(model, train_eval_loader, desc='Eval-train')
                    logger.add_scalar('train/sample_acc', train_metrics['cls_sample']['accuracy'], epoch)
                    logger.add_scalar('train/subject_acc', train_metrics['cls_subject']['accuracy'], epoch)
                    logger.add_scalar('train/subject_macro_f1', train_metrics['cls_subject']['macro_f1'], epoch)

                    # Log training metrics to CSV
                    csv_writer.writerow([
                        epoch, 'train',
                        train_metrics['cls_sample']['accuracy'], train_metrics['cls_sample']['precision'], 
                        train_metrics['cls_sample']['recall'], train_metrics['cls_sample']['f1'], 
                        train_metrics['cls_sample']['macro_f1'],
                        train_metrics['cls_subject']['accuracy'], train_metrics['cls_subject']['precision'], 
                        train_metrics['cls_subject']['recall'], train_metrics['cls_subject']['f1'], 
                        train_metrics['cls_subject']['macro_f1']
                    ])

                # Evaluate on validation set
                val_metrics = self.evaluate_subject_level(model, loaders['val'], desc='Eval-val')
                logger.add_scalar('val/sample_acc', val_metrics['cls_sample']['accuracy'], epoch)
                logger.add_scalar('val/subject_acc', val_metrics['cls_subject']['accuracy'], epoch)
                logger.add_scalar('val/subject_macro_f1', val_metrics['cls_subject']['macro_f1'], epoch)

                # Log validation metrics to CSV
                csv_writer.writerow([
                    epoch, 'val',
                    val_metrics['cls_sample']['accuracy'], val_metrics['cls_sample']['precision'], 
                    val_metrics['cls_sample']['recall'], val_metrics['cls_sample']['f1'], 
                    val_metrics['cls_sample']['macro_f1'],
                    val_metrics['cls_subject']['accuracy'], val_metrics['cls_subject']['precision'], 
                    val_metrics['cls_subject']['recall'], val_metrics['cls_subject']['f1'], 
                    val_metrics['cls_subject']['macro_f1']
                ])

                # Save checkpoint for last epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics
                }, os.path.join(run_dir, 'model_dir', 'last.pth'))

                # Save validation predictions if enabled
                if self.cfg.save_preds:
                    _write_csv(os.path.join(run_dir, 'predicts', f'val_preds_epoch{epoch}.csv'),
                              [[sid, int(pred), int(lab)] for sid, pred, lab in 
                               zip(val_metrics['subject_ids'], val_metrics['subject_preds'], val_metrics['subject_labels'])],
                              header=['subject_id', 'pred', 'label'])

                # Update best model checkpoint based on validation macro F1 score
                if val_metrics['cls_subject']['macro_f1'] > best_val_subject_f1:
                    best_val_subject_f1 = val_metrics['cls_subject']['macro_f1']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'metrics': val_metrics
                    }, best_path)
                    _write_json(os.path.join(run_dir, 'metrics', 'val_best_metrics.json'), val_metrics)

                # Always save last epoch validation metrics JSON
                _write_json(os.path.join(run_dir, 'metrics', 'val_last_metrics.json'), val_metrics)

        metrics_csv.close()
        logger.close()

        # Load best checkpoint for final test evaluation
        best_ckpt = torch.load(best_path)
        model.load_state_dict(best_ckpt['model_state_dict'])
        test_metrics = self.evaluate_subject_level(model, loaders['test'], desc='Test')
        _write_json(os.path.join(run_dir, 'metrics', 'test_metrics.json'), test_metrics)
        
        # Save test predictions if enabled
        if self.cfg.save_preds:
            _write_csv(os.path.join(run_dir, 'predicts', 'test_preds.csv'),
                      [[sid, int(pred), int(lab)] for sid, pred, lab in 
                       zip(test_metrics['subject_ids'], test_metrics['subject_preds'], test_metrics['subject_labels'])],
                      header=['subject_id', 'pred', 'label'])

        print(f"Fold {fold_id} completed. Best val macro F1: {best_val_subject_f1:.4f}, "
              f"Test macro F1: {test_metrics['cls_subject']['macro_f1']:.4f}")

    def run(self):
        """
        Execute training and evaluation across all folds.

        Performs stratified k-fold splitting and runs training/evaluation for each fold.
        """
        folds = self.split_kfolds()
        for fold_id, (tr_idx, val_idx, test_idx) in enumerate(folds):
            print(f"\n=== Starting Fold {fold_id} ===")
            self.run_fold(fold_id, tr_idx, val_idx, test_idx)
        print("\n=== All folds completed ===")
