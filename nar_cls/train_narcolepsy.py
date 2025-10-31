#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for finetuning the LPSGM model on narcolepsy diagnosis.

This script configures and initiates the training process for narcolepsy detection 
using the Large Polysomnography Model (LPSGM). It parses command-line arguments 
to set up model architecture, data loading parameters, training hyperparameters, 
and runtime options. The script then creates a training configuration object 
and launches the NarcolepsyTrainer to perform model training and evaluation.

This file is part of the LPSGM project, which leverages large-scale polysomnography 
data for sleep staging and mental disorder diagnosis through a dual-branch CNN 
epoch encoder and transformer-based sequence encoder.
"""

import os
import argparse
from datetime import datetime

from .trainer_narcolepsy import TrainConfig, NarcolepsyTrainer


def parse_args():
    """
    Parse command-line arguments for narcolepsy finetuning configuration.

    Returns:
        argparse.Namespace: Parsed arguments with training and model parameters.
    """
    p = argparse.ArgumentParser(description='LPSGM Narcolepsy Finetuning')

    # Model architecture and hyperparameters
    p.add_argument('--pretrained_path', type=str, required=True,
                   help='Path to the pretrained LPSGM model checkpoint')
    p.add_argument('--architecture', type=str, default='cat_cls',
                   help='Model architecture variant to use')
    p.add_argument('--epoch_encoder_dropout', type=float, default=0.0,
                   help='Dropout rate for the epoch encoder')
    p.add_argument('--transformer_num_heads', type=int, default=8,
                   help='Number of attention heads in transformer blocks')
    p.add_argument('--transformer_dropout', type=float, default=0.0,
                   help='Dropout rate for transformer layers')
    p.add_argument('--transformer_attn_dropout', type=float, default=0.0,
                   help='Dropout rate for transformer attention layers')
    p.add_argument('--ch_num', type=int, default=9,
                   help='Number of input channels for the model')
    p.add_argument('--seq_len', type=int, default=20,
                   help='Sequence length (number of epochs per input sequence)')
    p.add_argument('--ch_emb_dim', type=int, default=32,
                   help='Embedding dimension for channel features')
    p.add_argument('--seq_emb_dim', type=int, default=64,
                   help='Embedding dimension for sequence features')
    p.add_argument('--num_transformer_blocks', type=int, default=4,
                   help='Number of transformer blocks in the sequence encoder')
    p.add_argument('--clamp_value', type=float, default=10.0,
                   help='Clamp value to limit feature magnitudes')

    # Data loading parameters
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for training and evaluation')
    p.add_argument('--num_workers', type=int, default=32,
                   help='Number of worker threads for data loading')
    p.add_argument('--kfolds', type=int, default=5,
                   help='Number of folds for cross-validation')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--train_stride', type=int, default=1,
                   help='Stride for sliding window during training data sampling')
    p.add_argument('--val_stride', type=int, default=1,
                   help='Stride for sliding window during validation data sampling')
    p.add_argument('--test_stride', type=int, default=1,
                   help='Stride for sliding window during test data sampling')
    p.add_argument('--merge_NT1', action='store_true',
                   help='Flag to merge NT1 sleep stage with other classes during training')

    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=20,
                   help='Total number of training epochs')
    p.add_argument('--warmup_epochs', type=int, default=3,
                   help='Number of warmup epochs for learning rate scheduling')
    p.add_argument('--lr_backbone', type=float, default=1e-5,
                   help='Learning rate for the backbone (pretrained) network')
    p.add_argument('--lr_head', type=float, default=1e-4,
                   help='Learning rate for the classification head')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='Weight decay (L2 regularization) coefficient')
    p.add_argument('--eta_min', type=float, default=1e-8,
                   help='Minimum learning rate for scheduler')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='Maximum gradient norm for gradient clipping')
    p.add_argument('--class_weight', type=str, default='none',
                   help='Class weighting scheme for loss calculation')
    p.add_argument('--amp', action='store_true',
                   help='Enable automatic mixed precision training')
    p.add_argument('--freeze_backbone', action='store_true',
                   help='Freeze backbone parameters during finetuning')
    p.add_argument('--eval_every', type=int, default=1,
                   help='Frequency (in epochs) to perform evaluation during training')

    # Helper function to parse boolean values from string inputs
    def str2bool(v):
        """
        Convert a string to a boolean value.

        Args:
            v (str or bool): Input value to convert.

        Returns:
            bool: Converted boolean value.

        Raises:
            argparse.ArgumentTypeError: If input cannot be interpreted as boolean.
        """
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ('yes', 'true', 't', 'y', '1'):
            return True
        if v in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    p.add_argument('--enable_train_eval', type=str2bool, default=True,
                   help='Enable evaluation during training')

    # Loss weighting parameters
    p.add_argument('--cls_loss_w', type=float, default=1.0,
                   help='Weight for classification loss component')

    # Runtime and output options
    p.add_argument('--run_root', type=str, default='./run_nar',
                   help='Root directory for saving training runs')
    p.add_argument('--save_preds', action='store_true',
                   help='Flag to save model predictions during evaluation')

    args = p.parse_args()
    return args


def main():
    """
    Main entry point for narcolepsy finetuning training script.

    Parses command-line arguments, constructs training configuration,
    initializes the trainer, and starts the training process.
    """
    args = parse_args()

    # Generate a timestamped directory for the current training run
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    run_root = os.path.join(args.run_root, ts)

    # Create a training configuration object with parsed arguments
    cfg = TrainConfig(
        pretrained_path=args.pretrained_path,
        architecture=args.architecture,
        epoch_encoder_dropout=args.epoch_encoder_dropout,
        transformer_num_heads=args.transformer_num_heads,
        transformer_dropout=args.transformer_dropout,
        transformer_attn_dropout=args.transformer_attn_dropout,
        ch_num=args.ch_num,
        seq_len=args.seq_len,
        ch_emb_dim=args.ch_emb_dim,
        seq_emb_dim=args.seq_emb_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        clamp_value=args.clamp_value,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kfolds=args.kfolds,
        seed=args.seed,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        grad_clip=args.grad_clip,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride,
        class_weight=args.class_weight,
        amp=args.amp,
        freeze_backbone=args.freeze_backbone,
        eval_every=args.eval_every,
        enable_train_eval=args.enable_train_eval,
        cls_loss_w=args.cls_loss_w,
        run_root=run_root,
        save_preds=args.save_preds,
        merge_NT1=args.merge_NT1,
    )

    # Initialize the narcolepsy trainer with the configuration
    trainer = NarcolepsyTrainer(cfg)

    # Start the training and evaluation process
    trainer.run()


if __name__ == '__main__':
    main()
