# -*- coding: utf-8 -*-
"""
train.py

CLI entry point for MNC narcolepsy fine-tuning (3-class or Normal vs Abnormal).

Delegates all training logic to ``cls_core.trainer.PooledTrainer``. This
module only defines argparse defaults specific to the MNC tasks and wires in
the task's ``load_subjects`` implementation and label names.
"""

import argparse
import os

from cls_core.trainer import PooledTrainer, TrainConfig

from .dataset import DEFAULT_DATA_ROOT, TASKS, load_subjects


def parse_args():
    p = argparse.ArgumentParser(description='MNC narcolepsy classification')

    # -- Model architecture (match pretrained LPSGM backbone) --
    p.add_argument('--architecture', type=str, default='cat_cls')
    p.add_argument('--ch_emb_dim', type=int, default=32)
    p.add_argument('--seq_emb_dim', type=int, default=64)
    p.add_argument('--num_transformer_blocks', type=int, default=4)
    p.add_argument('--transformer_num_heads', type=int, default=8)
    p.add_argument('--transformer_dropout', type=float, default=0.0)
    p.add_argument('--transformer_attn_dropout', type=float, default=0.0)
    p.add_argument('--epoch_encoder_dropout', type=float, default=0.0)
    p.add_argument('--ch_num', type=int, default=9)
    p.add_argument('--seq_len', type=int, default=20)
    p.add_argument('--clamp_value', type=float, default=10.0)

    # -- Task --
    p.add_argument('--task', type=str, default='3class', choices=sorted(TASKS),
                   help="'3class' (Control / NT1 / Other Hypersomnia) or "
                        "'binary' (Normal vs Abnormal); sets the number of classes.")

    # -- Data --
    p.add_argument('--label_csv', type=str, default='')
    p.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--train_stride', type=int, default=5)
    p.add_argument('--val_stride', type=int, default=5)
    p.add_argument('--test_stride', type=int, default=5)

    # -- Cross-validation --
    p.add_argument('--kfolds', type=int, default=5)
    p.add_argument('--val_fraction', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fold_ids', type=str, default='',
                   help='Optional comma-separated fold indices (e.g. "0,1,2"); '
                        'defaults to running all folds.')

    # -- Optimization --
    p.add_argument('--pretrained_path', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..',
                                        'weights', 'ched32_seqed64_ch9_seql20_block4.pth'))
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--lr_backbone', type=float, default=1e-6)
    p.add_argument('--lr_head', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--eta_min', type=float, default=1e-8)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--class_weight', type=str, default='auto',
                   help="'none', 'auto', or 'w0,w1,...' comma-separated weights.")
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--freeze_backbone', action='store_true', default=False)

    # -- Output --
    p.add_argument('--run_root', type=str, default=None,
                   help="Defaults to './run_nar' (3class) or './run_nar_binary' (binary).")
    p.add_argument('--save_preds', action='store_true', default=True)

    return p.parse_args()


def main():
    args = vars(parse_args())
    task = args.pop('task')
    if args['run_root'] is None:
        args['run_root'] = './run_nar' if task == '3class' else './run_nar_binary'
    num_classes, label_names = TASKS[task]
    cfg = TrainConfig(num_classes=num_classes, **args)
    subjects = load_subjects(cfg.data_root, task=task)
    trainer = PooledTrainer(cfg, subjects, label_names=label_names)
    trainer.run()


if __name__ == '__main__':
    main()
