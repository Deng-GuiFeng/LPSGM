#!/bin/bash
# MNC Narcolepsy Classification
#   TASK=3class (default): Non-narcolepsy Control / Type 1 Narcolepsy / Other Hypersomnia
#   TASK=binary:           Normal vs Abnormal (Type 1 Narcolepsy and Other Hypersomnia merged)
# 5-fold CV with 25% of each fold's training set held out for validation;
# pooled LPSGM backbone fine-tune -> frozen-backbone linear probing.
#
# Usage:
#   bash nar_cls/run_nar.sh                 # 3-class, run all 5 folds
#   TASK=binary bash nar_cls/run_nar.sh     # Normal vs Abnormal, run all 5 folds
#   bash nar_cls/run_nar.sh 0,1,2           # only folds 0, 1, 2
#   CUDA_VISIBLE_DEVICES=1 bash nar_cls/run_nar.sh 3,4
set -e

FOLD_IDS="${1:-}"
TASK="${TASK:-3class}"
if [ "$TASK" = "binary" ]; then
    RUN_ROOT="./run_nar_binary"
    NUM_CLASSES=2
else
    RUN_ROOT="./run_nar"
    NUM_CLASSES=3
fi

# --- Stage 1: backbone fine-tuning (5-fold + 25% val split) ---
python -m nar_cls.train \
    --task "$TASK" \
    --run_root "$RUN_ROOT" \
    --fold_ids "$FOLD_IDS" \
    --kfolds 5 \
    --val_fraction 0.25 \
    --epochs 5 \
    --warmup_epochs 1 \
    --batch_size 32 \
    --num_workers 16 \
    --lr_backbone 1e-6 \
    --lr_head 1e-4 \
    --train_stride 5 \
    --val_stride 5 \
    --test_stride 5 \
    --class_weight auto

# --- Stage 2: frozen-backbone linear probing ---
if [ -z "$FOLD_IDS" ]; then
    EVAL_FOLDS="0 1 2 3 4"
else
    EVAL_FOLDS="${FOLD_IDS//,/ }"
fi

for FOLD in $EVAL_FOLDS; do
    python -m nar_cls.simple_eval \
        --task "$TASK" \
        --fold "$FOLD" \
        --run-root "$RUN_ROOT" \
        --num-classes "$NUM_CLASSES" \
        --kfolds 5 \
        --seed 42
done
