#!/bin/bash
# APPLES Depression Classification (binary: Depressed vs Non-depressed)
# 5-fold CV with 25% of each fold's training set held out for validation;
# pooled LPSGM backbone fine-tune -> frozen-backbone linear probing.
#
# Usage:
#   bash dep_cls/run_dep.sh                 # run all 5 folds
#   bash dep_cls/run_dep.sh 0,1,2           # only folds 0, 1, 2
#   CUDA_VISIBLE_DEVICES=1 bash dep_cls/run_dep.sh 3,4
set -e

FOLD_IDS="${1:-}"
RUN_ROOT="./run_dep"

# --- Stage 1: backbone fine-tuning (5-fold + 25% val split) ---
python -m dep_cls.train \
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
    --val_stride 1 \
    --test_stride 1 \
    --class_weight auto

# --- Stage 2: frozen-backbone linear probing ---
if [ -z "$FOLD_IDS" ]; then
    EVAL_FOLDS="0 1 2 3 4"
else
    EVAL_FOLDS="${FOLD_IDS//,/ }"
fi

for FOLD in $EVAL_FOLDS; do
    python -m dep_cls.simple_eval \
        --fold "$FOLD" \
        --run-root "$RUN_ROOT" \
        --num-classes 2 \
        --kfolds 5 \
        --seed 42
done
