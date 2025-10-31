#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m nar_cls.train_narcolepsy \
  --seq_len 20 \
  --batch_size 128 \
  --num_workers 64 \
  --kfolds 5 \
  --seed 42 \
  --train_stride 1 \
  --val_stride 1 \
  --test_stride 1 \
  --pretrained_path weights/ched32_seqed64_ch9_seql20_block4.pth \
  --architecture cat_cls \
  --epoch_encoder_dropout 0.1 \
  --transformer_num_heads 8 \
  --transformer_dropout 0.1 \
  --transformer_attn_dropout 0.1 \
  --ch_num 9 \
  --ch_emb_dim 32 \
  --seq_emb_dim 64 \
  --num_transformer_blocks 4 \
  --clamp_value 10.0 \
  --epochs 5 \
  --warmup_epochs 3 \
  --lr_backbone 1e-5 \
  --lr_head 1e-3 \
  --weight_decay 1e-4 \
  --eta_min 1e-8 \
  --grad_clip 1.0 \
  --class_weight auto \
  --amp \
  --eval_every 1 \
  --enable_train_eval False \
  --cls_loss_w 1.0 \
  --run_root ./run_nar \
  --save_preds \
  --merge_NT1
