# !/bin/bash

python ./finetune.py \
    --seed 20010613 \
    --architecture 'cat_cls' \
    --epoch_encoder_dropout 0.1 \
    --transformer_num_heads 8 \
    --transformer_dropout 0.1 \
    --transformer_attn_dropout 0.1 \
    --ch_num 9 \
    --seq_len 20 \
    --ch_emb_dim 32 \
    --seq_emb_dim 64 \
    --num_transformer_blocks 4 \
    --eval_size 0.2 \
    --batch_size 32 \
    --num_workers 64 \
    --num_processes 100 \
    --epochs 100 \
    --warmup_epochs 15 \
    --lr0 1e-4 \
    --weight_decay 1e-4 \
    --eta_min 1e-8 \
    --clip_value 1 \
    --clamp_value 10 \
    --random_shift_len 0 \
    --cache_root .cache \
    --state_dict_file weights/ched32_seqed64_ch9_seql20_block4.pth \
    --ft_center 'HANG7' \
    --kfolds 5 \
    --n_fold 0 \
    --run_id 'HANG7/fold0' \
    --save_pred 
    # --save_dir 

