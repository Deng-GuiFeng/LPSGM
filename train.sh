# !/bin/bash

python train.py \
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
    --num_transformer_blocks 6 \
    --eval_size 0.1 \
    --batch_size 32 \
    --num_workers 32 \
    --num_processes 100 \
    --epochs 25 \
    --warmup_epochs 10 \
    --lr0 1e-4 \
    --weight_decay 1e-4 \
    --eta_min 1e-8 \
    --clip_value 1 \
    --clamp_value 10 \
    --target_domain HANG7,SYSU \
    --cache_root .cache \
    --random_shift_len 0 \
    --source_domains APPLES,DCSM,DOD-H,DOD-O,HMC,ISRUC,P2018,SHHS-1,SHHS-2,STAGES-BOGN,STAGES-STNF,STAGES-MSTR,STAGES-GSDV,STAGES-GSBB,STAGES-GSLH,STAGES-GSSA,STAGES-GSSW,STAGES-MSMI,STAGES-MSNF,STAGES-MSQW,STAGES-MSTH,STAGES-STLK,ABC,NCHSDB,HOMEPAP,SVUH,CHAT,CCSHS,CFS,MROS  \
    --save_epoch \
    --save_pred 
    # --save_dir 



