#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import warnings
import sys
from math import ceil
from datetime import datetime
import time
import re

from model.model import LPSGM
from preprocess_data.utils import *
from data.dataset import *
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


def sequence_voting(logits, N, seq_len):
    voted_logits = np.zeros((N, 5))
    for i in range(len(logits)):
        logits_i = logits[i]
        for j in range(seq_len):
            if i + j < N:
                voted_logits[i + j] += logits_i[j]
    voted_predicts = np.argmax(voted_logits, axis=-1)
    return voted_predicts.tolist()


@torch.no_grad()
def test_recodings(sig, ch_id, model, args, desc=None):

    # 准备输入序列，注意维度变化
    seq = sig.transpose(0, 2, 1)  # (N, cn, 3000)
    seq = np.stack([seq[i:i + args.seq_len] for i in range(len(seq) - args.seq_len + 1)], axis=0)  # (seqn, seql, cn, 3000)
    seq = seq.reshape(-1, args.seq_len * len(ch_id), 3000)  # (seqn, seql*cn, 3000)
    batch_num = ceil(len(seq) / args.batch_size)

    model.eval()
    prediction = []
    for batch_idx in tqdm(range(batch_num), desc):
        seq_batch = seq[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        batch_size = len(seq_batch)
        # 构造 seq_idx 和 ch_idx
        seq_idx = np.arange(args.seq_len).reshape(1, args.seq_len, 1)  # (1, seql, 1)
        seq_idx = np.tile(seq_idx, (batch_size, 1, len(ch_id)))  # (batch_size, seql, cn)
        seq_idx = np.reshape(seq_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn, )
        ch_idx = ch_id[np.newaxis, np.newaxis, :]  # (1, 1, cn)
        ch_idx = np.tile(ch_idx, (batch_size, args.seq_len, 1))  # (batch_size, seql, cn)
        ch_idx = np.reshape(ch_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn, )
        # 转 tensor 并放到 GPU 上
        seq_batch = torch.tensor(seq_batch, dtype=torch.float32).cuda()
        seq_idx = torch.tensor(seq_idx, dtype=torch.int64).cuda()
        ch_idx = torch.tensor(ch_idx, dtype=torch.int64).cuda()
        mask = torch.zeros((batch_size, args.seq_len * len(ch_id)), dtype=torch.int64).bool().cuda()
        ori_len = [args.seq_len * len(ch_id)] * batch_size
        logits = model(seq_batch, mask, ch_idx, seq_idx, ori_len)  # (seqn, seql, 5)
        logits = torch.softmax(logits, dim=-1)  # (seqn, seql, 5)
        logits = logits.cpu().numpy()
        prediction.append(logits)

    prediction = np.concatenate(prediction, axis=0)  # (seqn, seql, 5)
    prediction = sequence_voting(prediction, len(sig), args.seq_len)  # (N, )
    return prediction


def Tester(args, model, channels=None, logger_type='txt', verbose=False):
    if channels is None:
        channels = range(args.ch_num)

    # log test subjects
    print(f"Test Subjects: {len(args.test_subjects)}")
    print("\n"*10, "Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    start_time = time.time()

    PREDICTS, LABELS = [], []
    for sub_dir in [sub for (sub, _) in args.test_subjects]:
        predicts, labels = [], []
        for seq_name in os.listdir(sub_dir):
            seq_path = os.path.join(sub_dir, seq_name)
            npz = np.load(seq_path)
            sig, ano, ch_id = npz['sig'].astype(np.float32), npz['ano'].astype(np.int64), npz['ch_id'].astype(np.int64)

            channel_indices = np.where(np.isin(ch_id, channels))[0]
            sig = sig[:, :, channel_indices]  # (N, 3000, cn')
            ch_id = ch_id[channel_indices]  # (cn', )

            try:
                pred = test_recodings(sig, ch_id, model, args, seq_path)  # (N, )
            except Exception as e:
                print(f"Error: {seq_path}, {e}")
                continue

            predicts.extend(pred)
            labels.extend(ano.tolist())

        if verbose:
            predicts = np.array(predicts).squeeze()
            labels = np.array(labels).squeeze()
            acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = get_metric(labels, predicts)

            if logger_type == 'txt':
                # txt logger
                print("="*50)
                print(f"{sub_dir}:\nacc: {acc}, f1: {f1} \ncm:\n{cm} \nwake_f1: {wake_f1}, n1_f1: {n1_f1}, n2_f1: {n2_f1}, n3_f1: {n3_f1}, rem_f1: {rem_f1}, kappa: {kappa}")
            else:
                # csv logger
                print(f"{sub_dir.split('/')[-1]}, {acc}, {f1}, {kappa}, {wake_f1}, {n1_f1}, {n2_f1}, {n3_f1}, {rem_f1}")

        if args.save_pred:
            predicts = np.array(predicts).squeeze()
            labels = np.array(labels).squeeze()
            save_sub_dir = os.path.join(args.save_dir, os.path.basename(sub_dir).split('.')[0])
            os.makedirs(save_sub_dir, exist_ok=True)
            np.save(os.path.join(save_sub_dir, "predicts.npy"), predicts)
            np.save(os.path.join(save_sub_dir, "labels.npy"), labels)

        PREDICTS.extend(predicts)
        LABELS.extend(labels)

    PREDICTS = np.array(PREDICTS).squeeze()
    LABELS = np.array(LABELS).squeeze()
    
    acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = get_metric(LABELS, PREDICTS)

    end_time = time.time()

    if logger_type == 'txt':
        # txt logger
        print('\n'*3, "="*30, "Inference", "="*30)
        print(f"Channels: {channels}")
        print("acc: {:.5f}, f1: {:.5f}, kappa: {:.5f}".format(acc, f1, kappa))
        print("cm: \n{}".format(cm))
        print("wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
            wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,
        ))
        print(f"Time: {(end_time-start_time)/len(args.test_subjects):.2f} s")
        print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(acc, f1, kappa, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,))

    else:
        # csv logger
        print(f"\nOverall, {acc}, {f1}, {kappa}, {wake_f1}, {n1_f1}, {n2_f1}, {n3_f1}, {rem_f1}")

    return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa

 
def pt_test(args: object, 
            center: str, 
            weights_file: str,
            channels: tuple,
            verbose=False,
            logger_type='txt',
        ):
    
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.cache_root=os.path.join(args.cache_root, current_time)

    if center == "HANG7":
        subjects, subtypes = get_HANG7_subjects()
    elif center == "SYSU":
        subjects, subtypes = get_SYSU_datasets()
    elif center == "QU3":
        subjects, subtypes = get_QU3_subjects()

    args.test_subjects = [(sub, center) for sub in subjects]

    model = nn.DataParallel(LPSGM(args)).cuda()
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])

    print(f"Model Weights: {weights_file}")

    Tester(args, model, 
        channels, 
        logger_type=logger_type,
        verbose = verbose,
    )


def ft_test(args: object, 
            center: str, kfolds: int, n_fold: int, seed: int,
            weights_file: str,
            channels: tuple,
            verbose=False,
            logger_type='txt',
        ):
    
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.cache_root=os.path.join(args.cache_root, current_time)

    if center == "HANG7":
        _, _, args.test_subjects = HANG7_StratifiedKFold(kfolds, n_fold, seed)
    elif center == "SYSU":
        _, _, args.test_subjects = SYSU_StratifiedKFold(kfolds, n_fold, seed)
    elif center == "QU3":
        _, _, args.test_subjects = QU3_StratifiedKFold(kfolds, n_fold, seed)

    model = nn.DataParallel(LPSGM(args)).cuda()
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])

    print(f"Model Weights: {weights_file}")

    Tester(args, model, 
        channels, 
        logger_type=logger_type,
        verbose = verbose,
    )


if __name__ == "__main__":

    class args(object):
        architecture = 'cat_cls'    # 'cat_cls'
        epoch_encoder_dropout = 0
        transformer_num_heads = 8
        transformer_dropout = 0
        transformer_attn_dropout = 0
        ch_num = 8
        seq_len = 20
        ch_emb_dim =32
        seq_emb_dim = 64
        num_transformer_blocks = 4
        clamp_value = 10

        batch_size = 128
        num_workers = 128
        num_processes = 100
        cache_root = r"/nvme2/denggf/cache/"

        save_pred = True
        save_dir = None
    

################################ Pretrained Test ################################
    logger_type = '.csv'
    center = "HANG7"
    channels = (0, 1, 2, 3, 4, 5, 6, 7)

    run_name = r"/home/denggf/Desktop/Workspace/LPSGM-main/run/Mar21_01-56-50"
    weights_file = os.path.join(run_name, 'model_dir/eval_best.pth') 
    sys.stdout = open( os.path.join(run_name, f'{center}_metrics.{logger_type}'), 'w')

    if args.save_dir is None:
        args.save_dir = os.path.join(run_name, center)

    pt_test(
        args,
        center = center,
        weights_file = weights_file,
        channels = channels,
        verbose = True,
        logger_type=logger_type,
    )


################################ Pretrained Test ################################
    logger_type = '.csv'
    center = "QU3"
    channels = (0, 1, 2, 3, 4, 5, 6, 7)

    run_name = r"/home/denggf/Desktop/Workspace/LPSGM-main/run/Mar21_01-56-50"
    weights_file = os.path.join(run_name, 'model_dir/eval_best.pth') 
    sys.stdout = open( os.path.join(run_name, f'{center}_metrics.{logger_type}'), 'w')

    if args.save_dir is None:
        args.save_dir = os.path.join(run_name, center)

    pt_test(
        args,
        center = center,
        weights_file = weights_file,
        channels = channels,
        verbose = True,
        logger_type=logger_type,
    )



# ################################ Finetuned Test ################################
#     logger_type = 'csv'
#     center = "QU3"
#     channels = (0, 1, 2, 3, 4 ,5, 6, 7)

#     kfolds = 5
#     seed = 666
#     task_dir = r"/home/denggf/Desktop/LPSGM-main/run/QU3_V2FT"

#     run_name_list = [os.path.join(task_dir, f"fold{i}", run_name) for i in range(kfolds) for run_name in os.listdir(os.path.join(task_dir, f"fold{i}"))]
    
#     for run_name in run_name_list:
#         if args.save_dir is None:
#             args.save_dir = os.path.join(run_name, "predicts")

#         n_fold = int(re.search(r'fold(\d+)', run_name).group(1))
#         weights_file = os.path.join(run_name, 'model_dir/eval_best.pth') 
#         sys.stdout = open( os.path.join(run_name, f'test.{logger_type}'), 'w')

#         ft_test(
#             args,
#             center = center,
#             kfolds = kfolds,
#             n_fold = n_fold,
#             seed = seed,
#             weights_file = weights_file,
#             channels = channels,
#             verbose = True,
#             logger_type=logger_type,
#         )