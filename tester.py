#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tester.py

This script provides an iterative testing and evaluation framework for the LPSGM (Large Polysomnography Model) project.
It supports loading pretrained LPSGM models to perform sleep staging inference on polysomnography (PSG) recordings.
Key functionalities include:
- Preparing input sequences from multi-channel PSG signals for model inference.
- Performing model inference with sequence-level voting to improve prediction robustness.
- Computing detailed sleep staging metrics per subject and overall.
- Logging structured evaluation results in JSONL format for traceability and reproducibility.
- Supporting flexible channel configurations and batch processing for efficient evaluation.
- Designed to handle large-scale PSG datasets with multiple subjects and recordings.

This file plays a critical role in validating model performance on unseen test data, facilitating cross-center generalization assessment,
and enabling comprehensive benchmarking of sleep staging and mental disorder diagnosis capabilities.
"""

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
import json

from model.model import LPSGM
from preprocess.utils import *
from dataset.dataset import *
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


def sequence_voting(logits, N, seq_len):
    """
    Perform sequence-level voting by aggregating overlapping logits to produce final epoch predictions.

    Args:
        logits (np.ndarray): Array of shape (num_sequences, seq_len, 5) containing model output logits per sequence.
        N (int): Total number of epochs in the original signal.
        seq_len (int): Length of each input sequence (number of epochs).

    Returns:
        list: Final predicted sleep stage labels for each epoch after voting.
    """
    voted_logits = np.zeros((N, 5))  # Initialize accumulation array for logits per epoch and class
    for i in range(len(logits)):
        logits_i = logits[i]
        for j in range(seq_len):
            if i + j < N:
                voted_logits[i + j] += logits_i[j]  # Aggregate logits for overlapping epochs
    voted_predicts = np.argmax(voted_logits, axis=-1)  # Select class with highest aggregated logit per epoch
    return voted_predicts.tolist()


@torch.no_grad()
def test_recodings(sig, ch_id, model, args, desc=None):
    """
    Perform inference on PSG recordings by preparing input sequences, running the model, and aggregating predictions.

    Args:
        sig (np.ndarray): PSG signal array of shape (num_epochs, 3000, num_channels).
        ch_id (np.ndarray): Array of channel indices corresponding to the channels in 'sig'.
        model (torch.nn.Module): Pretrained LPSGM model for inference.
        args (argparse.Namespace): Configuration arguments including sequence length and batch size.
        desc (str, optional): Description string for progress bar display.

    Returns:
        list: Predicted sleep stage labels for each epoch in the recording.
    """
    # Transpose signal to shape (num_epochs, num_channels, 3000) for processing
    seq = sig.transpose(0, 2, 1)  # (N, cn, 3000)

    # Construct overlapping sequences of length seq_len along the epoch dimension
    seq = np.stack([seq[i:i + args.seq_len] for i in range(len(seq) - args.seq_len + 1)], axis=0)  # (seqn, seql, cn, 3000)

    # Reshape sequences to merge channel and sequence length dimensions for model input
    seq = seq.reshape(-1, args.seq_len * len(ch_id), 3000)  # (seqn, seql*cn, 3000)

    batch_num = ceil(len(seq) / args.batch_size)  # Calculate total number of batches

    model.eval()  # Set model to evaluation mode
    prediction = []
    for batch_idx in tqdm(range(batch_num), desc):
        seq_batch = seq[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        batch_size = len(seq_batch)

        # Construct sequence indices tensor indicating position within each sequence
        seq_idx = np.arange(args.seq_len).reshape(1, args.seq_len, 1)  # (1, seql, 1)
        seq_idx = np.tile(seq_idx, (batch_size, 1, len(ch_id)))  # (batch_size, seql, cn)
        seq_idx = np.reshape(seq_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn)

        # Construct channel indices tensor indicating channel identity for each segment
        ch_idx = ch_id[np.newaxis, np.newaxis, :]  # (1, 1, cn)
        ch_idx = np.tile(ch_idx, (batch_size, args.seq_len, 1))  # (batch_size, seql, cn)
        ch_idx = np.reshape(ch_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn)

        # Convert numpy arrays to torch tensors and move to GPU
        seq_batch = torch.tensor(seq_batch, dtype=torch.float32).cuda()
        seq_idx = torch.tensor(seq_idx, dtype=torch.int64).cuda()
        ch_idx = torch.tensor(ch_idx, dtype=torch.int64).cuda()

        # Create mask tensor indicating valid positions (all valid here)
        mask = torch.zeros((batch_size, args.seq_len * len(ch_id)), dtype=torch.int64).bool().cuda()

        ori_len = [args.seq_len * len(ch_id)] * batch_size  # Original lengths for each sequence in batch

        # Forward pass through the model to obtain logits for each epoch
        logits = model(seq_batch, mask, ch_idx, seq_idx, ori_len)  # (batch_size, seql*cn, 5)

        logits = torch.softmax(logits, dim=-1)  # Apply softmax to obtain probabilities

        logits = logits.cpu().numpy()  # Move logits back to CPU for further processing
        prediction.append(logits)

    prediction = np.concatenate(prediction, axis=0)  # Concatenate batch predictions (seqn, seql, 5)

    # Aggregate overlapping sequence predictions to produce final epoch-level predictions
    prediction = sequence_voting(prediction, len(sig), args.seq_len)  # (N, )
    return prediction


def Tester(args, model, channels=None, verbose=False):
    """
    Main testing function to evaluate the LPSGM model on specified test subjects and channels.

    This function:
    - Maps channel names to indices if necessary.
    - Iterates over test subjects and their recordings.
    - Loads PSG data and hypnograms.
    - Performs model inference and collects predictions.
    - Computes per-subject and overall evaluation metrics.
    - Saves predictions if configured.
    - Logs detailed results in JSONL format for reproducibility.

    Args:
        args (argparse.Namespace): Configuration arguments including test subjects, batch size, etc.
        model (torch.nn.Module): Pretrained LPSGM model for inference.
        channels (list or None): List of channel names or indices to use; defaults to all channels.
        verbose (bool): If True, prints detailed per-subject metrics.

    Returns:
        tuple: Overall evaluation metrics including accuracy, F1 score, confusion matrix, class-wise F1 scores, and Cohen's kappa.
    """
    # Preserve original channel argument for logging
    original_channels_arg = channels

    # Map channel names to indices if necessary
    if channels is not None:
        mapped = []
        for ch in channels:
            if isinstance(ch, str):
                mapped.append(CHANNEL_TO_INDEX[ch])
            else:
                mapped.append(ch)
        channels = mapped

    # Default to all available channels if none specified
    if channels is None:
        channels = range(args.ch_num)

    # Log test subjects information
    print(f"Test Subjects: {len(args.test_subjects)}")
    print("Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    start_time = time.time()

    PREDICTS, LABELS = [], []
    subject_records = []  # Store metrics for each subject
    center_name = None

    # Iterate over each test subject directory and its associated center
    for sub_dir in [sub for (sub, _) in args.test_subjects]:
        predicts, labels = [], []

        # Iterate over all sequences (recordings) in the subject directory
        for seq_name in os.listdir(sub_dir):
            seq_path = os.path.join(sub_dir, seq_name)
            npz = np.load(seq_path)

            # Load hypnogram labels (sleep stages) for the recording
            ano = npz['Hypnogram'].astype(np.int64)  # (L, )

            # Collect available channels and their data from the NPZ file
            available_channels = []
            channel_data_list = []
            channel_indices = []

            for ch_name in ALL_CHANNELS:
                if ch_name in npz.files:
                    available_channels.append(ch_name)
                    channel_data_list.append(npz[ch_name].astype(np.float32))  # (L, 3000)
                    channel_indices.append(CHANNEL_TO_INDEX[ch_name])

            if len(available_channels) == 0:
                print(f"Warning: No valid channels found in {seq_path}")
                continue

            # Stack channel data to form signal array of shape (L, 3000, cn)
            sig = np.stack(channel_data_list, axis=2)  # (L, 3000, cn)
            ch_id = np.array(channel_indices, dtype=np.int64)  # (cn, )

            # Select channels according to the specified channel indices
            channel_indices = np.where(np.isin(ch_id, channels))[0]
            sig = sig[:, :, channel_indices]  # (L, 3000, cn')
            ch_id = ch_id[channel_indices]  # (cn', )

            try:
                # Perform inference on the current recording
                pred = test_recodings(sig, ch_id, model, args, seq_path)  # (L, )
            except Exception as e:
                print(f"Error: {seq_path}, {e}")
                continue

            predicts.extend(pred)
            labels.extend(ano.tolist())

        # Compute and save per-subject metrics regardless of verbosity to maintain complete logs
        predicts_arr = np.array(predicts).squeeze()
        labels_arr = np.array(labels).squeeze()
        if len(predicts_arr) > 0 and len(labels_arr) > 0:
            acc_s, f1_s, cm_s, wake_f1_s, n1_f1_s, n2_f1_s, n3_f1_s, rem_f1_s, kappa_s = get_metric(labels_arr, predicts_arr)
            subject_record = {
                'subject_dir': sub_dir,
                'subject_id': os.path.basename(sub_dir).split('.')[0],
                'num_epochs': int(len(predicts_arr)),
                'metrics': {
                    'acc': float(acc_s),
                    'f1': float(f1_s),
                    'kappa': float(kappa_s),
                    'wake_f1': float(wake_f1_s),
                    'n1_f1': float(n1_f1_s),
                    'n2_f1': float(n2_f1_s),
                    'n3_f1': float(n3_f1_s),
                    'rem_f1': float(rem_f1_s),
                    'confusion_matrix': np.array(cm_s).astype(int).tolist(),
                }
            }
            subject_records.append(subject_record)

            if verbose:
                print("="*50)
                print(f"{sub_dir}:\nacc: {acc_s}, f1: {f1_s} \ncm:\n{cm_s} \nwake_f1: {wake_f1_s}, n1_f1: {n1_f1_s}, n2_f1: {n2_f1_s}, n3_f1: {n3_f1_s}, rem_f1: {rem_f1_s}, kappa: {kappa_s}")

        # Save predictions and labels per subject if configured
        if args.save_pred:
            save_sub_dir = os.path.join(args.save_dir, os.path.basename(sub_dir).split('.')[0])
            os.makedirs(save_sub_dir, exist_ok=True)
            np.save(os.path.join(save_sub_dir, "predicts.npy"), predicts_arr)
            np.save(os.path.join(save_sub_dir, "labels.npy"), labels_arr)

        PREDICTS.extend(predicts)
        LABELS.extend(labels)

    # Aggregate all predictions and labels across subjects for overall metrics
    PREDICTS = np.array(PREDICTS).squeeze()
    LABELS = np.array(LABELS).squeeze()

    acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = get_metric(LABELS, PREDICTS)

    end_time = time.time()

    # Console output with unified formatting for overall inference results
    print('\n'*3, "="*30, "Inference", "="*30)
    print(f"Channels(indices): {channels}")
    print("acc: {:.5f}, f1: {:.5f}, kappa: {:.5f}".format(acc, f1, kappa))
    print("cm: \n{}".format(cm))
    print("wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
        wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,
    ))
    print(f"Avg Time per Subject: {(end_time-start_time)/len(args.test_subjects):.2f} s")

    # -------- JSON log writing -------- #
    # Infer center name from test subjects if not set
    if center_name is None and len(args.test_subjects) > 0:
        center_name = args.test_subjects[0][1]
    center_name = center_name or 'UNKNOWN'

    # Determine run root directory based on weights file or save directory
    weights_file = getattr(args, 'weights_file', None)
    run_root = None
    if weights_file is not None:
        wf_dir = os.path.dirname(weights_file)
        if os.path.basename(wf_dir) == 'model_dir':
            run_root = os.path.dirname(wf_dir)
        else:
            run_root = wf_dir
    else:
        # Fallback to parent directory of save_dir if available
        if getattr(args, 'save_dir', None):
            run_root = os.path.dirname(args.save_dir)
    run_root = run_root or os.getcwd()

    log_file = os.path.join(run_root, f"{center_name}_metrics.json")

    overall_record = {
        'acc': float(acc),
        'f1': float(f1),
        'kappa': float(kappa),
        'wake_f1': float(wake_f1),
        'n1_f1': float(n1_f1),
        'n2_f1': float(n2_f1),
        'n3_f1': float(n3_f1),
        'rem_f1': float(rem_f1),
        'confusion_matrix': np.array(cm).astype(int).tolist(),
    }

    # Helper function to safely serialize args to JSON-compatible format
    def _safe(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        # Convert test_subjects list of tuples to list of lists for JSON compatibility
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
            return [list(item) for item in v]
        return str(v)

    args_snapshot = {k: _safe(v) for k, v in args.__dict__.items() if not k.startswith('__')}

    run_record = {
        'timestamp': datetime.now().isoformat(),
        'center': center_name,
        'weights_file': weights_file,
        'channels_original': list(original_channels_arg) if original_channels_arg is not None else None,
        'channel_indices': list(channels),
        'num_subjects': len(args.test_subjects),
        'duration_seconds': float(end_time - start_time),
        'avg_time_per_subject_sec': float((end_time - start_time)/len(args.test_subjects)),
        'args': args_snapshot,
        'save_pred': bool(getattr(args, 'save_pred', False)),
        'predict_save_dir': getattr(args, 'save_dir', None),
        'per_subject': subject_records,
        'overall': overall_record,
    }

    try:
        # Read existing records if file exists, otherwise start with empty list
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        else:
            records = []
        # Append new record and write back with indentation for readability
        records.append(run_record)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Results saved to: {log_file}")
    except Exception as e:
        print(f"[LOG][ERROR] Failed to write log: {e}")

    return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa

 
def pt_test(args: object, 
            center: str, 
            weights_file: str,
            channels: tuple,
            verbose=False,
        ):
    """
    Load pretrained LPSGM model and perform testing on specified center and channels.

    Args:
        args (object): Configuration object containing parameters and paths.
        center (str): Name of the dataset center to test on.
        weights_file (str): Path to the pretrained model weights file.
        channels (tuple): Tuple of channel names to use for testing.
        verbose (bool): If True, enables detailed logging during testing.

    Returns:
        None
    """
    # Append current timestamp to cache root directory to avoid conflicts
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.cache_root = os.path.join(args.cache_root, current_time)

    # Retrieve list of subject directories for the specified center
    subjects, _ = get_datasets_subjects([center])

    # Prepare test subjects list as tuples of (subject_dir, center)
    args.test_subjects = [(sub, center) for sub in subjects]

    # Initialize model and load pretrained weights
    model = nn.DataParallel(LPSGM(args)).cuda()
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])

    print(f"Model Weights: {weights_file}")

    # Store weights file path in args for logging purposes
    args.weights_file = weights_file

    # Run testing procedure
    Tester(args, model, 
        channels, 
        verbose=verbose,
    )


if __name__ == "__main__":

    class args(object):
        # Model architecture and hyperparameters
        architecture = 'cat_cls'    # Model architecture type
        epoch_encoder_dropout = 0
        transformer_num_heads = 8
        transformer_dropout = 0
        transformer_attn_dropout = 0
        ch_num = 9
        seq_len = 20
        ch_emb_dim = 32
        seq_emb_dim = 64
        num_transformer_blocks = 4
        clamp_value = 10

        # Data loading and processing parameters
        batch_size = 128
        num_workers = 64
        num_processes = 100
        cache_root = r".cache/"

        # Prediction saving options
        save_pred = False
        save_dir = None
    


################################ Pretrained Test ################################
    center = "HANG7"
    channels = ('F3','F4','C3','C4','O1','O2','E1','E2','Chin')

    run_name = r"run/Oct31_20-25-22" 
    weights_file = r"run/Oct31_20-25-22/model_dir/eval_best.pth" 

    if args.save_dir is None:
        args.save_dir = os.path.join(run_name, center)

    pt_test(
        args,
        center=center,
        weights_file=weights_file,
        channels=channels,
        verbose=False,
    )


################################ Pretrained Test ################################
    center = "SYSU"
    channels = ('F3','F4','C3','C4','O1','O2','E1','E2','Chin')

    run_name = r"run/Oct31_20-25-22" 
    weights_file = r"run/Oct31_20-25-22/model_dir/eval_best.pth" 

    if args.save_dir is None:
        args.save_dir = os.path.join(run_name, center)

    pt_test(
        args,
        center=center,
        weights_file=weights_file,
        channels=channels,
        verbose=False,
    )
