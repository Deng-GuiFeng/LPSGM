# -*- coding: utf-8 -*-
"""
inference_backend.py

This module provides the backend functionality for performing inference using the LPSGM (Large Polysomnography Model) 
for sleep staging and mental disorder diagnosis. It includes model loading, data preprocessing, batch inference, 
and post-processing of predictions into hypnograms. The module supports multi-GPU setups and optional subprocess 
execution to manage CUDA contexts efficiently.

Key functionalities:
- Mapping EEG/EOG/EMG channel names to model spatial embedding indices.
- Loading and preparing the LPSGM model with specified architecture and weights.
- Processing multi-channel polysomnography signals into model input format.
- Performing batch inference with sequence voting to improve prediction robustness.
- Managing device memory and CUDA cache cleanup.
- Supporting multiprocessing for inference isolation.

This file is a critical component in the LPSGM pipeline, enabling scalable and efficient inference on PSG recordings.
"""

import os
import gc
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from math import ceil
import multiprocessing as mp

from model.model import LPSGM

# Mapping from channel names to their corresponding spatial embedding indices used in the model
CHANNEL_TO_INDEX = {
    'F3': 0,
    'F4': 1,
    'C3': 2,
    'C4': 3,
    'O1': 4,
    'O2': 5,
    'E1': 6,
    'E2': 7,
    'Chin': 8,
}

# Mapping from numeric sleep stage labels to their string representations
STAGE_TO_INDEX = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

class args:
    # Configuration parameters for the LPSGM model architecture and inference
    architecture = 'cat_cls'           # Model architecture type
    ch_num = 9                        # Number of input channels
    ch_emb_dim = 32                   # Channel embedding dimension
    seq_emb_dim = 64                  # Sequence embedding dimension
    seq_len = 20                     # Length of input sequence (number of epochs)
    num_transformer_blocks = 4       # Number of transformer blocks in sequence encoder
    transformer_num_heads = 8        # Number of attention heads in transformer
    transformer_dropout = 0           # Dropout rate for transformer layers
    transformer_attn_dropout = 0      # Attention dropout rate
    epoch_encoder_dropout = 0         # Dropout rate in epoch encoder
    batch_size = 64                  # Batch size for inference
    clamp_value = 10                 # Clamp value for model inputs (if applicable)
    weights = r"weights/ched32_seqed64_ch9_seql20_block4.pth"  # Path to pretrained model weights


def sequence_voting(logits, N, seq_len):
    """
    Perform sequence voting on overlapping sequence predictions to produce final epoch predictions.

    Args:
        logits (np.ndarray): Array of shape (seqn, seql, 5) containing raw model logits for each sequence and epoch.
        N (int): Total number of epochs in the original signal.
        seq_len (int): Length of each input sequence (number of epochs).

    Returns:
        list[int]: List of predicted sleep stage indices for each epoch after voting.
    """
    voted_logits = np.zeros((N, 5))  # Initialize array to accumulate logits per epoch
    for i in range(len(logits)):
        logits_i = logits[i]
        for j in range(seq_len):
            if i + j < N:
                voted_logits[i + j] += logits_i[j]  # Aggregate logits from overlapping sequences
    voted_predicts = np.argmax(voted_logits, axis=-1)  # Select class with highest aggregated logit per epoch
    return voted_predicts.tolist()


def _build_model(device: torch.device):
    """
    Instantiate and load the LPSGM model with pretrained weights onto the specified device.

    Args:
        device (torch.device): Device on which to load the model (CPU or CUDA).

    Returns:
        torch.nn.Module: The loaded LPSGM model ready for inference.
    """
    model = LPSGM(args)
    # Wrap model with DataParallel if multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # Load pretrained weights, supporting checkpoint dict format with 'model_state_dict' key
    state = torch.load(args.weights, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model


def _inference_core(sig_dict, device: str | torch.device | None = None):
    """
    Core inference function that processes input signals, performs batch inference, and returns hypnogram predictions.

    Args:
        sig_dict (dict[str, np.ndarray]): Dictionary mapping channel names to numpy arrays of shape (N, 3000),
                                         where N is number of epochs and 3000 is samples per epoch.
        device (str or torch.device or None): Device identifier (e.g., 'cuda:0', 'cpu') or None for automatic selection.

    Returns:
        list[str]: List of predicted sleep stages as strings for each epoch.
    """
    start_time = time.time()
    # Determine device for inference
    device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    with torch.no_grad():
        model = _build_model(device)

        end_time = time.time()
        print(f"Model loaded, time cost: {end_time - start_time:.2f}s")

        ch_idx_list, sig_list = [], []
        # Collect channel indices and corresponding signals in order
        for ch_name, ch_sig in sig_dict.items():
            ch_idx_list.append(CHANNEL_TO_INDEX[ch_name])
            sig_list.append(ch_sig)
        ch_id = np.array(ch_idx_list)  # Array of channel indices, shape (cn,)
        sig = np.stack(sig_list, axis=0)  # Stack signals into array of shape (cn, N, 3000)

        # Prepare input sequences for the model
        seq = sig.transpose(1, 0, 2)  # Transpose to (N, cn, 3000)
        # Create overlapping sequences of length seq_len along epochs dimension
        seq = np.stack([seq[i:i + args.seq_len] for i in range(len(seq) - args.seq_len + 1)], axis=0)  # (seqn, seql, cn, 3000)
        # Reshape sequences to merge channel and sequence length dimensions for model input
        seq = seq.reshape(-1, args.seq_len * len(ch_id), 3000)  # (seqn, seql*cn, 3000)
        batch_num = ceil(len(seq) / args.batch_size)  # Number of batches for inference

        model.eval()
        prediction = []
        # Iterate over batches for inference
        for batch_idx in tqdm(range(batch_num), desc="Batch Inference"):
            seq_batch_np = seq[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_size = len(seq_batch_np)
            # Construct sequence indices tensor for positional embeddings
            seq_idx_np = np.arange(args.seq_len).reshape(1, args.seq_len, 1)  # (1, seql, 1)
            seq_idx_np = np.tile(seq_idx_np, (batch_size, 1, len(ch_id)))  # (batch_size, seql, cn)
            seq_idx_np = np.reshape(seq_idx_np, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn)
            # Construct channel indices tensor for spatial embeddings
            ch_idx_np = ch_id[np.newaxis, np.newaxis, :]  # (1, 1, cn)
            ch_idx_np = np.tile(ch_idx_np, (batch_size, args.seq_len, 1))  # (batch_size, seql, cn)
            ch_idx_np = np.reshape(ch_idx_np, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn)

            # Convert numpy arrays to torch tensors on the target device
            seq_batch_t = torch.as_tensor(seq_batch_np, dtype=torch.float32, device=device)
            seq_idx_t = torch.as_tensor(seq_idx_np, dtype=torch.int64, device=device)
            ch_idx_t = torch.as_tensor(ch_idx_np, dtype=torch.int64, device=device)
            mask = torch.zeros((batch_size, args.seq_len * len(ch_id)), dtype=torch.bool, device=device)  # No masking applied

            # Forward pass through the model to obtain logits
            logits_t = model(seq_batch_t, mask, ch_idx_t, seq_idx_t, None)  # Output shape: (batch_size, seql, 5)
            # Apply softmax to obtain probabilities and move to CPU numpy array
            logits_cpu = torch.softmax(logits_t, dim=-1).detach().to('cpu').numpy()
            prediction.append(logits_cpu)

            # Clear GPU memory for this batch to avoid memory leaks
            if device.type == 'cuda':
                del seq_batch_t, seq_idx_t, ch_idx_t, mask, logits_t
                torch.cuda.synchronize()

        # Concatenate batch predictions along sequence dimension
        prediction = np.concatenate(prediction, axis=0)  # (seqn, seql, 5)
        # Apply sequence voting to aggregate overlapping predictions into final epoch predictions
        prediction = sequence_voting(prediction, sig.shape[1], args.seq_len)  # (N,)
        # Convert numeric stage indices to string labels
        hypnogram = [STAGE_TO_INDEX[idx] for idx in prediction]

        # Release GPU memory by moving model to CPU and clearing caches
        if device.type == 'cuda':
            model.to('cpu')
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        end_time = time.time()
        print(f"Inference done, time cost: {end_time - start_time:.2f}s")

        return hypnogram


def _worker_entry(sig_dict, device, queue: mp.Queue):
    """
    Worker function to run inference in a subprocess and communicate results via a multiprocessing queue.

    Args:
        sig_dict (dict[str, np.ndarray]): Input signals dictionary.
        device (str or torch.device or None): Device identifier for inference.
        queue (mp.Queue): Multiprocessing queue to send back results or errors.
    """
    try:
        res = _inference_core(sig_dict, device)
        queue.put(("ok", res))
    except Exception as e:
        import traceback
        # Send error message and traceback to parent process
        queue.put(("err", f"{e}\n{traceback.format_exc()}"))


def inference_recodings(sig_dict, use_subprocess: bool = True, device: str | torch.device | None = None):
    """
    Perform inference on polysomnography recordings, optionally using a subprocess to isolate CUDA context.

    Args:
        sig_dict (dict[str, np.ndarray]): Dictionary mapping channel names to numpy arrays of shape (N, 3000).
        use_subprocess (bool): Whether to run inference in a subprocess (default True). This helps ensure CUDA 
                               contexts are properly released after inference.
        device (str or torch.device or None): Device identifier (e.g., 'cuda:0', 'cpu') or None for automatic selection.

    Returns:
        list[str]: List of predicted sleep stages as strings for each epoch.
    """
    if use_subprocess and torch.cuda.is_available():
        ctx = mp.get_context('spawn')  # Use spawn context for subprocess to avoid CUDA issues
        q: mp.Queue = ctx.Queue()
        p = ctx.Process(target=_worker_entry, args=(sig_dict, device, q))
        p.start()
        status, payload = q.get()
        p.join()
        if status == 'ok':
            return payload
        raise RuntimeError(f"Inference subprocess failed:\n{payload}")
    else:
        # Run inference in the current process
        return _inference_core(sig_dict, device)


if __name__ == "__main__":
    # Example usage with random data for testing
    sig_dict = {
        'F3': np.random.randn(500, 3000),
        'F4': np.random.randn(500, 3000),
        'C3': np.random.randn(500, 3000),
        'C4': np.random.randn(500, 3000),
        # 'O1': np.random.randn(500, 3000),
        # 'O2': np.random.randn(500, 3000),
        # 'E1': np.random.randn(500, 3000),
        'E2': np.random.randn(500, 3000),
        'Chin': np.random.randn(500, 3000),
    }

    for _ in range(5):
        hypnogram = inference_recodings(sig_dict)
        # print(hypnogram, len(hypnogram))
