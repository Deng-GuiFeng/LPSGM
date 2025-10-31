# -*- coding: utf-8 -*-
"""
online_dataloader.py

This module provides on-the-fly data loading and windowing functionality for narcolepsy diagnosis 
within the LPSGM (Large Polysomnography Model) project. It defines a PyTorch Dataset class that 
dynamically extracts fixed-length windows from large polysomnography (PSG) recordings stored in 
.npz files without caching, enabling efficient training on large-scale datasets. 

Key features include:
- Support for multi-channel PSG signals with flexible channel selection.
- Sliding window extraction with configurable sequence length and stride.
- Labeling windows with subject-level diagnosis for 3-class narcolepsy classification.
- Custom collate function for padding variable-length batches.
- Construction of DataLoader objects for training, validation, and testing phases.

This module plays a critical role in preparing raw PSG data for input into the LPSGM model's 
epoch and sequence encoders, facilitating large-scale training and evaluation.
"""

import os
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import CHANNEL_TO_INDEX


def load_subject_npzs(subject_dir: str) -> List[str]:
    """
    Load all .npz file paths for a given subject directory.

    Args:
        subject_dir (str): Path to the subject's data directory.

    Returns:
        List[str]: List of full paths to .npz files within the directory.
    """
    return [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('.npz')]


class WindowedDatasetNar(Dataset):
    """
    PyTorch Dataset for on-the-fly windowed data loading without caching, tailored for 
    narcolepsy diagnosis (3-class classification). Each sample corresponds to a window 
    of fixed length (seq_len) extracted from continuous PSG signals with a specified stride.

    Labels are assigned per epoch window and correspond to the subject-level diagnosis, 
    repeated across all epochs in the window.

    Attributes:
        seq_len (int): Length of each window in epochs.
        stride (int): Step size for sliding window extraction.
        samples (List[Tuple]): List of tuples containing (npz_path, start_index, label, subject_id).
    """
    def __init__(self, subject_dirs: List[str], subject_labels: List[int], seq_len: int, stride: int = 1):
        """
        Initialize the dataset by enumerating all valid windows across subjects.

        Args:
            subject_dirs (List[str]): List of directories for each subject.
            subject_labels (List[int]): Corresponding diagnosis labels for each subject.
            seq_len (int): Number of epochs per window.
            stride (int, optional): Sliding window stride. Defaults to 1.
        """
        self.seq_len = seq_len
        self.stride = max(1, int(stride))  # Ensure stride is at least 1
        self.samples = []  # Stores tuples of (npz_path, window_start_idx, subject_label, subject_id)

        # Iterate over each subject directory and label pair
        for sd, lab in zip(subject_dirs, subject_labels):
            npzs = load_subject_npzs(sd)  # Load all npz files for the subject
            sid = os.path.basename(sd)    # Extract subject ID from directory name
            for p in npzs:
                with np.load(p) as npz:
                    # Identify channels present in the current npz file that are recognized by CHANNEL_TO_INDEX
                    channels_present = [ch for ch in CHANNEL_TO_INDEX.keys() if ch in npz]
                    if len(channels_present) == 0:
                        continue  # Skip files with no known channels
                    # Determine the length of the signal along the time dimension for the first available channel
                    L = int(npz[channels_present[0]].shape[0])
                if L < seq_len:
                    continue  # Skip files shorter than the window length
                # Generate sliding windows with specified stride
                for st in range(0, L - seq_len + 1, self.stride):
                    self.samples.append((p, st, int(lab), sid))

    def __len__(self):
        """
        Return the total number of windowed samples available.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a windowed sample and its associated metadata by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - x (torch.FloatTensor): Flattened signal tensor of shape (seq_len * num_channels, 3000).
                - y (torch.LongTensor): Per-epoch label vector of length seq_len.
                - seq_idx (torch.LongTensor): Sequence indices repeated per channel.
                - ch_idx (torch.LongTensor): Channel indices repeated per epoch.
                - mask (torch.BoolTensor): Placeholder mask tensor (all False).
                - ori_len (torch.LongTensor): Original sequence length (seq_len).
                - sid (str): Subject identifier.
        """
        path, st, lab, sid = self.samples[idx]
        with np.load(path) as npz:
            # Identify known channels present in the npz file
            channels_present = [ch for ch in CHANNEL_TO_INDEX.keys() if ch in npz]
            if len(channels_present) == 0:
                raise RuntimeError(f"No known channels in {path}")
            # Sort channels by their predefined index to maintain consistent ordering
            channels_present.sort(key=lambda c: CHANNEL_TO_INDEX[c])

            # Extract signal segments for each channel within the window [st : st + seq_len]
            # Each channel array shape: (seq_len, 3000) - 3000 samples per epoch
            ch_arrays = [npz[ch][st: st + self.seq_len] for ch in channels_present]
            # Stack channels along axis=1 to get shape (seq_len, num_channels, 3000)
            sig = np.stack(ch_arrays, axis=1).astype(np.float32)

            seql, cn = sig.shape[:2]  # seql: number of epochs, cn: number of channels

            # Create sequence indices: array of epoch indices repeated for each channel
            seq_idx = np.tile(np.arange(seql).reshape(seql, 1), (1, cn))
            # Map channel names to their integer indices
            ch_ids = np.array([CHANNEL_TO_INDEX[ch] for ch in channels_present], dtype=np.int64)
            # Create channel indices array repeated for each epoch
            ch_idx = np.tile(ch_ids.reshape(1, cn), (seql, 1))

            # Flatten signal for input to backbone network: shape (seql * cn, 3000)
            x = sig.reshape(seql * cn, -1)
            seq_idx = seq_idx.reshape(seql * cn)
            ch_idx = ch_idx.reshape(seql * cn)
            ori_len = seql  # Original sequence length (number of epochs)

            # Create per-epoch label vector filled with the subject diagnosis label
            y = np.full((seql,), int(lab), dtype=np.int64)

            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(seq_idx),
                torch.from_numpy(ch_idx),
                torch.zeros((seql * cn,), dtype=torch.bool),  # Placeholder mask; actual mask created in collate
                torch.tensor(ori_len, dtype=torch.long),
                sid,
            )


def collate_fn_nar(batch):
    """
    Custom collate function to batch variable-length windowed samples for narcolepsy diagnosis.

    Pads sequences and associated indices to the maximum length in the batch and creates masks 
    indicating padded elements.

    Args:
        batch (List[Tuple]): List of samples returned by WindowedDatasetNar.__getitem__.

    Returns:
        Tuple containing:
            - x (torch.FloatTensor): Padded signal tensor of shape (batch_size, max_seq_len, 3000).
            - y (torch.LongTensor): Padded per-epoch label tensor.
            - seq_idx (torch.LongTensor): Padded sequence indices.
            - ch_idx (torch.LongTensor): Padded channel indices.
            - mask (torch.BoolTensor): Boolean mask indicating padded positions (True for padding).
            - ori_len (torch.LongTensor): Original sequence lengths before padding.
            - names (Tuple[str]): Subject identifiers for each sample in the batch.
    """
    xs, ys, seq_ids, ch_ids, _masks_in, ori_lens, names = zip(*batch)

    # Pad signal tensors along the batch dimension with zeros
    x = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in xs], batch_first=True, padding_value=0.0)
    # Pad sequence index tensors with zeros
    seq_idx = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(s) for s in seq_ids], batch_first=True, padding_value=0)
    # Pad channel index tensors with zeros
    ch_idx = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(c) for c in ch_ids], batch_first=True, padding_value=0)

    # Create base masks of zeros (no padding) for each sample
    base_masks = [torch.zeros((x_i.shape[0],), dtype=torch.bool) for x_i in xs]
    # Pad masks with True indicating padded positions
    mask = torch.nn.utils.rnn.pad_sequence(base_masks, batch_first=True, padding_value=True)

    # Pad label tensors with zeros
    y = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(y_i) for y_i in ys], batch_first=True, padding_value=0)
    # Stack original sequence lengths into a tensor
    ori_len = torch.stack(list(ori_lens), dim=0)
    return x, y, seq_idx, ch_idx, mask, ori_len, names


def build_dataloaders_nar(train_subjects: List[str], train_labels: List[int],
                          val_subjects: List[str], val_labels: List[int],
                          test_subjects: List[str], test_labels: List[int],
                          seq_len: int, batch_size: int, num_workers: int,
                          train_stride: int = 1, val_stride: int = 1, test_stride: int = 1) -> Dict[str, DataLoader]:
    """
    Construct DataLoader objects for training, validation, and testing datasets for narcolepsy diagnosis.

    Each DataLoader uses the WindowedDatasetNar with appropriate stride and batch configurations.

    Args:
        train_subjects (List[str]): List of training subject directories.
        train_labels (List[int]): Corresponding training labels.
        val_subjects (List[str]): List of validation subject directories.
        val_labels (List[int]): Corresponding validation labels.
        test_subjects (List[str]): List of test subject directories.
        test_labels (List[int]): Corresponding test labels.
        seq_len (int): Window length in epochs.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        train_stride (int, optional): Sliding window stride for training. Defaults to 1.
        val_stride (int, optional): Sliding window stride for validation. Defaults to 1.
        test_stride (int, optional): Sliding window stride for testing. Defaults to 1.

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoader objects.
    """
    train_set = WindowedDatasetNar(train_subjects, train_labels, seq_len, stride=train_stride)
    val_set = WindowedDatasetNar(val_subjects, val_labels, seq_len, stride=val_stride)
    test_set = WindowedDatasetNar(test_subjects, test_labels, seq_len, stride=test_stride)

    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn_nar),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=True, drop_last=False, collate_fn=collate_fn_nar),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True, drop_last=False, collate_fn=collate_fn_nar),
    }
    return loaders
