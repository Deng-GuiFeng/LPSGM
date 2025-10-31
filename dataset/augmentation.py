# -*- coding: utf-8 -*-
"""
augmentation.py

This module provides data augmentation functions for polysomnography (PSG) signals used in the LPSGM project.
These augmentations enhance model robustness by introducing variability in channel selection, temporal alignment,
and sequence segmentation. The functions support multi-channel PSG data and associated labels, enabling flexible
preprocessing for sleep staging and mental disorder diagnosis tasks.

Functions:
- random_channel_crop: Randomly selects a subset of channels from the input sequence and associated data.
- random_temporal_shift: Applies a random temporal shift to the sequence data to simulate temporal variability.
- random_split_sample: Splits long sequences into smaller segments with optional random offset for training.
"""

import numpy as np


def random_channel_crop(seq, label, seq_idx, ch_idx, spg):
    """
    Randomly crops channels from the input PSG sequence and corresponding metadata.

    Args:
        seq (np.ndarray): PSG signal sequence of shape (sequence_length, channel_num, 3000).
        label (np.ndarray or None): Labels corresponding to each channel, shape (sequence_length, channel_num) or None.
        seq_idx (np.ndarray): Sequence indices, shape (sequence_length, channel_num).
        ch_idx (np.ndarray): Channel indices, shape (sequence_length, channel_num).
        spg (np.ndarray or None): Spectrogram data, shape (sequence_length, channel_num, 129, 29) or None.

    Returns:
        tuple: Tuple containing:
            - new_seq (np.ndarray): Sequence with randomly selected channels.
            - new_label (np.ndarray or None): Corresponding labels for selected channels or None.
            - new_seq_idx (np.ndarray): Sequence indices for selected channels.
            - new_ch_idx (np.ndarray): Channel indices for selected channels.
            - new_spg (np.ndarray or None): Spectrogram data for selected channels or None.
    """
    orig_cn = seq.shape[1]  # Original number of channels
    new_cn = np.random.randint(1, orig_cn + 1)  # Randomly choose number of channels to keep
    ch_indices = np.random.choice(orig_cn, new_cn, replace=False)  # Randomly select channel indices without replacement

    new_seq = seq[:, ch_indices, :]  # Select channels from sequence
    new_seq_idx = seq_idx[:, ch_indices]  # Select corresponding sequence indices
    new_ch_idx = ch_idx[:, ch_indices]  # Select corresponding channel indices
    
    if type(label) == np.ndarray:
        new_label = label[:, ch_indices]  # Select labels for chosen channels if available
    else:
        new_label = None

    if type(spg) == np.ndarray:
        new_spg = spg[:, ch_indices, :, :]  # Select spectrogram data for chosen channels if available
    else:
        new_spg = None

    return new_seq, new_label, new_seq_idx, new_ch_idx, new_spg


def random_temporal_shift(seq, shift_len=100):
    """
    Applies a random temporal circular shift to the PSG sequence to simulate temporal variability.

    Args:
        seq (np.ndarray): PSG signal sequence of shape (L, 3000, channel_num).
        shift_len (int, optional): Standard deviation for normal distribution to sample shift length. Default is 100.

    Returns:
        np.ndarray: Temporally shifted sequence of shape (L, 3000, channel_num).
    """
    seq = seq.reshape(-1, seq.shape[-1])  # Flatten time dimension: (L*3000, channel_num)
    shift_len = round(np.random.normal(0, shift_len, 1)[0])  # Sample shift length from normal distribution centered at 0
    shifted_seq = np.roll(seq, shift_len, axis=0)  # Circularly shift the flattened sequence along time axis
    shifted_seq = shifted_seq.reshape(-1, 3000, seq.shape[-1])  # Reshape back to original dimensions
    return shifted_seq  # Return shifted sequence


def random_split_sample(seq, labels, random: bool, seq_len):
    """
    Splits a long PSG sequence and corresponding labels into smaller segments with optional random offset.

    Args:
        seq (np.ndarray): PSG signal sequence of shape (L, channel_num, 3000).
        labels (np.ndarray or None): Labels corresponding to each sequence frame, shape (L,) or None.
        random (bool): Whether to apply a random starting offset before splitting.
        seq_len (int): Length of each segment after splitting.

    Returns:
        list or tuple:
            - If labels are provided and successfully processed, returns a tuple of two lists:
              (list of segmented sequences, list of segmented labels).
            - If labels are None or cannot be processed, returns a list of segmented sequences.
            - If no valid segments can be created, returns empty list(s).
    """
    if random:
        rn = np.random.randint(0, seq_len)  # Random offset for splitting
    else:
        rn = 0  # No offset

    seqn = (seq.shape[0] - rn) // seq_len  # Number of full segments that can be created

    if not seqn > 0:
        # Not enough data to create any segment
        if type(seq) == type(labels):  # Both seq and labels are of same type (likely np.ndarray)
            return [], []  # Return empty lists for sequences and labels
        else:
            return []  # Return empty list for sequences only
    
    # Crop sequence starting from offset to fit an integer number of segments
    seq = seq[rn : rn + seqn * seq_len]
    # Reshape sequence into segments: (num_segments, segment_length, channel_num, 3000)
    seq = seq.reshape(seqn, seq_len, seq.shape[1], seq.shape[2])
    # Split into list of arrays, each representing one segment, and remove single-dimensional entries
    splited_seq = [np.squeeze(arr) for arr in np.split(seq, seqn, axis=0)]

    try:
        # Process labels similarly if available
        labels = labels[rn : rn + seqn * seq_len]
        labels = labels.reshape(seqn, seq_len)  # Reshape labels into segments
        splited_labels = [np.squeeze(arr) for arr in np.split(labels, seqn, axis=0)]
    except:
        # If labels processing fails (e.g., labels is None), return only segmented sequences
        return splited_seq
    else:
        # Return both segmented sequences and labels
        return splited_seq, splited_labels
