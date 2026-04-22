# -*- coding: utf-8 -*-
"""
preparer.py

Shared per-subject data preparation for the Grad-CAM pipeline.

Each downstream stage (``save_raw`` / ``custom_gradcam`` / ``guided_backprop``)
starts from the same EDF pair and needs the same ``(num_epochs, num_channels,
3000)`` preprocessed tensor plus matching 30-second sleep-stage annotations,
so this module centralizes EDF loading, EDF+ annotation parsing, signal
alignment, filtering, resampling, optional normalization, and sliding-window
construction to avoid divergent copies across stages.

The default implementation targets the MASS-SS1 / MASS-SS3 public datasets.
To support other recordings, provide a different channel map to
``load_mass_subject`` and adapt or replace ``read_mass_annotations`` if the
annotation format differs.
"""

from typing import Dict, List, Tuple

import numpy as np

from .utils import load_sig, pre_process, read_mass_annotations


def _align_sig_by_annotations(sig_dict, stages, onsets, durations):
    """
    Segment and concatenate raw PSG signals so that they align exactly with
    30-second epochs scored by the sleep expert.

    MASS annotations do not necessarily start at t=0 and may contain gaps or
    unknown stretches (stage = -1). This function extracts the annotated
    regions at 30-second granularity and concatenates them, keeping the
    signals and labels one-to-one per epoch.

    Args:
        sig_dict (dict): ``{channel: {'sample_rate': sr, 'data': 1d np.ndarray}}``.
        stages (np.ndarray): 1-D int array of annotation stage labels.
        onsets (np.ndarray): 1-D float array of annotation onsets in seconds.
        durations (np.ndarray): 1-D float array of annotation durations in seconds.

    Returns:
        tuple:
            aligned_sig_dict (dict): ``sig_dict`` with ``data`` cropped/concatenated
                to cover only the annotated regions.
            aligned_stages (np.ndarray): 1-D int array of per-30s-epoch labels.
    """
    if not sig_dict:
        return sig_dict, np.array([], dtype=np.int32)

    any_ch = next(iter(sig_dict))
    sr = int(round(sig_dict[any_ch]['sample_rate']))

    # Keep only annotations with a known stage label in [0..4] and positive duration.
    valid_idx = np.where((stages >= 0) & (stages <= 4) & (durations > 0))[0]
    if valid_idx.size == 0:
        return sig_dict, np.array([], dtype=np.int32)

    epoch_samps = 30 * sr

    # First pass: compute segment (start_sample, num_epochs, stage_label).
    segs = []
    for i in valid_idx:
        onset_sec = float(onsets[i])
        dur_sec = float(durations[i])
        start_samp = int(round(onset_sec * sr))
        n_epochs = int(np.floor(dur_sec / 30.0 + 1e-6))
        if n_epochs <= 0:
            continue
        segs.append((start_samp, n_epochs, int(stages[i])))

    if not segs:
        return sig_dict, np.array([], dtype=np.int32)

    # Second pass: clamp each segment to the shortest available channel length.
    ch_lengths = {ch_name: len(ch['data']) for ch_name, ch in sig_dict.items()}
    segs_final = []
    for (s, n_ep, stg) in segs:
        candidates = []
        for L in ch_lengths.values():
            if s >= L:
                candidates.append(0)
            else:
                candidates.append((L - s) // epoch_samps)
        n_ep_final = min(n_ep, min(candidates) if candidates else 0)
        if n_ep_final > 0:
            segs_final.append((s, n_ep_final, stg))

    if not segs_final:
        empty = {
            ch_name: {'sample_rate': sr, 'data': np.array([], dtype=ch['data'].dtype)}
            for ch_name, ch in sig_dict.items()
        }
        return empty, np.array([], dtype=np.int32)

    # Concatenate aligned signal segments per channel.
    aligned_sig_dict = {}
    for ch_name, ch in sig_dict.items():
        data = ch['data']
        parts = [data[s:s + n_ep_final * epoch_samps]
                 for (s, n_ep_final, _) in segs_final]
        aligned_sig_dict[ch_name] = {
            'sample_rate': sr,
            'data': np.concatenate(parts, axis=0) if parts else np.array([], dtype=data.dtype),
        }

    # Expand per-segment stage labels into per-epoch labels.
    aligned_stages = np.concatenate(
        [np.full(n_ep_final, stg, dtype=np.int32) for (_, n_ep_final, stg) in segs_final],
        axis=0,
    )
    return aligned_sig_dict, aligned_stages


def load_mass_subject(
    psg_path: str,
    ano_path: str,
    channel_map: Dict[str, Tuple],
    resample_rate: int = 100,
    notch: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load one MASS subject and return aligned per-epoch signals plus stage labels.

    Args:
        psg_path: Path to ``{sub_id} PSG.edf`` (raw PSG signals).
        ano_path: Path to ``{sub_id} Base.edf`` (EDF+ sleep stage annotations).
        channel_map: Canonical channel → EDF channel options mapping.
        resample_rate: Target sampling rate in Hz.
        notch: Whether to apply a 50 Hz notch filter.
        normalize: Whether to z-score normalize each channel. Pass False to
                   retain the microvolt scale (e.g. for raw-waveform rendering).

    Returns:
        sig: ``(num_epochs, num_channels, 30 * resample_rate)`` preprocessed
             signal tensor.
        channels: Ordered list of canonical channel names actually loaded.
        annotation: ``(num_epochs,)`` integer sleep-stage labels aligned with sig.
    """
    _, sig_dict = load_sig(psg_path, channel_map)
    stages, onsets, durations = read_mass_annotations(ano_path)
    sig_dict, stages = _align_sig_by_annotations(sig_dict, stages, onsets, durations)

    sig_dict = pre_process(sig_dict, resample_rate=resample_rate, notch=notch, normalize=normalize)

    channels = list(sig_dict.keys())
    if not channels:
        return np.zeros((0, 0, 30 * resample_rate), dtype=np.float32), channels, stages

    # Verify all channels produced the same epoch count (pre_process truncates
    # to full epochs; alignment already guarantees equal raw lengths).
    epoch_counts = [sig_dict[ch].shape[0] for ch in channels]
    if len(set(epoch_counts)) != 1:
        # Fall back to the minimum epoch count to keep channels synchronized.
        n_min = min(epoch_counts)
        for ch in channels:
            sig_dict[ch] = sig_dict[ch][:n_min]
        stages = stages[:n_min]

    stacked = np.stack([sig_dict[ch] for ch in channels], axis=0)  # (cn, N, 3000)
    sig = stacked.transpose(1, 0, 2)  # (N, cn, 3000)

    # pre_process may have dropped trailing partial epochs; re-align annotations accordingly.
    annotation = stages[:sig.shape[0]].astype(np.int64)
    return sig, channels, annotation


def build_sequences(sig: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Slice per-epoch signals into overlapping sliding windows of ``seq_len`` epochs.

    The resulting array follows LPSGM's token layout: each token corresponds
    to one (epoch, channel) pair and the two axes are flattened into a single
    dimension.

    Args:
        sig: Array of shape ``(num_epochs, num_channels, 3000)``.
        seq_len: Number of epochs per window.

    Returns:
        Array of shape ``(num_windows, seq_len * num_channels, 3000)`` where
        ``num_windows = num_epochs - seq_len + 1`` (or 0 when too short).
    """
    num_epochs, cn, _ = sig.shape
    num_windows = num_epochs - seq_len + 1
    if num_windows <= 0:
        return np.zeros((0, seq_len * cn, 3000), dtype=sig.dtype)
    windows = np.stack([sig[i:i + seq_len] for i in range(num_windows)], axis=0)  # (num_windows, seq_len, cn, 3000)
    return windows.reshape(num_windows, seq_len * cn, 3000)
