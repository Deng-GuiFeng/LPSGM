# -*- coding: utf-8 -*-
"""
save_raw.py

Stage 1 of the Grad-CAM pipeline: for each subject, load the EDF pair, run
LPSGM inference, and persist the three arrays the downstream stages need:

- ``raw_signal.npy``: preprocessed but **unnormalized** signal of shape
  ``(num_epochs, num_channels, 3000)``. Used as the background waveform when
  rendering per-epoch Grad-CAM overlays.
- ``annotation.npy``: ``(num_epochs,)`` expert sleep-stage labels aligned with
  the raw signal.
- ``prediction.npy``: ``(num_windows,)`` LPSGM predictions for ``time_step=0``
  of each sliding window (the first epoch of each window), consistent with the
  Grad-CAM and Guided Backpropagation stages that also operate on ``time_step=0``.
- ``channels.txt``: ordered canonical channel names actually loaded.
"""

from math import ceil
from typing import Dict, Tuple, Optional
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from .preparer import load_mass_subject, build_sequences
from .utils import find_mass_subjects
from .wrapper import LPSGMInferenceWrapper


def _infer_predictions(
    wrapper: LPSGMInferenceWrapper,
    sequences_np: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run LPSGM inference over all windows for one subject and return per-window
    argmax predictions at ``time_step=0``.

    Args:
        wrapper: Loaded ``LPSGMInferenceWrapper`` in eval mode.
        sequences_np: Array of shape ``(num_windows, seq_len * num_channels, 3000)``.
        batch_size: Inference batch size.
        device: CUDA or CPU device.

    Returns:
        ``(num_windows,)`` int array of predicted sleep stage indices.
    """
    predictions = []
    num_batches = ceil(len(sequences_np) / batch_size)
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Inference"):
            batch_np = sequences_np[b * batch_size:(b + 1) * batch_size]
            batch_t = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            logits = wrapper(batch_t)                          # (B, seq_len, 5)
            probs = torch.softmax(logits[:, 0, :], dim=-1)     # first-epoch logits
            predictions.append(probs.argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def save_raw_for_subject(
    wrapper: LPSGMInferenceWrapper,
    sub_id: str,
    psg_path: str,
    ano_path: str,
    dst_root: str,
    channel_map: Dict[str, Tuple],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[int, int]:
    """
    Process one subject: export raw signals, expert annotations, and LPSGM predictions.

    Two preprocessing passes are performed on the same EDF pair: one with
    ``normalize=False`` (raw microvolt-scale waveforms for rendering) and one
    with ``normalize=True`` (z-scored tensors for model inference).

    Args:
        wrapper: Loaded LPSGM wrapper in eval mode.
        sub_id: Subject identifier used as the output subdirectory name.
        psg_path: Path to ``{sub_id} PSG.edf``.
        ano_path: Path to ``{sub_id} Base.edf``.
        dst_root: Root directory under which to write ``{dst_root}/{sub_id}/``.
        channel_map: Canonical channel → EDF channel options mapping.
        seq_len: LPSGM input window length in epochs.
        batch_size: Inference batch size.
        device: CUDA or CPU device.

    Returns:
        ``(num_saved_epochs, num_channels)``.
    """
    print(f"[save_raw] Processing {sub_id}")

    raw_signal, channels_raw, annotation = load_mass_subject(psg_path, ano_path, channel_map, normalize=False)
    model_input, channels_model, _ = load_mass_subject(psg_path, ano_path, channel_map, normalize=True)
    assert channels_raw == channels_model, "Channel order diverged between raw and normalized preprocessing"
    channels = channels_raw

    if raw_signal.shape[0] == 0:
        print(f"[save_raw] Skipping {sub_id}: no aligned epochs available")
        return 0, len(channels)

    sequences_np = build_sequences(model_input, seq_len)
    if sequences_np.shape[0] == 0:
        print(f"[save_raw] Skipping {sub_id}: not enough epochs for seq_len={seq_len}")
        return 0, len(channels)
    predictions = _infer_predictions(wrapper, sequences_np, batch_size, device)

    # Trim to the shortest length: raw/annotation are per-epoch (N), predictions are
    # per-window (N - seq_len + 1). Downstream stages index all three by window position,
    # so truncating to the minimum keeps them consistent.
    N = min(raw_signal.shape[0], predictions.shape[0], annotation.shape[0])
    raw_signal = raw_signal[:N]
    predictions = predictions[:N]
    annotation = annotation[:N]

    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)
    np.save(os.path.join(dst_sub_dir, "raw_signal.npy"), raw_signal.astype(np.float32))
    np.save(os.path.join(dst_sub_dir, "prediction.npy"), predictions.astype(np.int64))
    np.save(os.path.join(dst_sub_dir, "annotation.npy"), annotation.astype(np.int64))
    with open(os.path.join(dst_sub_dir, "channels.txt"), 'w') as f:
        f.write("\n".join(channels))

    return N, len(channels)


def run_save_raw(
    wrapper: LPSGMInferenceWrapper,
    src_root: str,
    dst_root: str,
    channel_map: Dict[str, Tuple],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    clean_dst: bool = True,
    subject_limit: Optional[int] = None,
    subject_ids: Optional[list] = None,
) -> None:
    """
    Iterate over every MASS subject under ``src_root`` and export raw outputs.

    Args:
        wrapper: Loaded LPSGM wrapper in eval mode.
        src_root: Directory containing MASS ``{sub_id} PSG.edf`` + ``{sub_id} Base.edf`` pairs.
        dst_root: Output root directory; per-subject subdirectories are written beneath it.
        channel_map: Canonical channel → EDF channel options mapping.
        seq_len: LPSGM input window length in epochs.
        batch_size: Inference batch size.
        device: CUDA or CPU device.
        clean_dst: If True, remove ``dst_root`` before writing.
        subject_limit: Optional cap on the number of subjects processed.
        subject_ids: Optional whitelist of subject IDs.
    """
    print(f"[save_raw] Source: {src_root}")
    print(f"[save_raw] Destination: {dst_root}")

    if clean_dst:
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    subjects = find_mass_subjects(src_root, subject_ids=subject_ids)
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    print(f"[save_raw] Found {len(subjects)} subjects with MASS EDF pairs")

    for sub_id, psg_path, ano_path in subjects:
        # Skip-if-exists: resume support.
        marker = os.path.join(dst_root, sub_id, "prediction.npy")
        if os.path.exists(marker):
            print(f"[save_raw] Skipping {sub_id}: already processed")
            continue
        try:
            save_raw_for_subject(
                wrapper, sub_id, psg_path, ano_path,
                dst_root, channel_map, seq_len, batch_size, device,
            )
        except Exception as exc:
            print(f"[save_raw] Failed on {sub_id}: {exc}")
