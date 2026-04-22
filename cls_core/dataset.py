# -*- coding: utf-8 -*-
"""
dataset.py

Shared subject-level windowed dataset for disorder classification.

All downstream tasks (narcolepsy / OSA / depression) store preprocessed PSG
data as one or more NPZ files per subject under ``data/<dataset>/<subject_id>/``,
with channels as separate arrays inside each NPZ file. This module provides:

- ``load_subjects_from_csv``: resolve subject NPZ directories + labels from a
  task-specific label CSV.
- ``SubjectWindowedDataset``: in-memory-preloaded PyTorch Dataset that slices
  each subject's recording into fixed-length windows of 30-second epochs.
- ``collate_fn``: padding-aware batch collation matching the LPSGM backbone's
  masking contract.
- ``build_dataloaders``: convenience factory producing train / val / test
  DataLoaders that share the preloaded NPZ cache.

Data is preloaded once in the main process during dataset construction and
then shared with DataLoader workers through Copy-on-Write (Linux fork
semantics), eliminating disk I/O during training.
"""

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.dataset import CHANNEL_TO_INDEX


def load_subjects_from_csv(
    csv_path: str,
    data_root: str,
    id_col: str = 'fileid',
    label_col: str = 'label',
) -> Tuple[List[str], List[int], List[str]]:
    """
    Resolve ``(subject_dirs, labels, subject_ids)`` from a label CSV and NPZ root.

    Each row of ``csv_path`` must contain ``id_col`` (subject identifier
    matching a subdirectory name under ``data_root``) and ``label_col`` (integer
    class label). Subjects without a matching NPZ-containing directory on
    disk are silently skipped.

    Args:
        csv_path: Path to the label CSV.
        data_root: Root directory containing ``{data_root}/{subject_id}/*.npz``.
        id_col: Column name in the CSV that holds the subject identifier.
        label_col: Column name that holds the integer class label.

    Returns:
        Tuple ``(subject_dirs, labels, subject_ids)`` where the three lists are
        aligned and sorted by subject_id.
    """
    id2label: Dict[str, int] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id2label[row[id_col]] = int(row[label_col])

    data_root = os.path.realpath(data_root)

    subject_dirs, subject_labels, subject_ids = [], [], []
    for sid in sorted(id2label.keys()):
        subj_dir = os.path.join(data_root, sid)
        if not os.path.isdir(subj_dir):
            continue
        npzs = [f for f in os.listdir(subj_dir) if f.endswith('.npz')]
        if not npzs:
            continue
        subject_dirs.append(subj_dir)
        subject_labels.append(id2label[sid])
        subject_ids.append(sid)
    return subject_dirs, subject_labels, subject_ids


# --------- module-level helper for multiprocessing pool ---------

def _load_single_npz(npz_path: str):
    """
    Load one NPZ file into a dict mapping channel name to a contiguous float32 array.

    Defined at module level so it can be pickled for ``ProcessPoolExecutor``.
    Arrays are copied out of the ``NpzFile`` so they remain valid after the
    file handle is closed.
    """
    with np.load(npz_path) as npz:
        data = {}
        for ch in CHANNEL_TO_INDEX.keys():
            if ch in npz:
                data[ch] = np.ascontiguousarray(npz[ch], dtype=np.float32)
    return npz_path, data


class SubjectWindowedDataset(Dataset):
    """
    In-memory preloaded windowed dataset for subject-level classification.

    All NPZ data is loaded in ``__init__`` (main process). DataLoader workers
    forked afterwards share the data via Copy-on-Write, so ``__getitem__`` is
    a pure memory slice with no file I/O.

    Args:
        subject_dirs: List of subject directory paths.
        subject_labels: List of integer labels aligned with ``subject_dirs``.
        seq_len: Window length in 30-second epochs.
        stride: Stride between consecutive windows (in epochs).
        n_load_workers: Parallelism for NPZ preloading.
        verbose: Whether to print progress during preloading.
        preload_data: Optional dict ``{npz_path: {channel: array}}`` of already
            loaded NPZ data. When provided, overlapping paths are reused and
            only the missing ones are loaded. Useful to share one cache across
            train / val / test datasets.
    """

    def __init__(
        self,
        subject_dirs: List[str],
        subject_labels: List[int],
        seq_len: int,
        stride: int = 1,
        n_load_workers: int = 32,
        verbose: bool = True,
        preload_data: Optional[Dict] = None,
    ):
        self.seq_len = seq_len
        self.stride = max(1, int(stride))

        # Collect all NPZ paths and remember (label, sid) per path.
        subject_npz_paths: List[str] = []
        npz_to_label_sid: Dict[str, Tuple[int, str]] = {}
        for sd, lab in zip(subject_dirs, subject_labels):
            sid = os.path.basename(sd)
            npzs = [os.path.join(sd, f) for f in sorted(os.listdir(sd)) if f.endswith('.npz')]
            for p in npzs:
                subject_npz_paths.append(p)
                npz_to_label_sid[p] = (int(lab), sid)

        # Preload all NPZ data (or reuse a shared cache).
        if preload_data is not None:
            self._data = preload_data
            n_reused = sum(1 for p in subject_npz_paths if p in self._data)
            n_missing = len(subject_npz_paths) - n_reused
            if n_missing > 0:
                if verbose:
                    print(f"[SubjectWindowedDataset] Reusing {n_reused} cached, loading {n_missing} new NPZs...")
                missing_paths = [p for p in subject_npz_paths if p not in self._data]
                self._load_parallel(missing_paths, n_load_workers, verbose)
            elif verbose:
                print(f"[SubjectWindowedDataset] Reusing all {n_reused} preloaded NPZs (zero I/O)")
        else:
            self._data = {}
            self._load_parallel(subject_npz_paths, n_load_workers, verbose)

        # Enumerate (npz_path, start_epoch_index, label, subject_id) per window.
        self.samples: List[Tuple[str, int, int, str]] = []
        for p in subject_npz_paths:
            lab, sid = npz_to_label_sid[p]
            chans = self._data[p]
            if not chans:
                continue
            first_ch = next(iter(chans.keys()))
            L = int(chans[first_ch].shape[0])
            if L < seq_len:
                continue
            for st in range(0, L - seq_len + 1, self.stride):
                self.samples.append((p, st, lab, sid))

        if verbose:
            total_mb = sum(
                sum(arr.nbytes for arr in d.values()) for d in self._data.values()
            ) / 1e6
            print(f"[SubjectWindowedDataset] Loaded {len(self._data)} NPZs ({total_mb:.0f} MB), "
                  f"{len(self.samples)} windows (seq_len={seq_len}, stride={self.stride})")

    def _load_parallel(self, npz_paths: List[str], n_workers: int, verbose: bool) -> None:
        """Parallel NPZ loading using ``ProcessPoolExecutor``."""
        t0 = time.perf_counter()
        if verbose:
            print(f"[SubjectWindowedDataset] Preloading {len(npz_paths)} NPZs with {n_workers} workers...")

        # Serial fallback for very small loads.
        if len(npz_paths) < 10 or n_workers == 1:
            for p in npz_paths:
                _, data = _load_single_npz(p)
                self._data[p] = data
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for path, data in pool.map(_load_single_npz, npz_paths, chunksize=4):
                    self._data[path] = data

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[SubjectWindowedDataset] Preload done in {elapsed:.1f}s "
                  f"({len(npz_paths) / max(elapsed, 1e-6):.0f} NPZs/sec)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, st, lab, sid = self.samples[idx]
        chans = self._data[path]

        # Order channels by the canonical LPSGM channel index.
        channels_present = sorted(chans.keys(), key=lambda c: CHANNEL_TO_INDEX[c])
        ch_arrays = [chans[ch][st: st + self.seq_len] for ch in channels_present]
        sig = np.stack(ch_arrays, axis=1)  # (seq_len, cn, epoch_length)

        seql, cn = sig.shape[:2]
        seq_idx = np.tile(np.arange(seql).reshape(seql, 1), (1, cn))
        ch_ids = np.array([CHANNEL_TO_INDEX[ch] for ch in channels_present], dtype=np.int64)
        ch_idx = np.tile(ch_ids.reshape(1, cn), (seql, 1))

        # Flatten (seq_len, cn) -> (seq_len * cn) tokens.
        x = sig.reshape(seql * cn, -1)
        seq_idx = seq_idx.reshape(seql * cn)
        ch_idx = ch_idx.reshape(seql * cn)
        y = np.full((seql,), int(lab), dtype=np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(seq_idx),
            torch.from_numpy(ch_idx),
            torch.zeros((seql * cn,), dtype=torch.bool),
            torch.tensor(seql, dtype=torch.long),
            sid,
        )


def collate_fn(batch):
    """
    Pad a batch of windows produced by ``SubjectWindowedDataset`` into dense tensors.

    The batch may contain samples with a varying number of tokens (when
    different subjects expose different channel subsets). Padding tokens are
    marked True in the returned ``mask``; real tokens remain False.

    Returns:
        Tuple ``(x, y, seq_idx, ch_idx, mask, ori_len, names)``.
    """
    xs, ys, seq_ids, ch_ids, _masks, ori_lens, names = zip(*batch)

    x = torch.nn.utils.rnn.pad_sequence(list(xs), batch_first=True, padding_value=0.0)
    seq_idx = torch.nn.utils.rnn.pad_sequence(list(seq_ids), batch_first=True, padding_value=0)
    ch_idx = torch.nn.utils.rnn.pad_sequence(list(ch_ids), batch_first=True, padding_value=0)

    # Padding tokens are marked True; real tokens are False.
    base_masks = [torch.zeros((xi.shape[0],), dtype=torch.bool) for xi in xs]
    mask = torch.nn.utils.rnn.pad_sequence(base_masks, batch_first=True, padding_value=True)

    y = torch.nn.utils.rnn.pad_sequence(list(ys), batch_first=True, padding_value=0)
    ori_len = torch.stack(list(ori_lens), dim=0)
    return x, y, seq_idx, ch_idx, mask, ori_len, names


def build_dataloaders(
    train_dirs: List[str],
    train_labels: List[int],
    val_dirs: List[str],
    val_labels: List[int],
    test_dirs: List[str],
    test_labels: List[int],
    seq_len: int,
    batch_size: int,
    num_workers: int,
    train_stride: int = 1,
    val_stride: int = 1,
    test_stride: int = 1,
    n_load_workers: int = 32,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders that share one preloaded NPZ cache.

    The train dataset loads its NPZs from disk; the val and test datasets reuse
    the same cache and only load the missing NPZ files. This avoids redundant
    I/O when folds overlap, while still handling fully disjoint splits correctly.

    Args:
        train_dirs / val_dirs / test_dirs: Per-split subject directory lists.
        train_labels / val_labels / test_labels: Aligned integer labels.
        seq_len: Window length in epochs.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes per split.
        train_stride / val_stride / test_stride: Window stride per split.
        n_load_workers: Parallelism for the initial NPZ preload.
        persistent_workers: Keep DataLoader workers alive across epochs when
            ``num_workers > 0``.
        prefetch_factor: Passed through to ``DataLoader`` when
            ``num_workers > 0``.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to DataLoaders.
    """
    train_set = SubjectWindowedDataset(
        train_dirs, train_labels, seq_len, stride=train_stride,
        n_load_workers=n_load_workers,
    )
    val_set = SubjectWindowedDataset(
        val_dirs, val_labels, seq_len, stride=val_stride,
        n_load_workers=n_load_workers, preload_data=train_set._data,
    )
    test_set = SubjectWindowedDataset(
        test_dirs, test_labels, seq_len, stride=test_stride,
        n_load_workers=n_load_workers, preload_data=val_set._data,
    )

    dl_kwargs = dict(
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    # prefetch_factor=None is invalid when num_workers==0, drop the key.
    if num_workers == 0:
        dl_kwargs.pop('prefetch_factor')
        dl_kwargs['persistent_workers'] = False

    return {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, **dl_kwargs),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False, **dl_kwargs),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, **dl_kwargs),
    }
