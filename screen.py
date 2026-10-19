# -*- coding: utf-8 -*-
"""
screen.py

Apply a fine-tuned LPSGM screening model to raw PSG EDF recordings.

Each released screening model corresponds to one cross-validation fold of one
task (MNC narcolepsy 3-class, MNC Normal vs Abnormal, APPLES OSA severity,
APPLES depression). A model directory contains:

    best.pth             fine-tuned encoder (Epoch Encoder + Sequence Encoder)
    screening_head.npz   feature standardization + logistic-regression head

For each recording the script loads and preprocesses the signals in the same
way as ``inference.py``, splits the night into non-overlapping 20-epoch
windows, averages the encoder features over all windows, standardizes the
resulting subject-level feature and applies the logistic-regression head.
Class probabilities are written to a CSV file, one row per recording.

Usage:
    python screen.py --model_dir weights/screening/apples_osa/fold0 \
        --edf_dir /path/to/edfs --out osa_fold0.csv

Edit ``CHANNEL_MAP`` below to match the channel names of your EDF files
(see ``inference.py`` for the mapping syntax).
"""

import argparse
import csv
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from cls_core.simple_eval import _load_backbone, load_screening_head
from dataset.dataset import CHANNEL_TO_INDEX
from web_demo.utils import find_files_with_suffix, load_sig, pre_process

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# LPSGM channel name -> options tried in order in the EDF file
# (single channel name, or a (channel, reference) pair for a differential signal).
CHANNEL_MAP = {
    'F3': (('F3', 'M2'), 'F3-M2', 'F3'),
    'F4': (('F4', 'M1'), 'F4-M1', 'F4'),
    'C3': (('C3', 'M2'), 'C3-M2', 'C3'),
    'C4': (('C4', 'M1'), 'C4-M1', 'C4'),
    'O1': (('O1', 'M2'), 'O1-M2', 'O1'),
    'O2': (('O2', 'M1'), 'O2-M1', 'O2'),
    'E1': (('E1', 'M2'), 'E1-M2', 'E1', 'LOC'),
    'E2': (('E2', 'M1'), 'E2-M1', 'E2-M2', 'E2', 'ROC'),
    'Chin': (('Chin1', 'Chin2'), 'Chin 1-Chin 2', 'Chin', 'EMG'),
}
RESAMPLE_RATE = 100


def recording_windows(processed: dict, seq_len: int, stride: int):
    """
    Build model inputs for all windows of one recording.

    Args:
        processed: ``{channel: array (num_epochs, 3000)}`` from ``pre_process``.
        seq_len: Window length in 30-second epochs.
        stride: Step between window starts (``seq_len`` = non-overlapping).

    Yields:
        Tuples ``(x, seq_idx, ch_idx)`` for single windows, laid out as in
        ``cls_core.dataset.SubjectWindowedDataset``.
    """
    channels = sorted(processed.keys(), key=lambda c: CHANNEL_TO_INDEX[c])
    n_epochs = min(processed[c].shape[0] for c in channels)
    cn = len(channels)
    ch_ids = np.array([CHANNEL_TO_INDEX[c] for c in channels], dtype=np.int64)
    seq_idx = np.tile(np.arange(seq_len).reshape(seq_len, 1), (1, cn)).reshape(-1)
    ch_idx = np.tile(ch_ids.reshape(1, cn), (seq_len, 1)).reshape(-1)
    for st in range(0, n_epochs - seq_len + 1, stride):
        sig = np.stack([processed[c][st: st + seq_len] for c in channels], axis=1)  # (seq_len, cn, 3000)
        yield sig.reshape(seq_len * cn, -1).astype(np.float32), seq_idx, ch_idx


@torch.no_grad()
def subject_feature(model, processed: dict, seq_len: int, stride: int, batch_size: int = 64):
    """Average the encoder's pooled feature over all windows of one recording."""
    feats, batch = [], []

    def flush():
        x = torch.from_numpy(np.stack([b[0] for b in batch])).cuda()
        s = torch.from_numpy(np.stack([b[1] for b in batch])).cuda()
        c = torch.from_numpy(np.stack([b[2] for b in batch])).cuda()
        mask = torch.zeros(s.shape, dtype=torch.bool, device=s.device)
        ori_len = [seq_len] * len(batch)
        feats.append(model.extract_features(x, mask, c, s, ori_len).cpu().numpy())
        batch.clear()

    for item in recording_windows(processed, seq_len, stride):
        batch.append(item)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if not feats:
        return None
    return np.concatenate(feats).astype(np.float64).mean(axis=0)


def main():
    p = argparse.ArgumentParser(description='Apply a fine-tuned LPSGM screening model to EDF recordings')
    p.add_argument('--model_dir', required=True,
                   help='Directory containing best.pth and screening_head.npz of one fold')
    p.add_argument('--edf_dir', required=True, help='Directory searched recursively for .edf files')
    p.add_argument('--out', required=True, help='Output CSV path')
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    scaler, clf = load_screening_head(os.path.join(args.model_dir, 'screening_head.npz'))
    head = np.load(os.path.join(args.model_dir, 'screening_head.npz'))
    seq_len, stride = int(head['seq_len']), int(head['stride'])
    model = _load_backbone(os.path.join(args.model_dir, 'best.pth'), num_classes=len(clf.classes_))

    rows = []
    for edf_path in tqdm(sorted(find_files_with_suffix(args.edf_dir, suffix='.edf')), desc='Screening'):
        _, sig_dict = load_sig(edf_path, CHANNEL_MAP)
        processed = pre_process(sig_dict, resample_rate=RESAMPLE_RATE, notch=False)
        feat = subject_feature(model, processed, seq_len, stride, args.batch_size)
        if feat is None:
            print(f'Skipped (shorter than {seq_len} epochs): {edf_path}')
            continue
        prob = clf.predict_proba(scaler.transform(feat[None, :]))[0]
        rows.append([os.path.basename(edf_path)] + [f'{v:.6f}' for v in prob])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['recording'] + [f'prob_class{int(c)}' for c in clf.classes_])
        w.writerows(rows)
    print(f'Saved: {args.out} ({len(rows)} recordings)')


if __name__ == '__main__':
    main()
