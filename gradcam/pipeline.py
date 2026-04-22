# -*- coding: utf-8 -*-
"""
pipeline.py

End-to-end Grad-CAM analysis pipeline for LPSGM.

All outputs are written under a single ``--output-root`` directory with the
following subdirectories (selectable via ``--stages``):

1. ``save_raw``: per-subject preprocessing + LPSGM inference; writes raw
   signals, expert annotations, and LPSGM predictions to ``{output-root}/raw/``.
2. ``gradcam``: dual-branch custom Grad-CAM; writes
   ``gradcam_branch{1,2}.npy`` to ``{output-root}/gradcam/``.
3. ``guided``: Guided Backpropagation; writes ``guided_backpropagation.npy``
   to ``{output-root}/guided/``.
4. ``render``: per-epoch PNG visualizations fusing Grad-CAM with guided
   saliency; writes per-subject PNGs to ``{output-root}/figures/``.

The default configuration targets the MASS-SS1 / MASS-SS3 public datasets.
``--src-root`` should point at a directory containing MASS-style EDF pairs:
for each subject, both ``{sub_id} PSG.edf`` and ``{sub_id} Base.edf`` must be
present.

Recommended invocation from the repository root:

    python -m gradcam.pipeline \\
        --src-root <path_to_mass_edf_root> \\
        --weights weights/ched32_seqed64_ch9_seql20_block4.pth \\
        --output-root gradcam_output \\
        --stages save_raw,gradcam,guided,render
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .channel_maps import MASS_CHANNEL_MAP, SLEEP_STAGES
from .custom_gradcam import run_custom_gradcam
from .guided_backprop import run_guided_backprop
from .save_raw import run_save_raw
from .wrapper import build_wrapper_from_weights


DEFAULT_ARCH = dict(
    architecture='cat_cls',
    ch_num=9,
    ch_emb_dim=32,
    seq_emb_dim=64,
    num_transformer_blocks=4,
    transformer_num_heads=8,
    transformer_dropout=0.0,
    transformer_attn_dropout=0.0,
    epoch_encoder_dropout=0.0,
    clamp_value=10.0,
)

ALL_STAGES = ['save_raw', 'gradcam', 'guided', 'render']


def _parse_args():
    p = argparse.ArgumentParser(description="LPSGM Grad-CAM analysis pipeline")
    p.add_argument('--src-root', type=str, required=True,
                   help="Directory containing MASS '{sub_id} PSG.edf' + '{sub_id} Base.edf' pairs.")
    p.add_argument('--weights', type=str, required=True,
                   help="Path to the pretrained LPSGM checkpoint.")
    p.add_argument('--output-root', type=str, default='gradcam_output',
                   help="Directory for all outputs. Subdirectories raw/, gradcam/, "
                        "guided/ and figures/ are created beneath it.")
    p.add_argument('--stages', type=str, default=','.join(ALL_STAGES),
                   help=f"Comma-separated subset of: {','.join(ALL_STAGES)}")
    p.add_argument('--seq-len', type=int, default=20, help="Window length in epochs.")
    p.add_argument('--batch-size', type=int, default=16, help="Inference batch size.")
    p.add_argument('--subject-limit', type=int, default=None,
                   help="Optional cap on the number of subjects (debugging).")
    p.add_argument('--subject-ids', type=str, default=None,
                   help="Optional comma-separated whitelist of subject IDs to process.")
    p.add_argument('--device', type=str, default=None,
                   help="Override device (e.g. 'cuda:0'). Default: auto-detect.")
    p.add_argument('--render-dpi', type=int, default=120,
                   help="DPI for per-epoch rendering (stage 'render').")
    p.add_argument('--no-clean', action='store_true',
                   help="Do not wipe cache directories before running (resume mode).")
    return p.parse_args()


def _build_model_args(seq_len: int):
    ns = SimpleNamespace(**DEFAULT_ARCH)
    ns.seq_len = seq_len
    return ns


def _select_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _render_stage(
    raw_root: str,
    gradcam_root: str,
    guided_root: str,
    output_root: str,
    dpi: int,
    subject_limit: Optional[int] = None,
) -> None:
    """Render per-epoch guided Grad-CAM PNGs (CPU-only, multiprocessing)."""
    from .render import generate_map

    channels_canonical = list(MASS_CHANNEL_MAP.keys())
    os.makedirs(output_root, exist_ok=True)

    sub_ids = sorted(os.listdir(raw_root)) if os.path.isdir(raw_root) else []
    if subject_limit is not None:
        sub_ids = sub_ids[:subject_limit]

    tasks = []
    for sub_id in sub_ids:
        raw_sub = os.path.join(raw_root, sub_id)
        gc_sub = os.path.join(gradcam_root, sub_id)
        gd_sub = os.path.join(guided_root, sub_id)
        required = [
            os.path.join(raw_sub, 'raw_signal.npy'),
            os.path.join(raw_sub, 'annotation.npy'),
            os.path.join(raw_sub, 'prediction.npy'),
            os.path.join(gc_sub, 'gradcam_branch1.npy'),
            os.path.join(gc_sub, 'gradcam_branch2.npy'),
            os.path.join(gd_sub, 'guided_backpropagation.npy'),
        ]
        if not all(os.path.exists(p) for p in required):
            print(f"[render] Skip {sub_id}: missing input files")
            continue

        raw_signal = np.load(required[0])
        annotation = np.load(required[1])
        prediction = np.load(required[2])
        gc_b1 = np.load(required[3])
        gc_b2 = np.load(required[4])
        guided = np.load(required[5])

        # Prefer the subject-specific channel order saved by save_raw if present.
        channels_txt = os.path.join(raw_sub, 'channels.txt')
        if os.path.exists(channels_txt):
            with open(channels_txt) as f:
                channels = [line.strip() for line in f if line.strip()]
        else:
            channels = channels_canonical

        N = min(raw_signal.shape[0], annotation.shape[0], prediction.shape[0],
                gc_b1.shape[0], gc_b2.shape[0], guided.shape[0])
        out_sub = os.path.join(output_root, sub_id)
        os.makedirs(out_sub, exist_ok=True)

        for i in range(N):
            stage = int(annotation[i])
            pred = int(prediction[i])
            if stage < 0 or stage > 4:
                continue
            label = SLEEP_STAGES[stage]
            predict = SLEEP_STAGES[pred]
            save_path = os.path.join(out_sub, f"{i:04d}_{label}_{predict}.png")
            tasks.append((
                guided[i], gc_b1[i], gc_b2[i], raw_signal[i],
                stage, pred, save_path, sub_id, channels, SLEEP_STAGES, dpi,
            ))

    if not tasks:
        print("[render] No rendering tasks found.")
        return
    print(f"[render] Rendering {len(tasks)} epochs with dpi={dpi}")
    num_procs = min(cpu_count(), 32)
    with Pool(processes=num_procs) as pool:
        for _ in tqdm(pool.imap_unordered(generate_map, tasks), total=len(tasks), desc="Render"):
            pass


def main():
    args = _parse_args()
    stages: List[str] = [s.strip() for s in args.stages.split(',') if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"Unknown stages: {unknown}; valid: {ALL_STAGES}")

    device = _select_device(args.device)
    print(f"[pipeline] Device: {device}")
    print(f"[pipeline] Stages: {stages}")

    raw_root = os.path.join(args.output_root, 'raw')
    gradcam_root = os.path.join(args.output_root, 'gradcam')
    guided_root = os.path.join(args.output_root, 'guided')
    render_out_root = os.path.join(args.output_root, 'figures')

    clean = not args.no_clean
    model_args = _build_model_args(args.seq_len)
    subject_ids = None
    if args.subject_ids:
        subject_ids = [s.strip() for s in args.subject_ids.split(',') if s.strip()]

    # Build a shared inference wrapper for stages that don't require ReLU swapping.
    # Guided Backpropagation builds its own wrapper internally.
    wrapper = None
    if any(s in stages for s in ('save_raw', 'gradcam')):
        wrapper = build_wrapper_from_weights(model_args, args.weights, device)

    if 'save_raw' in stages:
        run_save_raw(
            wrapper=wrapper,
            src_root=args.src_root,
            dst_root=raw_root,
            channel_map=MASS_CHANNEL_MAP,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            clean_dst=clean,
            subject_limit=args.subject_limit,
            subject_ids=subject_ids,
        )

    if 'gradcam' in stages:
        run_custom_gradcam(
            wrapper=wrapper,
            src_root=args.src_root,
            dst_root=gradcam_root,
            channel_map=MASS_CHANNEL_MAP,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            clean_dst=clean,
            subject_limit=args.subject_limit,
            subject_ids=subject_ids,
        )

    if 'guided' in stages:
        run_guided_backprop(
            args=model_args,
            weights_path=args.weights,
            src_root=args.src_root,
            dst_root=guided_root,
            channel_map=MASS_CHANNEL_MAP,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            clean_dst=clean,
            subject_limit=args.subject_limit,
            subject_ids=subject_ids,
        )

    if 'render' in stages:
        _render_stage(
            raw_root=raw_root,
            gradcam_root=gradcam_root,
            guided_root=guided_root,
            output_root=render_out_root,
            dpi=args.render_dpi,
            subject_limit=args.subject_limit,
        )

    print("[pipeline] Done.")


if __name__ == '__main__':
    main()
