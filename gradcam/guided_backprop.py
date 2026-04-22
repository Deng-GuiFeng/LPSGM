# -*- coding: utf-8 -*-
"""
guided_backprop.py

Stage 3 of the Grad-CAM pipeline: Guided Backpropagation for LPSGM.

Guided Backpropagation produces input-resolution saliency maps by masking the
gradient flow through ReLU activations: on the backward pass, gradients are
kept only where both the forward activation was positive AND the incoming
gradient is positive. This implementation swaps every ``nn.ReLU`` inside LPSGM
with a custom autograd function before running the backward pass.

Because the ReLU replacement mutates the model in place, a **fresh** wrapper
is always constructed inside this module rather than reusing one built for
other stages. The Grad-CAM wrapper therefore remains untouched for downstream
work.
"""

from math import ceil
from typing import Dict, Tuple, Optional
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .preparer import load_mass_subject, build_sequences
from .utils import find_mass_subjects
from .wrapper import LPSGMInferenceWrapper, load_pretrained_into_wrapper


class GuidedBackpropReLU(torch.autograd.Function):
    """
    Custom ReLU whose backward pass zeros out gradients at positions where
    either the forward activation was non-positive or the incoming gradient
    is non-positive.
    """

    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros_like(input), input, positive_mask)
        ctx.save_for_backward(input, positive_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        grad_output_positive = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros_like(input),
            grad_output,
            positive_mask * grad_output_positive,
        )
        return grad_input


class GuidedBackpropReLUModule(nn.Module):
    """Thin ``nn.Module`` wrapper around ``GuidedBackpropReLU`` for drop-in replacement."""

    def forward(self, x):
        return GuidedBackpropReLU.apply(x)


def replace_relu_with_guided(module: nn.Module) -> None:
    """
    Recursively replace every ``nn.ReLU`` in ``module`` with ``GuidedBackpropReLUModule``.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedBackpropReLUModule())
        else:
            replace_relu_with_guided(child)


def _build_guided_wrapper(args, weights_path: str, device: torch.device) -> LPSGMInferenceWrapper:
    """Build a fresh LPSGM wrapper, load weights, and swap every ReLU for the guided variant."""
    wrapper = LPSGMInferenceWrapper(args)
    load_pretrained_into_wrapper(wrapper, weights_path, map_location='cpu')
    wrapper.to(device)
    wrapper.eval()
    replace_relu_with_guided(wrapper)
    return wrapper


def run_guided_backprop_for_subject(
    wrapper: LPSGMInferenceWrapper,
    sub_id: str,
    psg_path: str,
    ano_path: str,
    dst_root: str,
    channel_map: Dict[str, Tuple],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Produce input-resolution guided-backprop saliency maps for ``time_step=0`` of each window.

    Output under ``{dst_root}/{sub_id}/``:
    - ``guided_backpropagation.npy``: shape ``(num_windows, num_channels, 3000)``.
    """
    print(f"[guided_backprop] Processing {sub_id}")

    signals, channels, _ = load_mass_subject(psg_path, ano_path, channel_map, normalize=True)
    sequences_np = build_sequences(signals, seq_len)
    if sequences_np.shape[0] == 0:
        print(f"[guided_backprop] Skipping {sub_id}: not enough epochs")
        return
    num_channels = len(channels)

    # Batched Guided Backprop: process B windows per forward/backward pass.
    num_windows = sequences_np.shape[0]
    maps_list = []
    for b_start in tqdm(range(0, num_windows, batch_size), desc="Guided backprop"):
        b_end = min(b_start + batch_size, num_windows)
        B = b_end - b_start
        wrapper.zero_grad()
        batch_np = sequences_np[b_start:b_end]
        batch_t = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
        batch_t.requires_grad = True

        logits = wrapper(batch_t)                     # (B, seq_len, 5)
        preds = logits[:, 0, :].argmax(dim=-1)        # (B,)
        # Sum the predicted-class logit at time_step=0 across the batch, then backward.
        targets = logits[:, 0, :].gather(1, preds.view(-1, 1)).squeeze(-1)  # (B,)
        targets.sum().backward()

        grads_np = batch_t.grad.detach().cpu().numpy()  # (B, seq_len*cn, 3000)
        # Extract the first epoch of each window: reshape (B, seq_len, cn, 3000)[:, 0].
        first_epoch = grads_np.reshape(B, seq_len, num_channels, -1)[:, 0]  # (B, cn, 3000)
        maps_list.append(first_epoch)

    maps = np.concatenate(maps_list, axis=0)  # (num_windows, cn, 3000)

    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)
    np.save(os.path.join(dst_sub_dir, "guided_backpropagation.npy"), maps.astype(np.float32))


def run_guided_backprop(
    args,
    weights_path: str,
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
    Run Guided Backpropagation over every MASS subject under ``src_root``.

    A fresh wrapper with swapped ReLU modules is built internally, leaving any
    caller-owned wrapper untouched.

    Args:
        args: Config object passed to ``LPSGM``.
        weights_path: Path to pretrained LPSGM checkpoint.
        src_root: Directory containing MASS ``{sub_id} PSG.edf`` + ``{sub_id} Base.edf`` pairs.
        dst_root: Output root directory.
        channel_map: Canonical channel → EDF channel options mapping.
        seq_len: LPSGM input window length in epochs.
        batch_size: Inference / backward batch size.
        device: CUDA or CPU device.
        clean_dst: If True, remove ``dst_root`` before writing.
        subject_limit: Optional cap on the number of subjects processed.
        subject_ids: Optional whitelist of subject IDs.
    """
    print(f"[guided_backprop] Source: {src_root}")
    print(f"[guided_backprop] Destination: {dst_root}")

    if clean_dst:
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    guided_wrapper = _build_guided_wrapper(args, weights_path, device)

    subjects = find_mass_subjects(src_root, subject_ids=subject_ids)
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    print(f"[guided_backprop] Found {len(subjects)} subjects")

    for sub_id, psg_path, ano_path in subjects:
        marker = os.path.join(dst_root, sub_id, "guided_backpropagation.npy")
        if os.path.exists(marker):
            print(f"[guided_backprop] Skipping {sub_id}: already processed")
            continue
        try:
            run_guided_backprop_for_subject(
                guided_wrapper, sub_id, psg_path, ano_path,
                dst_root, channel_map, seq_len, batch_size, device,
            )
        except Exception as exc:
            print(f"[guided_backprop] Failed on {sub_id}: {exc}")
