# -*- coding: utf-8 -*-
"""
custom_gradcam.py

Stage 2 of the Grad-CAM pipeline: dual-branch Grad-CAM for LPSGM's Epoch Encoder.

For each 30-second epoch at the first position of a sliding window, two
Grad-CAM activation maps are computed - one per CNN branch - targeted at
``encoder_branch{1,2}[11]``, the last Conv1d of each branch, which produces
feature maps of temporal length 63 (branch 1) and 13 (branch 2). The per-branch
maps are saved as raw arrays and combined downstream in ``render.py`` via
element-wise summation after resampling to input resolution.

Target class policy: LPSGM's own prediction is used as the target class so
that Grad-CAM explains the model's decision rather than the expert label.
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
from .wrapper import LPSGMInferenceWrapper


class CustomGradCAM:
    """
    Grad-CAM extractor bound to a single target layer.

    Forward and backward hooks capture the target layer's activations and the
    gradient of the selected class score with respect to those activations.
    The weights are obtained by pooling the gradients along the temporal axis
    and then used to weight-sum the activations along the feature-channel axis.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def __call__(self, input_tensor: torch.Tensor, target_class: int, time_step: int) -> torch.Tensor:
        """
        Compute Grad-CAM for ``(target_class, time_step)`` on a single window.

        Args:
            input_tensor: Tensor of shape ``(1, seq_len * ch_num, 3000)``.
            target_class: Integer class index (0..4).
            time_step: Epoch index within the window (0..seq_len-1).

        Returns:
            Tensor of shape ``(seq_len * ch_num, feature_length)`` with non-negative activations.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)                     # (1, seq_len, 5)
        target = output[0, time_step, target_class]
        target.backward(retain_graph=True)

        gradients = self.gradients                             # (seq_len*ch_num, 256, L)
        activations = self.activations                         # (seq_len*ch_num, 256, L)

        # Global average pooling of gradients along the temporal axis yields
        # per-feature-channel weights.
        weights = torch.mean(gradients, dim=[2])               # (seq_len*ch_num, 256)
        # Weighted sum across feature channels, keeping the temporal axis.
        cam = torch.sum(weights[:, :, None] * activations, dim=1)  # (seq_len*ch_num, L)
        cam = torch.clamp(cam, min=0)                          # ReLU
        return cam

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


class BatchedGradCAM:
    """
    Batched Grad-CAM extractor: process B windows in one forward/backward pass
    instead of B separate passes, yielding a substantial speedup over the
    per-window implementation.

    Mathematically equivalent: because ``sum_b out[b, t, target_b]`` has an
    additive gradient structure, each sample's gradient flow is independent,
    so the hook's captured ``gradients[b]`` and ``activations[b]`` correctly
    correspond to window ``b``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._hook_handles.append(target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o)))
        self._hook_handles.append(target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0])))

    def __call__(
        self,
        input_tensor: torch.Tensor,     # (B, seq_len*cn, 3000)
        target_classes: torch.Tensor,   # (B,) int64
        seq_len: int,
        num_channels: int,
        time_step: int = 0,
    ) -> np.ndarray:
        """
        Compute batched Grad-CAM at a single ``time_step`` for every window.

        Returns:
            Numpy array of shape ``(B, num_channels, feature_length)``.
        """
        self.model.zero_grad()
        out = self.model(input_tensor)  # (B, seq_len, 5)
        # Gather the target-class logit at time_step for each element.
        targets = out[:, time_step, :].gather(1, target_classes.view(-1, 1)).squeeze(-1)  # (B,)
        targets.sum().backward(retain_graph=True)

        B = input_tensor.size(0)
        a = self.activations.view(B, seq_len, num_channels, 256, -1)
        g = self.gradients.view(B, seq_len, num_channels, 256, -1)
        a_t = a[:, time_step]                          # (B, cn, 256, L)
        g_t = g[:, time_step]                          # (B, cn, 256, L)
        weights = g_t.mean(dim=-1)                     # (B, cn, 256)
        cam = (weights.unsqueeze(-1) * a_t).sum(dim=2) # (B, cn, L)
        cam = torch.clamp(cam, min=0)
        return cam.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def run_custom_gradcam_for_subject(
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
    Produce per-window Grad-CAM activations for both branches and save them.

    Output files under ``{dst_root}/{sub_id}/``:
    - ``gradcam_branch1.npy``: shape ``(num_windows, num_channels, 63)``.
    - ``gradcam_branch2.npy``: shape ``(num_windows, num_channels, 13)``.
    """
    print(f"[custom_gradcam] Processing {sub_id}")

    signals, channels, _ = load_mass_subject(psg_path, ano_path, channel_map, normalize=True)
    sequences_np = build_sequences(signals, seq_len)
    if sequences_np.shape[0] == 0:
        print(f"[custom_gradcam] Skipping {sub_id}: not enough epochs")
        return
    num_channels = len(channels)

    # First pass: infer the target class per window (used as the Grad-CAM target class).
    predictions = []
    num_batches = ceil(len(sequences_np) / batch_size)
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Inference"):
            batch_np = sequences_np[b * batch_size:(b + 1) * batch_size]
            batch_t = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            logits = wrapper(batch_t)
            probs = torch.softmax(logits[:, 0, :], dim=-1)
            predictions.append(probs.argmax(dim=-1).cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)

    # Register Grad-CAM hooks on both branches (index 11 = last Conv1d).
    extractor_b1 = BatchedGradCAM(wrapper, wrapper.epoch_encoder.encoder_branch1[11])
    extractor_b2 = BatchedGradCAM(wrapper, wrapper.epoch_encoder.encoder_branch2[11])
    try:
        maps_b1 = []
        maps_b2 = []
        num_windows = len(predictions)
        for b_start in tqdm(range(0, num_windows, batch_size), desc="Grad-CAM"):
            b_end = min(b_start + batch_size, num_windows)
            batch_np = sequences_np[b_start:b_end]
            batch_t = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            tgt = torch.as_tensor(predictions[b_start:b_end], dtype=torch.long, device=device)
            cam_b1 = extractor_b1(batch_t, tgt, seq_len, num_channels, time_step=0)
            cam_b2 = extractor_b2(batch_t, tgt, seq_len, num_channels, time_step=0)
            maps_b1.append(cam_b1)
            maps_b2.append(cam_b2)
        maps_b1 = np.concatenate(maps_b1, axis=0)   # (num_windows, cn, 63)
        maps_b2 = np.concatenate(maps_b2, axis=0)   # (num_windows, cn, 13)
    finally:
        extractor_b1.remove_hooks()
        extractor_b2.remove_hooks()

    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)
    np.save(os.path.join(dst_sub_dir, "gradcam_branch1.npy"), maps_b1.astype(np.float32))
    np.save(os.path.join(dst_sub_dir, "gradcam_branch2.npy"), maps_b2.astype(np.float32))


def run_custom_gradcam(
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
    Run Grad-CAM over every MASS subject under ``src_root``.

    Args:
        wrapper: Loaded LPSGM wrapper (already in eval mode).
        src_root: Directory containing MASS ``{sub_id} PSG.edf`` + ``{sub_id} Base.edf`` pairs.
        dst_root: Output root directory (``{dst_root}/{sub_id}/gradcam_branch{1,2}.npy``).
        channel_map: Canonical channel → EDF channel options mapping.
        seq_len: LPSGM input window length in epochs.
        batch_size: Inference / backward batch size.
        device: CUDA or CPU device.
        clean_dst: If True, remove ``dst_root`` before writing.
        subject_limit: Optional cap on the number of subjects processed.
        subject_ids: Optional whitelist of subject IDs.
    """
    print(f"[custom_gradcam] Source: {src_root}")
    print(f"[custom_gradcam] Destination: {dst_root}")

    if clean_dst:
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    subjects = find_mass_subjects(src_root, subject_ids=subject_ids)
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    print(f"[custom_gradcam] Found {len(subjects)} subjects")

    for sub_id, psg_path, ano_path in subjects:
        marker = os.path.join(dst_root, sub_id, "gradcam_branch2.npy")
        if os.path.exists(marker):
            print(f"[custom_gradcam] Skipping {sub_id}: already processed")
            continue
        try:
            run_custom_gradcam_for_subject(
                wrapper, sub_id, psg_path, ano_path,
                dst_root, channel_map, seq_len, batch_size, device,
            )
        except Exception as exc:
            print(f"[custom_gradcam] Failed on {sub_id}: {exc}")
