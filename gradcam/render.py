# -*- coding: utf-8 -*-
"""
render.py

Stage 4 of the Grad-CAM pipeline: per-epoch PNG visualizations overlaying
guided Grad-CAM on the raw PSG waveforms.

The final visualization fuses the two saliency analyses produced upstream:

1. Grad-CAM from both branches of the Epoch Encoder is linearly resampled to
   the raw signal's time resolution (3000 samples per 30-second epoch) and
   combined by **element-wise summation**.
2. Guided Backpropagation saliency is transformed into a Hilbert envelope,
   Gaussian-smoothed (sigma=10), and resampled to the raw-signal resolution.
3. The two are combined by pointwise multiplication (``gradcam_fused * envelope``),
   producing a per-sample guided Grad-CAM intensity available for callers that
   need per-sample attribution.

Both outputs are min-max normalized to ``[0, 1]``. The default per-epoch
rendering uses ``gradcam_fused`` (step 1) as the per-sample background color
behind the raw waveform trace; ``guided_gradcam`` (step 3) is exposed through
``fuse_maps()`` for downstream callers.

This module performs no model operations; it is a pure post-processing step
applied to the ``.npy`` files produced by ``save_raw``, ``custom_gradcam``,
and ``guided_backprop``.
"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


def _interp_to_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolate a ``(cn, L)`` array along its time axis to ``(cn, target_length)``."""
    cn, L = x.shape
    out = np.zeros((cn, target_length), dtype=x.dtype)
    src = np.linspace(0, L - 1, L)
    dst = np.linspace(0, L - 1, target_length)
    for c in range(cn):
        out[c] = interp1d(src, x[c], kind='linear')(dst)
    return out


def fuse_maps(
    gradcam_branch1: np.ndarray,
    gradcam_branch2: np.ndarray,
    guided_bp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine per-branch Grad-CAM and Guided Backpropagation into the final saliency maps.

    Args:
        gradcam_branch1: ``(cn, 63)`` Grad-CAM from branch 1 (small filters).
        gradcam_branch2: ``(cn, 13)`` Grad-CAM from branch 2 (large filters).
        guided_bp: ``(cn, 3000)`` raw guided-backprop gradients at input resolution.

    Returns:
        Tuple of two ``(cn, 3000)`` arrays, both min-max normalized to ``[0, 1]``:
        - ``gradcam_fused``: sum of the two branches after interpolation to input resolution.
        - ``guided_gradcam``: ``gradcam_fused * envelope(guided_bp)``.
    """
    T = guided_bp.shape[1]
    gc_b1 = _interp_to_length(gradcam_branch1, T)
    gc_b2 = _interp_to_length(gradcam_branch2, T)
    gradcam_fused = gc_b1 + gc_b2  # element-wise sum across branches

    # Envelope of the guided backprop via Hilbert transform + Gaussian smoothing.
    envelope = np.abs(hilbert(guided_bp, axis=1))
    envelope = gaussian_filter1d(envelope, sigma=10, axis=1)

    guided_gradcam = gradcam_fused * envelope

    def _norm(x: np.ndarray) -> np.ndarray:
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    return _norm(gradcam_fused), _norm(guided_gradcam)


def generate_map(task_args):
    """
    Render a single epoch visualization as a PNG.

    ``task_args`` is a tuple packed by the multiprocessing caller:
        (guided_bp, gradcam_b1, gradcam_b2, raw, ano, pred, save_path,
         sub_id, channels, sleep_stages, dpi)

    The background color uses the **two-branch summed Grad-CAM**
    (``gradcam_fused``), not the guided-BP-modulated variant. At per-epoch
    scale the Guided Backpropagation envelope is dominated by sharp spikes;
    its element-wise product collapses the Grad-CAM into a sparse,
    filamentary pattern that is difficult to read. ``fuse_maps`` still
    exposes ``guided_gradcam`` for callers that need per-sample attribution.
    """
    (guided_bp, gradcam_b1, gradcam_b2, raw, ano, pred, save_path,
     sub_id, channels, sleep_stages, dpi) = task_args

    num_channels = len(channels)
    gradcam_fused, _ = fuse_maps(gradcam_b1, gradcam_b2, guided_bp)

    fig, axes = plt.subplots(num_channels, 1, figsize=(40, 20), sharex=True)
    fig.suptitle(
        f'{sub_id}  Label: {sleep_stages[ano]}  Predict: {sleep_stages[pred]}',
        fontsize=40,
    )

    T = np.linspace(0, 30, raw.shape[1])
    for i in range(num_channels):
        for j in range(raw.shape[1]):
            end = T[j + 1] if j + 1 < raw.shape[1] else T[j]
            axes[i].axvspan(T[j], end, color=plt.cm.cool(gradcam_fused[i, j]), alpha=0.95)
        axes[i].plot(T, raw[i], color='black')
        axes[i].set_ylabel(channels[i], fontsize=30)
        axes[i].set_xlim(0, 30)
        axes[i].set_ylim(raw[i].min(), raw[i].max())
        for t in range(5, 31, 5):
            axes[i].axvline(x=t, color='white', linestyle='--', linewidth=2.0)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[-1].set_xlabel('Time (s)', fontsize=30)
    axes[-1].tick_params(axis='x', labelsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=dpi)
    plt.close()
