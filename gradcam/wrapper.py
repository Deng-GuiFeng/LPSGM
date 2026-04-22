# -*- coding: utf-8 -*-
"""
wrapper.py

Inference wrapper around the main LPSGM model for Grad-CAM and Guided
Backpropagation analyses.

The training-time LPSGM (``model/model.py``) exposes a multi-argument forward
signature ``forward(x, mask, ch_idx, seq_idx, ori_len)`` to support the
padding/masking mechanism used when training across heterogeneous channel
configurations. Grad-CAM hooks and Guided Backpropagation routines are far
simpler to author against a single-argument ``forward(x)`` signature. This
wrapper derives the auxiliary tensors (``mask``, ``ch_idx``, ``seq_idx``,
``ori_len``) deterministically from the input shape, leaving the main LPSGM
untouched.

Design goals:
- Zero intrusion into ``model/``: the wrapper composes LPSGM, never modifies it.
- Transparent access to backbone submodules: ``wrapper.epoch_encoder`` proxies
  to ``wrapper.model.epoch_encoder`` so that Grad-CAM hooks written as
  ``model.epoch_encoder.encoder_branch1[11]`` keep working unchanged.
- Checkpoint compatibility: pre-trained weights (saved with top-level keys
  ``epoch_encoder.*``, ``seq_encoder.*``, ``classifier.*``) are loaded into the
  inner LPSGM, bypassing the ``model.`` prefix that wrapping would otherwise
  introduce.
"""

import torch
import torch.nn as nn

from model.model import LPSGM


class LPSGMInferenceWrapper(nn.Module):
    """
    Single-argument ``forward(x)`` wrapper around LPSGM for interpretability analyses.

    Args:
        args: Namespace-like config object passed through to ``LPSGM``. Must
              define ``seq_len``, ``architecture``, ``ch_num``, ``ch_emb_dim``,
              ``seq_emb_dim``, ``num_transformer_blocks``, ``transformer_num_heads``,
              ``transformer_dropout``, ``transformer_attn_dropout``,
              ``epoch_encoder_dropout``, ``clamp_value``.

    Input to ``forward``:
        x: Tensor of shape ``(batch_size, seq_len * ch_num, 3000)``.

    Output of ``forward``:
        Tensor of shape ``(batch_size, seq_len, 5)`` with per-epoch sleep stage logits.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = LPSGM(args)

    def forward(self, x):
        bz, seql_cn, _ = x.shape
        seql = self.args.seq_len
        cn = seql_cn // seql
        device = x.device

        # seq_idx[i] = which epoch position the i-th (epoch, channel) token belongs to
        seq_idx = torch.arange(seql, device=device, dtype=torch.long).view(seql, 1).expand(seql, cn).reshape(-1)
        seq_idx = seq_idx.unsqueeze(0).expand(bz, -1).contiguous()

        # ch_idx[i] = which channel the i-th (epoch, channel) token belongs to
        ch_idx = torch.arange(cn, device=device, dtype=torch.long).view(1, cn).expand(seql, cn).reshape(-1)
        ch_idx = ch_idx.unsqueeze(0).expand(bz, -1).contiguous()

        # All tokens are valid (no padding) under the fixed-channel Grad-CAM setting
        mask = torch.zeros(bz, seql_cn, dtype=torch.bool, device=device)
        ori_len = [seql_cn] * bz

        return self.model(x, mask, ch_idx, seq_idx, ori_len)

    @property
    def epoch_encoder(self):
        """Proxy for ``self.model.epoch_encoder`` to preserve hook paths like
        ``wrapper.epoch_encoder.encoder_branch1[11]``."""
        return self.model.epoch_encoder

    @property
    def seq_encoder(self):
        """Proxy for ``self.model.seq_encoder``."""
        return self.model.seq_encoder

    @property
    def classifier(self):
        """Proxy for ``self.model.classifier``."""
        return self.model.classifier


def load_pretrained_into_wrapper(wrapper: LPSGMInferenceWrapper, weights_path: str, map_location: str = 'cpu') -> None:
    """
    Load a pretrained LPSGM checkpoint into the inner model of the wrapper.

    Handles two common checkpoint layouts:
    - ``{'model_state_dict': {...}}``: standard format saved by the LPSGM training code.
    - Raw state dict ``{...}``: directly a mapping from parameter names to tensors.

    Also strips the ``module.`` prefix that ``nn.DataParallel`` prepends to keys.

    Args:
        wrapper: The ``LPSGMInferenceWrapper`` instance to populate.
        weights_path: Filesystem path to the ``.pth`` checkpoint.
        map_location: Device argument forwarded to ``torch.load``.
    """
    ckpt = torch.load(weights_path, map_location=map_location)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        cleaned[clean_key] = value

    # Load into the inner LPSGM because the checkpoint keys carry no ``model.`` prefix.
    missing, unexpected = wrapper.model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[wrapper] missing keys when loading checkpoint: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[wrapper] unexpected keys when loading checkpoint: {len(unexpected)} (first 5: {unexpected[:5]})")


def build_wrapper_from_weights(args, weights_path: str, device: torch.device) -> LPSGMInferenceWrapper:
    """
    Convenience factory: build the wrapper, load weights, move to device, set eval mode.

    Args:
        args: Config object consumed by ``LPSGM``.
        weights_path: Filesystem path to the ``.pth`` checkpoint.
        device: Target device for the model.

    Returns:
        A ready-to-use ``LPSGMInferenceWrapper`` in ``eval()`` mode on ``device``.
    """
    wrapper = LPSGMInferenceWrapper(args)
    load_pretrained_into_wrapper(wrapper, weights_path, map_location='cpu')
    wrapper.to(device)
    wrapper.eval()
    return wrapper
