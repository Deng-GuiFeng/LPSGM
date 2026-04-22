# -*- coding: utf-8 -*-
"""
model.py

Pool-then-classify subject-level classifier built on top of the LPSGM backbone.

The model reuses ``EpochEncoder`` and the configured ``TransformerEncoder``
from the main ``model/`` package and adds a mean-pooling step across the
sequence dimension followed by a single linear classifier. This architecture
is shared by every downstream disorder-classification task (narcolepsy, OSA,
depression): the only task-specific knob is ``num_classes``.

Pretrained LPSGM weights saved under top-level keys ``epoch_encoder.*`` and
``seq_encoder.*`` are loaded via ``load_from_pretrained`` into the backbone
while the task-specific classifier head is trained from scratch.
"""

import os
from typing import Dict

import torch
import torch.nn as nn

from model.epoch_encoder import EpochEncoder


class LPSGMPooledClassifier(nn.Module):
    """
    LPSGM backbone + mean-pooling + linear classifier for subject-level tasks.

    Forward pipeline (identical to training-time LPSGM except for the pooling head):

        x -> clamp -> flatten channels into tokens -> EpochEncoder
          -> SeqEncoder (per-token features)
          -> mean over sequence dimension
          -> Linear classifier -> logits of shape (batch, num_classes)

    Args:
        args: Namespace-like configuration object. Must define ``architecture``,
              ``epoch_encoder_dropout``, ``transformer_num_heads``,
              ``transformer_dropout``, ``transformer_attn_dropout``, ``ch_num``,
              ``seq_len``, ``ch_emb_dim``, ``seq_emb_dim``, ``num_transformer_blocks``
              and ``clamp_value`` (same fields as the training-time LPSGM).
        num_classes: Number of output classes.
    """

    def __init__(self, args, num_classes: int):
        super().__init__()
        self.args = args

        self.epoch_encoder = EpochEncoder(args.epoch_encoder_dropout)

        if args.architecture == 'cat_cls':
            from model.seq_encoder.seq_encoder_cat_cls import TransformerEncoder
        elif args.architecture == 'add_cls':
            from model.seq_encoder.seq_encoder_add_cls import TransformerEncoder
        elif args.architecture == 'cat_avg':
            from model.seq_encoder.seq_encoder_cat_avg import TransformerEncoder
        elif args.architecture == 'none_cls':
            from model.seq_encoder.seq_encoder_none_cls import TransformerEncoder
        else:
            raise NotImplementedError(f"Unknown architecture: {args.architecture}")

        self.seq_encoder = TransformerEncoder(
            ch_num=args.ch_num,
            seq_len=args.seq_len,
            num_heads=args.transformer_num_heads,
            hidden_dim=512,
            dropout=args.transformer_dropout,
            attention_dropout=args.transformer_attn_dropout,
            ch_emb_dim=args.ch_emb_dim,
            seq_emb_dim=args.seq_emb_dim,
            num_transformer_blocks=args.num_transformer_blocks,
        )

        # Feature dimension after the Transformer depends on architecture variant:
        # 'cat_*' variants concatenate channel and sequence embeddings, adding
        # ch_emb_dim + seq_emb_dim extra dimensions to the 512-dim token.
        if args.architecture in ['add_cls', 'none_cls']:
            feat_dim = 512
        else:
            feat_dim = 512 + args.ch_emb_dim + args.seq_emb_dim

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, mask, ch_idx, seq_idx, ori_len):
        """
        Args:
            x: Tensor of shape ``(batch, seq_len * ch_num, 3000)``.
            mask: Bool tensor of shape ``(batch, seq_len * ch_num)`` marking
                  padding tokens with True.
            ch_idx: Long tensor of channel indices, same shape as ``mask``.
            seq_idx: Long tensor of sequence-position indices, same shape as ``mask``.
            ori_len: Per-sample original token count before padding, ``list[int]``.

        Returns:
            Tensor of shape ``(batch, num_classes)`` with subject-level logits.
        """
        mask = mask.bool()
        ch_idx = ch_idx.long()
        seq_idx = seq_idx.long()
        x = torch.clamp(x, -self.args.clamp_value, self.args.clamp_value)

        bz, seql_cn, _ = x.shape
        x = x.view(bz * seql_cn, 1, -1)
        x = self.epoch_encoder(x)
        x = x.view(bz, seql_cn, -1)
        feat = self.seq_encoder(x, mask, ch_idx, seq_idx, ori_len)  # (bz, seq_len, feat_dim)

        pooled = feat.mean(dim=1)                  # (bz, feat_dim)
        logits = self.classifier(pooled)           # (bz, num_classes)
        return logits

    def extract_features(self, x, mask, ch_idx, seq_idx, ori_len):
        """
        Return the mean-pooled subject-level feature before the classifier head.

        Useful for post-hoc evaluators (``simple_eval`` / ``hierarchical_eval``)
        that train a separate classifier on frozen LPSGM features.

        Returns:
            Tensor of shape ``(batch, feat_dim)``.
        """
        mask = mask.bool()
        ch_idx = ch_idx.long()
        seq_idx = seq_idx.long()
        x = torch.clamp(x, -self.args.clamp_value, self.args.clamp_value)
        bz, seql_cn, _ = x.shape
        x = x.view(bz * seql_cn, 1, -1)
        x = self.epoch_encoder(x)
        x = x.view(bz, seql_cn, -1)
        feat = self.seq_encoder(x, mask, ch_idx, seq_idx, ori_len)
        return feat.mean(dim=1)

    def load_from_pretrained(self, ckpt_path: str) -> Dict[str, int]:
        """
        Load pretrained LPSGM backbone weights into this model.

        Only ``epoch_encoder.*`` and ``seq_encoder.*`` keys are copied; the
        classifier head is initialized from scratch. The ``module.`` prefix
        that ``nn.DataParallel`` prepends to checkpoint keys is stripped.

        Args:
            ckpt_path: Filesystem path to the ``.pth`` checkpoint.

        Returns:
            Summary dict with counts of loaded, skipped, missing, and unexpected keys.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        sd = state.get('model_state_dict', state)

        new_sd = {}
        loaded = skipped = 0
        for k, v in sd.items():
            k = k.replace('module.', '')
            if k.startswith('epoch_encoder.') or k.startswith('seq_encoder.'):
                new_sd[k] = v
                loaded += 1
            else:
                skipped += 1

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        return {
            'loaded': loaded,
            'skipped': skipped,
            'missing': len(missing),
            'unexpected': len(unexpected),
        }
