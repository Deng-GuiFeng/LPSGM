# -*- coding: utf-8 -*-
"""
model.py

Main LPSGM model integrating epoch encoder, sequence encoder, and classifier.

This module defines the LPSGM class, which implements the core architecture of the Large Polysomnography Model (LPSGM) for sleep staging and mental disorder diagnosis. 
The model consists of three main components:
1. Epoch Encoder: A dual-branch CNN that extracts features from 30-second PSG epochs.
2. Sequence Encoder: Transformer-based encoder that captures temporal dependencies across sequences of epochs.
3. Classifier: Predicts sleep stages or mental disorder classes based on encoded features.

The model supports multiple architectural variants for sequence encoding and flexible channel configurations, enabling robust representation learning from multi-channel PSG data.
"""

import torch
import torch.nn as nn

from model.classifier import Classifier
from model.epoch_encoder import EpochEncoder


class LPSGM(nn.Module):
    def __init__(self, args):
        """
        Initialize the LPSGM model with specified configurations.

        Args:
            args (Namespace): Configuration parameters including architecture type, dropout rates,
                              embedding dimensions, sequence length, channel number, and transformer settings.
        """
        super(LPSGM, self).__init__()

        self.args = args

        # Initialize the epoch encoder with dropout rate for regularization
        self.epoch_encoder = EpochEncoder(args.epoch_encoder_dropout)

        # Dynamically import the appropriate TransformerEncoder variant based on architecture type
        if args.architecture == 'cat_cls':
            from model.seq_encoder.seq_encoder_cat_cls import TransformerEncoder
            # Feature dimension includes epoch encoder output plus channel and sequence embeddings concatenated
            feat_dim = 512 + args.ch_emb_dim + args.seq_emb_dim
        elif args.architecture == 'add_cls':
            from model.seq_encoder.seq_encoder_add_cls import TransformerEncoder
            # Feature dimension is only the epoch encoder output (512)
            feat_dim = 512
        elif args.architecture == 'cat_avg':
            from model.seq_encoder.seq_encoder_cat_avg import TransformerEncoder
            # Feature dimension includes epoch encoder output plus channel and sequence embeddings concatenated
            feat_dim = 512 + args.ch_emb_dim + args.seq_emb_dim
        elif args.architecture == 'none_cls':
            from model.seq_encoder.seq_encoder_none_cls import TransformerEncoder
            # Feature dimension is only the epoch encoder output (512)
            feat_dim = 512
        else:
            # Raise error if architecture type is not supported
            raise NotImplementedError

        # Initialize the sequence encoder (Transformer) with specified hyperparameters
        self.seq_encoder = TransformerEncoder(
            ch_num = args.ch_num,                            # Number of PSG channels
            seq_len = args.seq_len,                          # Length of the input sequence (number of epochs)
            num_heads = args.transformer_num_heads,          # Number of attention heads in Transformer
            hidden_dim = 512,                                # Hidden dimension size for Transformer layers
            dropout = args.transformer_dropout,              # Dropout rate for Transformer layers
            attention_dropout = args.transformer_attn_dropout, # Dropout rate for attention weights
            ch_emb_dim = args.ch_emb_dim,                    # Dimension of channel embeddings
            seq_emb_dim = args.seq_emb_dim,                  # Dimension of sequence embeddings
            num_transformer_blocks = args.num_transformer_blocks, # Number of Transformer blocks
        )

        # Initialize the classifier to output predictions for 5 classes (sleep stages)
        self.classifier = Classifier(
            feat_dim = feat_dim,  # Input feature dimension to classifier
            num_classes = 5,      # Number of output classes for classification
            )
        

    def forward(self, x, mask, ch_idx, seq_idx, ori_len):
        """
        Forward pass of the LPSGM model.

        Args:
            x (torch.Tensor): Input PSG data tensor of shape (batch_size, seq_len * ch_num, 3000),
                              where 3000 corresponds to 30-second epoch samples.
            mask (torch.Tensor): Attention mask tensor indicating valid positions, shape (batch_size, seq_len * ch_num).
            ch_idx (torch.Tensor): Channel index tensor indicating channel identities, shape (batch_size, seq_len * ch_num).
            seq_idx (torch.Tensor): Sequence index tensor indicating position of epochs in sequence, shape (batch_size, seq_len * ch_num).
            ori_len (torch.Tensor): Original sequence lengths before padding, shape (batch_size,).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, seq_len, 5), representing class scores per epoch.
        """
        # Clamp input signal values to a specified range to reduce noise and outliers
        x = torch.clamp(x, -self.args.clamp_value, self.args.clamp_value)

        bz, seql_cn, _ = x.shape  # Batch size, sequence length times channel number, number of samples per epoch

        # Reshape input to merge batch and sequence-channel dimensions for epoch encoding
        x = x.view(bz*seql_cn, 1, -1)   # Shape: (batch_size * seq_len * ch_num, 1, 3000)

        # Extract features from each 30-second epoch using the epoch encoder
        x = self.epoch_encoder(x)       # Shape: (batch_size * seq_len * ch_num, hidden_dim)

        # Reshape back to separate batch and sequence-channel dimensions
        x_epoch = x.view(bz, seql_cn, -1)   # Shape: (batch_size, seq_len * ch_num, hidden_dim)

        # Encode temporal dependencies and contextual information across the sequence using the Transformer encoder
        feat = self.seq_encoder(x_epoch, mask, ch_idx, seq_idx, ori_len)   # Shape: (batch_size, seq_len, hidden_dim)

        # Classify each epoch's feature representation into one of the 5 sleep stages
        out = self.classifier(feat)    # Shape: (batch_size, seq_len, 5)

        return out