# -*- coding: utf-8 -*-
"""
seq_encoder_none_cls.py

This module defines the TransformerEncoder class, which implements a sequence encoder 
based on Transformer blocks without using a traditional CLS token for classification. 
It is designed for the LPSGM project to capture temporal dependencies in polysomnography 
(PSG) data sequences. The encoder incorporates spatial and temporal embeddings and applies 
multiple Transformer blocks to extract meaningful sequence-level features for downstream 
tasks such as sleep staging and mental disorder diagnosis.

Key features:
- Spatial embedding for channel information
- Temporal embedding for sequence position encoding
- Multiple Transformer blocks with multi-head self-attention
- Dropout and layer normalization for regularization and stability

This encoder processes input sequences of PSG features and outputs refined temporal features 
without appending a CLS token for classification.
"""

import torch
import torch.nn as nn
from model.seq_encoder.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    """Transformer-based sequence encoder without a CLS token for PSG data sequences."""

    def __init__(self,
                 ch_num: int,
                 seq_len: int,
                 num_heads: int,
                 hidden_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 ch_emb_dim: None,
                 seq_emb_dim: None,
                 num_transformer_blocks: int,
                 ):
        """
        Initialize the TransformerEncoder.

        Args:
            ch_num (int): Number of input channels (e.g., PSG channels).
            seq_len (int): Length of the input sequence.
            num_heads (int): Number of attention heads in each Transformer block.
            hidden_dim (int): Dimensionality of hidden embeddings.
            dropout (float): Dropout rate applied after embeddings and within Transformer blocks.
            attention_dropout (float): Dropout rate applied to attention weights.
            ch_emb_dim (None): Placeholder for channel embedding dimension (unused).
            seq_emb_dim (None): Placeholder for sequence embedding dimension (unused).
            num_transformer_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Learnable spatial embedding for each channel, inspired by Vision Transformer (ViT)
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, hidden_dim))

        # Learnable temporal embedding for each position in the sequence
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))

        # Learnable CLS token per sequence position (not used as a single CLS token)
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim))

        # Dropout layer applied after adding embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks for modeling temporal dependencies
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim,
                hidden_dim,
                dropout,
                attention_dropout,
            ) for _ in range(num_transformer_blocks)
        ])
        
        # Layer normalization applied after Transformer blocks
        self.ln = nn.LayerNorm(hidden_dim)


    def forward(self, input, mask, ch_idx, seq_idx, ori_len):
        """
        Forward pass of the TransformerEncoder.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len * ch_num, hidden_dim),
                                  representing concatenated features for each channel and sequence step.
            mask (torch.Tensor): Boolean mask tensor of shape (batch_size, seq_len * ch_num),
                                 indicating valid positions (True) or padding (False).
            ch_idx (torch.Tensor): Channel indices tensor (unused in this implementation).
            seq_idx (torch.Tensor): Sequence indices tensor (unused in this implementation).
            ori_len (torch.Tensor): Original sequence lengths tensor (unused in this implementation).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim),
                          representing the encoded sequence features.
        """
        bz, seql_cn = input.shape[:2]

        # Flatten input to 2D tensor for processing (batch_size * seq_len * ch_num, hidden_dim)
        x_flat = input.view(bz * seql_cn, -1)

        # Reshape back to 3D tensor (batch_size, seq_len * ch_num, hidden_dim)
        x = x_flat.view(bz, seql_cn, -1)

        # Expand CLS tokens and add temporal embeddings for each sequence position
        cls_tokens = self.cls_token.unsqueeze(0).expand(bz, -1, -1) + self.temporal_embedding.unsqueeze(0).expand(bz, -1, -1)

        # Concatenate CLS tokens at the beginning of the sequence dimension
        x = torch.cat([cls_tokens, x], dim=1)   # Shape: (batch_size, seq_len * ch_num + seq_len, hidden_dim)

        # Create padding mask for CLS tokens (all False since CLS tokens are valid)
        padding_mask = torch.zeros(bz, self.seq_len, dtype=torch.bool, device=mask.device)

        # Concatenate padding mask for CLS tokens with input mask
        mask = torch.cat([padding_mask, mask.bool()], dim=1)

        # Apply dropout to embeddings
        x = self.dropout(x)

        # Pass through each Transformer block with attention masking
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, attn_mask=None)

        # Apply layer normalization to the output features
        feat = self.ln(x)   # Shape: (batch_size, seq_len * ch_num + seq_len, hidden_dim)

        # Extract features corresponding to the CLS tokens (first seq_len positions)
        feat = feat[:, :self.seq_len, :]   # Shape: (batch_size, seq_len, hidden_dim)

        return feat
