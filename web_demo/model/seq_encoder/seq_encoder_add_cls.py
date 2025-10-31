# -*- coding: utf-8 -*-
"""
Sequence Encoder with Additive CLS Token Aggregation.

This module defines a Transformer-based sequence encoder used in the LPSGM project for modeling temporal dependencies in polysomnography data sequences. 
It incorporates learnable spatial and temporal embeddings, and utilizes an additive CLS token mechanism to aggregate sequence information.
The encoder processes input features extracted from epochs across multiple channels and time steps, enhancing representation for downstream tasks such as sleep staging and mental disorder diagnosis.

Key components:
- Spatial embedding for channel-specific feature encoding
- Temporal embedding for position-specific feature encoding
- Additive CLS tokens for sequence-level aggregation
- Stacked Transformer blocks for capturing complex temporal relationships
"""

import torch
import torch.nn as nn
from model.seq_encoder.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    """Transformer Encoder with additive CLS token aggregation for sequence modeling.

    This encoder applies spatial and temporal embeddings to input features, prepends additive CLS tokens for sequence-level representation, 
    and processes the sequence through multiple Transformer blocks.

    Args:
        ch_num (int): Number of input channels.
        seq_len (int): Length of the input sequence (number of epochs).
        num_heads (int): Number of attention heads in each Transformer block.
        hidden_dim (int): Dimensionality of hidden embeddings.
        dropout (float): Dropout rate applied after embedding addition.
        attention_dropout (float): Dropout rate applied within attention layers.
        ch_emb_dim (None): Placeholder parameter, not used in this implementation.
        seq_emb_dim (None): Placeholder parameter, not used in this implementation.
        num_transformer_blocks (int): Number of stacked Transformer blocks.
    """

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
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Learnable spatial embeddings for each channel (channel-wise feature encoding)
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, hidden_dim))  # Inspired by Vision Transformer (ViT)

        # Learnable temporal embeddings for each position in the sequence (epoch-wise position encoding)
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))

        # Learnable CLS tokens for each position in the sequence, used for additive aggregation
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim))

        # Dropout layer applied after embedding addition
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks for capturing temporal dependencies
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
        Forward pass of the Transformer Encoder.

        Args:
            input (torch.Tensor): Input features of shape (batch_size, seq_len * ch_num, hidden_dim).
            mask (torch.Tensor): Boolean mask indicating valid positions, shape (batch_size, seq_len * ch_num).
            ch_idx (torch.Tensor): Channel indices for each input token, shape (batch_size, seq_len * ch_num).
            seq_idx (torch.Tensor): Sequence position indices for each input token, shape (batch_size, seq_len * ch_num).
            ori_len (torch.Tensor): Original lengths of sequences in the batch, shape (batch_size,).

        Returns:
            torch.Tensor: Output features after Transformer encoding, shape (batch_size, seq_len, hidden_dim).
        """
        bz, seql_cn = input.shape[:2]

        # Flatten batch and sequence-channel dimensions for embedding lookup
        input_flat = input.view(bz * seql_cn, -1)  # Shape: (batch_size * seq_len * ch_num, hidden_dim)
        ch_idx_flat = ch_idx.view(bz * seql_cn,)   # Shape: (batch_size * seq_len * ch_num,)
        seq_idx_flat = seq_idx.view(bz * seql_cn,) # Shape: (batch_size * seq_len * ch_num,)

        # Retrieve spatial embeddings corresponding to channel indices
        SE = self.spatial_embedding[ch_idx_flat]

        # Retrieve temporal embeddings corresponding to sequence position indices
        TE = self.temporal_embedding[seq_idx_flat]

        # Add input features with spatial and temporal embeddings (element-wise addition)
        input_sum = input_flat + SE + TE

        # Reshape back to (batch_size, seq_len * ch_num, hidden_dim)
        x = input_sum.view(bz, seql_cn, -1)

        # Prepare additive CLS tokens by combining CLS token embeddings with temporal embeddings
        # CLS tokens have shape (batch_size, seq_len, hidden_dim)
        cls_tokens = self.cls_token.unsqueeze(0).expand(bz, -1, -1) + self.temporal_embedding.unsqueeze(0).expand(bz, -1, -1)

        # Concatenate CLS tokens at the beginning of the sequence dimension
        # Resulting shape: (batch_size, seq_len * ch_num + seq_len, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)

        # Create padding mask for CLS tokens (all False since CLS tokens are always valid)
        padding_mask = torch.zeros(bz, self.seq_len, dtype=torch.bool, device=mask.device)

        # Concatenate padding mask for CLS tokens with original mask
        mask = torch.cat([padding_mask, mask.bool()], dim=1)

        # Apply dropout after embedding addition and concatenation
        x = self.dropout(x)

        # Pass through each Transformer block sequentially
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, attn_mask=None)

        # Apply layer normalization to the output features
        feat = self.ln(x)   # Shape: (batch_size, seq_len * ch_num + seq_len, hidden_dim)

        # Extract features corresponding to CLS tokens only (aggregated sequence representation)
        feat = feat[:, :self.seq_len, :]   # Shape: (batch_size, seq_len, hidden_dim)

        return feat
