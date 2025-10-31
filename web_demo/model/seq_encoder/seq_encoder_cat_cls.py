# -*- coding: utf-8 -*-
"""
Sequence Encoder with Concatenated CLS Token for LPSGM Project.

This module implements a Transformer-based sequence encoder designed to process 
polysomnography (PSG) data sequences. It integrates spatial (channel) embeddings, 
temporal embeddings, and a concatenated CLS token to capture both spatial and temporal 
dependencies across multi-channel PSG epochs. The encoder applies multiple Transformer 
blocks to model complex temporal relationships, facilitating downstream tasks such as 
sleep staging and mental disorder diagnosis.

Key Features:
- Learnable spatial embeddings representing PSG channels
- Learnable temporal embeddings representing sequence positions
- Concatenated CLS token for sequence-level representation
- Multi-head self-attention Transformer blocks with dropout and layer normalization
- Flexible handling of input masks for variable-length sequences

This component is a critical part of the LPSGM architecture, enabling effective 
sequence modeling of PSG data for robust feature extraction.
"""

import torch
import torch.nn as nn
from model.seq_encoder.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    """Transformer Encoder module for sequence representation with concatenated CLS token."""

    def __init__(self,
                 ch_num: int,
                 seq_len: int,
                 num_heads: int,
                 hidden_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 ch_emb_dim: int,
                 seq_emb_dim: int,
                 num_transformer_blocks: int,
                 ):
        """
        Initialize the TransformerEncoder.

        Args:
            ch_num (int): Number of input channels (spatial dimension).
            seq_len (int): Length of the input sequence.
            num_heads (int): Number of attention heads in multi-head self-attention.
            hidden_dim (int): Dimensionality of the hidden representation per token.
            dropout (float): Dropout rate applied after embeddings and within Transformer blocks.
            attention_dropout (float): Dropout rate applied to attention weights.
            ch_emb_dim (int): Dimensionality of channel (spatial) embeddings.
            seq_emb_dim (int): Dimensionality of temporal (sequence position) embeddings.
            num_transformer_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Learnable spatial embeddings for each channel (ch_num x ch_emb_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, ch_emb_dim))

        # Learnable temporal embeddings for each sequence position (seq_len x seq_emb_dim)
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, seq_emb_dim))

        # Learnable CLS token per sequence position, concatenated with embeddings
        # Shape: (seq_len, hidden_dim + ch_emb_dim)
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim + ch_emb_dim))

        # Dropout layer applied after embedding concatenation and within Transformer blocks
        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer blocks with input and output dimensions adjusted for concatenated embeddings
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim + ch_emb_dim + seq_emb_dim,  # input dimension per token
                hidden_dim + ch_emb_dim + seq_emb_dim,  # output dimension per token
                dropout,
                attention_dropout,
            ) for _ in range(num_transformer_blocks)
        ])

        # Layer normalization applied after the final Transformer block
        self.ln = nn.LayerNorm(hidden_dim + ch_emb_dim + seq_emb_dim)


    def forward(self, input, mask, ch_idx, seq_idx, ori_len):
        """
        Forward pass of the TransformerEncoder.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len * ch_num, hidden_dim),
                                  representing encoded features per channel per sequence step.
            mask (torch.Tensor): Boolean mask tensor of shape (batch_size, seq_len * ch_num),
                                 indicating valid (False) or padded (True) positions.
            ch_idx (torch.Tensor): Channel indices tensor of shape (batch_size, seq_len * ch_num),
                                   mapping each token to its channel index.
            seq_idx (torch.Tensor): Sequence position indices tensor of shape (batch_size, seq_len * ch_num),
                                    mapping each token to its sequence position.
            ori_len (torch.Tensor): Original lengths tensor of shape (batch_size,),
                                    representing the true sequence lengths before padding.

        Returns:
            torch.Tensor: Output features of shape (batch_size, seq_len, hidden_dim + ch_emb_dim + seq_emb_dim),
                          representing the encoded sequence with CLS token embeddings.
        """
        bz, seql_cn = input.shape[:2]  # batch size and total tokens per batch (seq_len * ch_num)

        # Flatten batch and token dimensions for embedding lookup and concatenation
        input_flat = input.view(bz * seql_cn, -1)  # (batch_size * seq_len * ch_num, hidden_dim)
        ch_idx_flat = ch_idx.view(bz * seql_cn,)   # (batch_size * seq_len * ch_num,)
        seq_idx_flat = seq_idx.view(bz * seql_cn,) # (batch_size * seq_len * ch_num,)

        # Retrieve spatial embeddings for each token based on channel index
        SE = self.spatial_embedding[ch_idx_flat]   # (batch_size * seq_len * ch_num, ch_emb_dim)

        # Retrieve temporal embeddings for each token based on sequence position index
        TE = self.temporal_embedding[seq_idx_flat] # (batch_size * seq_len * ch_num, seq_emb_dim)

        # Concatenate input features with spatial and temporal embeddings
        input_cat = torch.cat([input_flat, SE, TE], dim=1)   # (batch_size * seq_len * ch_num, hidden_dim + ch_emb_dim + seq_emb_dim)

        # Reshape back to (batch_size, seq_len * ch_num, combined_embedding_dim)
        input_cat = input_cat.view(bz, seql_cn, -1)

        # Prepare concatenated CLS tokens:
        # CLS token expanded across batch and concatenated with temporal embeddings
        # Shape after concatenation: (batch_size, seq_len, hidden_dim + ch_emb_dim + seq_emb_dim)
        cls_tokens = torch.cat([
            self.cls_token.unsqueeze(0).expand(bz, -1, -1),          # CLS token per sequence position
            self.temporal_embedding.unsqueeze(0).expand(bz, -1, -1)  # Temporal embeddings for CLS token
        ], dim=-1)

        # Concatenate CLS tokens with input tokens along sequence dimension
        # Resulting shape: (batch_size, seq_len + seq_len * ch_num, hidden_dim + ch_emb_dim + seq_emb_dim)
        x = torch.cat([cls_tokens, input_cat], dim=1)

        # Create padding mask for CLS tokens (all False, as CLS tokens are valid)
        padding_mask = torch.zeros(bz, self.seq_len, dtype=torch.bool, device=mask.device)

        # Concatenate CLS padding mask with input mask
        # Final mask shape: (batch_size, seq_len + seq_len * ch_num)
        mask = torch.cat([padding_mask, mask.bool()], dim=1)

        # Apply dropout to the concatenated embeddings before Transformer blocks
        x = self.dropout(x)

        # Pass through each Transformer block with the combined mask
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, attn_mask=None)

        # Apply layer normalization to the output of the Transformer blocks
        feat = self.ln(x)   # (batch_size, seq_len + seq_len * ch_num, hidden_dim + ch_emb_dim + seq_emb_dim)

        # Extract features corresponding to the CLS tokens (first seq_len tokens)
        feat = feat[:, :self.seq_len, :]   # (batch_size, seq_len, hidden_dim + ch_emb_dim + seq_emb_dim)

        return feat
