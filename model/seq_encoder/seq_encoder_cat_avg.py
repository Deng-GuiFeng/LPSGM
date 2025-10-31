# -*- coding: utf-8 -*-
"""
Sequence Encoder with Averaged Feature Concatenation for LPSGM.

This module defines a Transformer-based sequence encoder that processes 
multi-channel polysomnography (PSG) epoch features. It incorporates learnable 
spatial (channel) and temporal embeddings, concatenates them with input features, 
and applies multiple Transformer blocks to capture temporal dependencies. 

Key functionality includes averaging features across channels for each time step, 
enabling flexible channel configurations and robust sequence representation. 

This encoder is a critical component of the LPSGM model, facilitating effective 
sleep staging and mental disorder diagnosis by modeling temporal dynamics in PSG data.
"""

import torch
import torch.nn as nn
from model.seq_encoder.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for sequence feature extraction with channel and temporal embeddings."""

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
        Initialize the TransformerEncoder module.

        Args:
            ch_num (int): Number of input channels.
            seq_len (int): Length of the input sequence (number of epochs).
            num_heads (int): Number of attention heads in Transformer blocks.
            hidden_dim (int): Dimension of the hidden features.
            dropout (float): Dropout rate applied after embeddings.
            attention_dropout (float): Dropout rate within attention layers.
            ch_emb_dim (int): Dimension of the learnable channel (spatial) embeddings.
            seq_emb_dim (int): Dimension of the learnable temporal embeddings.
            num_transformer_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Learnable spatial embeddings for each channel (inspired by Vision Transformer)
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, ch_emb_dim))

        # Learnable temporal embeddings for each position in the sequence
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, seq_emb_dim))

        # Learnable classification token (not used explicitly in forward, but initialized)
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim + ch_emb_dim))

        # Dropout layer applied after embedding concatenation
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks with specified dimensions and dropout rates
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim + ch_emb_dim + seq_emb_dim,
                hidden_dim + ch_emb_dim + seq_emb_dim,
                dropout,
                attention_dropout,
            ) for _ in range(num_transformer_blocks)
        ])
        
        # Layer normalization applied after Transformer blocks
        self.ln = nn.LayerNorm(hidden_dim + ch_emb_dim + seq_emb_dim)


    def forward(self, input, mask, ch_idx, seq_idx, ori_len):
        """
        Forward pass of the TransformerEncoder.

        Args:
            input (torch.Tensor): Input features of shape (batch_size, seq_len * ch_num, hidden_dim).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len * ch_num).
            ch_idx (torch.Tensor): Channel indices for each input feature, shape (batch_size, seq_len * ch_num).
            seq_idx (torch.Tensor): Sequence position indices for each input feature, shape (batch_size, seq_len * ch_num).
            ori_len (torch.Tensor): Original valid lengths per batch element, shape (batch_size,).

        Returns:
            torch.Tensor: Encoded features of shape (batch_size, seq_len, hidden_dim + ch_emb_dim + seq_emb_dim),
                          where features are averaged across channels for each time step.
        """
        bz, seql_cn = input.shape[:2]

        # Flatten batch and sequence-channel dimensions for embedding lookup
        input = input.view(bz * seql_cn, -1)  # (batch_size * seq_len * ch_num, hidden_dim)
        ch_idx = ch_idx.view(bz * seql_cn)    # (batch_size * seq_len * ch_num,)
        seq_idx = seq_idx.view(bz * seql_cn)  # (batch_size * seq_len * ch_num,)

        # Retrieve spatial embeddings for each channel index
        SE = self.spatial_embedding[ch_idx]   # (batch_size * seq_len * ch_num, ch_emb_dim)

        # Retrieve temporal embeddings for each sequence position index
        TE = self.temporal_embedding[seq_idx] # (batch_size * seq_len * ch_num, seq_emb_dim)

        # Concatenate input features with spatial and temporal embeddings
        input = torch.cat([input, SE, TE], dim=1)  # (batch_size * seq_len * ch_num, hidden_dim + ch_emb_dim + seq_emb_dim)

        # Reshape back to (batch_size, seq_len * ch_num, combined_feature_dim)
        input = input.view(bz, seql_cn, -1)

        # Apply dropout after embedding concatenation
        input = self.dropout(input)

        # Pass through stacked Transformer blocks with attention mask
        for transformer_block in self.transformer_blocks:
            input = transformer_block(input, mask, attn_mask=None)

        # Apply layer normalization to the Transformer output
        feat = self.ln(input)  # (batch_size, seq_len * ch_num, hidden_dim + ch_emb_dim + seq_emb_dim)

        feat_list = []
        # Process each batch element individually to handle variable original lengths
        for i, ft in enumerate(feat):
            # Select only valid features up to original length for this batch element
            ft = ft[:ori_len[i]]  # (valid_seq_len * ch_num, feature_dim)

            seql_cn = ft.shape[0]
            # Calculate number of channels by dividing total length by sequence length
            cn = seql_cn // self.seq_len

            # Reshape to separate sequence length and channels
            ft = ft.view(self.seq_len, cn, -1)  # (seq_len, ch_num, feature_dim)

            # Average features across channels for each time step
            ft = ft.mean(1)  # (seq_len, feature_dim)

            # Append batch element with added batch dimension
            feat_list.append(ft.unsqueeze(0))  # (1, seq_len, feature_dim)

        # Concatenate all batch elements back into a single tensor
        feat = torch.concat(feat_list, dim=0)  # (batch_size, seq_len, feature_dim)

        return feat
