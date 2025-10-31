# -*- coding: utf-8 -*-
"""
transformer_block.py

This module defines the TransformerBlock and its associated MLPBlock used within the LPSGM model.
The TransformerBlock captures temporal dependencies in sequential polysomnography (PSG) data
by employing multi-head self-attention and a feed-forward network. It serves as a core component
of the sequence encoder, enabling the model to learn contextual relationships across epochs.

Classes:
    MLPBlock: Implements the feed-forward network within the Transformer block.
    TransformerBlock: Implements a single Transformer encoder block with layer normalization,
                      multi-head self-attention, dropout, and residual connections.
"""

import torch
import torch.nn as nn


class MLPBlock(nn.Sequential):
    """Feed-forward MLP block used inside the Transformer encoder block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        """
        Initialize the MLP block with two linear layers, GELU activation, and dropout.

        Args:
            in_dim (int): Input and output feature dimension.
            mlp_dim (int): Hidden layer dimension within the MLP.
            dropout (float): Dropout rate applied after each linear layer.
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)  # First linear projection
        self.act = nn.GELU()                        # GELU activation function
        self.dropout_1 = nn.Dropout(dropout)       # Dropout after first linear layer
        self.linear_2 = nn.Linear(mlp_dim, in_dim) # Second linear projection back to input dimension
        self.dropout_2 = nn.Dropout(dropout)       # Dropout after second linear layer

        # Initialize weights with Xavier uniform distribution for stable training
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        # Initialize biases with small normal noise to avoid zero initialization
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block for modeling temporal dependencies."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        ):
        """
        Initialize the Transformer encoder block.

        Args:
            num_heads (int): Number of attention heads in multi-head self-attention.
            hidden_dim (int): Dimensionality of input and output features.
            mlp_dim (int): Hidden dimension size of the feed-forward MLP.
            dropout (float): Dropout rate applied after attention and MLP layers.
            attention_dropout (float): Dropout rate applied within the attention mechanism.
        """
        super().__init__()
        self.num_heads = num_heads

        # Layer normalization before self-attention
        self.ln_1 = nn.LayerNorm(hidden_dim)
        # Multi-head self-attention module with dropout
        self.self_attention = nn.MultiheadAttention(hidden_dim, 
                                                    num_heads, 
                                                    dropout=attention_dropout, 
                                                    batch_first=True)
        # Dropout applied after self-attention output
        self.dropout = nn.Dropout(dropout)

        # Layer normalization before MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        # Feed-forward MLP block
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, key_padding_mask, attn_mask):
        """
        Forward pass of the Transformer encoder block.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
            key_padding_mask (torch.Tensor or None): Boolean mask indicating padded positions
                                                     in the input sequence with shape (batch_size, seq_length).
                                                     True values are positions that should be masked.
            attn_mask (torch.Tensor or None): Attention mask to prevent attention to certain positions,
                                              typically used for causal or local attention.

        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, seq_length, hidden_dim).
        """
        # Assert input tensor has 3 dimensions: batch size, sequence length, hidden dimension
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        # Convert key_padding_mask to boolean type if not already, as required by MultiheadAttention
        if key_padding_mask is not None and key_padding_mask.dtype is not torch.bool:
            key_padding_mask = key_padding_mask.bool()
        
        # Apply layer normalization before self-attention
        x = self.ln_1(input)
        # Compute multi-head self-attention; discard attention weights
        x, _ = self.self_attention(query=x, key=x, value=x, 
                                  key_padding_mask=key_padding_mask, 
                                  need_weights=False, 
                                  attn_mask=attn_mask)
        # Apply dropout after attention
        x = self.dropout(x)
        # Residual connection adding input to attention output
        x = x + input

        # Apply layer normalization before feed-forward MLP
        y = self.ln_2(x)
        # Pass through MLP block
        y = self.mlp(y)
        # Final residual connection adding MLP output to previous sum
        return x + y
