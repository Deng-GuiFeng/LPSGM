# -*- coding: utf-8 -*-
"""
epoch_encoder.py

This module defines the EpochEncoder class, a dual-branch convolutional neural network (CNN) designed 
to extract robust feature representations from 30-second polysomnography (PSG) epochs. The encoder 
processes raw PSG signals through two parallel convolutional branches with different receptive fields 
to capture multi-scale temporal features. The extracted features serve as input embeddings for subsequent 
sequence modeling in the LPSGM framework, facilitating sleep staging and mental disorder diagnosis tasks.

The design is adapted and modified from the DeepSleepNet architecture, optimized for large-scale PSG data.

Author: LPSGM Project
"""

import torch
import torch.nn as nn


class EpochEncoder(nn.Module):
    def __init__(self, dropout):
        """
        Initializes the dual-branch CNN epoch encoder.

        Args:
            dropout (float): Dropout rate applied after initial convolutional layers to prevent overfitting.
        """
        super(EpochEncoder, self).__init__()
        
        # First convolutional branch with smaller kernel sizes and strides to capture fine-grained temporal features
        self.encoder_branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, padding=24),  # Input length 3000 -> 500 after conv
            nn.BatchNorm1d(64),  # Batch normalization for stable training
            nn.ReLU(),  # Non-linear activation
            
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),  # Downsample to length 63
            
            nn.Dropout(dropout),  # Regularization
            
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same'),  # Maintain length 63
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'),  # Maintain length 63
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding='same'),  # Maintain length 63
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),  # Downsample to length 16
            nn.AdaptiveAvgPool1d(1),  # Global average pooling to length 1
        )

        # Second convolutional branch with larger kernel sizes and strides to capture long-range temporal dependencies
        self.encoder_branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50),  # Input length 3000 -> 53 after conv
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.MaxPool1d(kernel_size=4, stride=4),  # Downsample to length 13
            
            nn.Dropout(dropout),  # Regularization
            
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding='same'),  # Maintain length 13
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding='same'),  # Maintain length 13
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding='same'),  # Maintain length 13
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample to length 6
            nn.AdaptiveAvgPool1d(1),  # Global average pooling to length 1
        )

    def forward(self, x):
        """
        Forward pass through the dual-branch CNN encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * sequence_length * channels, 1, 3000),
                              representing raw PSG epoch signals.

        Returns:
            torch.Tensor: Concatenated feature tensor of shape (batch_size * sequence_length * channels, 512),
                          combining embeddings from both branches.
        """
        # Process input through first branch and remove singleton dimension
        x1 = self.encoder_branch1(x).squeeze()
        # Process input through second branch and remove singleton dimension
        x2 = self.encoder_branch2(x).squeeze()
        # Concatenate features from both branches along the last dimension
        x = torch.concat([x1, x2], dim=-1)
        return x


if __name__ == "__main__":
    # Example usage and sanity check for the EpochEncoder
    
    bz, seql, cn = 64, 20, 8  # Batch size, sequence length, number of channels
    x = torch.randn((bz, seql, cn, 3000))  # Simulated input tensor with raw PSG data
    
    # Reshape input to merge batch, sequence, and channel dimensions for CNN processing
    x = x.view((bz * seql * cn, 1, -1))
    print(x.size())  # Expected shape: (bz*seql*cn, 1, 3000)
    
    model = EpochEncoder(0.5)  # Initialize encoder with 50% dropout
    print(sum(p.numel() for p in model.parameters()))  # Print total number of parameters
    
    y = model(x)  # Forward pass
    print(y.size())  # Expected output shape: (bz*seql*cn, 512)
