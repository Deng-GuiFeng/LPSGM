# -*- coding: utf-8 -*-
"""
model.py

This module defines the LPSGMNar class, a specialized model architecture for sleep disorder diagnosis within the LPSGM project.
LPSGMNar extends the base LPSGM model by adapting its classifier head to a 3-class classification task, suitable for diagnosing sleep disorders.
The model leverages the pretrained backbone of LPSGM for feature extraction from polysomnography data and modifies the classification layer accordingly.
"""

import torch
import torch.nn as nn
from model.model import LPSGM as BaseLPSGM
from model.classifier import Classifier as BaseClassifier


class LPSGMNar(nn.Module):
    def __init__(self, args, num_classes: int = 3):
        """
        Initialize the LPSGMNar model with a modified classifier for sleep disorder diagnosis.

        Args:
            args (Namespace): Configuration arguments containing model hyperparameters and architecture settings.
            num_classes (int): Number of output classes for classification. Defaults to 3 for sleep disorder categories.
        """
        super().__init__()
        # Instantiate the base LPSGM backbone model for feature extraction
        self.backbone = BaseLPSGM(args)
        # Determine the feature dimension based on the specified architecture in args
        feat_dim = self._infer_feat_dim(args)
        # Replace the original classifier with a new classifier head tailored for 3-class classification
        self.backbone.classifier = BaseClassifier(feat_dim=feat_dim, num_classes=num_classes)

    def _infer_feat_dim(self, args):
        """
        Infer the feature dimension of the backbone's output based on the architecture type.

        Args:
            args (Namespace): Configuration arguments containing the 'architecture' attribute and embedding dimensions.

        Returns:
            int: The inferred feature dimension size.
        
        Raises:
            ValueError: If the architecture type specified in args is unknown.
        """
        if args.architecture in ['add_cls', 'none_cls']:
            # Architectures that use a single 512-dimensional feature vector
            return 512
        elif args.architecture in ['cat_cls', 'cat_avg']:
            # Architectures that concatenate channel and sequence embeddings to the base 512-dimensional feature vector
            return 512 + args.ch_emb_dim + args.seq_emb_dim
        else:
            # Raise error for unsupported architecture types
            raise ValueError(f"Unknown architecture: {args.architecture}")

    def forward(self, x, mask, ch_idx, seq_idx, ori_len):
        """
        Forward pass through the LPSGMNar model.

        Args:
            x (Tensor): Input tensor representing PSG data.
            mask (Tensor): Mask tensor indicating valid data points.
            ch_idx (Tensor): Channel indices tensor for flexible channel configuration.
            seq_idx (Tensor): Sequence indices tensor for temporal encoding.
            ori_len (Tensor): Original lengths of sequences before padding.

        Returns:
            Tensor: Output logits from the classifier corresponding to sleep disorder classes.
        """
        # Delegate forward computation to the backbone model
        return self.backbone(x, mask, ch_idx, seq_idx, ori_len)
