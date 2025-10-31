# -*- coding: utf-8 -*-
"""
classifier.py

This module defines the classification head used in the LPSGM project for sleep staging and mental disorder diagnosis.
It implements a simple linear layer that maps extracted features to class logits, supporting multi-class classification
tasks such as 5-class sleep stage classification. This component receives feature representations from upstream encoders
and outputs predictions for downstream diagnostic tasks.
"""

import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes=5):
        """
        Initializes the Classifier module with a linear classification layer.

        Args:
            feat_dim (int): Dimension of the input feature vector.
            num_classes (int, optional): Number of output classes. Defaults to 5.
        """
        super(Classifier, self).__init__()
        # Linear layer projecting input features to class logits
        self.ln = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, feat_dim).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
        """
        return self.ln(x)


if __name__ == "__main__":
    # Instantiate the classifier with default parameters for testing
    model = Classifier(feat_dim=128)  # Example feature dimension; adjust as needed
    # Print total number of parameters in the model
    print(sum(p.numel() for p in model.parameters()))
