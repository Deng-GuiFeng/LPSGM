# -*- coding: utf-8 -*-
"""
Configuration settings for narcolepsy classification tasks within the LPSGM project.

This file defines dataset paths, training hyperparameters, model architecture parameters,
and diagnosis label mappings specific to narcolepsy classification. It supports configuration
of data directories, training routines, model backbone settings, and label encoding.

The configurations here are used to control the training and evaluation of the Large Polysomnography
Model (LPSGM) tailored for narcolepsy diagnosis, facilitating reproducible experiments and
consistent model behavior.
"""

import os
from dataset.dataset import ALL_CHANNELS, CHANNEL_TO_INDEX

# Directories containing MNC (Multi-Night Cohort) datasets used for narcolepsy classification
MNC_DIRS = [
    r"./data/MNC-CNC/",
    r"./data/MNC-DHC/",
    r"./data/MNC-FHC/",
    r"./data/MNC-IHC/",
    r"./data/MNC-KHC/",
    r"./data/MNC-SSC/",
]

# Root directory for saving output files such as model checkpoints and logs
OUTPUT_ROOT = os.path.join(os.getcwd(), 'run_nar')

# Training configuration parameters
EPOCHS = 30                  # Number of training epochs
BATCH_SIZE = 16              # Number of samples per training batch
LR = 3e-4                    # Initial learning rate for optimizer
WEIGHT_DECAY = 1e-4          # Weight decay (L2 regularization) coefficient
SEQ_LEN = 20                 # Length of input sequence (number of epochs per sequence)
SHIFT_LEN = 5                # Step size for sliding window over sequences
NUM_WORKERS = 8              # Number of subprocesses for data loading
NUM_PROCESSES = 8            # Number of parallel processes for training or evaluation
SEED = 2025                  # Random seed for reproducibility
KFOLDS = 5                   # Number of folds for cross-validation
FREEZE_BACKBONE = False      # Whether to freeze backbone weights during training
BACKBONE_CKPT = ''           # File path to pre-trained backbone checkpoint (optional)

# Backbone model hyperparameters matching expectations in model/model.py
ARCHITECTURE = 'add_cls'     # Method to combine channel embeddings: options include
                             # 'cat_cls' (concatenate with class token),
                             # 'add_cls' (add with class token),
                             # 'cat_avg' (concatenate with average pooling),
                             # 'none_cls' (no class token)
CH_EMB_DIM = 32              # Dimension of channel embedding vectors
SEQ_EMB_DIM = 32             # Dimension of sequence embedding vectors
TRANSFORMER_NUM_HEADS = 8    # Number of attention heads in Transformer blocks
TRANSFORMER_DROPOUT = 0.1    # Dropout rate applied after Transformer layers
TRANSFORMER_ATTN_DROPOUT = 0.1  # Dropout rate applied to attention weights
NUM_TRANSFORMER_BLOCKS = 4   # Number of Transformer blocks in sequence encoder
CLAMP_VALUE = 20.0           # Clamping value to limit activation magnitudes

# Mapping of diagnosis labels to integer class indices for classification
DIAGNOSIS_MAPPING = {
    'NON-NARCOLEPSY CONTROL': 0,
    'T1 NARCOLEPSY': 1,
    'OTHER HYPERSOMNIA': 2,
}
