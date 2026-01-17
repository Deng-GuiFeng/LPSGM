# -*- coding: utf-8 -*-
"""
train.py

Main training script for the Large Polysomnography Model (LPSGM) project.

This script orchestrates the entire training pipeline for LPSGM, a large-scale model designed for sleep staging and mental disorder diagnosis using polysomnography data. It handles dataset preparation, configuration logging, model training, and evaluation/testing. The script supports flexible channel configurations, domain-adaptive pre-training, and is designed to work with multiple public PSG datasets.

Key functionalities:
- Parsing and managing command-line arguments for model configuration and training parameters.
- Preparing training, evaluation, and testing datasets based on specified source and target domains.
- Logging configuration details and dataset splits for reproducibility.
- Training the model via the Trainer class and evaluating it using the Tester class.
- Managing output directories and saving model checkpoints and predictions.

This file is central to running experiments and benchmarking the LPSGM model.
"""

import os
from datetime import datetime
import sys
import shutil
import argparse
import warnings

from utils import *
from train.trainer import Trainer
from dataset.dataset import *
from tester import Tester

# Suppress user warnings to reduce console clutter during training/testing
warnings.filterwarnings("ignore", category=UserWarning)

# Timestamp string for naming output directories and logs
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
# Root directory of the script execution path
root = os.path.split(sys.argv[0])[0]


def prepare_data(args):
    """
    Prepare training, evaluation, and testing subject lists based on source and target domains.

    Args:
        args (argparse.Namespace): Command-line arguments containing domain and dataset info.

    Returns:
        None: Modifies args in-place to add train_subjects, eval_subjects, and test_subjects.
    """
    # Retrieve subjects and their corresponding domains from source domains
    source_subjects, source_domains = get_datasets_subjects(args.source_domains)
    # Pair each subject with its domain for consistent splitting
    source_subjects = [(sub, dom) for sub, dom in zip(source_subjects, source_domains)]
    # Split source subjects into training and evaluation sets with stratification by domain
    args.train_subjects, args.eval_subjects = train_test_split(
        source_subjects, 
        test_size=args.eval_size, 
        random_state=args.seed,
        shuffle=True,
        stratify=source_domains,
    )
    # Retrieve subjects and domains from the target domain for testing
    test_subjects, test_domains = get_datasets_subjects(args.target_domain)
    # Pair test subjects with their domains
    args.test_subjects = [(sub, dom) for sub, dom in zip(test_subjects, test_domains)]


def log_config(args):
    """
    Backup the current configuration and log dataset splits and training parameters.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and dataset info.

    Returns:
        None
    """
    # Create directory for current run if it doesn't exist
    os.makedirs(args.run_name, exist_ok=True)
    # Backup this script file for reproducibility
    shutil.copy2(
        os.path.realpath(__file__), 
        os.path.join(args.run_name, "config_backup")
    )

    # Redirect stdout to data.txt to log dataset splits
    sys.stdout = open(os.path.join(args.run_name, 'data.txt'), 'a')
    print(f"Train Subjects: {len(args.train_subjects)}, Eval Subjects: {len(args.eval_subjects)}, Test Subjects: {len(args.test_subjects)}")
    
    # Log training subjects
    print("\n"*10, "Train Subjects:")
    for sub in args.train_subjects:
        print(sub)
    # Log evaluation subjects
    print("\n"*10, "Eval Subjects:")
    for sub in args.eval_subjects:
        print(sub)
    # Log testing subjects
    print("\n"*10, "Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    # Redirect stdout to train.txt to log training parameters
    sys.stdout = open(os.path.join(args.run_name, 'train.txt'), 'a')
    print(
        '--architecture', args.architecture, '\n',
        '--epoch_encoder_dropout', args.epoch_encoder_dropout, '\n',
        '--transformer_num_heads', args.transformer_num_heads, '\n',
        '--transformer_dropout', args.transformer_dropout, '\n',
        '--transformer_attn_dropout', args.transformer_attn_dropout, '\n',
        '--ch_num', args.ch_num, '\n',
        '--seq_len', args.seq_len, '\n',
        '--ch_emb_dim', args.ch_emb_dim, '\n',
        '--seq_emb_dim', args.seq_emb_dim, '\n',
        '--num_transformer_blocks', args.num_transformer_blocks, '\n',

        '--seed', args.seed, '\n',
        '--batch_size', args.batch_size, '\n',
        '--num_workers', args.num_workers, '\n',
        '--num_processes', args.num_processes, '\n',
        '--epochs', args.epochs, '\n',
        '--warmup_epochs', args.warmup_epochs, '\n',
        '--lr0', args.lr0, '\n',
        '--weight_decay', args.weight_decay, '\n',
        '--eta_min', args.eta_min, '\n',
        '--clip_value', args.clip_value, '\n',
        '--save_epoch', args.save_epoch, '\n',

        '--source_domains', args.source_domains, '\n',
        '--target_domain', args.target_domain, '\n',
        '--state_dict_file', args.state_dict_file, '\n',
        '--run_name', args.run_name, '\n',
        '--model_dir', args.model_dir, '\n',
        '--log_dir', args.log_dir, '\n',
        '--cache_root', args.cache_root, '\n',
        '--random_shift_len', args.random_shift_len, '\n',
    )


def run(args):
    """
    Execute the full training and testing pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments with all configurations.

    Returns:
        None
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    # Prepare dataset splits for training, evaluation, and testing
    prepare_data(args)
    # Log configuration and dataset information
    log_config(args)

    # Start training process
    print("\n\nHere Start the Training ...\n\n")
    model = Trainer(args).train()

    # Load the best model checkpoint for evaluation
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "eval_best.pth"))['model_state_dict'])
    print("\n\nHere Start the Testing ...\n\n")
    # Redirect stdout to test.txt to log testing outputs
    sys.stdout = open(os.path.join(args.run_name, 'test.txt'), 'a')
    # Run the testing process with the trained model
    Tester(args, model)


def main():
    """
    Parse command-line arguments, set up directories, and start the training/testing process.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Large PSG Model')

    # General parameters
    parser.add_argument('--seed', type=int, default=0)

    # Model architecture parameters
    parser.add_argument('--architecture', type=str, default='cat_cls')  # Options: 'cat_cls', 'add_cls', 'cat_avg', 'none_cls'
    parser.add_argument('--epoch_encoder_dropout', type=float, default=0)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--transformer_dropout', type=float, default=0)
    parser.add_argument('--transformer_attn_dropout', type=float, default=0)
    parser.add_argument('--ch_num', type=int, default=8)  # Number of input channels
    parser.add_argument('--seq_len', type=int, default=20)  # Sequence length for transformer input
    parser.add_argument('--ch_emb_dim', type=int, default=32)  # Channel embedding dimension
    parser.add_argument('--seq_emb_dim', type=int, default=64)  # Sequence embedding dimension
    parser.add_argument('--num_transformer_blocks', type=int, default=6)  # Number of transformer blocks

    # Training parameters
    parser.add_argument('--eval_size', type=float, default=0.1)  # Fraction of source data for evaluation
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=256)  # DataLoader workers
    parser.add_argument('--num_processes', type=int, default=200)  # Parallel processes for data loading/preprocessing
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=15)  # Number of warmup epochs for learning rate scheduling
    parser.add_argument('--lr0', type=float, default=1e-4)  # Initial learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=1e-8)  # Minimum learning rate for scheduler
    parser.add_argument('--clip_value', type=float, default=1)  # Gradient clipping value
    parser.add_argument('--clamp_value', type=float, default=10)  # Additional clamp value (usage context not specified)
    parser.add_argument('--state_dict_file', type=str, default=None)  # Path to pretrained model weights
    parser.add_argument('--save_epoch', action='store_true')  # Flag to save model at each epoch

    # Data parameters
    parser.add_argument(
        '--source_domains', type=str_to_set, 
        default={
            'APPLES', 'DCSM', 'DOD-H', 'DOD-O', 'HMC', 'ISRUC', 'P2018', 'SHHS-1', 'SHHS-2', 
            'STAGES-BOGN', 'STAGES-STNF', 'STAGES-MSTR', 'STAGES-GSDV', 'STAGES-GSBB', 'STAGES-GSLH', 
            'STAGES-GSSA', 'STAGES-GSSW', 'STAGES-MSMI', 'STAGES-MSNF', 'STAGES-MSQW', 'STAGES-MSTH', 'STAGES-STLK', 
            'ABC', 'NCHSDB', 'HOMEPAP', 'SVUH', 'CHAT', 'CCSHS', 'CFS', 'MROS'
        }
    )
    parser.add_argument('--target_domain', type=str_to_set, default={'HANG7', 'SYSU'})
    parser.add_argument('--cache_root', type=str, default=None)  # Root directory for caching processed data
    parser.add_argument('--random_shift_len', type=int, default=100)  # Max random shift length for data augmentation

    # Testing parameters
    parser.add_argument('--save_pred', action='store_true')  # Flag to save predictions during testing
    parser.add_argument('--save_dir', type=str, default=None)  # Directory to save predictions

    # Parse arguments from command line
    args = parser.parse_args()

    # Define run-specific directories for outputs and logs
    args.run_name = os.path.join(root, "run", current_time)
    args.model_dir = os.path.join(args.run_name, "model_dir")
    args.log_dir = os.path.join(args.run_name, "log_dir")
    args.cache_root = os.path.join(args.cache_root, current_time) if args.cache_root else None

    # Set prediction save directory if saving predictions and no directory specified
    if args.save_pred and args.save_dir is None:
        args.save_dir = os.path.join(args.run_name, "predicts")

    # Start the training and testing process
    run(args)


if __name__ == "__main__":
    main()
