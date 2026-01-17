# -*- coding: utf-8 -*-
"""
finetune.py

Fine-tuning script for domain adaptation of the Large Polysomnography Model (LPSGM).
This script handles dataset preparation, configuration logging, model fine-tuning,
and evaluation on a target dataset from a specified center. It supports k-fold cross-validation,
configurable model and training parameters, and logs relevant information for reproducibility.

Key functionalities:
- Prepare train, evaluation, and test splits based on the target center and fold.
- Backup configuration and log dataset splits and training parameters.
- Initialize and run the fine-tuning process using the Finetuner class.
- Load the best fine-tuned model and perform testing using the Tester class.
- Command-line interface for flexible configuration of model architecture, training,
  dataset, and runtime options.
"""

import os
from datetime import datetime
import sys
import shutil
import argparse
import torch
import warnings

from utils import *
from finetune.finetuner import Finetuner
from dataset.dataset import *
from tester import Tester

# Suppress user warnings to reduce console clutter during execution
warnings.filterwarnings("ignore", category=UserWarning)

# Current timestamp for unique run identification
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
# Root directory of the script for relative path construction
root = os.path.split(sys.argv[0])[0]


def prepare_data(args):
    """
    Prepare train, evaluation, and test subject splits based on the specified target center
    and k-fold cross-validation parameters.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing dataset and fold info.

    Returns:
        None: Updates args with train_subjects, eval_subjects, and test_subjects attributes.
    """
    # Select stratified k-fold splitting method based on the target center
    if args.ft_center == "HANG7":
        train_subjects, eval_subjects, test_subjects = HANG7_StratifiedKFold(
            args.kfolds, args.n_fold, args.seed, args.eval_size)
    elif args.ft_center == "SYSU":
        train_subjects, eval_subjects, test_subjects = SYSU_StratifiedKFold(
            args.kfolds, args.n_fold, args.seed, args.eval_size)

    # Attach center information to each subject for consistent dataset handling
    args.train_subjects = [(sub, args.ft_center) for sub, _ in train_subjects]
    args.eval_subjects = [(sub, args.ft_center) for sub, _ in eval_subjects]
    args.test_subjects = [(sub, args.ft_center) for sub, _ in test_subjects]


def log_config(args):
    """
    Backup the current configuration script and log dataset splits and training parameters
    into separate files within the run directory for reproducibility and auditing.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with run and dataset info.

    Returns:
        None
    """
    # Create the run directory if it does not exist
    os.makedirs(args.run_name, exist_ok=True)
    # Backup this script file as a record of the configuration used
    shutil.copy2(
        os.path.realpath(__file__), 
        os.path.join(args.run_name, "config_backup")
        )

    # Redirect stdout to data.txt to log dataset splits
    sys.stdout = open(os.path.join(args.run_name, 'data.txt'), 'a')
    print(f"Train Subjects: {len(args.train_subjects)}, Eval Subjects: {len(args.eval_subjects)}, Test Subjects: {len(args.test_subjects)}")
    print("\n"*10, "Train Subjects:")
    for sub in args.train_subjects:
        print(sub)
    print("\n"*10, "Eval Subjects:")
    for sub in args.eval_subjects:
        print(sub)
    print("\n"*10, "Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    # Redirect stdout to train.txt to log training parameters
    sys.stdout = open(os.path.join(args.run_name, 'train.txt'), 'a')
    print(
        '--ft_center', args.ft_center, '\n',
        '--kfolds', args.kfolds, '\n',
        '--n_fold', args.n_fold, '\n',

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

        '--state_dict_file', args.state_dict_file, '\n',
        '--run_name', args.run_name, '\n',
        '--model_dir', args.model_dir, '\n',
        '--log_dir', args.log_dir, '\n',
        '--cache_root', args.cache_root, '\n',
        )


def run(args):
    """
    Execute the fine-tuning workflow:
    - Set random seed for reproducibility
    - Prepare dataset splits
    - Log configuration and dataset information
    - Fine-tune the model on the target dataset
    - Load the best fine-tuned model and perform testing

    Args:
        args (argparse.Namespace): Parsed command-line arguments with all configurations.

    Returns:
        None
    """
    # Initialize random seed for reproducibility
    set_seed(args.seed)
    # Prepare train, eval, and test splits
    prepare_data(args)
    # Log configuration and dataset splits
    log_config(args)

    # Start fine-tuning process
    print("\n"*5, "Here Start the Finetuning ...", "\n")
    model = Finetuner(args).finetune()
    
    # Load the best model checkpoint for evaluation
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "eval_best.pth"))['model_state_dict'])
    print("\n\nHere Start the Testing ...\n\n")
    # Redirect stdout to test.txt to log testing outputs
    sys.stdout = open(os.path.join(args.run_name, 'test.txt'), 'a')
    # Run testing with the fine-tuned model
    Tester(args, model)


def main():
    """
    Parse command-line arguments, set up directory paths, and initiate the fine-tuning run.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Finetuning LPSGM on Target Dataset')

    """ General parameters """
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--run_id', type=str, help='Unique identifier for the run')

    """ Model architecture parameters """
    parser.add_argument('--architecture', type=str, default='cat_cls', 
                        help="Model architecture type: 'cat_cls', 'add_cls', 'cat_avg', or 'none_cls'")
    parser.add_argument('--epoch_encoder_dropout', type=float, default=0, help='Dropout rate for epoch encoder')
    parser.add_argument('--transformer_num_heads', type=int, default=8, help='Number of attention heads in transformer')
    parser.add_argument('--transformer_dropout', type=float, default=0, help='Dropout rate in transformer blocks')
    parser.add_argument('--transformer_attn_dropout', type=float, default=0, help='Attention dropout rate in transformer')
    parser.add_argument('--ch_num', type=int, default=8, help='Number of input channels')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length for transformer input')
    parser.add_argument('--ch_emb_dim', type=int, default=32, help='Channel embedding dimension')
    parser.add_argument('--seq_emb_dim', type=int, default=64, help='Sequence embedding dimension')
    parser.add_argument('--num_transformer_blocks', type=int, default=6, help='Number of transformer blocks')

    """ Training parameters """
    parser.add_argument('--eval_size', type=float, default=0.1, help='Fraction of training data used for evaluation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=256, help='Number of data loader workers')
    parser.add_argument('--num_processes', type=int, default=200, help='Number of parallel processes')
    parser.add_argument('--epochs', type=int, default=50, help='Total number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='Number of warmup epochs for learning rate')
    parser.add_argument('--lr0', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--eta_min', type=float, default=1e-8, help='Minimum learning rate for scheduler')
    parser.add_argument('--clip_value', type=float, default=1, help='Gradient clipping value')
    parser.add_argument('--clamp_value', type=float, default=10, help='Value to clamp gradients')

    """ Dataset parameters """
    parser.add_argument('--ft_center', type=str, help='Target dataset center for fine-tuning')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--n_fold', type=int, default=0, help='Index of the current fold')
    parser.add_argument('--cache_root', type=str, help='Root directory for cached data')
    parser.add_argument('--random_shift_len', type=int, default=100, help='Random shift length for data augmentation')

    """ Testing parameters """
    parser.add_argument('--save_pred', action='store_true', help='Flag to save prediction outputs')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save predictions')

    args = parser.parse_args()

    # Construct run directory paths with timestamp for organization
    args.run_name = os.path.join(root, "run", args.run_id, current_time)
    args.model_dir = os.path.join(args.run_name, "model_dir")
    args.log_dir = os.path.join(args.run_name, "log_dir")
    args.cache_root = os.path.join(args.cache_root, current_time)

    # If saving predictions but no directory specified, create default save directory
    if args.save_pred and args.save_dir is None:
        args.save_dir = os.path.join(args.run_name, "predicts")

    # Execute the fine-tuning and testing workflow
    run(args)


if __name__ == "__main__":
    main()
