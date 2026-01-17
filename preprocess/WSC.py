# -*- coding: utf-8 -*-
"""
WSC.py

This module handles the preprocessing of the WSC polysomnography (PSG) dataset for the LPSGM project. 
It loads raw PSG signals and their corresponding sleep stage annotations, performs necessary preprocessing 
such as resampling and label adjustment, removes unknown labels, and saves the processed data in a standardized format. 

Key functionalities include:
- Parsing XML annotation files to extract sleep stages and remapping specific stage labels.
- Loading and preprocessing multi-channel PSG signals with configurable channel mappings.
- Synchronizing signal epochs and annotation lengths, handling discrepancies by truncation.
- Parallel processing of multiple recordings to accelerate dataset preparation.
- Integration with utility functions for signal loading, preprocessing, unknown label removal, and data saving.

This preprocessing step is critical for preparing the WSC dataset for downstream sleep staging and mental disorder diagnosis tasks.
"""

import os
import numpy as np
import warnings
import shutil
from multiprocessing import Pool

from utils import *

# warnings.filterwarnings("ignore", category=UserWarning)

.
def load_ano(ano_path):
    """
    Load and convert sleep stage annotations from a .stg.txt file into a numpy array of integer labels.

    Args:
        ano_path (str): Path to the annotation file containing sleep stages.

    Returns:
        np.ndarray: Array of sleep stage labels with shape (number_of_epochs,), where each label is an integer.
                    Sleep stages are mapped as: W=0, N1=1, N2=2, N3=3, REM=4, Unknown=-1.
    """
    # WSC dataset stage mapping: 0=W, 1=N1, 2=N2, 3=N3, 5=REM
    # All other values are treated as Unknown (-1)
    stage_dict = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

    sleep_stages = []
    with open(ano_path, 'r') as f:
        lines = f.readlines()

    # Skip header line and parse each epoch
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            # Use User-Defined Stage column (index 1)
            stage = int(parts[1])
            sleep_stages.append(stage_dict.get(stage, -1))

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id):
    """
    Process a single PSG recording: load signals and annotations, preprocess signals,
    synchronize epochs, remove unknown labels, and save the processed data.

    Args:
        sub_id (str): Subject identifier.

    Returns:
        None
    """
    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}.stg.txt")

    # Load raw signals and their start time using predefined channel configuration
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dictionary of channel signals
    # Load and preprocess sleep stage annotations
    ano = load_ano(ano_path)    

    # Resample and preprocess signals to the target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signal dictionary

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ, truncate to minimal length
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        # Truncate all channel signals to the minimal epoch count
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        # Truncate annotations accordingly
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    channel_names = list(channel_id.keys())
    # Save the cleaned and preprocessed data to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, sig_path, ano_path).

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing without stopping the entire pipeline
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run the preprocessing pipeline on all subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # Get unique subject IDs by listing files and removing extensions, excluding subjects to remove
    subjects = set([f.split(".")[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)
    # Filter subjects to only include those with both signal and annotation files
    subjects = [sub_id for sub_id in subjects 
                if os.path.exists(os.path.join(src_root, f"{sub_id}.edf")) 
                and os.path.exists(os.path.join(src_root, f"{sub_id}.stg.txt"))]

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing WSC Dataset"):
            pass


def test():
    """
    Placeholder for potential testing functions.

    Returns:
        None
    """
    subjects = set([f.split(".")[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)
    # Filter subjects to only include those with both signal and annotation files
    subjects = [sub_id for sub_id in subjects 
                if os.path.exists(os.path.join(src_root, f"{sub_id}.edf")) 
                and os.path.exists(os.path.join(src_root, f"{sub_id}.stg.txt"))]

    for sub_id in subjects:
        print(f"Processing {sub_id} for test...")
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING WSC DATASET', '='*30)

    # Define source and destination directories for raw and processed data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/wsc/files/polysomnography/"
    dst_root = r"./data/WSC/"
    # Remove existing processed data directory to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel mappings: keys are channel names used in processing,
    # values are tuples of tuples specifying raw channel pairs for referencing
    channel_id = {
        'F3': ('F3_M2', 'F3_M1', 'F3_AVG'), 
        'F4': ('F4_M1', 'F4_M2', 'F4_AVG'), 
        'C3': ('C3_M2', 'C3_M1', 'C3_AVG'), 
        'C4': ('C4_M1', 'C4_M2', 'C4_AVG'), 
        'O1': ('O1_M2', 'O1_M1', 'O1_AVG'), 
        'O2': ('O2_M1', 'O2_M2'), 
        'E1': ('E1',), 
        'E2': ('E2',), 
        
        'Chin': ('chin', 'cchin_l', 'cchin_r', 'rchin_l'), 
    }

    # Execute preprocessing with 100 parallel worker processes
    run(100)
    # test()

    # Perform formatting checks on the processed dataset to ensure consistency
    formatting_check(dst_root)

    # Placeholder for testing function call
    # test()
