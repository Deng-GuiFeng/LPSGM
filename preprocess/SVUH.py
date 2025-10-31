# -*- coding: utf-8 -*-
"""
SVUH Dataset Preprocessing Module for LPSGM Project

This script handles the preprocessing of the SVUH polysomnography dataset,
which is part of the Large Polysomnography Model (LPSGM) project. It performs
the following key functions:

- Loading and converting sleep stage annotations to the standardized format used
  in LPSGM (mapping stages 4 and 5 to 3 and 4 respectively).
- Extracting and preprocessing PSG signal data from recordings.
- Synchronizing signal epochs with annotation lengths.
- Removing epochs with unknown or invalid labels.
- Saving the processed data in a structured format for downstream model training.
- Supporting parallel processing to efficiently handle large datasets.

The module relies on utility functions from the 'utils' package for signal loading,
preprocessing, and saving. It is designed to be run as a standalone script or integrated
into larger preprocessing pipelines.

Author: LPSGM Team
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and convert sleep stage annotations from a text file.

    Args:
        ano_path (str): Path to the annotation text file.

    Returns:
        np.ndarray: Array of sleep stage labels with standardized mapping.
                    Original stage 4 is mapped to 3, and stage 5 is mapped to 4.
    """
    with open(ano_path, "r") as f:
        lines = f.readlines()
    sleep_stages = [int(l.strip()) for l in lines]

    # Map original stage 4 to 3 and stage 5 to 4 to align with LPSGM labeling scheme
    for i, ss in enumerate(sleep_stages):
        if ss == 4:
            sleep_stages[i] = 3
        if ss == 5:
            sleep_stages[i] = 4

    ano = np.array(sleep_stages, dtype=np.int32)
    return ano


def process_recording(sub_id):
    """
    Process a single subject's recording: load signals, preprocess, synchronize
    with annotations, remove unknown labels, and save processed data.

    Args:
        sub_id (str): Subject identifier corresponding to the recording file.
    """
    # Construct paths for .rec and corresponding .edf files
    rec_path = os.path.join(src_root, sub_id + ".rec")
    edf_path = rec_path.replace('.rec', '.edf')

    # Copy .rec file to .edf format for compatibility with signal loading utilities
    shutil.copyfile(rec_path, edf_path)

    # Load signal data and start time from the EDF file for specified channels
    start_time, sig_dict = load_sig(edf_path, channel_id)

    # Remove temporary EDF file after loading
    os.remove(edf_path)

    # Load sleep stage annotations from corresponding text file
    txt_path = os.path.join(src_root, sub_id + "_stage.txt")
    ano = load_ano(txt_path)

    # Preprocess signals: filtering, resampling, normalization as defined in utils
    sig_dict = pre_process(sig_dict, resample_rate)

    # Ensure all channels have the same number of epochs and match annotation length
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts:
        # Use the minimum epoch count to synchronize signals and annotations
        epochN = min(epoch_counts[0], len(ano))
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid sleep stage labels
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())

    # Save processed signals and annotations to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording, typically (sub_id,).
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all subject recordings using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes.
    """
    # Identify all subject IDs by listing .rec files and excluding removed subjects
    subjects = set([f.split(".")[0] for f in os.listdir(src_root) if f.endswith('.rec')]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing SVUH Dataset"):
            pass


def test():
    """
    Sequentially process all subject recordings for testing or debugging purposes.
    """
    subjects = set([f.split(".")[0] for f in os.listdir(src_root) if f.endswith('.rec')]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING SVUH DATASET', '=' * 30)

    # Define source directory containing raw SVUH dataset files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/SVUH/"

    # Define destination directory for saving preprocessed data
    dst_root = r"./data/SVUH/"

    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Set signal resampling rate in Hz
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty by default)
    SUB_REMOVE = []

    # Define channel mapping: keys are standardized channel names used in LPSGM,
    # values are tuples of original channel names in the dataset
    channel_id = {
        'C3': ('C3A2',),
        'C4': ('C4A1',),
        'E1': ('Lefteye',),
        'E2': ('RightEye',),
        'Chin': ('EMG',),
    }

    # Execute preprocessing with 100 parallel processes
    run(100)

    # Perform formatting checks on the saved preprocessed data
    formatting_check(dst_root)

    # Uncomment to run sequential test processing
    # test()
