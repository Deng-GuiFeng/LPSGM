# -*- coding: utf-8 -*-
"""
ISRUC Dataset Preprocessing Module for LPSGM Project

This script preprocesses the ISRUC polysomnography dataset to prepare it for training
and evaluation within the Large Polysomnography Model (LPSGM) framework. It handles:
- Loading and converting raw signal files (.rec) to EDF format
- Reading and processing sleep stage annotations
- Resampling and preprocessing EEG/EOG/EMG signals
- Synchronizing signal epochs with annotation labels
- Removing unknown or invalid sleep stage labels
- Saving the cleaned and formatted data for downstream model training

The ISRUC dataset contains multiple subgroups and sessions, each processed in parallel
to accelerate preprocessing. The script supports flexible channel configurations
and ensures consistency between signals and annotations.

This module is a critical component in the data pipeline enabling robust sleep staging
and mental disorder diagnosis using LPSGM.
"""

import numpy as np
import os
import shutil
from mne.io import read_raw_edf
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and process sleep stage annotations from a text file.

    Args:
        ano_path (str): Path to the annotation text file.

    Returns:
        np.ndarray: Array of sleep stage labels as integers, with stage '5' replaced by '4'.
    """
    with open(ano_path, 'r') as f:
        lines = f.readlines()

    # Strip whitespace and remove empty lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    # Replace sleep stage label '5' with '4' for consistency
    sleep_stages = [line.replace('5', '4') for line in lines]

    # Convert to numpy array of integers
    ano = np.array(sleep_stages, dtype=np.int32)
    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject/session recording by loading signals and annotations,
    preprocessing signals, synchronizing epochs, and saving the processed data.

    Args:
        sub_id (str): Unique identifier for the subject/session.
        sig_path (str): Path to the raw signal file (.rec).
        ano_path (str): Path to the annotation file (.txt).
    """
    # Convert .rec file to .edf format by copying and renaming
    shutil.copyfile(sig_path, sig_path.replace('.rec', '.edf'))
    sig_path = sig_path.replace('.rec', '.edf')

    # Load signal data and recording start time using utility function
    start_time, sig_dict = load_sig(sig_path, channel_id)

    # Remove temporary .edf file after loading
    os.remove(sig_path)

    # Load sleep stage annotations
    ano = load_ano(ano_path)

    # Preprocess signals including resampling to target frequency
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal epochs and annotation epochs mismatch
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        # Truncate signals and annotations to the minimal number of epochs
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid sleep stage labels
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())

    # Save the processed signals and annotations to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, sig_path, ano_path).
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Main function to orchestrate preprocessing of all ISRUC dataset recordings
    using multiprocessing for parallel execution.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    task_inputs = []

    # Process Subgroup 1 and Subgroup 3 recordings
    for i in [1, 3]:
        subgroup_dir = os.path.join(src_root, f"Subgroup_{i}")
        for sub_id in os.listdir(subgroup_dir):
            sig_path = os.path.join(subgroup_dir, sub_id, f"{sub_id}.rec")
            ano_path = os.path.join(subgroup_dir, sub_id, f"{sub_id}_1.txt")  # Annotation from 1st expert
            sub_id_full = f"Subgroup_{i}-{sub_id}"
            task_inputs.append((sub_id_full, sig_path, ano_path))

    # Process Subgroup 2 recordings with two sessions per subject
    subgroup_dir = os.path.join(src_root, f"Subgroup_2")
    for sub_id in os.listdir(subgroup_dir):
        # Session 1 paths and IDs
        sig_path_1 = os.path.join(subgroup_dir, sub_id, "1", "1.rec")
        ano_path_1 = os.path.join(subgroup_dir, sub_id, "1", "1_1.txt")  # Annotation from 1st expert
        sub_id_1 = f"Subgroup_2-{sub_id}-S1"
        task_inputs.append((sub_id_1, sig_path_1, ano_path_1))
        # Session 2 paths and IDs
        sig_path_2 = os.path.join(subgroup_dir, sub_id, "2", "2.rec")
        ano_path_2 = os.path.join(subgroup_dir, sub_id, "2", "2_1.txt")  # Annotation from 1st expert
        sub_id_2 = f"Subgroup_2-{sub_id}-S2"
        task_inputs.append((sub_id_2, sig_path_2, ano_path_2))

    # Filter out subjects/sessions marked for removal
    task_inputs = [task_input for task_input in task_inputs if task_input[0] not in SUB_REMOVE]

    # Parallel processing with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(task_inputs), desc="Processing ISRUC Dataset") as pbar:
            for args in task_inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING ISRUC DATASET', '='*30)

    # Source directory containing raw ISRUC dataset files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/ISRUC/unrar/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/ISRUC/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for all signals (Hz)
    resample_rate = 100

    # List of subject/session IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel names to their corresponding electrode pairs in different subgroups
    channel_id = {
        'F3': (('F3', 'A2'), 'F3-A2', 'F3-M2'),
        'F4': (('F4', 'A1'), 'F4-A1', 'F4-M1'),
        'C3': (('C3', 'A2'), 'C3-A2', 'C3-M2'),
        'C4': (('C4', 'A1'), 'C4-A1', 'C4-M1'),
        'O1': (('O1', 'A2'), 'O1-A2', 'O1-M2'),
        'O2': (('O2', 'A1'), 'O2-A1', 'O2-M1'),
        'E1': (('LOC', 'A2'), 'LOC-A2', 'E1-M2'),
        'E2': (('ROC', 'A1'), 'ROC-A1', 'E2-M1'),

        'Chin': ('X1', '24'),
    }

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform final formatting checks on the preprocessed dataset
    formatting_check(dst_root)
