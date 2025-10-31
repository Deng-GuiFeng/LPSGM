# -*- coding: utf-8 -*-
"""
DOD-H Dataset Preprocessing Module

This script handles the preprocessing of the DOD-H polysomnography dataset, which is part of the LPSGM project.
It reads raw PSG data from HDF5 files, extracts and resamples multi-channel signals, aligns them with hypnogram annotations,
removes unknown sleep stage labels, and saves the cleaned and formatted data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities include:
- Reading multi-channel PSG signals and hypnogram annotations from HDF5 files
- Resampling signals to a uniform sampling rate
- Synchronizing signal epochs with annotation epochs
- Removing epochs with unknown labels
- Parallel processing of multiple subject recordings for efficient preprocessing
"""

import numpy as np
import os
import h5py
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def read_h5(h5_path):
    """
    Read multi-channel PSG signals and hypnogram annotations from an HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file containing PSG data.

    Returns:
        dict: Dictionary mapping channel names to dicts with 'sample_rate' and 'data' keys.
        np.ndarray: Hypnogram annotation array.
        int: Calculated sample rate of the signals.
    """
    h5_file = h5py.File(h5_path, 'r')

    sig_dict = {}
    sample_rate = None
    
    # Extract signal data for each predefined channel
    for ch_name, h5_path_str in channel_id.items():
        sig_data = h5_file[h5_path_str][:]
        sig_dict[ch_name] = {
            'sample_rate': None,  # Placeholder, will be set after sample rate calculation
            'data': sig_data
        }
    
    # Load hypnogram annotations (sleep stage labels)
    ano = h5_file["/hypnogram"][:]
    
    # Calculate sample rate based on the ratio of signal length to annotation length and epoch duration (30s)
    if sig_dict:
        first_channel_data = list(sig_dict.values())[0]['data']
        sample_rate = int(first_channel_data.shape[0] / ano.shape[0] / 30)
        
        # Assign calculated sample rate to all channels
        for ch_data in sig_dict.values():
            ch_data['sample_rate'] = sample_rate
    
    h5_file.close()
    return sig_dict, ano, sample_rate


def process_recording(sub_id):
    """
    Process a single subject's recording: read data, preprocess signals, synchronize epochs, remove unknown labels, and save.

    Args:
        sub_id (str): Subject identifier corresponding to the HDF5 filename (without extension).
    """
    # Construct path to the subject's HDF5 file
    h5_path = os.path.join(src_root, f"{sub_id}.h5")

    # Read raw signals and annotations from file
    sig_dict, ano, sample_rate = read_h5(h5_path)

    # Preprocess signals: resample and apply any additional preprocessing steps defined in utils.pre_process
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs as annotations
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if there is a mismatch and truncate to minimal epoch count to maintain alignment
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown sleep stage labels to ensure clean dataset
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names in consistent order
    channel_names = list(channel_id.keys())

    # Save preprocessed signals and annotations to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function for processing a single recording with error handling.

    Args:
        *args: Arguments passed to process_recording, typically a single subject ID.
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all subjects in the source directory using parallel processing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # List all subject IDs by extracting filenames without extensions, excluding those in SUB_REMOVE
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing DOD-H Dataset"):
            pass


def test():
    """
    Sequentially process all subjects for testing/debugging purposes.
    """
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING DOD-H DATASET', '='*30)

    # Define source directory containing raw DOD-H HDF5 files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/DOD/DOD-H/"
    # Define destination directory for saving preprocessed data
    dst_root = r"./data/DOD-H/"
    # Remove existing destination directory to ensure clean preprocessing output
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate (Hz) for all signals after preprocessing
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel names to their corresponding HDF5 dataset paths
    channel_id = {
        'F3': '/signals/eeg/F3_M2',
        'F4': '/signals/eeg/F4_M1',
        'C3': '/signals/eeg/C3_M2',
        'E1': '/signals/eog/EOG1',
        'E2': '/signals/eog/EOG2',
        'Chin': '/signals/emg/EMG',
    }

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the saved preprocessed data
    formatting_check(dst_root)

    # Uncomment to run sequential test processing
    # test()
