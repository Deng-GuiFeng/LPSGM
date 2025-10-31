# -*- coding: utf-8 -*-
"""
DOD-O Dataset Preprocessing Module for LPSGM Project

This script handles the preprocessing of the DOD-O polysomnography dataset, which is part of the Large Polysomnography Model (LPSGM) project. 
It reads raw PSG data stored in HDF5 files, extracts relevant EEG, EOG, and EMG channels, resamples signals, aligns them with hypnogram annotations, 
removes unknown sleep stage labels, and saves the processed data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Reading multi-channel PSG signals and hypnogram annotations from HDF5 files
- Resampling signals to a uniform sampling rate
- Synchronizing signal epochs with annotation epochs
- Removing epochs with unknown or invalid labels
- Parallel processing of multiple recordings for efficient dataset preparation

This preprocessing step is critical for ensuring data consistency and quality before training or evaluating the LPSGM model.
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
    Reads PSG signals and hypnogram annotations from an HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file containing PSG data.

    Returns:
        dict: Dictionary mapping channel names to their data and sample rate.
        np.ndarray: Hypnogram annotation array.
        int: Sampling rate of the signals (Hz).
    """
    h5_file = h5py.File(h5_path, 'r')

    sig_dict = {}
    sample_rate = None
    
    # Extract signal data for each predefined channel from the HDF5 file
    for ch_name, h5_path_str in channel_id.items():
        sig_data = h5_file[h5_path_str][:]
        sig_dict[ch_name] = {
            'sample_rate': None,  # To be computed below
            'data': sig_data
        }
    
    # Load hypnogram annotations (sleep stage labels per 30-second epoch)
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
    Processes a single subject's PSG recording: reads data, preprocesses signals, aligns epochs with annotations,
    removes unknown labels, and saves the processed data.

    Args:
        sub_id (str): Subject identifier corresponding to the filename (without extension).
    """
    # Construct full path to the subject's HDF5 file
    h5_path = os.path.join(src_root, f"{sub_id}.h5")
    
    # Read raw signals and annotations from the HDF5 file
    sig_dict, ano, sample_rate = read_h5(h5_path)

    # Preprocess signals: resample and apply any necessary filtering or normalization
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that the number of signal epochs matches the number of annotation epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Use the minimal number of epochs to synchronize signals and annotations
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid sleep stage labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    
    # Save the processed data to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a recording with exception handling for multiprocessing.

    Args:
        *args: Arguments to pass to process_recording (expects sub_id as first argument).
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Executes parallel preprocessing of all subject recordings in the source directory.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # Identify all subject IDs by listing files in source directory and excluding removed subjects
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # Use multiprocessing Pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing DOD-H Dataset"):
            pass


def test():
    """
    Sequentially processes all subject recordings for testing or debugging purposes.
    """
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING DOD-O DATASET', '='*30)

    # Define source directory containing raw DOD-O HDF5 files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/DOD/DOD-O/"
    # Define destination directory for saving processed data
    dst_root = r"./data/DOD-O/"
    # Remove existing processed data directory to avoid conflicts
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target sampling rate for resampling signals (Hz)
    resample_rate = 100

    # List of subjects to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel names to their HDF5 dataset paths within the file
    channel_id = {
        'F3': '/signals/eeg/F3_M2', 
        'C3': '/signals/eeg/C3_M2', 
        'C4': '/signals/eeg/C4_M1', 
        'O1': '/signals/eeg/O1_M2', 
        'O2': '/signals/eeg/O2_M1', 
        'E1': '/signals/eog/EOG1', 
        'E2': '/signals/eog/EOG2', 
        'Chin': '/signals/emg/EMG',
    }

    # Start parallel preprocessing with 100 worker processes
    run(100)
    
    # Perform formatting checks on the processed data directory
    formatting_check(dst_root)

    # Uncomment below to run sequential test processing
    # test()
