# -*- coding: utf-8 -*-
"""
MESA.py

This module handles the preprocessing of the MESA (Multi-Ethnic Study of Atherosclerosis) polysomnography dataset 
for the LPSGM project. It includes loading raw EDF signals and corresponding sleep stage annotations, 
signal preprocessing (including resampling), synchronization of signal epochs with annotation labels, 
removal of unknown labels, and saving the processed data in a structured format.

Key functionalities:
- Parsing sleep stage annotations from XML files
- Loading and preprocessing multi-channel PSG signals
- Ensuring alignment between signal epochs and annotation labels
- Parallel processing of multiple subject recordings for efficient dataset preparation
- Integration with utility functions for saving and formatting checks

This preprocessing step is critical for preparing the MESA dataset to be used for sleep staging and mental disorder diagnosis 
using the LPSGM model.
"""

import numpy as np
import os
from mne.io import read_raw_edf
from scipy import signal
from scipy.interpolate import interp1d
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and parse sleep stage annotations from a Profusion XML file.

    Args:
        ano_path (str): Path to the annotation XML file.

    Returns:
        np.ndarray: Array of sleep stage labels mapped to integer classes.
                    Unknown or unmapped stages are assigned -1.
                    Stage mapping: '0'->0, '1'->1, '2'->2, '3' and '4'->3, '5'->4
    """
    # Mapping from original annotation labels to standardized sleep stage classes
    stage_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4}

    # Parse all SleepStage elements from the XML annotation file
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    # Map each sleep stage text to its corresponding integer label; assign -1 if unknown
    sleep_stages = [stage_dict.get(sleep_stage.get_text(), -1) for sleep_stage in sleep_stages]

    # Convert list to numpy array of int32 type
    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's PSG recording and corresponding annotations.

    Steps:
    - Load raw signals and their start time from EDF file
    - Load sleep stage annotations from XML file
    - Preprocess signals (e.g., resampling)
    - Align the number of epochs between signals and annotations
    - Remove epochs with unknown labels
    - Save the processed signals and labels to the destination directory

    Args:
        sub_id (str): Subject identifier
        sig_path (str): Path to the EDF signal file
        ano_path (str): Path to the annotation XML file
    """
    # Load raw signals and start time for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dict of channel signals

    # Load sleep stage annotations as integer labels
    ano = load_ano(ano_path)   

    # Preprocess signals: e.g., resample to target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signal dict

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal epochs and annotation epochs mismatch; truncate to minimal length
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names in order for saving
    channel_names = list(channel_id.keys())

    # Save processed signals and annotations to destination root directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, sig_path, ano_path)
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Print error message without interrupting the multiprocessing pool
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run the preprocessing pipeline over all subjects in the MESA dataset using multiprocessing.

    Steps:
    - Collect all subject IDs and corresponding EDF and annotation file paths
    - Filter out subjects listed in SUB_REMOVE
    - Use multiprocessing Pool to process recordings in parallel with progress bar

    Args:
        num_processes (int): Number of worker processes to use for parallel processing
    """
    Inputs = []
    # Iterate over all EDF files in the dataset directory
    for file in os.listdir(edf_root):
        sub_id = file.split('.')[0]
        edf_path = os.path.join(edf_root, sub_id + ".edf")
        ano_path = os.path.join(ano_root, sub_id + "-profusion.xml")
        Inputs.append((sub_id, edf_path, ano_path))

    # Exclude subjects specified in SUB_REMOVE list
    Inputs = [subject for subject in Inputs if subject[0] not in SUB_REMOVE]

    # Initialize multiprocessing pool and progress bar for parallel processing
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing MESA Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder for testing functionality.
    """
    pass


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING MESA DATASET', '='*30)

    # Source directory containing raw MESA polysomnography files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mesa/files/polysomnography/"

    # Destination directory for saving preprocessed data
    dst_root = r"./data/MESA/"

    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target sampling rate for resampling signals
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of desired channel names to their corresponding channel labels in EDF files
    channel_id = {
        'C4': ('EEG3',),      # EEG channel C4
        'E1': ('EOG-L',),     # Left EOG channel
        'E2': ('EOG-R',),     # Right EOG channel
        'Chin': ('EMG',),     # Chin EMG channel
    }

    # Paths to EDF files and annotation XML files
    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    # Execute preprocessing with 200 parallel processes
    run(200)

    # Perform formatting checks on the saved preprocessed dataset
    formatting_check(dst_root)

    # Uncomment to run tests if implemented
    # test()
