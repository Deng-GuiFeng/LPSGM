# -*- coding: utf-8 -*-
"""
CFS.py

This script handles preprocessing of the Cleveland Family Study (CFS) polysomnography dataset 
for the LPSGM project. It loads raw EDF signals and corresponding XML annotations, performs 
resampling and label adjustments, removes unknown sleep stage labels, and saves the processed 
data in a structured format. The preprocessing pipeline supports parallel processing to 
accelerate handling of the large dataset.

Key functionalities include:
- Loading and parsing sleep stage annotations from XML files
- Loading and preprocessing multi-channel PSG signals from EDF files
- Synchronizing signal epochs and annotation lengths
- Removing epochs with unknown or invalid labels
- Saving cleaned data for downstream sleep staging and mental disorder diagnosis tasks
- Parallel processing with progress tracking

This preprocessing is critical for ensuring data quality and consistency before training 
the LPSGM model.
"""

import numpy as np
import os
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and process sleep stage annotations from an XML file.

    Args:
        ano_path (str): Path to the XML annotation file.

    Returns:
        np.ndarray: Array of sleep stage labels with remapped stages:
                    Stage 4 mapped to 3, Stage 5 mapped to 4.
    """
    # Parse XML and extract all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    # Convert text labels to float and remap stages for consistency
    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        if ss == 4:
            ss = 3  # Remap stage 4 to 3
        elif ss == 5:
            ss = 4  # Remap stage 5 to 4
        sleep_stages[i] = ss
    ano = np.array(sleep_stages)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's PSG recording and corresponding annotation.

    This includes loading signals and annotations, preprocessing signals,
    synchronizing epoch counts, removing unknown labels, and saving the processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF signal file.
        ano_path (str): Path to the XML annotation file.
    """
    # Load raw signals and start time from EDF file for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)   # returns start_time and sig_dict

    # Load and process sleep stage annotations
    ano = load_ano(ano_path)

    # Preprocess signals (e.g., resampling) to target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Use minimal epoch count to synchronize signals and annotations
        epochN = min(epoch_counts[0], ano.shape[0])
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
    Wrapper function to process a single recording with error handling.

    Args:
        *args: Arguments to pass to process_recording function.
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Print error message if processing fails for a subject
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all subjects in the CFS dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes.
    """
    Inputs = []

    # Iterate over all files in EDF root directory
    for sub_id in os.listdir(edf_root):
        sub_id = sub_id.split('.')[0]  # Extract subject ID without extension

        if sub_id in SUB_REMOVE:
            # Skip subjects marked for removal
            continue
        
        # Construct full paths for EDF and annotation files
        edf_path = os.path.join(edf_root, sub_id+".edf")
        ano_path = os.path.join(ano_root, sub_id+"-profusion.xml")
        Inputs.append((sub_id, edf_path, ano_path))
    
    # Create a multiprocessing pool and process recordings asynchronously
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing CFS Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder test function for future testing implementations.
    """
    pass
        

if __name__ == "__main__":
    print('='*30, 'PREPROCESSING CFS DATASET', '='*30)

    # Source directory containing raw CFS polysomnography data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/cfs/polysomnography/"
    # Destination directory for saving preprocessed data
    dst_root = r"./data/CFS/"
    # Remove existing destination directory to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel configurations and their corresponding electrode pairs
    channel_id = {
        'C3': (('C3','M2'), ),
        'C4': (('C4','M1'), ),
        'E1': (('LOC','M2'), ),
        'E2': (('ROC','M1'), ),

        'Chin': (('EMG1','EMG2'), ),
    }

    # Paths to EDF signal files and annotation XML files
    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the saved preprocessed data
    formatting_check(dst_root)

    # test()
