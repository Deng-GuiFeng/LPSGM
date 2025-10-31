# -*- coding: utf-8 -*-
"""
CCSHS.py

This module handles the preprocessing of the CCSHS (Chinese Clinical Sleep Health Study) dataset for the LPSGM project.
It includes loading and processing polysomnography (PSG) signals and corresponding sleep stage annotations,
resampling signals, removing unknown labels, and saving the processed data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Parsing XML annotation files to extract sleep stages and remapping stages to standard labels
- Loading PSG signal data from EDF files with specified channel configurations
- Preprocessing signals including resampling
- Synchronizing signal epochs and annotation epochs, handling mismatches
- Removing epochs with unknown or invalid labels
- Multiprocessing support for efficient dataset processing
- Final formatting checks on the processed dataset

This preprocessing step is critical for preparing the large-scale CCSHS dataset to be compatible with the LPSGM model pipeline.
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
    Load and parse sleep stage annotations from an XML file.

    Args:
        ano_path (str): Path to the XML annotation file.

    Returns:
        np.ndarray: Array of sleep stage labels with remapped stages:
                    Stage 4 is remapped to 3, Stage 5 is remapped to 4,
                    following the standard 5-class sleep staging convention.
    """
    # Parse XML and find all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        # Remap stage 4 to 3
        if ss == 4:
            ss = 3
        # Remap stage 5 to 4
        elif ss == 5:
            ss = 4
        sleep_stages[i] = ss
    ano = np.array(sleep_stages)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's recording by loading signals and annotations,
    preprocessing signals, synchronizing epochs, removing unknown labels,
    and saving the processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF file containing PSG signals.
        ano_path (str): Path to the XML annotation file.
    """
    # Load signal data and start time from EDF file for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)    # returns start_time and dict of signals keyed by channel

    # Load sleep stage annotations and remap stages
    ano = load_ano(ano_path)

    # Preprocess signals: resample to target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Use the minimal number of epochs to synchronize signals and annotations
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save processed signals and annotations to destination root directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording function.
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run the preprocessing pipeline on the CCSHS dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel processes to use.
    """
    Inputs = []

    # Iterate over all subject files in the EDF root directory
    for sub_id in os.listdir(edf_root):
        sub_id = sub_id.split('.')[0]

        # Skip subjects listed in SUB_REMOVE
        if sub_id in SUB_REMOVE:
            continue
        
        edf_path = os.path.join(edf_root, sub_id+".edf")
        ano_path = os.path.join(ano_root, sub_id+"-profusion.xml")
        Inputs.append((sub_id, edf_path, ano_path))
    
    # Create a multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing CCSHS Dataset") as pbar:
            for args in Inputs:
                # Asynchronously apply processing function with progress bar update on completion
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder test function for CCSHS preprocessing module.
    """
    pass
        

if __name__ == "__main__":
    print('='*30, 'PREPROCESSING CCSHS DATASET', '='*30)

    # Source directory containing raw CCSHS polysomnography data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/ccshs/ccshs/polysomnography/"
    # Destination directory for saving processed data
    dst_root = r"./data/CCSHS/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate in Hz for all signals
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel configurations for signal extraction
    # Each key is a channel name; value is a tuple of tuples specifying bipolar channel pairs
    channel_id = {
        'C3': (('C3','A2'),  ),
        'C4': (('C4','A1'),  ),
        'E1': (('LOC','A2'), ),
        'E2': (('ROC','A1'), ),

        'Chin': (('EMG1', 'EMG2'),),
    }

    # Paths to EDF files and annotation XML files
    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the processed dataset to ensure consistency
    formatting_check(dst_root)

    # test()
