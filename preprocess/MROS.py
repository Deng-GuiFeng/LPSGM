# -*- coding: utf-8 -*-
"""
MROS.py

This module handles the preprocessing of the MROS (Multi-Resolution Overnight Sleep) dataset for the LPSGM project.
It includes loading and parsing sleep stage annotations, signal loading and preprocessing, synchronization of epochs,
removal of unknown labels, and saving processed data in a structured format. The preprocessing pipeline supports
multiprocessing for efficient handling of large datasets.

Key functionalities:
- Parsing XML annotation files to extract sleep stages and remap stage labels to a unified scheme.
- Loading polysomnography signals from EDF files and resampling to a target frequency.
- Ensuring alignment between signal epochs and annotation epochs.
- Removing epochs with unknown or invalid sleep stage labels.
- Saving preprocessed signals and annotations for downstream model training.
- Multiprocessing support for scalable dataset processing.

This preprocessing is critical for preparing the MROS dataset for sleep staging and mental disorder diagnosis tasks
within the LPSGM framework.
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
    Load and parse sleep stage annotations from a Profusion XML file.

    Args:
        ano_path (str): Path to the annotation XML file.

    Returns:
        np.ndarray: Array of sleep stage labels with remapped classes:
                    Stage 4 mapped to 3, Stage 5 mapped to 4, others unchanged.
    """
    # Parse XML to find all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        # Remap stage 4 to 3 (NREM3), stage 5 to 4 (REM) for consistency with LPSGM labeling
        if ss == 4:
            ss = 3
        elif ss == 5:
            ss = 4
        sleep_stages[i] = ss
    # Convert list of stages to numpy array
    ano = np.array(sleep_stages)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's recording: load signals and annotations, preprocess signals,
    synchronize epochs, remove unknown labels, and save processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF signal file.
        ano_path (str): Path to the annotation XML file.

    Returns:
        None
    """
    # Load signals and their start time from EDF file for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dictionary of channel signals

    # Load and parse annotation sleep stages
    ano = load_ano(ano_path)

    # Preprocess signals: resampling, filtering, normalization as defined in utils.pre_process
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signals dictionary

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Print warning if number of signal epochs does not match annotation epochs
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Use the minimal number of epochs to synchronize signals and annotations
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid sleep stage labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names from the channel_id dictionary keys
    channel_names = list(channel_id.keys())
    # Save the processed signals and annotations to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with error handling.

    Args:
        *args: Arguments to be passed to process_recording (sub_id, sig_path, ano_path).

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Print error message if processing fails for a subject
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Main function to run preprocessing on the entire MROS dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel processes to use.

    Returns:
        None
    """
    Inputs = []
    # Iterate over dataset groups (folders) in EDF root directory
    for group in os.listdir(edf_root):
        edf_group_dir = os.path.join(edf_root, group)
        ano_group_dir = os.path.join(ano_root, group)

        # Iterate over each subject file in the group directory
        for sub_id in os.listdir(edf_group_dir):
            sub_id = sub_id.split('.')[0]  # Remove file extension to get subject ID
            if sub_id in SUB_REMOVE:
                # Skip subjects listed for removal
                continue
            edf_path = os.path.join(edf_group_dir, sub_id+".edf")
            ano_path = os.path.join(ano_group_dir, sub_id+"-profusion.xml")
            Inputs.append((sub_id, edf_path, ano_path))
        
    # Use multiprocessing Pool to process recordings in parallel
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing MROS Dataset") as pbar:
            for args in Inputs:
                # Apply single_process asynchronously with progress bar update on completion
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder test function for future testing of preprocessing pipeline.

    Returns:
        None
    """
    pass
        

if __name__ == "__main__":
    print('='*30, 'PREPROCESSING MROS DATASET', '='*30)

    # Source directory containing raw MROS polysomnography data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mros/files/polysomnography/"
    # Destination directory for saving preprocessed data
    dst_root = r"./data/MROS/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # List of subject IDs to exclude from processing
    SUB_REMOVE = ["mros-visit1-aa2931", ]

    # Target resampling rate for signals in Hz
    resample_rate = 100

    # Dictionary defining channel mappings for signal extraction
    # Keys are channel labels used in processing; values are tuples of possible channel name variants in EDF files
    channel_id = {
        'C3': ('C3-A2', ('C3','M2'), ('C3','A2')),
        'C4': ('C4-A1', ('C4','M1'), ('C4','A1')),
        'E1': (('E1','M2'), ('E1','A2'), ('LOC','M2'), ('LOC','A2'), 'LOC'), 
        'E2': (('E2','M1'), ('E2','A1'), ('ROC','M1'), ('ROC','A1'), 'ROC'),

        'Chin': ('L Chin-R Chin', ('L Chin','R Chin'), ('L Chin','RChin'), ('LChin','R Chin'), ('LChin','RChin'), ),
    }
    
    # Paths to EDF files and annotation XML files
    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    # Run preprocessing with specified number of parallel processes
    run(200)

    # Perform formatting check on the preprocessed dataset to ensure data integrity
    formatting_check(dst_root)

    # Placeholder for future test invocation
    # test()
