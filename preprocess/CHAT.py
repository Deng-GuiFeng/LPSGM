# -*- coding: utf-8 -*-
"""
CHAT dataset preprocessing module for the LPSGM project.

This script processes the Cleveland Hospital and Associated Technologies (CHAT) polysomnography dataset,
which includes EEG and other biosignal recordings along with expert sleep stage annotations.
It performs the following key functions:
- Loading and converting sleep stage annotations to a standardized 5-class format
- Loading and preprocessing multi-channel PSG signals with resampling
- Synchronizing signal epochs with annotation epochs, handling mismatches
- Removing epochs with unknown or invalid labels
- Saving the processed data in a structured format for downstream model training
- Parallel processing of multiple recordings to accelerate dataset preparation

This preprocessing is critical for preparing the CHAT dataset to be compatible with the LPSGM model,
which requires consistent epoch lengths, channel configurations, and label formats.
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
    Load and convert sleep stage annotations from the CHAT dataset XML file.

    Args:
        ano_path (str): Path to the annotation XML file.

    Returns:
        np.ndarray: Array of sleep stage labels converted to 5-class format.
                    Original stages 4 and 5 are remapped to 3 and 4 respectively.
    """
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        # Remap stage 4 to 3 and stage 5 to 4 to conform to 5-class sleep staging
        if ss == 4:
            ss = 3
        elif ss == 5:
            ss = 4
        sleep_stages[i] = ss
    ano = np.array(sleep_stages)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single PSG recording and its corresponding annotation file.

    This includes loading signals, preprocessing (e.g., resampling),
    loading and aligning annotations, removing unknown labels, and saving the processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF signal file.
        ano_path (str): Path to the annotation XML file.
    """
    # Load raw signals and recording start time from EDF file for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)    # returns start_time and dict of channel signals

    # Load and convert sleep stage annotations
    ano = load_ano(ano_path)

    # Preprocess signals (e.g., resample to target frequency)
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Log warning if signal and annotation epoch counts differ
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Truncate to minimal epoch count to ensure alignment
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save processed signals and annotations to destination directory
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
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Main function to process all recordings in the CHAT dataset using multiprocessing.

    It collects all subject IDs and their corresponding EDF and annotation paths,
    then processes them in parallel.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    Inputs = []
    # Iterate over dataset groups (subdirectories)
    for group in os.listdir(edf_root):
        edf_group_dir = os.path.join(edf_root, group)
        ano_group_dir = os.path.join(ano_root, group)

        # Iterate over subject recordings in each group
        for sub_id in os.listdir(edf_group_dir):
            if sub_id in SUB_REMOVE:
                # Skip subjects marked for removal
                continue

            sub_id = sub_id.split('.')[0]
            edf_path = os.path.join(edf_group_dir, sub_id+".edf")
            ano_path = os.path.join(ano_group_dir, sub_id+"-profusion.xml")
            Inputs.append((sub_id, edf_path, ano_path))
        
    # Use multiprocessing pool to process recordings in parallel with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing CHAT Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder for test function.
    """
    pass


if __name__ == "__main__":
    # Source directory containing raw CHAT PSG dataset files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/chat/files/polysomnography/"
    # Destination directory for saving processed data
    dst_root = r"./data/CHAT/"
    # Remove existing processed data directory if present to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for all signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel mappings and alternative names for each channel of interest
    channel_id = {
        'F3': ('F3-M2', 'F3-A2', ('F3','M2'), ('F3','m2')), 
        'F4': ('F4-M1', 'F4-A1', ('F4','M1')), 
        'C3': ('C3-M2', 'C3-A2', ('C3','M2'), ('C3','m2')), 
        'C4': ('C4-M1', 'C4-A1', ('C4','M1')), 
        'O1': ('O1-M2', 'O1-A2', ('O1','M2'), ('O1','m2')), 
        'O2': ('O2-M1', 'O2-A1', ('O2','M1')), 
        'E1': (('E1', 'M2'), ('E1', 'm2'), 'E1'), 
        'E2': (('E2', 'M1'), 'E2'), 

        'Chin': (
            ('LCHIN','CCHIN'), ('LCHIN','CChin'), ('LCHIN','Cchin'), ('LCHIN','cchin'),
            ('LChin','CCHIN'), ('LChin','CChin'), ('LChin','Cchin'), ('LChin','cchin'),
            ('Lchin','CCHIN'), ('Lchin','CChin'), ('Lchin','Cchin'), ('Lchin','cchin'),
            ('Lchin','Rchin'), 'LChin-Rchin',
        ),
    }

    # Paths to raw EDF signal files and annotation XML files
    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    # Execute dataset processing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the processed dataset directory
    formatting_check(dst_root)

    # test()
