# -*- coding: utf-8 -*-
"""
HOMEPAP.py

This module handles the preprocessing of the HOMEPAP polysomnography dataset for the LPSGM project.
It reads raw EDF signal files and corresponding XML annotation files, extracts and processes EEG and related signals,
resamples them, aligns signal epochs with sleep stage annotations, removes unknown labels, and saves the cleaned data.
The preprocessing supports multi-channel configurations and parallel processing to efficiently handle large datasets.

Key functionalities:
- Loading and parsing sleep stage annotations from XML files
- Loading raw EDF signals with flexible channel mapping
- Signal preprocessing including resampling and epoch segmentation
- Synchronizing signal epochs with annotation epochs and handling mismatches
- Removing epochs with unknown or invalid labels
- Saving processed data in a structured format for downstream model training
- Parallel processing with progress tracking for scalability

This preprocessing is critical for preparing the HOMEPAP dataset for sleep staging and mental disorder diagnosis tasks.
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
    Load sleep stage annotations from a Profusion XML annotation file.

    Args:
        ano_path (str): Path to the XML annotation file.

    Returns:
        np.ndarray: Array of integer sleep stage labels corresponding to 30-second epochs.
                    Mapping: 0 (Wake), 1 (N1), 2 (N2), 3 (N3), 4 (REM), 6 (Unknown/Other).
    """
    # Mapping from annotation string labels to integer sleep stage codes
    stage_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4, '6': 6}

    # Parse XML and extract all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')
    # Convert string labels to integers using the mapping dictionary
    sleep_stages = [stage_dict[sleep_stage.get_text()] for sleep_stage in sleep_stages]

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's recording by loading signals and annotations, preprocessing signals,
    aligning epochs, removing unknown labels, and saving the processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF signal file.
        ano_path (str): Path to the XML annotation file.
    """
    # Load raw signals and their start time from EDF file for specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dict of channel signals

    # Load sleep stage annotations from XML file
    ano = load_ano(ano_path)   

    # Preprocess signals: resample and segment into epochs
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signals dictionary

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ and truncate to minimal length
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        # Truncate all channel signals to minimal epoch count
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        # Truncate annotations accordingly
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names from channel_id keys
    channel_names = list(channel_id.keys())
    # Save the cleaned and processed data to the destination directory
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
    Main function to run preprocessing on all subjects in the HOMEPAP dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel processes to use.
    """
    Inputs = []
    # Iterate over dataset groups (e.g., different recording sessions or batches)
    for group in os.listdir(edf_root):
        edf_group_dir = os.path.join(edf_root, group)
        ano_group_dir = os.path.join(ano_root, group)

        # Collect subject IDs and corresponding EDF and annotation file paths
        for file in os.listdir(edf_group_dir):
            sub_id = file.split('.')[0]
            edf_path = os.path.join(edf_group_dir, sub_id+".edf")
            ano_path = os.path.join(ano_group_dir, sub_id+"-profusion.xml")
            Inputs.append((sub_id, edf_path, ano_path))

    # Filter out subjects to be removed due to data quality or other reasons
    Inputs = [subject for subject in Inputs if subject[0] not in SUB_REMOVE]

    # Use multiprocessing Pool to parallelize processing with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing HOMEPAP Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder test function for future unit tests or debugging.
    """
    pass


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING HOMEPAP DATASET', '='*30)

    # Source directory containing raw HOMEPAP polysomnography data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/homepap/polysomnography/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/HOMEPAP/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling frequency in Hz for all signals
    resample_rate = 100

    # List of subject IDs to exclude from preprocessing due to data issues
    SUB_REMOVE = ["homepap-lab-full-1600047", "homepap-lab-full-1600197"]

    # Dictionary mapping canonical channel names to possible channel labels in EDF files
    # Supports multiple referencing schemes and alternative names for robustness
    channel_id = {
        'F3': ('F3-M2', ('F3','M2'), ('F3','A2')),
        'F4': ('F4-M1', ('F4','M1'), ('F4','A1')),
        'C3': ('C3-M2', ('C3','M2'), ('C3','A2')),
        'C4': ('C4-M1', ('C4','M1'), ('C4','A1')), 
        'O1': ('O1-M2', ('O1','M2'), ('O1','A2')), 
        'O2': ('O2-M1', ('O2','M1'), ('O2','A1')),
        'E1': ('E1-M2', 'E1-E2', ('E-1','M2'), ('E-1','A2'), ('E1','M2'), ('E1', 'A2'), ('L-EOG','M2'), ('L-EOG', 'A2'), ('LOC','M2'), ('LOC', 'A2'), 'E-1', 'E1', 'L-EOG', 'LOC'), 
        'E2': ('E2-M1', ('E-2','M1'), ('E-2','A1'), ('E2','M1'), ('E2','A1'), ('R-EOG','M1'), ('R-EOG','A1'), ('ROC','M1'), ('ROC','A1'), 'E-2', 'E2', 'R-EOG', 'ROC'),

        'Chin': (
                    ('CCHIN','LCHIN'), ('CCHIN','LChin'), ('CCHIN','Lchin'), ('CCHIN','Chin1'), ('CCHIN','EMG1'),
                    ('CChin','LCHIN'), ('CChin','LChin'), ('CChin','Lchin'), ('CChin','Chin1'), ('CChin','EMG1'),
                    ('Cchin','LCHIN'), ('Cchin','LChin'), ('Cchin','Lchin'), ('Cchin','Chin1'), ('Cchin','EMG1'),
                    ('Chin2','LCHIN'), ('Chin2','LChin'), ('Chin2','Lchin'), ('Chin2','Chin1'), ('Chin2','EMG1'),
                    ('EMG2','LCHIN'), ('EMG2','LChin'), ('EMG2','Lchin'), ('EMG2','Chin1'), ('EMG2','EMG1'),
                    ('L.','C.'), ('Lchin','Rchin'), ('LChin', 'RChin'), 'Lchin-Cchin', 'Chin1-Chin2', 'CHIN', 'Chin', 'Chin EMG', 
                ),
    }

    # Root directories for EDF signals and annotation XML files
    edf_root = os.path.join(src_root, "edfs/lab/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/lab/")

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the saved preprocessed data
    formatting_check(dst_root)

    # test()  # Uncomment to run tests if implemented
