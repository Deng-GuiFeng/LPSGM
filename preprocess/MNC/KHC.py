# -*- coding: utf-8 -*-
"""
KHC.py

This script handles the preprocessing of the MNC-KHC cohort dataset for the LPSGM project.
It loads raw PSG recordings in EDF format, applies signal preprocessing including resampling,
maps subject diagnoses, and saves the processed data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Load and preprocess PSG signals from EDF files
- Map subject IDs to cohort and diagnosis information
- Parallel processing of multiple recordings for efficiency
- Flexible channel configuration for signal extraction
- Save preprocessed signals and diagnosis labels in a structured format

This preprocessing step is critical to ensure data consistency and quality before model training and evaluation.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_KHC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save PSG signals for a single subject recording.

    Args:
        sub_id (str): Subject identifier
        edf_path (str): Path to the EDF file containing raw PSG data

    Returns:
        None
    """
    # Load signal data and start time from EDF file using specified channel configuration
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Apply preprocessing steps such as filtering and resampling to the signals
    sig_dict = pre_process(sig_dict, resample_rate)
    # Map subject ID to diagnosis label using cohort and diagnosis mappings
    diagnosis = Diagnosis_Mapping[Cohort_KHC[sub_id]]
    # Save the preprocessed signals and diagnosis label to the destination directory
    save(dst_root, sub_id, sig_dict, diagnosis)


def single_process(*args):
    """
    Wrapper function to process a single recording with error handling.

    Args:
        *args: Arguments for process_recording function (sub_id, edf_path)

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing of a subject
        print(f"Error processing {args[0]}: {e}")


def map_sub_id(sub_id):
    """
    Extract the base subject ID by removing dataset-specific suffixes.

    Args:
        sub_id (str): Original subject ID string (e.g., "10777568_p61-nsrr")

    Returns:
        str: Base subject ID (e.g., "10777568_p61")
    """
    return sub_id.split('-')[0]


def run(num_processes):
    """
    Main function to coordinate preprocessing of all subject recordings using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    # List all subject files in the source directory (remove file extensions)
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)])
    # Exclude subjects listed in SUB_REMOVE
    subjects = [s for s in subjects if s not in SUB_REMOVE]
    # Filter subjects to those present in the cohort mapping after ID normalization
    subjects = [s for s in subjects if map_sub_id(s) in Cohort_KHC.keys()]
    # Prepare input arguments as tuples of (subject_id, edf_file_path)
    Inputs = [
        (map_sub_id(s), os.path.join(src_root, s + ".edf")) for s in subjects
    ]

    # Create a multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        # Use tqdm progress bar to monitor processing status
        with tqdm(total=len(Inputs), desc="Processing MNC-KHC Dataset") as pbar:
            for args in Inputs:
                # Submit asynchronous processing tasks with progress update callback
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING MNC-KHC DATASET', '=' * 30)

    # Define source directory containing raw EDF files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/khc/"
    # Define destination directory for saving preprocessed data
    dst_root = r"./data/MNC-KHC/"
    # Remove existing destination directory and its contents if present
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for all signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty here)
    SUB_REMOVE = []

    # Channel configuration dictionary specifying EEG and other channels to extract
    # Each key maps to a tuple defining the channel names in EDF and the standardized channel label
    channel_id = {
        # EEG channels referenced to contralateral mastoids
        # 'F3': (('F3','M2'), 'F3'),
        # 'F4': (('F4','M1'), 'F4'),
        'C3': (('C3', 'M2'), 'C3'),
        'C4': (('C4', 'M1'), 'C4'),
        'O1': (('O1', 'M2'), 'O1'),
        'O2': (('O2', 'M1'), 'O2'),
        # EOG channels
        'E1': (('E1', 'M2'), 'E1'),
        'E2': (('E2', 'M1'), 'E2'),
        # Chin EMG channel
        'Chin': ('chin',),
    }

    # Execute preprocessing with 100 parallel worker processes
    run(100)
