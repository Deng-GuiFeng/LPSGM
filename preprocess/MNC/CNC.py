# -*- coding: utf-8 -*-
"""
CNC.py

This script handles the preprocessing of the MNC-CNC cohort dataset for the LPSGM project.
It loads raw PSG recordings in EDF format, applies signal preprocessing including resampling,
maps subject IDs to cohort information and diagnoses, and saves the processed data for downstream
sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Loading raw PSG signals from EDF files
- Preprocessing signals (e.g., filtering, resampling)
- Mapping subject IDs to cohort and diagnosis labels
- Parallel processing of multiple recordings to accelerate preprocessing
- Saving the processed data in a structured format

This preprocessing step is critical to ensure data consistency and quality before model training.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_CNC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save a single PSG recording.

    Args:
        sub_id (str): Subject identifier mapped to cohort keys
        edf_path (str): File path to the EDF recording

    Returns:
        None
    """
    # Load raw signal and start time from EDF file for specified channels
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Apply preprocessing steps such as filtering and resampling to signals
    sig_dict = pre_process(sig_dict, resample_rate)
    # Map subject ID to diagnosis label using cohort and diagnosis mappings
    diagnosis = Diagnosis_Mapping[Cohort_CNC[sub_id]]
    # Save the preprocessed signals and diagnosis label to destination directory
    save(dst_root, sub_id, sig_dict, diagnosis)


def single_process(*args):
    """
    Wrapper function to safely process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, edf_path)

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Print error message if processing fails for a subject
        print(f"Error processing {args[0]}: {e}")


def map_sub_id(sub_id):
    """
    Normalize subject ID by extracting the base ID and converting to uppercase.

    Args:
        sub_id (str): Original subject ID string (e.g., "chc001-nsrr")

    Returns:
        str: Normalized subject ID (e.g., "CHC001")
    """
    return sub_id.split('-')[0].upper()


def run(num_processes):
    """
    Main function to coordinate parallel preprocessing of all subjects in the source directory.

    Args:
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    # List all subject files (without extension) in source directory
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)])
    # Exclude subjects listed in SUB_REMOVE
    subjects = [s for s in subjects if s not in SUB_REMOVE]
    # Filter subjects to those present in the cohort mapping after normalization
    subjects = [s for s in subjects if map_sub_id(s) in Cohort_CNC.keys()]
    # Prepare input arguments as tuples of (normalized_sub_id, edf_file_path)
    Inputs = [
        (map_sub_id(s), os.path.join(src_root, s + ".edf")) for s in subjects
    ]

    # Create a multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        # Use tqdm progress bar to monitor processing status
        with tqdm(total=len(Inputs), desc="Processing MNC-CNC Dataset") as pbar:
            for args in Inputs:
                # Apply asynchronous processing with callback to update progress bar
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING MNC-CNC DATASET', '=' * 30)

    # Source directory containing raw EDF files for MNC-CNC cohort
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/cnc/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/MNC-CNC/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target sampling rate for resampling signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from preprocessing (currently empty)
    SUB_REMOVE = []

    # Dictionary defining channel mappings for signal extraction
    # Keys are standardized channel names, values are tuples of possible channel labels in EDF
    channel_id = {
        'F3': ('F3',),
        'F4': ('F4',),
        'C3': ('C3',),
        'C4': ('C4',),
        'O1': ('O1',),
        'O2': ('O2',),
        'E1': ('E1',),
        'E2': ('E2',),

        # Chin EMG channels may have alternative labels
        'Chin': ('cchin_l', 'chin'),
    }

    # Execute preprocessing using 100 parallel processes
    run(100)
