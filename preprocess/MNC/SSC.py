# -*- coding: utf-8 -*-
"""
SSC.py

This script preprocesses the MNC-SSC cohort dataset for the LPSGM project. It handles loading raw PSG recordings,
applying signal preprocessing including resampling, mapping diagnoses, and saving the processed data for downstream
sleep staging and mental disorder diagnosis tasks. The preprocessing pipeline supports parallel processing to
efficiently handle large-scale data.

Key functionalities:
- Load PSG signals from EDF files with specified channel configurations
- Preprocess signals (e.g., filtering, resampling)
- Map subject IDs and diagnoses according to cohort metadata
- Save processed signals and labels in a structured format
- Parallelize processing across multiple CPU cores for scalability
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_SSC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save PSG signals and diagnosis for a single subject.

    Args:
        sub_id (str): Subject identifier mapped to cohort keys
        edf_path (str): File path to the subject's EDF recording
    """
    # Load raw signals and start time from EDF file using predefined channel configuration
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Apply preprocessing steps including resampling to target frequency
    sig_dict = pre_process(sig_dict, resample_rate)
    # Retrieve diagnosis label mapped from cohort subject ID
    diagnosis = Diagnosis_Mapping[Cohort_SSC[sub_id]]
    # Save the preprocessed signals and diagnosis label to destination directory
    save(dst_root, sub_id, sig_dict, diagnosis)


def single_process(*args):
    """
    Wrapper function to safely process a single recording with error handling.

    Args:
        *args: Arguments to be passed to process_recording (sub_id, edf_path)
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log error without interrupting the overall processing pipeline
        print(f"Error processing {args[0]}: {e}")


def map_sub_id(sub_id):
    """
    Extract and standardize subject ID from raw filename.

    Args:
        sub_id (str): Raw subject identifier string (e.g., "chc001-nsrr")

    Returns:
        str: Standardized subject ID in uppercase (e.g., "CHC001")
    """
    return sub_id.split('-')[0].upper()


def run(num_processes):
    """
    Main function to coordinate preprocessing of all subjects using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to spawn
    """
    # List all subject files in source directory (without file extensions)
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)])
    # Filter out subjects explicitly removed from processing
    subjects = [s for s in subjects if map_sub_id(s) not in SUB_REMOVE]
    # Keep only subjects present in the cohort metadata dictionary
    subjects = [s for s in subjects if map_sub_id(s) in Cohort_SSC.keys()]
    # Prepare input argument tuples for multiprocessing pool
    Inputs = [
        (map_sub_id(s), os.path.join(src_root, s + ".edf")) for s in subjects
    ]

    # Initialize multiprocessing pool and progress bar for concurrent processing
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing MNC-SSC Dataset") as pbar:
            for args in Inputs:
                # Apply processing asynchronously with progress bar update callback
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING MNC-SSC DATASET', '=' * 30)

    # Source directory containing raw EDF files for MNC-SSC cohort
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/ssc/"
    # Destination directory where preprocessed data will be saved
    dst_root = r"./data/MNC-SSC/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate in Hz for all signals
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty by default)
    SUB_REMOVE = []

    # Channel configuration dictionary specifying bipolar derivations and reference channels
    channel_id = {
        # EEG channels with bipolar montage and reference
        # 'F3': (('F3','M2'), 'F3'),
        # 'F4': (('F4','M1'), 'F4'),
        'C3': (('C3', 'M2'), 'C3'),
        'C4': (('C4', 'M1'), 'C4'),
        'O1': (('O1', 'M2'), 'O1'),
        'O2': (('O2', 'M1'), 'O2'),
        # EOG channels
        'E1': (('E1', 'M2'), 'E1'),
        'E2': (('E2', 'M1'), 'E2'),
        # Chin EMG channels
        'Chin': (('lchin', 'cchin'), 'cchin'),
    }

    # Execute preprocessing with 100 parallel worker processes
    run(100)
