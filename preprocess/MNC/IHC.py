# -*- coding: utf-8 -*-
"""
IHC.py

This script handles the preprocessing of the MNC IHC cohort dataset for the LPSGM project.
It loads raw PSG recordings in EDF format, applies signal preprocessing including resampling,
maps subject IDs to cohort diagnosis labels, and saves the processed signals and labels
in a structured format. The preprocessing pipeline supports parallel processing to efficiently
handle large datasets.

Key functionalities:
- Load raw PSG signals from EDF files
- Preprocess signals with resampling and filtering
- Map subject IDs to diagnosis labels based on cohort information
- Save processed data for downstream modeling
- Parallelize processing using multiprocessing Pool with progress tracking

This preprocessing step is critical for preparing the MNC IHC dataset for sleep staging and
mental disorder diagnosis model training and evaluation.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_IHC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save PSG signals for a single subject recording.

    Args:
        sub_id (str): Subject identifier
        edf_path (str): Path to the EDF file containing raw PSG signals

    Returns:
        None
    """
    # Load raw signals and start time from EDF file for specified channels
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Preprocess signals including resampling to target frequency
    sig_dict = pre_process(sig_dict, resample_rate)
    # Map subject ID to diagnosis label using cohort mapping
    diagnosis = Diagnosis_Mapping[Cohort_IHC[sub_id]]
    # Save preprocessed signals and diagnosis label to destination directory
    save(dst_root, sub_id, sig_dict, diagnosis)


def single_process(*args):
    """
    Wrapper function to safely process a single recording with error handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, edf_path)

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing without stopping the pipeline
        print(f"Error processing {args[0]}: {e}")


def map_sub_id(sub_id):
    """
    Extract base subject ID by removing any suffix after a hyphen.

    Args:
        sub_id (str): Original subject ID string (e.g., "20378prnotte-nsrr")

    Returns:
        str: Base subject ID (e.g., "20378prnotte")
    """
    return sub_id.split('-')[0]


def run(num_processes):
    """
    Main function to orchestrate preprocessing of all subjects in the source directory.

    Args:
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    # List all subject files (without extension) in the source directory
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)])
    # Exclude subjects listed in SUB_REMOVE
    subjects = [s for s in subjects if s not in SUB_REMOVE]
    # Filter subjects to only those present in the cohort mapping after ID normalization
    subjects = [s for s in subjects if map_sub_id(s) in Cohort_IHC.keys()]
    # Prepare input arguments as tuples of (subject_id, edf_file_path)
    Inputs = [
        (map_sub_id(s), os.path.join(src_root, s + ".edf")) for s in subjects
    ]

    # Create a multiprocessing pool to parallelize preprocessing
    with Pool(num_processes) as pool:
        # Use tqdm progress bar to monitor processing progress
        with tqdm(total=len(Inputs), desc="Processing MNC-IHC Dataset") as pbar:
            for args in Inputs:
                # Apply processing asynchronously with progress bar update callback
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING MNC-IHC DATASET', '=' * 30)

    # Source directory containing raw EDF files for MNC IHC cohort
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/ihc/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/MNC-IHC/"
    # Remove existing destination directory if it exists to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling frequency in Hz for all signals
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty here)
    SUB_REMOVE = []

    # Dictionary defining EEG and EMG channel configurations for signal loading
    channel_id = {
        'F3': (('F3', 'M2'),),
        'F4': (('F4', 'M1'),),
        'C3': (('C3', 'M2'),),
        'C4': (('C4', 'M1'),),
        'O1': (('O1', 'M2'),),
        'O2': (('O2', 'M1'),),
        'E1': (('E1', 'M2'),),
        'E2': (('E2', 'M1'),),

        'Chin': ('chin',),
    }

    # Execute preprocessing with 100 parallel worker processes
    run(100)
