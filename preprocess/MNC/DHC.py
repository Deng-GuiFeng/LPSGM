# -*- coding: utf-8 -*-
"""
DHC.py

This script performs preprocessing for the MNC DHC cohort dataset, which is part of the LPSGM project.
It processes raw PSG recordings by loading signals, applying preprocessing steps including resampling,
mapping diagnoses, and saving the processed data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Load raw PSG signal data from EDF files
- Preprocess signals with resampling and filtering
- Map subject IDs to diagnosis labels using cohort information
- Parallel processing of multiple recordings to accelerate preprocessing
- Save processed data in a structured format for model training and evaluation

This preprocessing step ensures data consistency and quality for the large-scale polysomnography model.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_DHC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save PSG signal data for a single subject.

    Args:
        sub_id (str): Subject identifier
        edf_path (str): Path to the EDF file containing raw PSG signals

    Returns:
        None
    """
    # Load raw signals and start time from EDF file using predefined channel configuration
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Apply preprocessing steps including resampling to target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)
    # Map subject ID to diagnosis label using cohort mapping
    diagnosis = Diagnosis_Mapping[Cohort_DHC[sub_id]]
    # Save preprocessed signals and diagnosis label to destination directory
    save(dst_root, sub_id, sig_dict, diagnosis)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

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
    Extract the base subject ID by removing any suffix after a hyphen.

    Args:
        sub_id (str): Original subject ID string (e.g., "Sub76-nsrr")

    Returns:
        str: Base subject ID (e.g., "Sub76")
    """
    return sub_id.split('-')[0]


def run(num_processes):
    """
    Main function to coordinate preprocessing of all subjects in the dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    Inputs = []
    # Iterate over predefined dataset groups (training, test subsets)
    for group in groups:
        group_root = os.path.join(src_root, group)
        # List all subject files in the group directory (without file extensions)
        subjects = set([f.split('.')[0] for f in os.listdir(group_root)])
        # Remove subjects listed in SUB_REMOVE to exclude problematic or unwanted data
        subjects = [s for s in subjects if s not in SUB_REMOVE]
        # Filter subjects to include only those present in the cohort mapping
        subjects = [s for s in subjects if map_sub_id(s) in Cohort_DHC.keys()]
        # Create tuples of (mapped subject ID, full EDF file path) for processing
        subjects = [
            (map_sub_id(s), os.path.join(group_root, s + ".edf")) for s in subjects
        ]
        Inputs.extend(subjects)

    # Use multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        # Progress bar to monitor preprocessing progress
        with tqdm(total=len(Inputs), desc="Processing MNC-DHC Dataset") as pbar:
            for args in Inputs:
                # Asynchronously apply processing function to each subject
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('=' * 30, 'PREPROCESSING MNC-DHC DATASET', '=' * 30)

    # Source directory containing raw EDF files organized by groups
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/dhc/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/MNC-DHC/"
    # Remove existing destination directory to ensure clean preprocessing output
    shutil.rmtree(dst_root, ignore_errors=True)

    # Dataset groups to process: training and multiple test subsets
    groups = (
        "training",
        "test/controls",
        "test/nc-lh",
        "test/nc-nh",
    )

    # Target sampling rate for resampling signals during preprocessing
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty by default)
    # SUB_REMOVE = ['Sub13-nsrr', 'Sub87-nsrr']
    SUB_REMOVE = []

    # Dictionary defining channel IDs and their corresponding signal labels to load
    channel_id = {
        'F3': ('F3',),
        'F4': ('F4',),
        'C3': ('C3',),
        'C4': ('C4',),
        'O1': ('O1',),
        'O2': ('O2',),
        'E1': ('E1',),
        'E2': ('E2',),

        'Chin': ('chin',),
    }

    # Start preprocessing using 100 parallel worker processes
    run(100)
