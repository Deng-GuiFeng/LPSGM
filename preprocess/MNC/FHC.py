# -*- coding: utf-8 -*-
"""
FHC.py

This script preprocesses the MNC FHC cohort data for the LPSGM project, which focuses on sleep staging and mental disorder diagnosis using polysomnography (PSG) data. 
It handles loading raw EDF recordings, applying signal preprocessing including resampling, mapping subject IDs to cohort metadata, and saving the processed signals along with diagnosis labels.
The script supports parallel processing to efficiently handle large datasets and ensures data consistency by filtering and mapping subject identifiers.

Key functionalities:
- Load raw PSG signals from EDF files
- Preprocess signals (e.g., resampling)
- Map subject IDs to diagnosis labels using cohort metadata
- Save preprocessed data for downstream model training and evaluation
- Parallelize processing for scalability

This file is a critical preprocessing step for preparing the MNC-FHC dataset within the LPSGM pipeline.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from cohort_info import Cohort_FHC, Diagnosis_Mapping
from utils import load_sig, pre_process, save


def process_recording(sub_id, edf_path):
    """
    Load, preprocess, and save the PSG recording for a given subject.

    Args:
        sub_id (str): Subject identifier mapped to cohort keys
        edf_path (str): File path to the subject's EDF recording

    Returns:
        None
    """
    # Load raw signals and start time from EDF file for specified channels
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Apply preprocessing steps such as filtering and resampling to the signals
    sig_dict = pre_process(sig_dict, resample_rate)
    # Map the subject ID to a diagnosis label using cohort metadata
    diagnosis = Diagnosis_Mapping[Cohort_FHC[sub_id]]
    # Save the preprocessed signals and diagnosis label to the destination directory
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
    Normalize subject IDs by removing suffixes and converting to uppercase.

    Args:
        sub_id (str): Original subject ID string (e.g., "171001b-c-nsrr")

    Returns:
        str: Normalized subject ID (e.g., "171001B-C")
    """
    return sub_id.replace('-nsrr', '').upper()


def run(num_processes):
    """
    Main function to coordinate preprocessing of all subjects using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    # List all subject files in source directory and extract base filenames
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)])
    # Filter out subjects explicitly marked for removal
    subjects = [s for s in subjects if s not in SUB_REMOVE]
    # Keep only subjects present in the cohort metadata after normalization
    subjects = [s for s in subjects if map_sub_id(s) in Cohort_FHC.keys()]
    # Prepare input arguments as tuples of (normalized_sub_id, edf_file_path)
    Inputs = [
        (map_sub_id(s), os.path.join(src_root, s+".edf")) for s in subjects
    ]

    # Create a multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        # Use tqdm progress bar to monitor processing status
        with tqdm(total=len(Inputs), desc="Processing MNC-FHC Dataset") as pbar:
            for args in Inputs:
                # Asynchronously apply processing function with callback to update progress bar
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING MNC-FHC DATASET', '='*30)

    # Source directory containing raw EDF files for MNC-FHC cohort
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/fhc/"
    # Destination directory to save preprocessed data
    dst_root = r"./data/MNC-FHC/"
    # Remove existing destination directory to ensure clean preprocessing output
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target sampling rate for signal resampling during preprocessing
    resample_rate = 100

    # List of subject IDs to exclude from processing (empty by default)
    SUB_REMOVE = []

    # Dictionary defining channel mappings to extract from EDF files
    # Keys are channel labels; values are tuples of channel names in EDF
    channel_id = {
        'F3': ('F3',),
        'F4': ('F4',),
        'C3': ('C3',),
        'C4': ('C4',),
        'O1': ('O1',),
        'O2': ('O2',),
        'E1': ('E1',),
        'E2': ('E2',),

        # Chin EMG channels combined as a tuple of tuples
        'Chin': (('lchin','cchin'),),
    }

    # Execute preprocessing with 100 parallel worker processes
    run(100)
