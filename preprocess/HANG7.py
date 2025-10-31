# -*- coding: utf-8 -*-
"""
HANG7 Dataset Preprocessing Module

This module handles the preprocessing of the HANG7 polysomnography dataset for the LPSGM project.
It reads raw EDF signal files and corresponding sleep stage annotations in XML or TXT formats,
performs signal preprocessing including resampling, aligns signals with annotations, removes unknown labels,
and saves the processed data in a structured format.

Key functionalities:
- Parsing sleep stage annotations from XML and TXT files with standardized stage mapping
- Loading and preprocessing multi-channel PSG signals
- Synchronizing signal epochs with annotation epochs, handling discrepancies
- Parallel processing of multiple recordings for efficient dataset preparation
- Integration with project utilities for signal loading, preprocessing, and saving

This preprocessing is critical for preparing the HANG7 dataset for downstream sleep staging and mental disorder diagnosis tasks.
"""

import numpy as np
import os
import warnings
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm

from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


def read_xml(xml_path):
    """
    Parse sleep stage annotations from an XML file and map stages to standardized numeric labels.

    Args:
        xml_path (str): Path to the XML annotation file.

    Returns:
        np.ndarray: Array of sleep stage labels as integers, with unknown stages marked as 9.
    """
    # Extract all SleepStage elements from the XML file
    id_stages = BeautifulSoup(open(xml_path), features="xml").find_all('SleepStage')
    for i in range(len(id_stages)):
        ss = float(id_stages[i].get_text())
        # Map stage 5 to 4 to unify REM stage labeling
        if ss == 5:
            ss = 4
        # Mark stages outside 0-4 range as unknown (9)
        if not(ss >= 0 and ss <= 4):
            ss = 9
        id_stages[i] = ss
    id_stages = np.array(id_stages)
    return id_stages


def read_txt(txt_path):
    """
    Parse sleep stage annotations from a TXT file and map string labels to standardized numeric labels.

    Args:
        txt_path (str): Path to the TXT annotation file.

    Returns:
        np.ndarray: Array of sleep stage labels as integers, with unknown stages marked as 9.
    """
    # Mapping from various string labels to standardized numeric sleep stages
    mapping = {
        'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4,
        'WK': 0, 'NN1': 1, 'NN2': 2, 'NN3': 3, 'REM': 4,
        '0': 0, '1': 1, '2': 2, '3': 3, '5': 4,
        '?' : 9, 'M': 9,
    }
    # Load the annotation strings from the TXT file
    str_stages = np.loadtxt(txt_path, dtype=str)
    # Vectorize the mapping function for efficient conversion
    vectorized_map = np.vectorize(lambda s: mapping[s])
    id_stages = vectorized_map(str_stages)
    return id_stages


def process_recording(sub_id, edf_path, xml_path, txt_path):
    """
    Process a single PSG recording: load signals, read annotations, preprocess signals,
    synchronize epochs, remove unknown labels, and save processed data.

    Args:
        sub_id (str): Subject identifier.
        edf_path (str): Path to the EDF signal file.
        xml_path (str): Path to the XML annotation file.
        txt_path (str): Path to the TXT annotation file.
    """
    # Load raw PSG signals and recording start time for specified channels
    start_time, sig_dict = load_sig(edf_path, channel_id)  # returns start_time and dict of signals per channel

    # Read sleep stage annotations from TXT file (preferred due to XML errors in some subjects)
    ano = read_txt(txt_path)

    # Preprocess signals: resample and apply filtering as defined in utils.pre_process
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signals dictionary

    # Verify that all channels have the same number of epochs and match annotation length
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ, then truncate to minimal length
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove epochs with unknown labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save the processed signals and annotations to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function for processing a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording function.
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing of a recording
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all recordings in the HANG7 dataset using parallel processing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    Inputs = []
    # Iterate over all subject groups and collect input parameters for processing
    for group in groups:
        group_dir = os.path.join(src_root, group)
        for sub_id in os.listdir(group_dir):
            if sub_id in SUB_REMOVE:
                continue
            edf_path = os.path.join(group_dir, sub_id, sub_id + ".edf")
            xml_path = os.path.join(group_dir, sub_id, sub_id + ".edf.XML")
            txt_path = os.path.join(group_dir, sub_id, sub_id + ".txt")
            Inputs.append((sub_id, edf_path, xml_path, txt_path))

    # Use multiprocessing Pool to process recordings in parallel with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing HANG7 Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Test function to sequentially process all recordings without multiprocessing.
    Useful for debugging or verifying processing steps.
    """
    subjects = []
    # Collect all subject input parameters
    for group in groups:
        group_dir = os.path.join(src_root, group)
        for sub_id in os.listdir(group_dir):
            if sub_id in SUB_REMOVE:
                continue
            edf_path = os.path.join(group_dir, sub_id, sub_id + ".edf")
            xml_path = os.path.join(group_dir, sub_id, sub_id + ".edf.XML")
            txt_path = os.path.join(group_dir, sub_id, sub_id + ".txt")
            subjects.append((sub_id, edf_path, xml_path, txt_path))
    
    # Process each recording sequentially
    for (sub_id, edf_path, xml_path, txt_path) in subjects:
        process_recording(sub_id, edf_path, xml_path, txt_path)


if __name__ == '__main__':
    print('='*30, 'PREPROCESSING HANG7 DATASET', '='*30)

    # Define source directory containing raw HANG7 dataset files organized by group
    src_root = r"/nvme1/denggf/PSG_datasets/private_datasets/HQ_collation/"
    # Define destination directory for saving processed data
    dst_root = r"./data/HANG7/"
    # Remove existing processed data directory if present to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Set target resampling rate for signal preprocessing (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Define subject groups within the dataset
    groups = (
        "depression",
        "healthy_control",
        "narcolepsy",
    )

    # Define channel configurations: keys are channel names, values specify bipolar derivations
    channel_id = {
        'F3': (('F3', 'M2'), 'F3-M2'),
        'F4': (('F4', 'M1'), 'F4-M1'),
        'C3': (('C3', 'M2'), 'C3-M2'),
        'C4': (('C4', 'M1'), 'C4-M1'),
        'O1': (('O1', 'M2'), 'O1-M2'),
        'O2': (('O2', 'M1'), 'O2-M1'),
        'E1': (('E1', 'M2'), 'E1-M2'),
        'E2': (('E2', 'M1'), 'E2-M2', 'E2-M1'),

        'Chin': ('Chin 1-Chin 2', ('Chin 1','Chin 2')),
    }

    # Execute preprocessing pipeline with specified number of parallel processes
    run(127)

    # Perform formatting checks on the processed dataset directory
    formatting_check(dst_root)

    # Uncomment to run test processing sequentially
    # test()
