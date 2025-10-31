# -*- coding: utf-8 -*-
"""
ABC.py

This module handles the preprocessing of the ABC polysomnography (PSG) dataset for the LPSGM project. 
It loads raw PSG signals and their corresponding sleep stage annotations, performs necessary preprocessing 
such as resampling and label adjustment, removes unknown labels, and saves the processed data in a standardized format. 

Key functionalities include:
- Parsing XML annotation files to extract sleep stages and remapping specific stage labels.
- Loading and preprocessing multi-channel PSG signals with configurable channel mappings.
- Synchronizing signal epochs and annotation lengths, handling discrepancies by truncation.
- Parallel processing of multiple recordings to accelerate dataset preparation.
- Integration with utility functions for signal loading, preprocessing, unknown label removal, and data saving.

This preprocessing step is critical for preparing the ABC dataset for downstream sleep staging and mental disorder diagnosis tasks.
"""

import os
import numpy as np
from bs4 import BeautifulSoup
import shutil
from multiprocessing import Pool

from utils import *


def load_ano(ano_path):
    """
    Load and process sleep stage annotations from an XML file.

    Args:
        ano_path (str): Path to the annotation XML file.

    Returns:
        np.ndarray: Array of integer sleep stage labels with remapped stages.
    """
    # Parse the XML file to extract all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        # Remap sleep stage 4 to 3 and stage 5 to 4 to conform to 5-class sleep staging
        if ss == 4:
            ss = 3
        elif ss == 5:
            ss = 4
        sleep_stages[i] = ss
    # Convert list of stages to numpy array of integers
    ano = np.array(sleep_stages, dtype=np.int32)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single PSG recording: load signals and annotations, preprocess signals,
    synchronize epochs, remove unknown labels, and save the processed data.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the EDF signal file.
        ano_path (str): Path to the annotation XML file.

    Returns:
        None
    """
    # Load raw signals and their start time using predefined channel configuration
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dictionary of channel signals
    # Load and preprocess sleep stage annotations
    ano = load_ano(ano_path)    

    # Resample and preprocess signals to the target sampling rate
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signal dictionary

    # Verify that all channels have the same number of epochs
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ, truncate to minimal length
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], ano.shape[0])
        # Truncate all channel signals to the minimal epoch count
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        # Truncate annotations accordingly
        ano = ano[:epochN]

    # Remove epochs with unknown or invalid labels from signals and annotations
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    channel_names = list(channel_id.keys())
    # Save the cleaned and preprocessed data to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (sub_id, sig_path, ano_path).

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing without stopping the entire pipeline
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Main function to orchestrate parallel preprocessing of all recordings in the ABC dataset.

    Args:
        num_processes (int): Number of parallel worker processes to use.

    Returns:
        None
    """
    Inputs = []
    # Iterate over dataset groups (subdirectories) containing EDF and annotation files
    for group in os.listdir(edf_root):
        edf_group = os.path.join(edf_root, group)
        xml_group = os.path.join(xml_root, group)

        # Iterate over each subject file in the group
        for sub_id in os.listdir(edf_group):
            sub_id = sub_id.split(".")[0]  # Remove file extension to get subject ID
            if sub_id in SUB_REMOVE:
                # Skip subjects marked for removal
                continue
            edf_path = os.path.join(edf_group, sub_id+".edf")
            xml_path = os.path.join(xml_group, sub_id+"-profusion.xml")

            Inputs.append((sub_id, edf_path, xml_path))

    # Use multiprocessing pool to process recordings in parallel
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing ABC Dataset") as pbar:
            for args in Inputs:
                # Apply processing asynchronously and update progress bar upon completion
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Placeholder for potential testing functions.

    Returns:
        None
    """
    pass


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING ABC DATASET', '='*30)

    # Define source and destination directories for raw and processed data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/abc/"
    dst_root = r"./data/ABC/"
    # Remove existing processed data directory to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel mappings: keys are channel names used in processing,
    # values are tuples of tuples specifying raw channel pairs for referencing
    channel_id = {
        'F3': (('F3','M2'),), 
        'F4': (('F4','M1'),), 
        'C3': (('C3','M2'),), 
        'C4': (('C4','M1'),), 
        'O1': (('O1','M2'),), 
        'O2': (('O2','M1'),), 
        'E1': (('E1','M2'),), 
        'E2': (('E2','M1'),), 
        
        'Chin': (('Chin1','Chin2'),),
    }

    # Paths to raw EDF signal files and XML annotation files
    edf_root = os.path.join(src_root, "polysomnography/edfs/")
    xml_root = os.path.join(src_root, "polysomnography/annotations-events-profusion/")

    # Execute preprocessing with 100 parallel worker processes
    run(100)

    # Perform formatting checks on the processed dataset to ensure consistency
    formatting_check(dst_root)

    # Placeholder for testing function call
    # test()
