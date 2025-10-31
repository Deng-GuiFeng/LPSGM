# -*- coding: utf-8 -*-
"""
STAGES_2.py

This script performs preprocessing of the STAGES polysomnography (PSG) datasets for the LPSGM project.
It processes raw PSG recordings and their annotations from multiple STAGES sub-datasets, extracting
and aligning EEG/EOG/EMG signal epochs with corresponding sleep stage labels. The preprocessing includes
signal loading, timestamp synchronization, epoch extraction, signal resampling, and saving the processed
data in a standardized format for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Load raw EDF signals and CSV annotations for each subject
- Synchronize annotation timestamps with signal start times
- Extract signal epochs per channel based on sleep stage events
- Resample and preprocess signals to a uniform sampling rate
- Split continuous signal segments based on annotation discontinuities
- Parallel processing of multiple subjects for efficient dataset preparation
- Support for multiple STAGES sub-datasets with configurable source and destination paths

This file is part of the LPSGM data preprocessing pipeline and is critical for ensuring data quality
and consistency across diverse PSG datasets used for model training and evaluation.
"""

import numpy as np
import os
from tqdm import tqdm
import shutil
import pandas as pd
from multiprocessing import Pool

from utils import *


def process_recording(sub_id):
    """
    Process a single subject's PSG recording and annotations to extract preprocessed signal epochs
    and aligned sleep stage labels.

    Args:
        sub_id (str): Subject identifier corresponding to the filename prefix of EDF and CSV files.

    Returns:
        None: Processed data is saved to disk; function prints warnings on errors or empty data.
    """
    # Construct full file paths for EDF signal and CSV annotation files
    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}.csv")

    # Load signal data and recording start time; sig_dict maps channel names to signal data and metadata
    start_time, sig_dict = load_sig(sig_path, channel_id)    # returns start_time and sig_dict
    
    # Read annotation CSV with relevant columns: start time, duration, and event label
    labels = pd.read_csv(ano_path, usecols=['Start Time', 'Duration (seconds)', 'Event'])

    # Convert 'Start Time' strings to datetime objects, aligning date with signal start_time
    labels['Start Time'] = pd.to_datetime(labels['Start Time'], format='%H:%M:%S').apply(
        lambda dt: dt.replace(year=start_time.year, month=start_time.month, day=start_time.day))

    # Adjust annotation timestamps if the first event occurs before the recording start time (crossing midnight)
    if labels.loc[0, 'Start Time'] < start_time:
        for i in range(len(labels)):
            labels.loc[i, 'Start Time'] += pd.Timedelta(days=1)
    
    # Ensure chronological order by adding a day to any timestamp earlier than its predecessor
    for i in range(1, len(labels)):
        if labels.loc[i, 'Start Time'] < labels.loc[i - 1, 'Start Time']:
            labels.loc[i, 'Start Time'] += pd.Timedelta(days=1)

    # Filter annotations to only include recognized sleep stage events
    labels = labels[labels['Event'].isin(SleepStages.keys())]

    # Define start and end times for each annotation event
    labels['start'] = labels['Start Time']
    labels['end'] = labels['Start Time'] + pd.to_timedelta(labels['Duration (seconds)'], unit='s')

    # Validate signal data availability
    if not sig_dict:
        print(f"Empty signal data for {sub_id}")
        return
    
    # Retrieve sample rate from the first available channel (assumed uniform across channels)
    sample_rate = list(sig_dict.values())[0]['sample_rate']
    
    # Initialize containers for extracted epochs per channel and annotation epochs
    channel_epochs = {ch_name: [] for ch_name in sig_dict.keys()}
    AnoEpoch = []
    StartIndex, EndIndex = [], []
    
    # Iterate over each annotation event to extract corresponding signal epochs and labels
    for i, row in labels.iterrows():
        # Calculate start and end times in seconds relative to recording start
        start_seconds = (row['start'] - start_time).total_seconds()
        end_seconds = (row['end'] - start_time).total_seconds()

        # Convert times to sample indices for slicing signal arrays
        start_index, end_index = int(start_seconds * sample_rate), int(end_seconds * sample_rate)
        ano_epoch = SleepStages[row['Event']]

        StartIndex.append(start_index)
        EndIndex.append(end_index)
        
        # Extract signal epoch for each channel based on calculated indices
        for ch_name, ch_data in sig_dict.items():
            sig_epoch = ch_data['data'][start_index : end_index]
            channel_epochs[ch_name].append(sig_epoch)

        # Extend annotation epoch list by repeating the stage label per 30-second segment
        AnoEpoch.extend([ano_epoch] * int(row['Duration (seconds)'] // 30))

    # Check if any annotation epochs were extracted
    if len(AnoEpoch) == 0:
        print(f"Empty data for {sub_id}")
        return
    
    # Concatenate extracted epochs for each channel into continuous arrays
    concatenated_sig_dict = {}
    for ch_name in sig_dict.keys():
        concatenated_sig_dict[ch_name] = {
            'data': np.concatenate(channel_epochs[ch_name], axis=0),
            'sample_rate': sample_rate
        }
    
    # Apply preprocessing and resampling to the concatenated signals
    sig_dict_processed = pre_process(concatenated_sig_dict, resample_rate)

    # Convert annotation epochs to numpy array of int64
    ano = np.array(AnoEpoch, dtype=np.int64)
    # Truncate annotation array to match length of processed signal data (first channel)
    first_ch_processed = list(sig_dict_processed.values())[0]
    ano = ano[:first_ch_processed.shape[0]]

    # Split signals and annotations into continuous segments based on discontinuities in StartIndex and EndIndex
    sig_list, ano_list = [], []
    last_idx = 0
    for i in range(1, len(StartIndex)):
        # Detect discontinuity where current start index does not match previous end index
        if StartIndex[i] != EndIndex[i-1]:
            sig_segment = {}
            for ch_name, sig_data in sig_dict_processed.items():
                # Slice signal segment from last split index to current index
                sig_segment[ch_name] = sig_data[last_idx : i]
            sig_list.append(sig_segment)
            # Slice corresponding annotation segment
            ano_list.append(ano[last_idx : i])
            last_idx = i
    # Append final segment if remaining data exists
    if last_idx < len(ano):
        sig_segment = {}
        for ch_name, sig_data in sig_dict_processed.items():
            sig_segment[ch_name] = sig_data[last_idx:]
        sig_list.append(sig_segment)
        ano_list.append(ano[last_idx:])
        
    # Retrieve list of channel names for saving
    channel_names = list(channel_id.keys())
    # Save processed signal and annotation segments to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to be passed to process_recording (expects sub_id).

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use.

    Returns:
        None
    """
    # Identify all subject IDs by listing CSV files and removing excluded subjects
    subjects = set([os.path.splitext(f)[0] for f in os.listdir(src_root) if f.endswith(".csv")]) - set(SUB_REMOVE)

    # Create a multiprocessing pool and process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects)):
            pass
   

def test():
    """
    Placeholder test function for future unit tests or debugging.

    Returns:
        None
    """
    pass


if __name__ == "__main__":

    # Dictionary mapping channel names to possible aliases and reference pairs for signal loading
    # This mapping supports flexible channel identification across different recording setups
    # The commented block shows alternative simpler mapping; current mapping includes tuples of aliases
    # and reference electrode pairs for robust channel selection.

    # channel_id = {
    #     'F3M2': 0, 'EEG_F3-A2': 0, 'F3': 0, 'EEG_F3-A1': 0, 'F3-M2': 0, 
    #     'F4M1': 1, 'EEG_F4-A1': 1, 'F4': 1, 'EEG_F4-A2': 1, 'F4-M1': 1, 
    #     'C3M2': 2, 'EEG_C3-A2': 2, 'C3': 2, 'EEG_C3-A1': 2, 'C3-M2': 2, 
    #     'C4M1': 3, 'EEG_C4-A1': 3, 'C4': 3, 'EEG_C4-A2': 3, 'C4-M1': 3, 
    #     'O1M2': 4, 'EEG_O1-A2': 4, 'O1': 4, 'EEG_O1-A1': 4, 'O1-M2': 4, 
    #     'O2M1': 5, 'EEG_O2-A1': 5, 'O2': 5, 'EEG_O2-A2': 5, 'O2-M1': 5,
    #     'E1M2': 6, 'EOG_LOC-A2': 6, 'E1': 6, 'EOG1': 6, 'L-EOG': 6, 'LOC': 6, 'E1_(LEOG)': 6, 
    #     'E2M2': 7, 'EOG_ROC-A2': 7, 'E2': 7, 'EOG2': 7, 'R-EOG': 7, 'EOG_ROC-A1': 7, 'ROC': 7, 'E2_(REOG)': 7, 
    #     'M1': -1, 'A1': -1, 
    #     'M2': -2, 'A2': -2, 
    # }

    channel_id = {
        'F3': ('F3M2', 'EEG_F3-A2', 'EEG_F3-A1', 'F3-M2', ('F3','M2'), ('F3','A2')),
        'F4': ('F4M1', 'EEG_F4-A1', 'EEG_F4-A2', 'F4-M1', ('F4','M1'), ('F4','A1')),
        'C3': ('C3M2', 'EEG_C3-A2', 'EEG_C3-A1', 'C3-M2', ('C3','M2'), ('C3','A2')),
        'C4': ('C4M1', 'EEG_C4-A1', 'EEG_C4-A2', 'C4-M1', ('C4','M1'), ('C4','A1')),
        'O1': ('O1M2', 'EEG_O1-A2', 'EEG_O1-A1', 'O1-M2', ('O1','M2'), ('O1','A2')),
        'O2': ('O2M1', 'EEG_O2-A1', 'EEG_O2-A2', 'O2-M1', ('O2','M1'), ('O2','A1')),
        'E1': ('E1M2', 'EOG_LOC-A2', 'EOG1', 'L-EOG', 'LOC', 'E1_(LEOG)', ('E1','M2'), ('E1','A2'), 'E1'),
        'E2': ('E2M2', 'EOG_ROC-A2', 'EOG2', 'R-EOG', 'ROC', 'E2_(REOG)', 'EOG_ROC-A1', ('E2','M1'), ('E2','A1'), 'E2'),

        'Chin': ('EMG_Chin', ('EMG1','EMG2'), ('EMG_#1','EMG_#2'), ('CHIN1','CHIN2'), 'Chin'),
    }

    # Target resampling rate for all signals after preprocessing
    resample_rate = 100
    # Mapping of sleep stage labels to integer codes
    SleepStages = {' Wake': 0, ' Stage1': 1, ' Stage2': 2, ' Stage3': 3, ' REM': 4}


    ### 3. GSDV ###
    print('='*30, 'PREPROCESSING STAGES-GSDV DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/GSDV/"
    dst_root = r"./data/STAGES-GSDV/"
    # Remove existing destination directory to ensure clean preprocessing output
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    # Run preprocessing with 100 parallel processes
    run(100)
    # Verify formatting and integrity of saved data
    formatting_check(dst_root)


    ### 4. MSTR ###
    print('='*30, 'PREPROCESSING STAGES-MSTR DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/MSTR/"
    dst_root = r"./data/STAGES-MSTR/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 5. GSBB ###
    print('='*30, 'PREPROCESSING STAGES-GSBB DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/GSBB/"
    dst_root = r"./data/STAGES-GSBB/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 6. GSLH ###
    print('='*30, 'PREPROCESSING STAGES-GSLH DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/GSLH/"
    dst_root = r"./data/STAGES-GSLH/"
    shutil.rmtree(dst_root, ignore_errors=True)
    # Exclude subject GSLH00003 from processing due to known issues
    SUB_REMOVE = ["GSLH00003", ]
    run(100)
    formatting_check(dst_root)


    ### 7. GSSA ###
    print('='*30, 'PREPROCESSING STAGES-GSSA DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/GSSA/"
    dst_root = r"./data/STAGES-GSSA/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 8. GSSW ###
    print('='*30, 'PREPROCESSING STAGES-GSSW DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/GSSW/"
    dst_root = r"./data/STAGES-GSSW/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 9. MSMI ###
    print('='*30, 'PREPROCESSING STAGES-MSMI DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/MSMI/"
    dst_root = r"./data/STAGES-MSMI/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 10. MSNF ###
    print('='*30, 'PREPROCESSING STAGES-MSNF DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/MSNF/"
    dst_root = r"./data/STAGES-MSNF/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 11. MSQW ###
    print('='*30, 'PREPROCESSING STAGES-MSQW DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/MSQW/"
    dst_root = r"./data/STAGES-MSQW/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 12. MSTH ###
    print('='*30, 'PREPROCESSING STAGES-MSTH DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/MSTH/"
    dst_root = r"./data/STAGES-MSTH/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)


    ### 13. STLK ###
    print('='*30, 'PREPROCESSING STAGES-STLK DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/STLK/"
    dst_root = r"./data/STAGES-STLK/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []
    run(100)
    formatting_check(dst_root)
