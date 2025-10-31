# -*- coding: utf-8 -*-
"""
STAGES_1.py

This script preprocesses the STAGES polysomnography (PSG) dataset for the LPSGM project. 
It processes raw PSG recordings and their corresponding annotations to extract continuous 
30-second epochs of multi-channel EEG/EOG/EMG signals along with their sleep stage labels.

Main functionalities:
- Load raw EDF signal files and CSV annotation files for each subject
- Convert annotation timestamps to datetime and handle day rollovers
- Extract 30-second epochs of signals aligned with sleep stage labels
- Concatenate epochs per channel and apply preprocessing including resampling
- Split processed signals and annotations into continuous segments without gaps
- Save the processed data in the specified destination directory
- Support parallel processing of multiple subjects for efficiency

This preprocessing is essential for preparing the STAGES dataset for downstream 
sleep staging and mental disorder diagnosis tasks using the LPSGM model.
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
    Process a single subject's PSG recording and annotations to extract preprocessed epochs.

    Args:
        sub_id (str): Subject identifier corresponding to the filename prefix

    Returns:
        None
    """
    # Construct file paths for EDF signal and CSV annotation files
    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}.csv")

    # Load signals and recording start time from EDF file
    start_time, sig_dict = load_sig(sig_path, channel_id)    # returns start_time and sig_dict
    
    # Read annotation CSV file, only loading 'Start Time' and 'Event' columns
    labels = pd.read_csv(ano_path, usecols=['Start Time', 'Event'])

    # Convert 'Start Time' strings to datetime objects, setting date to match recording start
    labels['Start Time'] = pd.to_datetime(labels['Start Time'], format='%H:%M:%S').apply(
        lambda dt: dt.replace(year=start_time.year, month=start_time.month, day=start_time.day))

    # Correct for day rollover in annotations by adding one day if timestamp decreases
    for i in range(1, len(labels)):
        if labels.loc[i, 'Start Time'] < labels.loc[i - 1, 'Start Time']:
            labels.loc[i, 'Start Time'] += pd.Timedelta(days=1)

    # Filter annotations to include only recognized sleep stage events
    labels = labels[labels['Event'].isin(SleepStages.keys())]

    # Define start and end times for each 30-second epoch
    labels['start'] = labels['Start Time']
    labels['end'] = labels['Start Time'] + pd.Timedelta(seconds=30)

    # Check if signal data is available
    if not sig_dict:
        print(f"Empty signal data for {sub_id}")
        return
    
    # Obtain sample rate from the first channel (assumed consistent across channels)
    sample_rate = list(sig_dict.values())[0]['sample_rate']
    
    # Initialize dictionary to hold epochs for each channel
    channel_epochs = {ch_name: [] for ch_name in sig_dict.keys()}
    AnoEpoch = []  # List to store sleep stage labels per epoch
    StartIndex, EndIndex = [], []  # Lists to store sample indices of epoch boundaries
    
    # Iterate over each annotation row to extract corresponding signal epochs
    for i, row in labels.iterrows():
        # Calculate start and end times in seconds relative to recording start
        start_seconds = (row['start'] - start_time).total_seconds()
        end_seconds = (row['end'] - start_time).total_seconds()

        # Convert times to sample indices
        start_index, end_index = int(start_seconds * sample_rate), int(end_seconds * sample_rate)
        ano_epoch = SleepStages[row['Event']]  # Map event string to integer label

        StartIndex.append(start_index)
        EndIndex.append(end_index)
        AnoEpoch.append(ano_epoch)
        
        # Extract signal epoch for each channel and append to list
        for ch_name, ch_data in sig_dict.items():
            sig_epoch = ch_data['data'][start_index : end_index]
            channel_epochs[ch_name].append(sig_epoch)

    # Check if any epochs were extracted
    if len(AnoEpoch) == 0:
        print(f"Empty data for {sub_id}")
        return
    
    # Concatenate all epochs for each channel into a single continuous array
    concatenated_sig_dict = {}
    for ch_name in sig_dict.keys():
        concatenated_sig_dict[ch_name] = {
            'data': np.concatenate(channel_epochs[ch_name], axis=0),
            'sample_rate': sample_rate
        }
    
    # Apply preprocessing steps such as filtering and resampling to target sample rate
    sig_dict_processed = pre_process(concatenated_sig_dict, resample_rate)

    # Convert annotation list to numpy array of int64 labels
    ano = np.array(AnoEpoch, dtype=np.int64)
    # Ensure annotation length matches processed signal length of first channel
    first_ch_processed = list(sig_dict_processed.values())[0]
    ano = ano[:first_ch_processed.shape[0]]

    # Split signals and annotations into continuous segments without gaps
    sig_list, ano_list = [], []
    last_idx = 0
    for i in range(1, len(StartIndex)):
        # Detect discontinuity between epochs by checking if current start != previous end
        if StartIndex[i] != EndIndex[i-1]:
            sig_segment = {}
            for ch_name, sig_data in sig_dict_processed.items():
                sig_segment[ch_name] = sig_data[last_idx : i]
            sig_list.append(sig_segment)
            ano_list.append(ano[last_idx : i])
            last_idx = i
    # Append the final segment after the loop
    if last_idx < len(ano):
        sig_segment = {}
        for ch_name, sig_data in sig_dict_processed.items():
            sig_segment[ch_name] = sig_data[last_idx:]
        sig_list.append(sig_segment)
        ano_list.append(ano[last_idx:])

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save the processed signal and annotation segments to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to safely process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording, typically (sub_id,)

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
        num_processes (int): Number of parallel worker processes to use

    Returns:
        None
    """
    # Identify subject IDs by listing CSV files and removing excluded subjects
    subjects = set([os.path.splitext(f)[0] for f in os.listdir(src_root) if f.endswith(".csv")]) - set(SUB_REMOVE)

    # Create a multiprocessing pool and process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects)):
            pass
   

def test():
    """
    Placeholder test function.

    Returns:
        None
    """
    pass


if __name__ == "__main__":

    # Channel ID mappings for signal extraction; supports multiple naming conventions per channel
    # Each key corresponds to a canonical channel name, values are tuples of possible aliases
    channel_id = {
        'F3': ('F3M2', 'EEG_F3-A2', 'EEG_F3-A1', 'F3-M2', ('F3','M2'), ('F3','A2')),
        'F4': ('F4M1', 'EEG_F4-A1', 'EEG_F4-A2', 'F4-M1', ('F4','M1'), ('F4','A1')),
        'C3': ('C3M2', 'EEG_C3-A2', 'EEG_C3-A1', 'C3-M2', ('C3','M2'), ('C3','A2')),
        'C4': ('C4M1', 'EEG_C4-A1', 'EEG_C4-A2', 'C4-M1', ('C4','M1'), ('C4','A1')),
        'O1': ('O1M2', 'EEG_O1-A2', 'EEG_O1-A1', 'O1-M2', ('O1','M2'), ('O1','A2')),
        'O2': ('O2M1', 'EEG_O2-A1', 'EEG_O2-A2', 'O2-M1', ('O2','M1'), ('O2','A1')),
        'E1': ('E1M2', 'EOG_LOC-A2', 'EOG1', 'L-EOG', 'LOC', 'E1_(LEOG)', ('E1','M2'), ('E1','A2')),
        'E2': ('E2M2', 'EOG_ROC-A2', 'EOG2', 'R-EOG', 'ROC', 'E2_(REOG)', 'EOG_ROC-A1', ('E2','M1'), ('E2','A1')),
        'Chin': ('EMG_Chin', 'CHIN', 'Chin'),
    }

    # Target resampling rate for all signals after preprocessing
    resample_rate = 100

    # Mapping of sleep stage annotation strings to integer labels
    SleepStages = {' Wake': 0, ' Stage1': 1, ' Stage2': 2, ' Stage3': 3, ' REM': 4}


    ### 1. BOGN Dataset Preprocessing ###
    print('='*30, 'PREPROCESSING STAGES-BOGN DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/BOGN/"
    dst_root = r"./data/STAGES-BOGN/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)
    # List of subjects to exclude from preprocessing
    SUB_REMOVE = ['BOGN00043', ]
    # Run preprocessing with 100 parallel processes
    run(100)
    # Check formatting of saved data for consistency
    formatting_check(dst_root)


    ### 2. STNF Dataset Preprocessing ###
    print('='*30, 'PREPROCESSING STAGES-STNF DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/STNF/"
    dst_root = r"./data/STAGES-STNF/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = ["STNF00373", ]
    run(100)
    formatting_check(dst_root)
