# -*- coding: utf-8 -*-
"""
NCHSDB Dataset Preprocessing Module

This module handles the preprocessing of the NCHSDB (National Center for Health Statistics Database) polysomnography dataset 
for the LPSGM project. It includes functions to load annotations, segment continuous signals based on sleep stage transitions, 
process individual recordings by extracting and preprocessing epochs, and manage multiprocessing for efficient dataset processing.

Key functionalities:
- Load and map sleep stage annotations to numerical labels
- Segment continuous PSG signals into epochs aligned with annotations
- Preprocess signals including resampling and filtering
- Save processed data for downstream sleep staging and mental disorder diagnosis tasks

The module supports flexible channel configurations and is designed to integrate with the overall LPSGM preprocessing pipeline.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and parse sleep stage annotations from a given annotation file.

    Args:
        ano_path (str): Path to the annotation file.

    Returns:
        tuple: 
            sleep_stages (np.ndarray): Array of sleep stage labels mapped to integers.
            start_seconds (list): List of start times (in seconds) for each annotated segment.
            end_seconds (list): List of end times (in seconds) for each annotated segment.
    """
    # Mapping of sleep stage descriptions to integer labels
    stage_dict = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}

    sleep_stages, start_seconds, end_seconds = [], [], []
    with open(ano_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split('\t')
        stage, onset, duration = line[0], float(line[1]), float(line[2])
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])
            start_seconds.append(onset)
            end_seconds.append(onset + duration)
    
    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages, start_seconds, end_seconds


def continue_split(sig_dict, ano, start, end):
    """
    Split continuous signals and annotations into segments separated by discontinuities in annotation timing.

    Args:
        sig_dict (dict): Dictionary of signal arrays per channel, each of shape (N, 3000).
        ano (np.ndarray): Array of annotation labels corresponding to the signal epochs.
        start (list): List of start times (seconds) for each annotation segment.
        end (list): List of end times (seconds) for each annotation segment.

    Returns:
        tuple:
            sig_list (list): List of dictionaries, each containing segmented signal data per channel.
            ano_list (list): List of annotation arrays corresponding to each segmented signal.
    """
    sig_list, ano_list = [], []

    last = 0
    # Iterate through annotation segments to detect discontinuities in timing
    for i in range(1, len(start)):
        if start[i] != end[i-1]:
            # Segment signals and annotations at discontinuity points
            sig_segment = {}
            for ch_name, sig_data in sig_dict.items():
                sig_segment[ch_name] = sig_data[last:i]  # Extract segment (N, 3000)
            sig_list.append(sig_segment)
            ano_list.append(ano[last:i])
            last = i
    # Append remaining segment after last discontinuity
    if last < len(ano):
        sig_segment = {}
        for ch_name, sig_data in sig_dict.items():
            sig_segment[ch_name] = sig_data[last:]  # Extract remaining segment
        sig_list.append(sig_segment)
        ano_list.append(ano[last:])
    
    return sig_list, ano_list

    
def process_recording(sub_id):
    """
    Process a single subject's PSG recording: load signals and annotations, extract epochs, preprocess signals, 
    split continuous data into segments, and save processed data.

    Args:
        sub_id (str): Subject identifier corresponding to the recording file names.
    """
    # Construct file paths for EDF signal and annotation files
    edf_path = os.path.join(src_root, sub_id + '.edf')
    annot_path = os.path.join(src_root, sub_id + '.annot')

    # Load raw signals and start time from EDF file
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Load sleep stage annotations and their timing
    ano, start_seconds, end_seconds = load_ano(annot_path)

    # Retrieve sample rate from the first available channel's data
    sample_rate = next(iter(sig_dict.values()))['sample_rate']
    
    # Convert annotation start and end times from seconds to sample indices
    start_index = [round(start*sample_rate) for start in start_seconds]
    end_index = [round(end*sample_rate) for end in end_seconds]

    # Extract signal epochs for each channel based on annotation timing
    sig_dict_epoched = {}
    for ch_name, ch_data in sig_dict.items():
        sig_data = ch_data['data']
        SigEpoch = []
        for start, end in zip(start_index, end_index):
            SigEpoch.append(sig_data[start:end])
        # Concatenate all epochs along the time axis
        sig_dict_epoched[ch_name] = {
            'sample_rate': ch_data['sample_rate'],
            'data': np.concatenate(SigEpoch, axis=0)
        }

    # Preprocess signals (e.g., filtering, resampling) to target resample_rate
    sig_dict_processed = pre_process(sig_dict_epoched, resample_rate)

    # Determine the number of epochs from the first processed channel's data
    first_ch_data = next(iter(sig_dict_processed.values()))
    ano = ano[:first_ch_data.shape[0]]

    # Split continuous signals and annotations into segments separated by timing discontinuities
    sig_list, ano_list = continue_split(sig_dict_processed, ano, start_seconds, end_seconds)

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save processed segments and annotations to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording function (expects sub_id as first argument).
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run the preprocessing pipeline on all subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # Get unique subject IDs by listing files and removing extensions, excluding subjects to remove
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing NCHSDB Dataset"):
            pass


def test():
    """
    Placeholder for test function.
    """
    pass



if __name__ == "__main__":
    print('='*30, 'PREPROCESSING NCHSDB DATASET', '='*30)

    # Define source directory containing raw EDF and annotation files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/nchsdb/sleep_data/"
    # Define destination directory for saving processed data
    dst_root = r"./data/NCHSDB/"
    # Remove existing destination directory and its contents if present
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target sampling rate for resampling signals
    resample_rate = 100

    # List of subject IDs to exclude due to data issues (e.g., NaN values)
    SUB_REMOVE = ['12334_19465', '1795_24829']    # Subjects with invalid data

    # Mapping of desired channel names to possible channel name variants in raw data
    channel_id = {
        'F3': ('EEG F3-M2',  ('EEG F3','EEG M2'), ('F3','EEG M2'), ('F3','M2')),     # Frontal EEG left
        'F4': ('EEG F4-M1',  'EEG F4-M2', ('EEG F4','EEG M1'), ('F4','EEG M1'), ('F4','M1')),    # Frontal EEG right
        'C3': ('EEG C3-M2',  ('EEG C3','EEG M2'), ('C3','EEG M2'), ('C3','M2')), # Central EEG left
        'C4': ('EEG C4-M1',  'EEG C4-M2', ('EEG C4','EEG M1'), ('C4','EEG M1'), ('C4','M1')),    # Central EEG right
        'O1': ('EEG O1-M2',  ('EEG O1','EEG M2'), ('O1','EEG M2'), ('O1','M2')), # Occipital EEG left
        'O2': ('EEG O2-M1',  ('EEG O2','EEG M1'), ('O2','EEG M1'), ('O2','M1')), # Occipital EEG right
        'E1': ('EOG LOC-M2', 'EEG LOC-M2', ('EEG E1','EEG M2'), ('LOC','EEG M2'), ('LOC','M2')),  # Left EOG
        'E2': ('EOG ROC-M1', 'EEG ROC-M1', 'EEG ROC-M2', ('EEG E2','EEG M1'), ('ROC','EEG M1'), ('ROC','M1')),    # Right EOG

        'Chin': ('EEG Chin1-Chin2', 'EMG CHIN1-CHIN2', 'EMG Chin1-Chin2', 'EMG Chin2-Chin1', ('EEG Chin1','EEG Chin2'), ('Chin1','Chin2'), 'EEG Chin1-Chin3', 'EEG Chin3-Chin2', 'EMG CHIN1-CHIN3', 'EMG Chin1-Chin3', 'EMG Chin3-Chin2'),  # Chin EMG
    }

    # Execute preprocessing with specified number of parallel processes
    run(100)

    # Perform formatting checks on the processed data directory
    formatting_check(dst_root)

    # test()
