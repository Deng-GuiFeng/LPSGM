# -*- coding: utf-8 -*-
"""
HMC.py

This module handles the preprocessing of the HMC (Hospital Medical Center) polysomnography dataset 
for the LPSGM project. It includes functions to load annotation files, segment continuous signals 
based on sleep stage annotations, preprocess signals (including resampling), and save the processed 
data in a structured format. The preprocessing pipeline supports parallel processing to efficiently 
handle large datasets.

Key functionalities:
- Loading sleep stage annotations and mapping them to numerical labels
- Segmenting continuous PSG signals into epochs aligned with annotations
- Preprocessing signals including resampling to a target frequency
- Saving processed signals and annotations for downstream model training
- Parallel processing support for scalability

This file is a critical component in preparing raw HMC PSG data for sleep staging and mental disorder 
diagnosis using the LPSGM model.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def load_ano(txt_path):
    """
    Load sleep stage annotations from a text file and convert them to numerical labels.

    Args:
        txt_path (str): Path to the annotation text file.

    Returns:
        tuple:
            np.ndarray: Array of sleep stage labels as integers.
            list: List of annotation start times in seconds.
            list: List of annotation end times in seconds.
    """
    # Mapping of sleep stage descriptions to integer labels
    stage_dict = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}

    sleep_stages, start_seconds, end_seconds = [], [], []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Parse each annotation line, skipping header
    for line in lines[1:]:
        line = line.strip().split(',')
        stage, onset, duration = line[4].strip(), line[2].strip(), line[3].strip()
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])  # Append numerical sleep stage label
            start_seconds.append(int(onset))        # Annotation start time in seconds
            end_seconds.append(int(onset) + int(duration))  # Annotation end time in seconds

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages, start_seconds, end_seconds


def continue_split(sig_dict, ano, start, end):
    """
    Split continuous signals and annotations into segments where there are discontinuities 
    in the annotation time intervals.

    Args:
        sig_dict (dict): Dictionary of signal arrays keyed by channel name.
        ano (np.ndarray): Array of annotation labels corresponding to the signal.
        start (list): List of annotation start times in seconds.
        end (list): List of annotation end times in seconds.

    Returns:
        tuple:
            list: List of segmented signal dictionaries.
            list: List of segmented annotation arrays.
    """
    sig_list, ano_list = [], []

    last = 0
    # Iterate through annotation intervals to detect discontinuities
    for i in range(1, len(start)):
        if start[i] != end[i-1]:
            # Segment signals and annotations at discontinuity
            sig_segment = {}
            for ch_name, sig_data in sig_dict.items():
                sig_segment[ch_name] = sig_data[last:i]
            sig_list.append(sig_segment)
            ano_list.append(ano[last:i])
            last = i
    # Append remaining segment after last discontinuity
    if last < len(ano):
        sig_segment = {}
        for ch_name, sig_data in sig_dict.items():
            sig_segment[ch_name] = sig_data[last:]
        sig_list.append(sig_segment)
        ano_list.append(ano[last:])
    
    return sig_list, ano_list


def process_recording(sub_id):
    """
    Process a single subject's PSG recording and corresponding annotations:
    - Load raw signals and annotations
    - Extract epochs based on annotation intervals
    - Preprocess signals including resampling
    - Split signals and annotations at discontinuities
    - Save processed data to destination directory

    Args:
        sub_id (str): Subject identifier.
    """
    # Construct file paths for signal and annotation files
    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}_sleepscoring.txt")
    
    # Load raw signals and their start time
    start_time, sig_dict = load_sig(sig_path, channel_id)
    # Load sleep stage annotations and their time intervals
    ano, start_seconds, end_seconds = load_ano(ano_path)
    
    # Retrieve sample rate from the first channel (assumed uniform across channels)
    sample_rate = list(sig_dict.values())[0]['sample_rate']
    
    # Extract signal epochs corresponding to annotation intervals for each channel
    processed_sig_dict = {}
    for ch_name, ch_data in sig_dict.items():
        sig = ch_data['data']
        SigEpoch = []
        for start, end in zip(start_seconds, end_seconds):
            # Slice signal data for the epoch based on sample rate and annotation times
            SigEpoch.append(sig[int(start*sample_rate):int(end*sample_rate)])
        # Concatenate all epochs into a continuous array
        concatenated_sig = np.concatenate(SigEpoch, axis=0)
        processed_sig_dict[ch_name] = {
            'sample_rate': sample_rate,
            'data': concatenated_sig
        }

    # Preprocess signals (e.g., filtering, resampling) to target sample rate
    sig_dict = pre_process(processed_sig_dict, resample_rate)
    
    # Split signals and annotations at any discontinuities in annotation intervals
    sig_list, ano_list = continue_split(sig_dict, ano, start_seconds, end_seconds)

    channel_names = list(channel_id.keys())
    # Save processed signal segments and corresponding annotations
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to be passed to process_recording function.
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes.
    """
    # Identify subject IDs by filtering filenames without underscores and excluding removed subjects
    subjects = set([f.split('.')[0] for f in os.listdir(src_root) if '_' not in f]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing HMC Dataset"):
            pass


def test():
    """
    Sequentially process all subjects for testing purposes without multiprocessing.
    """
    subjects = set([f.split('.')[0] for f in os.listdir(src_root) if '_' not in f]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING HMC DATASET', '='*30)

    # Define source directory containing raw HMC recordings and annotations
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/HMC/recordings/"
    # Define destination directory for saving preprocessed data
    dst_root = r"./data/HMC/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling frequency in Hz for all signals
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel short names to their corresponding full channel names in raw data
    channel_id = {
        'F4': ('EEG F4-M1',), 
        'C3': ('EEG C3-M2',), 
        'C4': ('EEG C4-M1',), 
        'O2': ('EEG O2-M1',), 
        'E1': ('EOG E1-M2',), 
        'E2': ('EOG E2-M2',),

        'Chin': ('EMG chin',),
    }

    # Execute preprocessing with specified number of parallel processes
    run(100)
    
    # Perform formatting checks on the preprocessed data directory
    formatting_check(dst_root)

    # Uncomment to run sequential test processing
    # test()
