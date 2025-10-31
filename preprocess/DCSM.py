# -*- coding: utf-8 -*-
"""
DCSM.py

This module handles the preprocessing of the DCSM (Dataset for Clinical Sleep Monitoring) dataset 
for the LPSGM project. It includes functions to load annotation files, remove extended wake periods, 
process individual PSG recordings, and manage multiprocessing for efficient dataset preparation.

Key functionalities:
- Load and convert sleep stage annotations into numerical labels.
- Remove prolonged wake segments to focus on sleep periods.
- Preprocess PSG signals including resampling and channel selection.
- Save processed data segments for downstream model training.
- Support parallel processing of multiple subjects.

This preprocessing is critical for preparing the DCSM dataset to be compatible with the LPSGM model's 
requirements for sleep staging and mental disorder diagnosis.
"""

import numpy as np
import os
from scipy.interpolate import interp1d
from multiprocessing import Pool
import shutil
from tqdm import tqdm

from utils import *


def load_ano(ano_path):
    """
    Load and convert sleep stage annotations from a file into a numpy array of integer labels.

    Args:
        ano_path (str): Path to the annotation file containing sleep stages and durations.

    Returns:
        np.ndarray: Array of sleep stage labels with shape (number_of_epochs,), where each label is an integer.
                    Sleep stages are mapped as: W=0, N1=1, N2=2, N3=3, REM=4.
    """
    stage_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

    sleep_stages = []
    with open(ano_path, 'r')as f:
        lines = f.readlines()

    for line in lines:
        s = line.replace('\n','').split(',')
        # duration in seconds for the stage segment
        duration = int(s[1])
        stage = s[2]
        # Extend the sleep stages list by repeating the stage label for each 30-second epoch in the duration
        sleep_stages.extend([stage_dict[stage]]*int(duration/30))
    sleep_stages = np.array(sleep_stages, dtype=np.int32)

    return sleep_stages


def remove_wake(sig_dict, ano):
    """
    Remove long continuous wake periods from the signal and annotation data to isolate sleep sessions.

    Args:
        sig_dict (dict): Dictionary mapping channel names to signal arrays of shape (EpochN, 3000).
        ano (np.ndarray): Array of sleep stage labels with shape (EpochN,), values in {0,1,2,3,4}.

    Returns:
        tuple:
            - list of dicts: Each dict contains channel data for a continuous sleep segment.
            - list of np.ndarray: Corresponding annotation arrays for each sleep segment.
    """
    EpochN = ano.shape[0]
    W_C = []  # List to store wake segments with onset, duration, and end epoch indices
    w_duration, w_onset = 0, 0

    # Identify continuous wake segments
    for i, ano_i in enumerate(ano):
        if ano_i == 0:  # Wake stage
            if w_duration == 0:
                w_onset = i
            w_duration += 1
        else:
            # Check if the wake segment is long enough to be considered for removal
            if (w_onset == 0 and w_duration > 60) or (w_onset != 0 and w_duration > 120):
                W_C.append({'onset': w_onset, 'duration': w_duration, 'end': w_onset + w_duration})
            w_duration, w_onset = 0, 0
    # Handle wake segment at the end of the recording
    if w_duration > 60:
        W_C.append({'onset': w_onset, 'duration': w_duration, 'end': w_onset + w_duration})

    Session_Start, Session_End = [] , []

    # Define sleep session boundaries based on wake segments
    for wc in W_C:
        onset = wc['onset']
        end = wc['end']

        if onset == 0:
            # If wake segment is at the start, session starts 60 epochs after wake end
            Session_Start.append(end - 60)
        elif end == EpochN:
            # If wake segment is at the end, session ends 60 epochs before wake onset
            Session_End.append(onset + 60)
        else:
            # For wake segments in the middle, define session end and start around wake boundaries
            Session_End.append(onset + 60)
            Session_Start.append(end - 60)

    # Handle case where first wake segment is not at start
    if len(W_C) > 0 and W_C[0]['onset'] != 0:
        Session_Start.insert(0, 0)
    elif len(W_C) == 0:
        # No wake periods found, use entire recording as one session
        Session_Start.append(0)
        Session_End.append(EpochN)

    # Ensure session start and end lists are balanced
    if len(Session_Start) > len(Session_End):
        Session_End.append(EpochN)

    sig_list = []
    # Extract annotation segments corresponding to sleep sessions
    ano_list = [ano[s:e] for s, e in zip(Session_Start, Session_End)]

    # Extract signal segments for each sleep session and channel
    for s, e in zip(Session_Start, Session_End):
        sig_segment = {}
        for ch_name, sig_data in sig_dict.items():
            sig_segment[ch_name] = sig_data[s:e]
        sig_list.append(sig_segment)

    return sig_list, ano_list


def process_recording(sub_id):
    """
    Process a single subject's PSG recording: load signals and annotations, preprocess, remove wake periods, and save.

    Args:
        sub_id (str): Subject identifier corresponding to a folder in the source dataset directory.

    Returns:
        None
    """
    # Construct file paths for EDF signal and hypnogram annotation files
    edf_path = os.path.join(src_root, sub_id, 'psg.edf')
    ids_path = os.path.join(src_root, sub_id, 'hypnogram.ids')

    # Load raw signals and start time from EDF file
    start_time, sig_dict = load_sig(edf_path, channel_id)
    # Load sleep stage annotations as numerical labels
    ano = load_ano(ids_path)

    # Preprocess signals: resample and apply any channel-specific processing
    sig_dict = pre_process(sig_dict, resample_rate)

    # Verify that all channels have the same number of epochs as annotations
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Use the minimal number of epochs to synchronize signals and annotations
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]

    # Remove long wake periods and segment signals and annotations accordingly
    sig_list, ano_list = remove_wake(sig_dict, ano)

    channel_names = list(channel_id.keys())
    # Save processed signal segments and annotations for the subject
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording, typically (sub_id,).

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
    # Get list of subject folders excluding those in SUB_REMOVE
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing DCSM Dataset"):
            pass


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING DCSM DATASET', '='*30)

    # Define source and destination directories for raw and processed data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/DCSM/"
    dst_root = r"./data/DCSM/"
    # Remove existing processed data directory to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subjects to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel logical names to EDF channel identifiers
    channel_id = {
        'F3': ('F3-M2',),
        'F4': ('F4-M1',),
        'C3': ('C3-M2',),
        'C4': ('C4-M1',),
        'O1': ('O1-M2',),
        'O2': ('O2-M1',),
        'E1': ('E1-M2',),
        'E2': ('E2-M2',),

        'Chin': ('CHIN',),
    }

    # Execute preprocessing with 100 parallel processes
    run(100)

    # Perform formatting checks on the processed dataset directory
    formatting_check(dst_root)

    # test()
