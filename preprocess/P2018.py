# -*- coding: utf-8 -*-
"""
P2018 Dataset Preprocessing Module for LPSGM Project

This module handles the preprocessing of the P2018 polysomnography (PSG) dataset,
which includes loading raw PSG signals and sleep stage annotations, resampling signals,
label processing, and saving the preprocessed data in a standardized format compatible
with the LPSGM pipeline.

Key functionalities:
- Load raw PSG signals and arousal annotations from .mat files
- Organize signals into a dictionary with channel-wise data and sampling rates
- Resample signals to a target sampling rate
- Convert multi-class sleep stage labels into a unified label array using majority voting per epoch
- Remove epochs with unknown labels
- Save processed signals and labels for downstream model training and evaluation
- Support multiprocessing for efficient dataset preprocessing

This preprocessing step is critical for preparing the P2018 dataset to be used in the
Large Polysomnography Model (LPSGM) for sleep staging and mental disorder diagnosis.
"""

import scipy.io as scio
import hdf5storage
import numpy as np
import os
import shutil
from scipy.stats import mode
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def process_recording(sub_id):
    """
    Process a single subject's PSG recording and corresponding sleep stage annotations.

    This function loads raw PSG signals and annotation files, organizes the data into a
    channel-wise dictionary, resamples signals, processes sleep stage labels into epoch-wise
    labels using majority voting, removes unknown labels, and saves the processed data.

    Args:
        sub_id (str): Subject identifier corresponding to a folder and file names.
    """
    # Construct file paths for raw signals and arousal annotations
    sig_path = os.path.join(src_root, sub_id, f"{sub_id}.mat")
    ano_path = os.path.join(src_root, sub_id, f"{sub_id}-arousal.mat")

    # Load raw PSG signal data from .mat file
    mat_data = scio.loadmat(sig_path)
    sig_raw = mat_data['val']
    
    # Initialize signal dictionary with channel names as keys
    # Each entry contains sample rate and corresponding signal data array
    sig_dict = {}
    for ch_name, ch_idx in channel_id.items():
        sig_dict[ch_name] = {
            'sample_rate': sample_rate,
            'data': sig_raw[ch_idx]
        }
    
    # Load sleep stage annotation data from arousal .mat file
    label_data_all = hdf5storage.loadmat(ano_path)['data']
    sleep_stages = label_data_all[0]['sleep_stages']
    wake = sleep_stages[0]['wake']
    nonrem1 = sleep_stages[0]['nonrem1']
    nonrem2 = sleep_stages[0]['nonrem2']
    nonrem3 = sleep_stages[0]['nonrem3']
    rem = sleep_stages[0]['rem']
    undefined = sleep_stages[0]['undefined']

    # Concatenate sleep stage labels along columns to form (N, 6) array
    label = np.concatenate([wake, nonrem1, nonrem2, nonrem3, rem, undefined], 1)
    # Map multi-class one-hot labels to integer labels using matrix multiplication
    label = label @ np.array([0, 1, 2, 3, 4, 9])

    # Resample signals to target sampling rate and perform preprocessing
    sig_dict = pre_process(sig_dict, resample_rate)
    
    # Determine total number of epochs from the first channel's data length
    # All channels should have the same number of epochs after preprocessing
    first_channel = list(sig_dict.keys())[0]
    EpochN = sig_dict[first_channel].shape[0]

    # Reshape label array to (EpochN, samples_per_epoch) for majority voting
    # Each epoch corresponds to 30 seconds of data at original sample rate
    ano = np.reshape(label[:EpochN * sample_rate * 30], (EpochN, sample_rate * 30))
    # Apply majority voting along each epoch to determine final epoch label
    ano = mode(ano, axis=1)[0]

    # Remove epochs with unknown labels and corresponding signal data
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)
    
    # List of channel names to maintain consistent order when saving
    channel_names = list(channel_id.keys())
    # Save processed signals and labels to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)
        

def single_process(*args):
    """
    Wrapper function for processing a single recording with exception handling.

    Args:
        *args: Arguments to be passed to process_recording function, typically (sub_id,)
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
    """
    # Get set of subject IDs by listing source directory and excluding removed subjects
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing P2018 Dataset"):
            pass


def test():
    """
    Sequentially process all subjects without multiprocessing for debugging purposes.
    """
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING P2018 DATASET', '='*30)

    # Define source directory containing raw P2018 dataset files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/P2018/training/"
    # Define destination directory to save preprocessed data
    dst_root = r"./data/P2018/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True) 

    # Original sampling rate of raw signals
    sample_rate = 200
    # Target sampling rate after resampling
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of channel names to their indices in raw signal array
    channel_id = {
        'F3': 0,  # F3-M2
        'F4': 1,  # F4-M1
        'C3': 2,  # C3-M2
        'C4': 3,  # C4-M1
        'O1': 4,  # O1-M2
        'O2': 5,  # O2-M1
        'E1': 6,  # E1-M2
        'Chin': 7 # Chin1-Chin2
    }

    # Start multiprocessing preprocessing with 100 parallel workers
    run(100)
    
    # Perform formatting check on the saved preprocessed data
    formatting_check(dst_root)

    # Uncomment below to run sequential test processing
    # test()
