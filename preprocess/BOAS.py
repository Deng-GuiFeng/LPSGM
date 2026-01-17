# -*- coding: utf-8 -*-
"""
BOAS.py

This module handles the preprocessing of the BOAS polysomnography (PSG) dataset for the LPSGM project. 
It loads raw PSG signals and their corresponding sleep stage annotations, performs necessary preprocessing 
such as resampling and label adjustment, removes unknown labels, and saves the processed data in a standardized format. 

Key functionalities include:
- Parsing XML annotation files to extract sleep stages and remapping specific stage labels.
- Loading and preprocessing multi-channel PSG signals with configurable channel mappings.
- Synchronizing signal epochs and annotation lengths, handling discrepancies by truncation.
- Parallel processing of multiple recordings to accelerate dataset preparation.
- Integration with utility functions for signal loading, preprocessing, unknown label removal, and data saving.

This preprocessing step is critical for preparing the BOAS dataset for downstream sleep staging and mental disorder diagnosis tasks.
"""

import os
import numpy as np
import warnings
import shutil
from multiprocessing import Pool

from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


def load_ano(ano_path):
    """
    Load annotation file in TSV format.
    
    TSV columns:
        onset, duration, begsample, endsample, offset, ai_hb
    """
    # 定义字段名称与索引的映射
    COLUMNS = {
        'onset': 0,
        'duration': 1,
        'begsample': 2,
        'endsample': 3,
        'offset': 4,
        'ai_hb': 5,  # annotation label (majority vote)
    }
    
    with open(ano_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过表头，读取所有数据行（过滤空行）
    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    
    # 解析数据
    data = {col: [] for col in COLUMNS}
    for line in data_lines:
        items = line.split('\t')
        for col, idx in COLUMNS.items():
            data[col].append(int(items[idx]))
    
    # 转换字段，begsample 从 1-indexed (Matlab) 转换为 0-indexed (Python)
    onset = np.array(data['onset'], dtype=np.int32)
    duration = np.array(data['duration'], dtype=np.int32)
    # begsample = [x - 1 for x in data['begsample']]  # 列表，用于切片索引
    # endsample = data['endsample']  # 列表，保持不变因为 Python 切片左闭右开
    # offset = np.array(data['offset'], dtype=np.int32)
    majority = np.array(data['ai_hb'], dtype=np.int32)
    
    return onset, duration, majority


def process_recording(sub_id):
    """
    Process a single PSG recording: load signals and annotations, preprocess signals,
    synchronize epochs, remove unknown labels, and save the processed data.

    Args:
        sub_id (str): Subject identifier.

    Returns:
        None
    """

    edf_path = os.path.join(src_root, sub_id, "eeg", f"{sub_id}_task-Sleep_acq-psg_eeg.edf")   
    # edf_path = os.path.join(src_root, sub_id, "eeg", f"{sub_id}_task-Sleep_acq-headband_eeg.edf")   
    tsv_path = os.path.join(src_root, sub_id, "eeg", f"{sub_id}_task-Sleep_acq-psg_events.tsv")
    
    # Load raw signals and their start time using predefined channel configuration
    start_time, sig_dict = load_sig(edf_path, channel_id)  # returns start_time and dictionary of channel signals
    # Load and preprocess sleep stage annotations
    _, _, ano = load_ano(tsv_path)    

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
    Run the preprocessing pipeline on all subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # Get unique subject IDs by listing files and removing extensions, excluding subjects to remove
    subjects = set([f"sub-{i}" for i in range(1, 128+1)]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing BOAS Dataset"):
            pass


def test():
    """
    Placeholder for potential testing functions.

    Returns:
        None
    """
    # Get unique subject IDs by listing files and removing extensions, excluding subjects to remove
    subjects = set([f"sub-{i}" for i in range(1, 128+1)]) - set(SUB_REMOVE)

    for sub_id in subjects:
        print(f"Processing {sub_id} for test...")
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING BOAS DATASET', '='*30)

    # Define source and destination directories for raw and processed data
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/BOAS/"
    dst_root = r"./data/BOAS/"
    # Remove existing processed data directory to start fresh
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Dictionary defining channel mappings: keys are channel names used in processing,
    # values are tuples of tuples specifying raw channel pairs for referencing
    channel_id = {
        'F3': ('PSG_F3',), 
        'F4': ('PSG_F4',), 
        'C3': ('PSG_C3',), 
        'C4': ('PSG_C4',), 
        'O1': ('PSG_O1',), 
        'O2': ('PSG_O2',), 
        'E1': ('PSG_EOG', 'PSG_EOGL'), 
        'E2': ('PSG_EOGR',), 
        
        'Chin': ('PSG_EMG',),

        # 'HB_1': ('HB_1',),
        # 'HB_2': ('HB_2',),
    }

    # Execute preprocessing with 100 parallel worker processes
    # run(16)
    test()

    # Perform formatting checks on the processed dataset to ensure consistency
    formatting_check(dst_root)

    # Placeholder for testing function call
    # test()
