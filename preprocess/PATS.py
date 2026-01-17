# -*- coding: utf-8 -*-
"""
PATS Dataset Preprocessing Module

This module handles the preprocessing of the PATS polysomnography (PSG) dataset for the LPSGM project.
It includes functions to load annotation files, process individual PSG recordings by extracting and 
preprocessing signal epochs based on sleep stage annotations, and save the processed data for downstream 
sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Parsing sleep stage annotations and converting them to numerical labels
- Loading PSG signals from EDF files with specified channel configurations
- Aligning and segmenting signals according to annotated sleep stages with time adjustments
- Resampling and preprocessing signals for model input
- Multiprocessing support for efficient dataset processing
- Dataset-specific configurations such as channel mappings and subject exclusions

This preprocessing step is crucial for preparing the PATS dataset to be compatible with the LPSGM model's 
input requirements.
"""

import numpy as np
import os
import shutil
from multiprocessing import Pool
from datetime import datetime, timedelta   
from utils import *


def load_ano(ano_path):
    """
    Load and parse sleep stage annotations from a given annotation file.

    Args:
        ano_path (str): Path to the annotation file.

    Returns:
        tuple: 
            - sleep_stages (np.ndarray): Array of integer sleep stage labels.
            - start_times (list of datetime): List of stage start times.
            - end_times (list of datetime): List of stage end times.
    """
    # Mapping of sleep stage strings to integer labels
    stage_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}

    sleep_stages, start_times, end_times = [], [], []
    with open(ano_path, 'r')as f:
        lines = f.readlines()

    for line in lines:
        line = line.split('\t')
        stage, start, end = line[0], line[3], line[4]
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])
            # Parse start and end times as datetime objects (time only)
            start_times.append(datetime.strptime(start, "%H:%M:%S"))
            end_times.append(datetime.strptime(end, "%H:%M:%S"))
    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages, start_times, end_times

    
def process_recording(sub_id):
    """
    Process a single subject's PSG recording by loading signals and annotations,
    aligning epochs, preprocessing signals, and saving the processed data.

    Args:
        sub_id (str): Subject identifier corresponding to file names.
    """
    # Construct file paths for EDF signal and annotation files
    edf_path = os.path.join(src_root, sub_id + '.edf')
    annot_path = os.path.join(src_root, sub_id + '.annot')

    # Load raw signals and recording start time from EDF file
    start_time, sig_dict = load_sig(edf_path, channel_id)

    # Load sleep stage annotations and corresponding start/end times
    ano, start_times, end_times = load_ano(annot_path)

    # Replace annotation times with full datetime objects using recording start date
    start_times = [st.replace(year=start_time.year, month=start_time.month, day=start_time.day) for st in start_times]
    end_times = [et.replace(year=start_time.year, month=start_time.month, day=start_time.day) for et in end_times]

    # Adjust times crossing midnight or day boundaries
    if start_times[0] > start_time and start_times[0].hour > 12 and start_times[0].hour < 12:
        start_times[0] += timedelta(days=1)
    if end_times[0] < start_times[0]:
        end_times[0] += timedelta(days=1)
    for i in range(1, len(start_times)):
        if start_times[i] < start_times[i-1]:
            start_times[i] += timedelta(days=1)
        if end_times[i] < end_times[i-1]:
            end_times[i] += timedelta(days=1)
    
    # Get sample rate assuming uniform sampling across channels
    sample_rate = list(sig_dict.values())[0]['sample_rate']
        
    # Initialize dictionary to hold segmented signal epochs per channel
    SigEpoch = {ch_name: [] for ch_name in sig_dict.keys()}
    StartIndex, EndIndex = [], []

    # Convert annotation times to sample indices and segment signals accordingly
    for st, et in zip(start_times, end_times):
        start_seconds = (st - start_time).total_seconds()
        end_seconds = (et - start_time).total_seconds()

        start_index, end_index = int(start_seconds * sample_rate), int(end_seconds * sample_rate)
        
        for ch_name, ch_data in sig_dict.items():
            sig_epoch = ch_data['data'][start_index : end_index]
            SigEpoch[ch_name].append(sig_epoch)
        
        StartIndex.append(start_index)
        EndIndex.append(end_index)

    # Concatenate segmented epochs for each channel into continuous arrays
    concatenated_sig_dict = {}
    for ch_name in sig_dict.keys():
        concatenated_sig_dict[ch_name] = {
            'data': np.concatenate(SigEpoch[ch_name], axis=0),
            'sample_rate': sample_rate
        }
    
    # Preprocess concatenated signals (e.g., filtering, resampling)
    processed_sig_dict = pre_process(concatenated_sig_dict, resample_rate)

    # Calculate indices adjusted for resampling and epoch length (30 seconds)
    resample_ratio = resample_rate / sample_rate
    ProcessedStartIndex = [int(idx * resample_ratio // (30 * resample_rate)) for idx in StartIndex]
    ProcessedEndIndex = [int(idx * resample_ratio // (30 * resample_rate)) for idx in EndIndex]

    sig_list, ano_list = [], []
    last_idx = 0

    # Segment processed signals and annotations into continuous blocks where indices are contiguous
    for i in range(1, len(ProcessedStartIndex)):
        if ProcessedStartIndex[i] != ProcessedEndIndex[i-1]:
            sig_segment = {}
            for ch_name, sig_data in processed_sig_dict.items():
                sig_segment[ch_name] = sig_data[last_idx : i]
            sig_list.append(sig_segment)
            ano_list.append(ano[last_idx : i])
            last_idx = i
    # Append remaining segments if any
    if last_idx < len(ano):
        sig_segment = {}
        for ch_name, sig_data in processed_sig_dict.items():
            sig_segment[ch_name] = sig_data[last_idx:]
        sig_list.append(sig_segment)
        ano_list.append(ano[last_idx:])
    
    # check len consistency and fix 
    for i in range(len(sig_list)):
        if sig_list[i][list(sig_list[i].keys())[0]].shape[0] != ano_list[i].shape[0]:
            min_len = min(sig_list[i][list(sig_list[i].keys())[0]].shape[0], ano_list[i].shape[0])
            for ch_name in sig_list[i]:
                sig_list[i][ch_name] = sig_list[i][ch_name][:min_len]
            ano_list[i] = ano_list[i][:min_len]

    # Extract channel names for saving
    channel_names = list(channel_id.keys())
    # Save processed signal and annotation segments to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (expects sub_id).
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run multiprocessing preprocessing on all subjects in the source directory,
    excluding specified subjects.

    Args:
        num_processes (int): Number of parallel worker processes.
    """
    # Collect unique subject IDs by file names, excluding those in SUB_REMOVE
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # Use multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing PATS Dataset"):
            pass


def test():
    """
    Sequentially process all subjects for testing or debugging purposes.
    """
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING PATS DATASET', '='*30)

    # Source directory containing raw PATS PSG EDF and annotation files
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/pats/polysomnography/"
    # Destination directory for saving processed data
    dst_root = r"./data/PATS/"
    # Remove existing processed data directory if present
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for all signals
    resample_rate = 100

    # List of subject IDs to exclude from processing
    SUB_REMOVE = []

    # Mapping of standardized channel names to raw channel identifiers in EDF files
    channel_id = {
        'F3': ('F3_M2',),
        'F4': ('F4_M1',),
        'C3': ('C3_M2',),
        'C4': ('C4_M1',),
        'O1': ('O1_M2',),
        'O2': ('O2_M1',),
        'E1': ('LOC',),
        'E2': ('ROC',),

        'Chin': ('EMG',)
    }

    # Execute preprocessing with specified number of parallel processes
    run(100)
    # test()

    # Perform formatting checks on the processed dataset
    formatting_check(dst_root)

    # Uncomment to run sequential test processing
    # test()
