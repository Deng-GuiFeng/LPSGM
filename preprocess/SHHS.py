# -*- coding: utf-8 -*-
import numpy as np
import os
from bs4 import BeautifulSoup
from multiprocessing import Pool
import shutil
from tqdm import tqdm

from utils import *

"""
SHHS.py

This module handles preprocessing of the Sleep Heart Health Study (SHHS) polysomnography (PSG) dataset for the LPSGM project.
It loads raw PSG signals and corresponding annotations, performs signal preprocessing and synchronization,
removes unwanted wake epochs at the start and end of recordings, filters out unknown sleep stage labels,
and saves the cleaned data for downstream sleep staging and mental disorder diagnosis tasks.

Key functionalities:
- Parsing XML annotation files to extract sleep stages
- Loading and preprocessing multi-channel PSG signals with resampling
- Synchronizing signal epochs with annotation epochs
- Removing excessive wake periods at sequence boundaries
- Multiprocessing support for efficient dataset processing
- Handling two SHHS dataset versions: SHHS-1 and SHHS-2

This preprocessing step is critical to ensure data quality and consistency for training and evaluation of the LPSGM model.
"""

def load_ano(ano_path):
    """
    Load sleep stage annotations from a Profusion XML file.

    Args:
        ano_path (str): Path to the annotation XML file.

    Returns:
        np.ndarray: Array of integer sleep stage labels for each epoch.
                    Mapping: '0'->0 (Wake), '1'->1 (N1), '2'->2 (N2), '3'->3 (N3),
                             '4'->3 (N3 alternative), '5'->4 (REM), unknown stages mapped to 9.
    """
    # Mapping from annotation text labels to integer sleep stages
    stage_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4}

    # Parse XML and extract all SleepStage elements
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')
    # Convert text labels to integers using the mapping, unknown labels mapped to 9
    sleep_stages = [stage_dict.get(sleep_stage.get_text(), 9) for sleep_stage in sleep_stages]

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id):
    """
    Process a single subject's PSG recording and annotations.

    This includes loading signals and annotations, preprocessing signals,
    synchronizing epochs, removing wake epochs at start/end, filtering unknown labels,
    and saving the cleaned data.

    Args:
        sub_id (str): Subject identifier corresponding to file names.
    """
    # Construct file paths for signal and annotation files
    sig_path = os.path.join(sig_root, f"{sub_id}.edf")
    ano_path = os.path.join(ano_root, f"{sub_id}-profusion.xml")

    # Load raw signals and start time, selecting specified channels
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and dict of channel signals
    # Load sleep stage annotations
    ano = load_ano(ano_path)   

    # Preprocess signals: filtering, resampling, normalization, etc.
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed signals
    
    # Verify that all channels have the same number of epochs and match annotation length
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != ano.shape[0]:
        # Warn if signal and annotation epoch counts differ
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {epoch_counts[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        # Truncate to minimal epoch count to ensure alignment
        epochN = min(epoch_counts[0], ano.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        ano = ano[:epochN]
    
    # Remove wake epochs at the start and end of the recording to reduce bias
    sig_dict, ano = remove_wake_start_end(sig_dict, ano)

    # If no valid data remains after wake removal, skip this recording
    if sig_dict is None or ano is None:
        print(f"Warning: {sub_id} has no valid data after removing wake periods")
        return

    # Remove epochs with unknown sleep stage labels (label 9)
    sig_list, ano_list = rm_unknown_label(sig_dict, ano)

    # Extract channel names to maintain consistent order for saving
    channel_names = list(channel_id.keys())
    # Save the cleaned and processed data to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)
    

def single_process(*args):
    """
    Wrapper function to safely process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording (expects sub_id as first argument).
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Log any errors encountered during processing without stopping the entire pipeline
        print(f"Error processing {args[0]}: {e}")

 
def run(num_processes):
    """
    Run preprocessing on all subjects in the dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel worker processes to use.
    """
    # Collect subject IDs by listing signal files and removing any excluded subjects
    subjects = set([f.split('.')[0] for f in os.listdir(sig_root)]) - set(SUB_REMOVE)

    # Use a multiprocessing pool to process subjects in parallel with progress bar
    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing SHHS Dataset"):
            pass


def test():
    """
    Placeholder for test function.
    """
    pass


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING SHHS DATASET', '='*30)

    # Define channel mappings for signal extraction: keys are target channel names,
    # values are tuples of possible channel name variants in raw data
    channel_id = {
        'C3': ('EEG', ),
        'C4': ('EEG(sec)', 'EEG2', 'EEG sec', 'EEG 2', 'EEG(SEC)'),
        'E1': ('EOG(L)', ),
        'E2': ('EOG(R)', ),

        'Chin': ('EMG',),
    }
    
    
    ### SHHS-1 ###
    # Set paths for SHHS-1 raw signals and annotations
    sig_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/files/polysomnography/edfs/shhs1/"
    ano_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/files/polysomnography/annotations-events-profusion/shhs1/"
    dst_root = r"./data/SHHS-1/"
    # Remove existing processed data directory to avoid conflicts
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subjects to exclude from processing
    SUB_REMOVE = [] 

    # Run preprocessing with 200 parallel processes
    run(200)

    # Perform formatting checks on saved data to ensure consistency
    formatting_check(dst_root)


    ### SHHS-2 ###
    # Set paths for SHHS-2 raw signals and annotations
    sig_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/files/polysomnography/edfs/shhs2/"
    ano_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/files/polysomnography/annotations-events-profusion/shhs2/"
    dst_root = r"./data/SHHS-2/"
    # Remove existing processed data directory to avoid conflicts
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for signals (Hz)
    resample_rate = 100

    # List of subjects to exclude from processing
    SUB_REMOVE = []

    # Run preprocessing with 200 parallel processes
    run(200)

    # Perform formatting checks on saved data to ensure consistency
    formatting_check(dst_root)
