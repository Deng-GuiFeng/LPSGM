# -*- coding: utf-8 -*-
"""
SYSU.py

This module handles the preprocessing of the SYSU dataset for the LPSGM project. 
It processes raw EEG and EOG EDF files along with their corresponding annotations, 
extracting and preparing epoch-based signals for subsequent model training and evaluation.

Key functionalities include:
- Loading and combining EEG and EOG signals
- Epoch segmentation and trimming based on start and end epochs
- Signal preprocessing including resampling
- Annotation conversion to numerical sleep stage labels
- Removal of unknown labels and saving processed data
- Parallel processing support for efficient dataset handling

The SYSU dataset contains both healthy subjects and depressed patients, and this module 
facilitates unified preprocessing for both groups.

"""

import numpy as np
import os
import shutil
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from utils import *


def process_recording(sub_id, eeg_edf_path, eog_edf_path, ano_xlsx_path, start_epoch, end_epoch):
    """
    Process a single subject's EEG and EOG recordings along with annotations.

    Args:
        sub_id (str): Subject identifier
        eeg_edf_path (str): Path to the EEG EDF file
        eog_edf_path (str): Path to the EOG EDF file
        ano_xlsx_path (str): Path to the annotation Excel file
        start_epoch (int): Starting epoch index for processing
        end_epoch (int): Ending epoch index for processing

    Returns:
        None
    """
    # Load EEG signal and its start time from EDF file using predefined EEG channel IDs
    start_time_eeg, sig_dict_eeg = load_sig(eeg_edf_path, eeg_channel_id)
    # Load EOG signal and its start time from EDF file using predefined EOG channel IDs
    start_time_eog, sig_dict_eog = load_sig(eog_edf_path, eog_channel_id)
    
    # Combine EEG and EOG signals into a single dictionary for unified processing
    combined_sig_dict = {}
    combined_sig_dict.update(sig_dict_eeg)
    combined_sig_dict.update(sig_dict_eog)
    
    # Retrieve sample rate from the first channel's metadata (assumed consistent across channels)
    sample_rate = list(combined_sig_dict.values())[0]['sample_rate']
    
    # Load annotation data from Excel file (assumed to be a single column of sleep stage labels)
    ano = pd.read_excel(ano_xlsx_path, header=None).iloc[:, 0].tolist()
    
    # Calculate epoch length in samples (30 seconds per epoch)
    first_channel_data = list(combined_sig_dict.values())[0]['data']
    EpochTN = sample_rate * 30
    # Total number of complete epochs available in the data
    EpochN = len(first_channel_data) // EpochTN
    
    # Trim signals to contain only complete epochs and slice according to start and end epochs
    for ch_name in combined_sig_dict:
        channel_data = combined_sig_dict[ch_name]['data']
        # Trim data to contain only complete epochs (discard incomplete trailing samples)
        trimmed_data = channel_data[:EpochN * EpochTN]
        # Reshape data into epochs (rows: epochs, columns: samples per epoch)
        epoch_data = trimmed_data.reshape(EpochN, EpochTN)
        # Slice epochs based on provided start and end indices (fixes mismatch in original data)
        sliced_data = epoch_data[start_epoch + 1:end_epoch]
        # Flatten sliced epochs back to 1D array for preprocessing
        combined_sig_dict[ch_name]['data'] = sliced_data.reshape(-1)
    
    # Slice annotation list to correspond with the selected epochs
    ano = ano[start_epoch:end_epoch - 1]
    
    # Apply preprocessing steps such as filtering and resampling to all channels
    processed_sig_dict = pre_process(combined_sig_dict, resample_rate)
    
    # Convert annotation labels to numerical sleep stage indices using predefined mapping
    ano = np.array([sleepstage[a] for a in ano], dtype=np.int64)
    
    # Remove unknown labels and prepare signals and annotations for saving
    sig_list, ano_list = rm_unknown_label(processed_sig_dict, ano)
    
    # Define channel names combining EEG and EOG keys for saving
    channel_names = list(eeg_channel_id.keys()) + list(eog_channel_id.keys())
    # Save the processed signals and annotations to the destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with error handling.

    Args:
        *args: Arguments to be passed to process_recording

    Returns:
        None
    """
    try:
        process_recording(*args)
    except Exception as e:
        # Print error message if processing fails for a subject
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Main function to run preprocessing on the entire SYSU dataset using multiprocessing.

    Args:
        num_processes (int): Number of parallel processes to use

    Returns:
        None
    """
    Inputs = []

    # Load healthy subjects' start and end epoch information from Excel
    healthy_xlsx = os.path.join(src_root, "健康被试数据-edf", "健康被试-整夜起止epoch统计.xlsx")
    for healthy_idx in range(80):
        sub_id = f"健康被试-{healthy_idx+1}"
        # Extract start and end epochs for the subject from Excel (rows start at index 2)
        (start_epoch, end_epoch) = pd.read_excel(healthy_xlsx, header=None).iloc[healthy_idx + 2, 1:3].tolist()
        # Construct file paths for EEG, EOG, and annotation files for healthy subjects
        eeg_path = os.path.join(src_root, "健康被试数据-edf", "EEG-健康-edf", f"{healthy_idx+1}.edf")
        eog_path = os.path.join(src_root, "健康被试数据-edf", "EOG-健康-edf", f"{healthy_idx+1}.edf")
        ano_path = os.path.join(src_root, "健康被试数据-edf", "标签-健康", f"{healthy_idx+1}.xlsx")

        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    # Load depressed patients' start and end epoch information from Excel
    depressed_xlsx = os.path.join(src_root, "抑郁患者数据-edf", "抑郁-开关灯时间统计.xlsx")
    for depressed_idx in range(24):
        sub_id = f"抑郁患者-{depressed_idx+1}"
        # Extract start and end epochs for the subject from Excel (rows start at index 2)
        (start_epoch, end_epoch) = pd.read_excel(depressed_xlsx, header=None).iloc[depressed_idx + 2, 1:3].tolist()
        # Construct file paths for EEG, EOG, and annotation files for depressed patients
        eeg_path = os.path.join(src_root, "抑郁患者数据-edf", "EEG-抑郁-edf", f"{depressed_idx+1}.edf")
        eog_path = os.path.join(src_root, "抑郁患者数据-edf", "EOG-抑郁-edf", f"{depressed_idx+1}.edf")
        ano_path = os.path.join(src_root, "抑郁患者数据-edf", "标签-抑郁", f"{depressed_idx+1}.xlsx")
        
        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    # Use multiprocessing pool to process all recordings in parallel with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing SYSU Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Test function to process all recordings sequentially without multiprocessing.

    Returns:
        None
    """
    Inputs = []

    # Load healthy subjects' start and end epoch information from Excel
    healthy_xlsx = os.path.join(src_root, "健康被试数据-edf", "健康被试-整夜起止epoch统计.xlsx")
    for healthy_idx in range(80):
        sub_id = f"健康被试-{healthy_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(healthy_xlsx, header=None).iloc[healthy_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "健康被试数据-edf", "EEG-健康-edf", f"{healthy_idx+1}.edf")
        eog_path = os.path.join(src_root, "健康被试数据-edf", "EOG-健康-edf", f"{healthy_idx+1}.edf")
        ano_path = os.path.join(src_root, "健康被试数据-edf", "标签-健康", f"{healthy_idx+1}.xlsx")

        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    # Load depressed patients' start and end epoch information from Excel
    depressed_xlsx = os.path.join(src_root, "抑郁患者数据-edf", "抑郁-开关灯时间统计.xlsx")
    for depressed_idx in range(24):
        sub_id = f"抑郁患者-{depressed_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(depressed_xlsx, header=None).iloc[depressed_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "抑郁患者数据-edf", "EEG-抑郁-edf", f"{depressed_idx+1}.edf")
        eog_path = os.path.join(src_root, "抑郁患者数据-edf", "EOG-抑郁-edf", f"{depressed_idx+1}.edf")
        ano_path = os.path.join(src_root, "抑郁患者数据-edf", "标签-抑郁", f"{depressed_idx+1}.xlsx")
        
        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))

    # Process each recording sequentially for testing purposes
    for args in Inputs:
        process_recording(*args)


if __name__ == "__main__":
    print('='*30, 'PREPROCESSING SYSU DATASET', '='*30)

    # Define source and destination root directories for raw and processed data
    src_root = r"/nvme1/denggf/PSG_datasets/private_datasets/SYSU/"
    dst_root = r"./data/SYSU/"
    # Remove existing processed data directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)

    # Target resampling rate for all signals (Hz)
    resample_rate = 100

    # Mapping of sleep stage labels to numerical indices for model compatibility
    sleepstage = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'U': 9, '?': 9}

    # EEG channel configurations with corresponding bipolar derivations
    eeg_channel_id = {
        'F3': ('F3-M2', 'F3-M1'),
        'F4': ('F4-M1', 'F4-M2'),
        'C3': ('C3-M2', 'C3-M1'),
        'C4': ('C4-M1', 'C4-M2'),
        'O1': ('O1-M2', 'O1-M1'),
        'O2': ('O2-M1', 'O2-M2'),
    }

    # EOG channel configurations with corresponding bipolar derivations
    eog_channel_id = {
        'E1': ('E1-M2', 'E1-M1'),
        'E2': ('E2-M1', 'E2-M2'),
    }
       
    # Run preprocessing using 104 parallel processes for efficiency
    run(104)

    # Perform formatting checks on the processed dataset to ensure data integrity
    formatting_check(dst_root)

    # Uncomment below to run test processing sequentially without multiprocessing
    # test()
