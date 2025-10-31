# -*- coding: utf-8 -*-
"""
utils.py

Preprocessing utilities for PSG data standardization in the LPSGM project.

This module provides functions for file handling, data integrity checks, signal preprocessing,
and data formatting specific to polysomnography (PSG) datasets. It supports operations such as
filtering, resampling, label cleaning, and saving processed data segments. These utilities
facilitate consistent data preparation for the Large Polysomnography Model (LPSGM), which
performs sleep staging and mental disorder diagnosis using multi-channel PSG recordings.
"""

import os
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from mne.io import read_raw_edf


def find_files_with_suffix(root_dir, suffix):
    """
    Recursively find all files under root_dir that end with the given suffix.

    Args:
        root_dir (str): Root directory to start searching from
        suffix (str): File suffix to match (e.g., '.npz')

    Returns:
        list: List of full file paths matching the suffix
    """
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files


def check_nan_and_inf(data):
    """
    Check if the input data contains any NaN or infinite values.

    Args:
        data (array-like): Input data to check

    Returns:
        tuple: (has_nan (bool), has_inf (bool)) indicating presence of NaN or Inf
    """
    arr = np.array(data)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    return has_nan, has_inf


def formatting_check(dataset_root):
    """
    Validate the formatting and integrity of PSG dataset files in the given root directory.

    This function checks for empty subject directories, verifies channel data presence,
    validates shapes of epoch data and hypnograms, detects NaN/Inf values, and ensures
    class labels are within the expected range [0-4]. It also computes class distribution
    statistics and channel count summaries.

    Args:
        dataset_root (str): Root directory containing subject subdirectories with PSG data
    """
    CH_COUNT = {}
    ClassNum = [0]*5
    SUM = 0
    subjects = os.listdir(dataset_root)
    print(f"Total Subjects: {len(subjects)}")

    for sub_id in tqdm(os.listdir(dataset_root)):
        sub_dir = os.path.join(dataset_root, sub_id)
        
        if len(os.listdir(sub_dir)) == 0:
            print(sub_id, "empty error")
            # shutil.rmtree(sub_dir)
            continue

        for seq_id in os.listdir(sub_dir):
            seq_path = os.path.join(sub_dir, seq_id)

            npz = np.load(seq_path)
            ano = npz['Hypnogram']
            
            # Extract channel keys excluding 'Hypnogram'
            channel_keys = [key for key in npz.keys() if key != 'Hypnogram']
            ChN = len(channel_keys)
            
            # Validate shape and check for NaN/Inf in first channel data
            if channel_keys:
                first_ch_data = npz[channel_keys[0]]
                EpochN1, TN = first_ch_data.shape
                
                has_nan, has_inf = check_nan_and_inf(first_ch_data)
                if has_nan or has_inf:
                    print(sub_id, seq_id, "nan or inf error")
            else:
                print(sub_id, seq_id, "no channel data found")
                continue
            
            # Verify epoch count consistency between channel data and hypnogram
            EpochN2 = ano.shape[0]

            if any([EpochN1 != EpochN2, TN != 3000]):
                print(sub_id, seq_id, "shape error", f"EpochN: {EpochN1}, TN: {TN}, ano_shape: {ano.shape}")

            SUM += EpochN2
            for i in range(5):
                ClassNum[i] += np.sum(ano == i)
            
            # Check that all classes are within the expected set {0,1,2,3,4}
            ClassSet = set(ano.tolist())
            if not ClassSet.issubset({0, 1, 2, 3, 4}):
                print(sub_id, "classes error", ClassSet)

            # Record channel count frequency
            CH_COUNT[ChN] = CH_COUNT.get(ChN, 0) + 1

    # Calculate and print class distribution ratios
    ClassRatio = [round(100 * ClassNum[c] / SUM, 1) for c in range(5)]
    print(f"Epochs Count: {SUM}, {ClassNum}")
    print(f"Epochs Ratio: {ClassRatio}")

    # Compute class rebalancing weights inversely proportional to class frequencies
    r_CalssNum = [1 / ClassNum[c] for c in range(5)]
    ClassWeight = r_CalssNum / np.sum(r_CalssNum)
    print(f"Classes Rebalance Weights: {ClassWeight}")

    print("Channel Count:", CH_COUNT)


def info(*mats):
    """
    Print shape, data type, minimum, and maximum values for each input matrix.

    Args:
        *mats: Variable length argument list of numpy arrays
    """
    for mat in mats:
        print(mat.shape, mat.dtype, mat.min(), mat.max())


def pre_process(sig_dict, resample_rate=100, norch=False):
    """
    Preprocess multi-channel PSG signals with filtering, optional notch filtering,
    resampling, normalization, and epoch segmentation.

    Args:
        sig_dict (dict): Dictionary of channel signals with structure
                         {channel_name: {'data': np.array, 'sample_rate': int}}
        resample_rate (int, optional): Target sampling rate for resampling. Defaults to 100.
        norch (bool, optional): Whether to apply 50Hz notch filter to remove powerline noise. Defaults to False.

    Returns:
        dict: Dictionary of processed signals segmented into epochs with shape (EpochN, EpochL)
              keyed by channel name
    """
    processed_dict = {}
    
    for ch_name, ch_data in sig_dict.items():
        sig = ch_data['data']
        sample_rate = ch_data['sample_rate']
        
        TN = len(sig)
        
        # Apply bandpass or highpass filtering based on channel type
        if ch_name in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2']:
            # 4th order Butterworth bandpass filter between 0.3 and 35 Hz
            b, a = signal.butter(N=4, Wn=[0.3 * 2 / sample_rate, 35 * 2 / sample_rate], btype='bandpass')
            sig = signal.filtfilt(b, a, sig)
        elif ch_name == 'Chin':
            # 4th order Butterworth highpass filter at 10 Hz
            b, a = signal.butter(N=4, Wn=10 * 2 / sample_rate, btype='highpass')
            sig = signal.filtfilt(b, a, sig)
        
        if norch:
            # Apply 50 Hz notch filter to reduce powerline interference
            b_notch, a_notch = signal.iirnotch(w0=50, Q=20, fs=sample_rate)
            sig = signal.filtfilt(b_notch, a_notch, sig)

        # Resample signal if target sampling rate differs from original
        if resample_rate != sample_rate:
            scaled_TN = round(resample_rate / sample_rate * TN)
            sig_r = interp1d(np.linspace(0, TN - 1, TN), sig, kind='linear')(
                np.linspace(0, TN - 1, scaled_TN))
        else:
            scaled_TN = TN
            sig_r = sig
            
        # Normalize signal using Z-score normalization
        sig_r = (sig_r - np.mean(sig_r)) / np.std(sig_r)

        # Segment signal into 30-second epochs
        EpochL = 30 * resample_rate
        EpochN = scaled_TN // EpochL
        sig_r = np.reshape(sig_r[:EpochN * EpochL], (EpochN, EpochL))
        
        processed_dict[ch_name] = sig_r

    return processed_dict


def rm_unknown_label(sig_dict, ano):
    """
    Remove segments containing unknown or invalid labels from the signal and annotation data.

    Args:
        sig_dict (dict): Dictionary of channel signals with shape (N, 3000) per channel
        ano (np.ndarray): 1D array of annotations/labels with length N

    Returns:
        tuple: (list of signal dictionaries, list of annotation arrays) where each list element
               corresponds to a continuous segment without unknown labels
    """
    # Identify indices where labels are outside the valid range [0,4]
    indices = np.where((ano > 4) | (ano < 0))[0]

    sig_list, ano_list = [], []
    start = 0

    # Split signals and annotations into segments excluding unknown label indices
    for i in indices:
        if i > start:
            sig_segment = {}
            for ch_name, sig_data in sig_dict.items():
                sig_segment[ch_name] = sig_data[start:i]
            sig_list.append(sig_segment)
            ano_list.append(ano[start:i])
        start = i + 1

    # Add the last segment after the final unknown label index
    if start < len(ano):
        sig_segment = {}
        for ch_name, sig_data in sig_dict.items():
            sig_segment[ch_name] = sig_data[start:]
        sig_list.append(sig_segment)
        ano_list.append(ano[start:])

    return sig_list, ano_list


def save(dst_root, sub_id, sig_list, ano_list, channel_names):
    """
    Save processed signal segments and corresponding annotations to disk in NPZ format.

    Args:
        dst_root (str): Destination root directory for saving processed data
        sub_id (str): Subject identifier used for directory and file naming
        sig_list (list): List of signal dictionaries for each segment
        ano_list (list): List of annotation arrays corresponding to each segment
        channel_names (list): List of channel names to save from each signal dictionary
    """
    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)

    for i, (sig_dict, ano) in enumerate(zip(sig_list, ano_list)):
        seq_path = os.path.join(dst_sub_dir, f"{sub_id}-s{i}.npz")
        ano = ano.astype(np.int32)
        
        # Prepare dictionary for saving with hypnogram and selected channels
        save_data = {'Hypnogram': ano}
        for ch_name in channel_names:
            if ch_name in sig_dict:
                save_data[ch_name] = sig_dict[ch_name].astype(np.float32)
        
        np.savez(seq_path, **save_data)


def get_ch_names(edf_path):
    """
    Retrieve channel names from an EDF file.

    Args:
        edf_path (str): Path to the EDF file

    Returns:
        list: List of channel names in the EDF file
    """
    sig_raw = read_raw_edf(edf_path, verbose=False)
    return sig_raw.ch_names


def load_sig(sig_path, channel_id):
    """
    Load PSG signals from an EDF file according to specified channel mappings.

    Supports differential channel pairs and single channels, selecting the first available
    option per target channel.

    Args:
        sig_path (str): Path to the EDF file
        channel_id (dict): Dictionary mapping target channel names to lists of possible
                           channel options or tuples representing differential pairs

    Returns:
        tuple: (start_time (datetime), sig_dict (dict)) where sig_dict maps target channel names
               to dicts with keys 'sample_rate' and 'data' (numpy array)
    """
    # Collect all possible channel names from channel_id options
    all_possible_channels = set()
    for channel_options in channel_id.values():
        for option in channel_options:
            if isinstance(option, tuple):
                all_possible_channels.update(option)
            else:
                all_possible_channels.add(option)
    
    # Read raw EDF data including all possible channels
    sig_raw = read_raw_edf(sig_path, include=list(all_possible_channels), verbose=False)
    available_channels = set(sig_raw.ch_names)
    sig_data = sig_raw.to_data_frame().to_numpy()  # Shape: (TN, C)
    
    # Estimate sample rate from time column differences
    sample_rate = round(1 / (sig_data[1, 0] - sig_data[0, 0]))
    
    # Extract measurement start time and remove timezone info
    start_time = sig_raw.info['meas_date']
    start_time = start_time.replace(tzinfo=None)
    
    sig_dict = {}
    
    # For each target channel, select the first valid channel option available
    for target_ch, channel_options in channel_id.items():
        channel_data = None
        
        for option in channel_options:
            if isinstance(option, tuple) and len(option) == 2:
                # Differential channel pair (e.g., ('F3', 'M2'))
                ch1, ch2 = option
                if ch1 in available_channels and ch2 in available_channels:
                    ch1_idx = sig_raw.ch_names.index(ch1)
                    ch2_idx = sig_raw.ch_names.index(ch2)
                    # Subtract signals of the two channels, skipping time column at index 0
                    channel_data = sig_data[:, ch1_idx + 1] - sig_data[:, ch2_idx + 1]
                    break
            else:
                # Single channel option (e.g., 'F3-M2')
                if option in available_channels:
                    ch_idx = sig_raw.ch_names.index(option)
                    channel_data = sig_data[:, ch_idx + 1]
                    break
        
        if channel_data is not None:
            sig_dict[target_ch] = {
                'sample_rate': sample_rate,
                'data': channel_data
            }
        else:
            print(f"Warning: Could not find any valid option for channel {target_ch}, existing channels: {get_ch_names(sig_path)}, {sig_path}")

    return start_time, sig_dict


def remove_wake_start_end(sig_dict, ano, duration=30):
    """
    Trim leading and trailing wake periods from the signals and annotations.

    Keeps a buffer of 'duration' epochs before the first non-wake epoch and after the last
    non-wake epoch to preserve context.

    Args:
        sig_dict (dict): Dictionary of signals keyed by channel name, each as a 1D numpy array
        ano (np.ndarray): 1D array of annotations/labels
        duration (int, optional): Number of epochs to keep as buffer before and after non-wake periods. Defaults to 30.

    Returns:
        tuple: (processed_sig_dict (dict), trimmed_ano (np.ndarray)) or (None, None) if no non-wake epochs found
    """
    # Find indices of epochs that are not wake (label != 0)
    non_wake_indices = np.where(ano != 0)[0]
    
    if len(non_wake_indices) == 0:
        return None, None
    
    first_non_wake = non_wake_indices[0]
    last_non_wake = non_wake_indices[-1]
    
    # Define start and end indices with buffer epochs included
    start = max(0, first_non_wake - duration * 2)
    end = min(len(ano), last_non_wake + duration * 2 + 1) 
    
    if start >= end:
        return None, None
    
    # Slice signals and annotations to the defined range
    processed_sig_dict = {}
    for ch_name, sig_data in sig_dict.items():
        processed_sig_dict[ch_name] = sig_data[start:end]
    
    return processed_sig_dict, ano[start:end]
