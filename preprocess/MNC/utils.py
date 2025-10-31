# -*- coding: utf-8 -*-
"""
utils.py

Utilities for preprocessing and handling the MNC (Multi-Night Cohort) dataset within the LPSGM project.

This module provides functions to load raw PSG signals from EDF files, preprocess signals with filtering,
resampling, and epoch segmentation, and save processed data in a structured format. It supports flexible
channel selection, including differential channel computation, and ensures signals are standardized for
subsequent modeling tasks such as sleep staging and mental disorder diagnosis.

Key functionalities:
- Loading raw signals with flexible channel mapping
- Signal filtering (bandpass, highpass, notch) tailored to channel types
- Resampling and segmentation into 30-second epochs
- Saving processed data with diagnosis labels
- Utility functions for inspecting signal shapes and channel names

These utilities facilitate consistent and reproducible preprocessing of large-scale PSG data for the LPSGM model.
"""

import os
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from mne.io import read_raw_edf


def info(*mats):
    """
    Print shape, data type, minimum, and maximum values of given numpy arrays.

    Args:
        *mats (np.ndarray): One or more numpy arrays to inspect.

    Returns:
        None
    """
    for mat in mats:
        print(mat.shape, mat.dtype, mat.min(), mat.max())


def pre_process(sig_dict, resample_rate=100, norch=False):
    """
    Preprocess raw signals by applying filtering, optional notch filtering, resampling, normalization,
    and segmentation into 30-second epochs.

    Args:
        sig_dict (dict): Dictionary where keys are channel names and values are dicts with keys:
                         'data' (np.ndarray): raw signal array,
                         'sample_rate' (int): original sampling rate.
        resample_rate (int, optional): Target sampling rate after resampling. Defaults to 100 Hz.
        norch (bool, optional): Whether to apply 50 Hz notch filter to remove powerline noise. Defaults to False.

    Returns:
        dict: Dictionary with channel names as keys and preprocessed signals as 2D numpy arrays of shape
              (num_epochs, epoch_length_samples).
    """
    processed_dict = {}
    
    for ch_name, ch_data in sig_dict.items():
        sig = ch_data['data']
        sample_rate = ch_data['sample_rate']
        
        TN = len(sig)  # Total number of samples in the signal
        
        # Apply channel-specific filtering
        if ch_name in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2']:
            # Bandpass filter between 0.3 Hz and 35 Hz for EEG/EOG channels
            b, a = signal.butter(N=4, Wn=[0.3 * 2 / sample_rate, 35 * 2 / sample_rate], btype='bandpass')
            sig = signal.filtfilt(b, a, sig)
        elif ch_name in ['Chin']:
            # Highpass filter at 10 Hz for Chin EMG channel
            b, a = signal.butter(N=4, Wn=10 * 2 / sample_rate, btype='highpass')
            sig = signal.filtfilt(b, a, sig)
        
        if norch:
            # Apply 50 Hz notch filter to remove powerline interference
            b_notch, a_notch = signal.iirnotch(w0=50, Q=20, fs=sample_rate)
            sig = signal.filtfilt(b_notch, a_notch, sig)

        # Resample signal if target rate differs from original
        if resample_rate != sample_rate:
            scaled_TN = round(resample_rate / sample_rate * TN)
            # Linear interpolation for resampling
            sig_r = interp1d(np.linspace(0, TN - 1, TN), sig, kind='linear')(
                np.linspace(0, TN - 1, scaled_TN))
        else:
            scaled_TN = TN
            sig_r = sig
            
        # Standardize signal to zero mean and unit variance (Z-score normalization)
        sig_r = (sig_r - np.mean(sig_r)) / np.std(sig_r)

        # Segment signal into 30-second epochs
        EpochL = 30 * resample_rate  # Number of samples per epoch
        EpochN = scaled_TN // EpochL  # Number of complete epochs
        sig_r = np.reshape(sig_r[:EpochN * EpochL], (EpochN, EpochL))
        
        processed_dict[ch_name] = sig_r

    return processed_dict


def save(dst_root, sub_id, sig_dict, diagnosis):
    """
    Save preprocessed signals and diagnosis label into a compressed .npz file organized by subject.

    Args:
        dst_root (str): Root directory where processed data will be saved.
        sub_id (str): Subject identifier used to name the subdirectory and file.
        sig_dict (dict): Dictionary of preprocessed signals with channel names as keys and 2D numpy arrays as values.
        diagnosis (str): Diagnosis label associated with the subject.

    Returns:
        None
    """
    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)
    npz_path = os.path.join(dst_sub_dir, f"{sub_id}.npz")
    
    # Prepare data dictionary for saving, converting signals to float32 for efficiency
    save_data = {'Diagnosis': diagnosis}
    for ch_name in sig_dict.keys():
        save_data[ch_name] = sig_dict[ch_name].astype(np.float32)

    np.savez(npz_path, **save_data)


def get_ch_names(edf_path):
    """
    Retrieve the list of channel names from an EDF file.

    Args:
        edf_path (str): Path to the EDF file.

    Returns:
        list: List of channel name strings contained in the EDF file.
    """
    sig_raw = read_raw_edf(edf_path, verbose=False)
    return sig_raw.ch_names


def load_sig(sig_path, channel_id):
    """
    Load raw signals from an EDF file based on flexible channel mappings, including support for differential channels.

    Args:
        sig_path (str): Path to the EDF file containing raw PSG data.
        channel_id (dict): Dictionary mapping target channel names to a list of possible source channels or tuples.
                           Each option can be a single channel name or a tuple representing a differential pair.

    Returns:
        tuple:
            datetime.datetime: Measurement start time with timezone removed.
            dict: Dictionary where keys are target channel names and values are dicts with keys:
                  'sample_rate' (int): Sampling rate of the signals,
                  'data' (np.ndarray): Loaded raw signal array.
    """
    # Collect all possible source channels from channel_id options
    all_possible_channels = set()
    for channel_options in channel_id.values():
        for option in channel_options:
            if isinstance(option, tuple):
                all_possible_channels.update(option)
            else:
                all_possible_channels.add(option)
    
    # Load raw EDF data including all possible channels to maximize availability
    sig_raw = read_raw_edf(sig_path, include=list(all_possible_channels), verbose=False)
    available_channels = set(sig_raw.ch_names)
    sig_data = sig_raw.to_data_frame().to_numpy()  # Data shape: (num_samples, num_channels + 1 time column)
    
    # Estimate sample rate from time column differences
    sample_rate = round(1 / (sig_data[1, 0] - sig_data[0, 0]))
    
    # Extract measurement start time and remove timezone info
    start_time = sig_raw.info['meas_date']
    start_time = start_time.replace(tzinfo=None)
    
    sig_dict = {}
    
    for target_ch, channel_options in channel_id.items():
        channel_data = None
        
        # Attempt to find a valid source channel or differential pair in order of preference
        for option in channel_options:
            if isinstance(option, tuple) and len(option) == 2:
                # Differential channel: subtract signals of two channels
                ch1, ch2 = option
                if ch1 in available_channels and ch2 in available_channels:
                    ch1_idx = sig_raw.ch_names.index(ch1)
                    ch2_idx = sig_raw.ch_names.index(ch2)
                    # +1 offset to skip time column in sig_data
                    channel_data = sig_data[:, ch1_idx + 1] - sig_data[:, ch2_idx + 1]
                    break
            else:
                # Single channel option
                if option in available_channels:
                    ch_idx = sig_raw.ch_names.index(option)
                    channel_data = sig_data[:, ch_idx + 1]  # +1 to skip time column
                    break
        
        if channel_data is not None:
            sig_dict[target_ch] = {
                'sample_rate': sample_rate,
                'data': channel_data
            }
        else:
            # Warn if no valid channel option found for the target channel
            print(f"Warning: Could not find any valid option for channel {target_ch}, existing channels: {get_ch_names(sig_path)}, {sig_path}")

    return start_time, sig_dict
