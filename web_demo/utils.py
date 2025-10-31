# -*- coding: utf-8 -*-
"""
utils.py

This module provides utility functions for the LPSGM project, which focuses on large-scale polysomnography (PSG) data processing for sleep staging and mental disorder diagnosis. The functionalities include file searching, data validation, signal preprocessing (filtering, resampling, normalization, epoch segmentation), EDF channel extraction, signal loading with flexible channel mapping, and calculation of sleep metrics from hypnograms. These utilities support the core model by preparing and validating PSG data and extracting relevant sleep parameters.
"""

import os
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from mne.io import read_raw_edf


def find_files_with_suffix(root_dir, suffix):
    """
    Recursively find all files with a given suffix in a directory tree.

    Args:
        root_dir (str): Root directory to start searching from.
        suffix (str): File suffix to match (e.g., '.edf').

    Returns:
        list: List of full file paths matching the suffix.
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
        data (array-like): Input data to check.

    Returns:
        tuple: (has_nan (bool), has_inf (bool)) indicating presence of NaN or Inf.
    """
    arr = np.array(data)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    return has_nan, has_inf


def info(*mats):
    """
    Print shape, data type, minimum, and maximum values for each input array.

    Args:
        *mats: Variable length argument list of numpy arrays.
    """
    for mat in mats:
        print(mat.shape, mat.dtype, mat.min(), mat.max())


def pre_process(sig_dict, resample_rate=100, notch=False):
    """
    Preprocess signals by applying filtering, optional notch filtering, resampling, normalization, and epoch segmentation.

    Args:
        sig_dict (dict): Dictionary of signals with channel names as keys and dicts containing 'data' and 'sample_rate'.
        resample_rate (int, optional): Target sampling rate after resampling. Defaults to 100 Hz.
        notch (bool, optional): Whether to apply a 50 Hz notch filter to remove powerline noise. Defaults to False.

    Returns:
        dict: Dictionary with channel names as keys and preprocessed signals segmented into epochs (shape: [num_epochs, epoch_length]).
    """
    processed_dict = {}
    
    for ch_name, ch_data in sig_dict.items():
        sig = ch_data['data']
        sample_rate = ch_data['sample_rate']
        
        TN = len(sig)
        
        # Apply bandpass or highpass filtering based on channel type
        if ch_name in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2']:
            # Apply 0.3-35 Hz bandpass filter for EEG and EOG channels
            b, a = signal.butter(N=4, Wn=[0.3 * 2 / sample_rate, 35 * 2 / sample_rate], btype='bandpass')
            sig = signal.filtfilt(b, a, sig)
        elif ch_name in ['Chin']:
            # Apply 10 Hz highpass filter for EMG (chin) channel
            b, a = signal.butter(N=4, Wn=10 * 2 / sample_rate, btype='highpass')
            sig = signal.filtfilt(b, a, sig)
        
        if notch:
            # Apply 50 Hz notch filter to remove powerline interference
            b_notch, a_notch = signal.iirnotch(w0=50, Q=20, fs=sample_rate)
            sig = signal.filtfilt(b_notch, a_notch, sig)

        # Resample signal if target rate differs from original
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
        EpochL = 30 * resample_rate  # samples per epoch
        EpochN = scaled_TN // EpochL  # number of complete epochs
        sig_r = np.reshape(sig_r[:EpochN * EpochL], (EpochN, EpochL))
        
        processed_dict[ch_name] = sig_r

    return processed_dict


def get_ch_names(edf_path):
    """
    Extract channel names from an EDF file.

    Args:
        edf_path (str): Path to the EDF file.

    Returns:
        list: List of channel names in the EDF file.
    """
    sig_raw = read_raw_edf(edf_path, verbose=False)
    return sig_raw.ch_names


def load_sig(sig_path, channel_id):
    """
    Load signals from an EDF file with flexible channel mapping including differential pairs.

    Args:
        sig_path (str): Path to the EDF file.
        channel_id (dict): Dictionary mapping target channel names to lists of possible channel options.
                           Each option can be a single channel name or a tuple representing a differential pair.

    Returns:
        tuple:
            datetime.datetime: Measurement start time without timezone info.
            dict: Dictionary with target channel names as keys and dicts containing 'sample_rate' and 'data' arrays.
    """
    # Collect all possible channels to include from channel_id options
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
    sig_data = sig_raw.to_data_frame().to_numpy()  # shape: (time_points, channels + 1 time column)
    sample_rate = round(1 / (sig_data[1, 0] - sig_data[0, 0]))  # Calculate sampling rate from time column
    
    # Extract measurement start time and remove timezone info
    start_time = sig_raw.info['meas_date']
    start_time = start_time.replace(tzinfo=None)
    
    sig_dict = {}
    
    # Map each target channel to available data using priority options
    for target_ch, channel_options in channel_id.items():
        channel_data = None
        
        # Iterate over channel options in order of preference
        for option in channel_options:
            if isinstance(option, tuple) and len(option) == 2:
                # Differential pair channel (e.g., ('F3', 'M2'))
                ch1, ch2 = option
                if ch1 in available_channels and ch2 in available_channels:
                    ch1_idx = sig_raw.ch_names.index(ch1)
                    ch2_idx = sig_raw.ch_names.index(ch2)
                    # Subtract signals of the two channels to form differential channel
                    channel_data = sig_data[:, ch1_idx + 1] - sig_data[:, ch2_idx + 1]  # +1 to skip time column
                    break
            else:
                # Single channel option (e.g., 'F3-M2' or 'F3')
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
            print(f"Warning: Could not find any valid option for channel {target_ch}, existing channels: {get_ch_names(sig_path)}, {sig_path}")

    return start_time, sig_dict


def calculate_hypnogram_metrics(hypnogram: np.ndarray, verbose=False):
    """
    Calculate standard sleep metrics from a hypnogram array representing sleep stages.

    Args:
        hypnogram (np.ndarray): 1D array of sleep stage labels per epoch.
                                Expected stage labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
        verbose (bool, optional): If True, print detailed metric results. Defaults to False.

    Returns:
        dict: Dictionary containing sleep metrics including latencies, durations, episode counts, and stage percentages.
    """
    # Duration of each epoch/frame in minutes (30 seconds = 0.5 minutes)
    time_per_frame = 0.5

    # Find index of first non-Wake epoch (start of sleep)
    try:
        first_non_w_index = np.where(hypnogram != 0)[0][0]
    except IndexError:
        raise ValueError("Hypnogram contains no non-Wake stages; cannot compute sleep latency.")

    def calculate_latency_after_n1(stage, first_n1_index):
        """
        Calculate latency (in minutes) to a specified sleep stage after the first non-Wake epoch.

        Args:
            stage (int): Sleep stage label to find latency for.
            first_n1_index (int): Index of first non-Wake epoch.

        Returns:
            float: Latency in minutes to the specified stage; NaN if stage not found.
        """
        try:
            target_index = np.where(hypnogram[first_n1_index:] == stage)[0][0] + first_n1_index
            return (target_index - first_n1_index) * time_per_frame
        except IndexError:
            return np.nan  # Stage not present in hypnogram

    # Calculate latencies to each sleep stage after sleep onset
    n1_latency = calculate_latency_after_n1(1, first_non_w_index)  # N1 latency
    n2_latency = calculate_latency_after_n1(2, first_non_w_index)  # N2 latency
    n3_latency = calculate_latency_after_n1(3, first_non_w_index)  # N3 latency
    rem_latency = calculate_latency_after_n1(4, first_non_w_index)  # REM latency

    # Calculate total sleep time (TST) and durations of each stage in minutes
    tst = np.sum(hypnogram > 0) * time_per_frame  # Total sleep time excluding Wake
    rem_duration = np.sum(hypnogram == 4) * time_per_frame  # REM sleep duration
    nrem_duration = np.sum((hypnogram == 1) | (hypnogram == 2) | (hypnogram == 3)) * time_per_frame  # NREM duration
    sws_duration = np.sum(hypnogram == 3) * time_per_frame  # Slow wave sleep (N3) duration

    # Calculate wake duration and number of wake episodes after sleep onset
    wake_after_sleep = hypnogram[first_non_w_index:]
    wake_duration = np.sum(wake_after_sleep == 0) * time_per_frame  # Wake time after sleep onset
    # Count transitions into Wake stage (episodes) after sleep onset
    wake_episodes = np.sum((wake_after_sleep[:-1] != 0) & (wake_after_sleep[1:] == 0))

    # Calculate durations of individual sleep stages
    n1_duration = np.sum(hypnogram == 1) * time_per_frame
    n2_duration = np.sum(hypnogram == 2) * time_per_frame
    n3_duration = sws_duration  # N3 duration equals slow wave sleep duration

    # Calculate percentage of total sleep time for each stage
    total_sleep_time = tst  # Sum of REM + N1 + N2 + N3
    rem_percentage = (rem_duration / total_sleep_time) * 100
    n1_percentage = (n1_duration / total_sleep_time) * 100
    n2_percentage = (n2_duration / total_sleep_time) * 100
    n3_percentage = (n3_duration / total_sleep_time) * 100

    if verbose:
        # Print detailed sleep metrics
        print("Sleep Latencies (minutes):")
        print(f"N1 latency: {n1_latency}")
        print(f"N2 latency: {n2_latency}")
        print(f"N3 latency: {n3_latency}")
        print(f"REM latency: {rem_latency}")

        print("\nSleep Durations (minutes):")
        print(f"Total Sleep Time (TST): {tst}")
        print(f"REM duration: {rem_duration}")
        print(f"NREM duration: {nrem_duration}")
        print(f"Slow Wave Sleep (N3) duration: {sws_duration}")

        print("\nSleep Staging:")
        print(f"Wake episodes (SPT): {wake_episodes}, Wake duration: {wake_duration}")
        print(f"REM duration: {rem_duration}")
        print(f"N1 duration: {n1_duration}")
        print(f"N2 duration: {n2_duration}")
        print(f"N3 duration: {n3_duration}")

        print("\nSleep Stage Percentages (% of TST):")
        print(f"REM: {rem_percentage:.2f}%")
        print(f"N1: {n1_percentage:.2f}%")
        print(f"N2: {n2_percentage:.2f}%")
        print(f"N3: {n3_percentage:.2f}%")

    return {
        "n1_latency": n1_latency,       # Latency to N1 stage
        "n2_latency": n2_latency,       # Latency to N2 stage
        "n3_latency": n3_latency,       # Latency to N3 stage
        "rem_latency": rem_latency,     # Latency to REM stage
        "tst": tst,                     # Total sleep time (minutes)
        "rem_duration": rem_duration,   # REM sleep duration (minutes)
        "nrem_duration": nrem_duration, # NREM sleep duration (minutes)
        "sws_duration": sws_duration,   # Slow wave sleep (N3) duration (minutes)
        "wake_duration": wake_duration, # Wake duration after sleep onset (minutes)
        "wake_episodes": wake_episodes, # Number of wake episodes after sleep onset
        "n1_duration": n1_duration,     # N1 stage duration (minutes)
        "n2_duration": n2_duration,     # N2 stage duration (minutes)
        "n3_duration": n3_duration,     # N3 stage duration (minutes)
        "rem_percentage": rem_percentage,   # REM stage percentage of TST
        "n1_percentage": n1_percentage,     # N1 stage percentage of TST
        "n2_percentage": n2_percentage,     # N2 stage percentage of TST
        "n3_percentage": n3_percentage,     # N3 stage percentage of TST
    }
