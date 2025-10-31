# -*- coding: utf-8 -*-
"""
MASS-SS1-SS3 Dataset Preprocessing Module

This module provides functionality to preprocess the MASS-SS1 and MASS-SS3 polysomnography (PSG) datasets
for use in the LPSGM project. It includes loading raw PSG signals and annotations, aligning signals with
sleep stage annotations, resampling, removing unknown labels, and saving the processed data in a standardized
format. The preprocessing ensures that the PSG signals and corresponding sleep stage labels are synchronized
and segmented into 30-second epochs, which are essential for subsequent sleep staging and mental disorder
diagnosis tasks.

Key functionalities:
- Loading and mapping sleep stage annotations to numerical labels
- Aligning continuous PSG signals with annotation epochs based on onset and duration
- Preprocessing signals including resampling and channel selection
- Handling multi-channel signals with flexible channel configurations
- Parallel processing support for efficient dataset preprocessing
- Dataset-specific handling for MASS-SS1 and MASS-SS3 subsets

This preprocessing step is critical to prepare the large-scale PSG data for training and evaluation of the
Large Polysomnography Model (LPSGM).
"""

import os
import numpy as np
from bs4 import BeautifulSoup
import shutil
from multiprocessing import Pool
from tqdm import tqdm
import mne

from utils import load_sig, pre_process, rm_unknown_label, save, formatting_check, find_files_with_suffix


def load_ano(ano_path):
    """
    Load and map sleep stage annotations from the annotation file.

    Args:
        ano_path (str): Path to the annotation file (e.g., EDF annotation file).

    Returns:
        tuple:
            stages (np.ndarray): Array of sleep stage labels mapped to integers (0 to 4), -1 for unknown.
            stages_onset (np.ndarray): Array of annotation onset times in seconds.
            stages_duration (np.ndarray): Array of annotation durations in seconds.
    """
    # Mapping from annotation description to sleep stage integer labels
    stages_mapping = {
        'Sleep stage W': 0, 
        'Sleep stage 1': 1, 
        'Sleep stage 2': 2,
        'Sleep stage 3': 3, 'Sleep stage 4': 3,  # Combine stages 3 and 4 as stage 3
        'Sleep stage R': 4, 
    }

    # Read annotations using MNE library
    ano_raw = mne.read_annotations(ano_path)
    ano_raw_dsp = ano_raw.description  # Annotation descriptions (stage names)
    stages_onset = ano_raw.onset       # Annotation onset times (seconds)
    stages_duration = ano_raw.duration # Annotation durations (seconds)

    # Vectorize mapping function to convert descriptions to integer labels, unknown mapped to -1
    vectorized_mapping = np.vectorize(lambda x: stages_mapping.get(x, -1))
    stages = vectorized_mapping(ano_raw_dsp)

    return stages, stages_onset, stages_duration


def align_sig_by_annotations(sig_dict, stages, stages_onset, stages_duration):
    """
    Align raw continuous PSG signals with sleep stage annotations by segmenting and concatenating
    signal epochs corresponding to annotated sleep stages.

    Args:
        sig_dict (dict): Dictionary with channel names as keys and dicts containing:
                         'sample_rate' (int) and 'data' (1D np.ndarray) as values.
        stages (np.ndarray): Array of sleep stage labels (0..4) or -1 for unknown, shape (N,).
        stages_onset (np.ndarray): Array of annotation onset times in seconds, shape (N,).
        stages_duration (np.ndarray): Array of annotation durations in seconds, shape (N,).

    Returns:
        tuple:
            aligned_sig_dict (dict): Dictionary with the same structure as sig_dict but with data
                                     cropped and concatenated to align with sleep stage epochs.
            aligned_stages (np.ndarray): Array of sleep stage labels aligned to 30-second epochs,
                                         shape (M,), where M is total number of epochs after alignment.

    Notes:
        - Only annotations with valid sleep stages (0 to 4) and positive duration are considered.
        - Durations are floored to the nearest multiple of 30 seconds to ensure epoch alignment.
        - Signal segments are trimmed to ensure all channels have consistent epoch counts.
    """
    if not sig_dict:
        # If no signal data provided, return empty results
        return sig_dict, np.array([], dtype=np.int32)

    # Obtain sampling rate from any channel (assumed consistent across channels in MASS-SS1)
    any_ch = next(iter(sig_dict))
    sr = int(round(sig_dict[any_ch]['sample_rate']))

    # Identify valid annotation indices: stages in [0..4] and positive duration
    valid_idx = np.where((stages >= 0) & (stages <= 4) & (stages_duration > 0))[0]
    if valid_idx.size == 0:
        # No valid annotations found
        return sig_dict, np.array([], dtype=np.int32)

    segs = []  # List to hold tuples of (start_sample, number_of_epochs, stage_label)
    epoch_samps = 30 * sr  # Number of samples per 30-second epoch

    for i in valid_idx:
        onset_sec = float(stages_onset[i])
        dur_sec = float(stages_duration[i])

        # Convert onset time to sample index, round to nearest integer
        start_samp = int(round(onset_sec * sr))
        # Compute number of full 30-second epochs in the duration (floor)
        n_epochs = int(np.floor(dur_sec / 30.0 + 1e-6))
        if n_epochs <= 0:
            # Skip segments with less than one full epoch
            continue
        segs.append((start_samp, n_epochs, int(stages[i])))

    if not segs:
        # No valid segments after filtering
        return sig_dict, np.array([], dtype=np.int32)

    # Determine available length per channel to ensure consistent epoch counts across channels
    ch_lengths = {ch_name: len(ch['data']) for ch_name, ch in sig_dict.items()}

    segs_final = []  # List for final segments with adjusted epoch counts
    for (s, n_ep, stg) in segs:
        # Calculate maximum epochs available per channel for this segment
        n_ep_candidates = []
        for L in ch_lengths.values():
            if s >= L:
                # Start sample beyond channel length, no epochs available
                n_ep_candidates.append(0)
            else:
                max_samps = L - s
                n_ep_candidates.append(max_samps // epoch_samps)
        # Final epoch count is minimum of requested and available epochs across all channels
        n_ep_final = min(n_ep, min(n_ep_candidates) if n_ep_candidates else 0)
        if n_ep_final > 0:
            segs_final.append((s, n_ep_final, stg))

    if not segs_final:
        # No usable segments after length adjustment
        aligned_sig_dict = {}
        for ch_name, ch in sig_dict.items():
            aligned_sig_dict[ch_name] = {
                'sample_rate': sr,
                'data': np.array([], dtype=ch['data'].dtype)
            }
        return aligned_sig_dict, np.array([], dtype=np.int32)

    # Concatenate aligned signal segments for each channel
    aligned_sig_dict = {}
    aligned_labels = []
    for ch_name, ch in sig_dict.items():
        data = ch['data']
        parts = []
        for (s, n_ep_final, stg) in segs_final:
            e = s + n_ep_final * epoch_samps  # End sample index for segment
            parts.append(data[s:e])
        if parts:
            aligned_data = np.concatenate(parts, axis=0)
        else:
            aligned_data = np.array([], dtype=data.dtype)
        aligned_sig_dict[ch_name] = {
            'sample_rate': sr,
            'data': aligned_data
        }

    # Generate aligned sleep stage labels, repeated for each epoch in segments
    for (_, n_ep_final, stg) in segs_final:
        aligned_labels.extend([stg] * n_ep_final)
    final_labels = np.asarray(aligned_labels, dtype=np.int32)

    return aligned_sig_dict, final_labels


def process_recording(sub_id, sig_path, ano_path):
    """
    Process a single subject's PSG recording and corresponding annotations.

    Steps:
    - Load raw signals and annotation data.
    - Align signals with annotations into 30-second epochs.
    - Preprocess signals (e.g., resampling).
    - Remove unknown labels and synchronize signal-label pairs.
    - Save the processed data to the destination directory.

    Args:
        sub_id (str): Subject identifier.
        sig_path (str): Path to the PSG signal file.
        ano_path (str): Path to the annotation file.
    """
    # Load raw signal and start time from signal file
    start_time, sig_dict = load_sig(sig_path, channel_id)  # returns start_time and sig_dict

    # Load sleep stage annotations
    stages, stages_onset, stages_duration = load_ano(ano_path)    

    # Align signals with annotations by segmenting and concatenating
    sig_dict, stages = align_sig_by_annotations(sig_dict, stages, stages_onset, stages_duration)

    # Preprocess signals (e.g., resampling to target rate)
    sig_dict = pre_process(sig_dict, resample_rate)  # returns processed sig_dict

    # Verify that all channels have the same number of epochs as labels
    epoch_counts = [sig_data.shape[0] for sig_data in sig_dict.values()]
    if epoch_counts and epoch_counts[0] != stages.shape[0]:
        # If mismatch, warn and truncate to minimal epoch count
        print(f"Warning: {sub_id} sig.shape[0] != stages.shape[0], sig.shape[0]: {epoch_counts[0]}, stages.shape[0]: {stages.shape[0]}, minimal epochN is used.")
        epochN = min(epoch_counts[0], stages.shape[0])
        for ch_name in sig_dict:
            sig_dict[ch_name] = sig_dict[ch_name][:epochN]
        stages = stages[:epochN]

    # Remove epochs with unknown labels and synchronize signal and label lists
    sig_list, ano_list = rm_unknown_label(sig_dict, stages)

    # Retrieve channel names for saving
    channel_names = list(channel_id.keys())
    # Save preprocessed data to destination directory
    save(dst_root, sub_id, sig_list, ano_list, channel_names)


def single_process(*args):
    """
    Wrapper function to process a single recording with exception handling.

    Args:
        *args: Arguments to pass to process_recording function.
    """
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    """
    Run preprocessing on all valid subjects in the source directory using multiprocessing.

    Args:
        num_processes (int): Number of parallel processes to use.
    """
    Inputs = []

    # Find all EDF files in the source directory
    edf_files = find_files_with_suffix(src_root, ".edf")
    # Extract unique subject IDs from filenames, excluding subjects to remove
    subjects = set(os.path.basename(f).split(' ')[0] for f in edf_files) - set(SUB_REMOVE)

    # Prepare input tuples for each subject with signal and annotation paths
    for sub_id in subjects:
        sig_path = os.path.join(src_root, f"{sub_id} PSG.edf")
        ano_path = os.path.join(src_root, f"{sub_id} Base.edf")

        if os.path.exists(sig_path) and os.path.exists(ano_path):
            Inputs.append((sub_id, sig_path, ano_path))

    # Use multiprocessing pool to process recordings in parallel with progress bar
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing MASS-SS1 Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    """
    Test function to process all valid subjects sequentially with progress indication.
    Useful for debugging or validating preprocessing on smaller scale.
    """
    Inputs = []

    # Find all EDF files in the source directory
    edf_files = find_files_with_suffix(src_root, ".edf")
    # Extract unique subject IDs from filenames, excluding subjects to remove
    subjects = set(os.path.basename(f).split(' ')[0] for f in edf_files) - set(SUB_REMOVE)

    # Prepare input tuples for each subject with signal and annotation paths
    for sub_id in subjects:
        sig_path = os.path.join(src_root, f"{sub_id} PSG.edf")
        ano_path = os.path.join(src_root, f"{sub_id} Base.edf")

        if os.path.exists(sig_path) and os.path.exists(ano_path):
            Inputs.append((sub_id, sig_path, ano_path))

    # Process each recording sequentially with progress bar
    for args in tqdm(Inputs, desc="Testing MASS-SS1 Dataset"):
        process_recording(*args)        


if __name__ == "__main__":
    # Define channel mappings for PSG signals: keys are target channel names,
    # values are tuples of possible source channel names to select from
    channel_id = {
        'F3': ('EEG F3-CLE', 'EEG F3-LER'), 
        'F4': ('EEG F4-CLE', 'EEG F4-LER'), 
        'C3': ('EEG C3-CLE', 'EEG C3-LER'), 
        'C4': ('EEG C4-CLE', 'EEG C4-LER'), 
        'O1': ('EEG O1-CLE', 'EEG O1-LER'), 
        'O2': ('EEG O2-CLE', 'EEG O2-LER'), 
        'E1': ('EOG Left Horiz',), 
        'E2': ('EOG Right Horiz',), 
        
        'Chin': (('EMG Chin1','EMG Chin2'),),
    }

    # Target resampling rate for all signals
    resample_rate = 100


    ### MASS-SS1 Dataset Preprocessing ###
    print('='*30, 'PREPROCESSING MASS-SS1 DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/MASS/SS1/"
    dst_root = r"./data/MASS-SS1/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []  # List of subject IDs to exclude from processing
    run(100)  # Run preprocessing with 100 parallel processes
    formatting_check(dst_root)  # Verify formatting of saved data


    ### MASS-SS3 Dataset Preprocessing ###
    print('='*30, 'PREPROCESSING MASS-SS3 DATASET', '='*30)

    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/MASS/SS3/"
    dst_root = r"./data/MASS-SS3/"
    # Remove existing destination directory to ensure clean preprocessing
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = []  # List of subject IDs to exclude from processing
    run(100)  # Run preprocessing with 100 parallel processes
    formatting_check(dst_root)  # Verify formatting of saved data
