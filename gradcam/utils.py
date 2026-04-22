# -*- coding: utf-8 -*-
"""
utils.py

Utility functions used by the Grad-CAM pipeline:
- EDF signal loading with flexible channel mapping (differential pairs or
  pre-referenced single channels).
- Per-channel signal preprocessing (filtering, resampling, optional z-score
  normalization, epoch segmentation).
- MASS-SS1/SS3 subject discovery and EDF+ annotation parsing.

These utilities intentionally duplicate the subset of functionality needed
locally (rather than importing from ``preprocess/``), so that the Grad-CAM
module stays self-contained and decoupled from the main preprocessing pipeline.
"""

import os
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from mne.io import read_raw_edf
import mne


def pre_process(sig_dict, resample_rate=100, notch=False, normalize=True):
    """
    Preprocess signals by applying filtering, optional notch filtering, resampling,
    optional z-score normalization, and 30-second epoch segmentation.

    Args:
        sig_dict (dict): Dictionary of signals with channel names as keys and dicts
                         containing 'data' and 'sample_rate'.
        resample_rate (int, optional): Target sampling rate after resampling. Defaults to 100 Hz.
        notch (bool, optional): Whether to apply a 50 Hz notch filter to remove powerline
                                noise. Defaults to False.
        normalize (bool, optional): Whether to z-score normalize each channel.
                                    Defaults to True. Pass False to retain the original
                                    microvolt scale (e.g. for raw-waveform rendering).

    Returns:
        dict: Dictionary with channel names as keys and preprocessed signals segmented
              into epochs of shape (num_epochs, 30 * resample_rate).
    """
    processed_dict = {}

    for ch_name, ch_data in sig_dict.items():
        sig = ch_data['data']
        sample_rate = ch_data['sample_rate']

        TN = len(sig)

        # Apply bandpass or highpass filtering based on channel type
        if ch_name in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2']:
            # 0.3-35 Hz bandpass filter for EEG and EOG channels
            b, a = signal.butter(N=4, Wn=[0.3 * 2 / sample_rate, 35 * 2 / sample_rate], btype='bandpass')
            sig = signal.filtfilt(b, a, sig)
        elif ch_name in ['Chin']:
            # 10 Hz highpass filter for the chin EMG channel
            b, a = signal.butter(N=4, Wn=10 * 2 / sample_rate, btype='highpass')
            sig = signal.filtfilt(b, a, sig)

        if notch:
            # 50 Hz notch filter to suppress powerline interference
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

        if normalize:
            sig_r = (sig_r - np.mean(sig_r)) / np.std(sig_r)

        # Segment into 30-second epochs
        EpochL = 30 * resample_rate
        EpochN = scaled_TN // EpochL
        sig_r = np.reshape(sig_r[:EpochN * EpochL], (EpochN, EpochL))

        processed_dict[ch_name] = sig_r

    return processed_dict


def get_ch_names(edf_path):
    """
    Return the list of channel names stored in an EDF file.

    Args:
        edf_path (str): Path to the EDF file.

    Returns:
        list: Channel names as reported by MNE.
    """
    sig_raw = read_raw_edf(edf_path, verbose=False)
    return sig_raw.ch_names


def load_sig(sig_path, channel_id):
    """
    Load PSG signals from an EDF file according to a flexible channel mapping.

    For each target channel, the first matching option (in listed order) is used.
    A tuple of two names denotes a differential pair (value = ch1 - ch2); any
    other value is treated as a pre-referenced single channel name.

    Args:
        sig_path (str): Path to the EDF file.
        channel_id (dict): Dictionary mapping target channel names to tuples of
                           options. Each option is either a string (pre-referenced
                           channel) or a two-element tuple (differential pair).

    Returns:
        tuple:
            datetime.datetime: Measurement start time with timezone stripped.
            dict: ``{target_ch: {'sample_rate': int, 'data': np.ndarray}}``.
    """
    # Collect every candidate raw channel name we may need to read
    all_possible_channels = set()
    for channel_options in channel_id.values():
        for option in channel_options:
            if isinstance(option, tuple):
                all_possible_channels.update(option)
            else:
                all_possible_channels.add(option)

    sig_raw = read_raw_edf(sig_path, include=list(all_possible_channels), verbose=False)
    available_channels = set(sig_raw.ch_names)
    sig_data = sig_raw.to_data_frame().to_numpy()  # (time_points, 1 + num_channels); column 0 is time
    sample_rate = round(1 / (sig_data[1, 0] - sig_data[0, 0]))

    start_time = sig_raw.info['meas_date']
    start_time = start_time.replace(tzinfo=None)

    sig_dict = {}

    for target_ch, channel_options in channel_id.items():
        channel_data = None

        for option in channel_options:
            if isinstance(option, tuple) and len(option) == 2:
                # Differential pair (e.g., ('EMG Chin1', 'EMG Chin2'))
                ch1, ch2 = option
                if ch1 in available_channels and ch2 in available_channels:
                    ch1_idx = sig_raw.ch_names.index(ch1)
                    ch2_idx = sig_raw.ch_names.index(ch2)
                    channel_data = sig_data[:, ch1_idx + 1] - sig_data[:, ch2_idx + 1]
                    break
            else:
                # Pre-referenced single channel (e.g., 'EEG F3-CLE')
                if option in available_channels:
                    ch_idx = sig_raw.ch_names.index(option)
                    channel_data = sig_data[:, ch_idx + 1]
                    break

        if channel_data is not None:
            sig_dict[target_ch] = {
                'sample_rate': sample_rate,
                'data': channel_data,
            }
        else:
            print(f"Warning: no matching option for channel {target_ch} in {sig_path}. "
                  f"Available channels: {get_ch_names(sig_path)}")

    return start_time, sig_dict


def find_mass_subjects(src_root, subject_ids=None):
    """
    Enumerate MASS-SS1/SS3 subject recordings under ``src_root``.

    A subject is recognized when both ``{sub_id} PSG.edf`` (signals) and
    ``{sub_id} Base.edf`` (EDF+ annotations) exist directly under ``src_root``.

    Args:
        src_root (str): Root directory containing the MASS EDF files.
        subject_ids (Iterable[str], optional): Whitelist of subject IDs. When
            provided, only matching subjects are returned.

    Returns:
        list: A sorted list of ``(sub_id, psg_path, ano_path)`` tuples.
    """
    out = []
    for fname in sorted(os.listdir(src_root)):
        if not fname.endswith(' PSG.edf'):
            continue
        sub_id = fname[:-len(' PSG.edf')]
        psg_path = os.path.join(src_root, fname)
        ano_path = os.path.join(src_root, f"{sub_id} Base.edf")
        if os.path.exists(ano_path):
            out.append((sub_id, psg_path, ano_path))

    if subject_ids is not None:
        whitelist = set(subject_ids)
        out = [row for row in out if row[0] in whitelist]
    return out


def read_mass_annotations(ano_path):
    """
    Parse sleep stage annotations from a MASS-SS1/SS3 EDF+ annotation file.

    The MASS dataset stores sleep stage annotations as EDF+ events. Stage
    descriptions are mapped to LPSGM's 5-class convention:

        'Sleep stage W'  -> 0
        'Sleep stage 1'  -> 1
        'Sleep stage 2'  -> 2
        'Sleep stage 3'  -> 3
        'Sleep stage 4'  -> 3  (merged into N3)
        'Sleep stage R'  -> 4

    Any other description is mapped to -1 (unknown) and filtered out during
    alignment.

    Args:
        ano_path (str): Path to the ``{sub_id} Base.edf`` annotation file.

    Returns:
        tuple:
            stages (np.ndarray): 1-D int array of stage labels (-1 for unknown).
            onsets (np.ndarray): 1-D float array of annotation onsets in seconds.
            durations (np.ndarray): 1-D float array of annotation durations in seconds.
    """
    stages_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    ano_raw = mne.read_annotations(ano_path)
    descriptions = ano_raw.description
    onsets = ano_raw.onset
    durations = ano_raw.duration

    vectorized = np.vectorize(lambda d: stages_mapping.get(d, -1))
    stages = vectorized(descriptions)
    return stages, onsets, durations
