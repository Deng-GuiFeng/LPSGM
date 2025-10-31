# -*- coding: utf-8 -*-
"""
dataset.py

This module provides dataset utilities for disorder classification within the LPSGM project.
It includes functionality to scan and load polysomnography (PSG) data stored in NPZ files,
organize data by subject, and extract multi-channel signal data for downstream model training
and evaluation. The dataset supports flexible channel selection and diagnostic label handling,
including options to merge certain diagnostic categories.

Key components:
- Recursive file search for NPZ files in specified dataset directories
- Subject-level data aggregation with diagnosis labels
- Loading and stacking of multi-channel PSG signals from NPZ files

This module plays a critical role in preparing and structuring raw PSG data for the
Large Polysomnography Model (LPSGM) to perform sleep staging and mental disorder diagnosis.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .config import MNC_DIRS, ALL_CHANNELS


def find_files_with_suffix(root_dir: str, suffix: str) -> List[str]:
    """
    Recursively find all files with a given suffix under the specified root directory.

    Args:
        root_dir (str): Root directory to start the search
        suffix (str): File suffix to match (e.g., '.npz')

    Returns:
        List[str]: List of full file paths matching the suffix
    """
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files


def scan_mnc_npz() -> List[str]:
    """
    Scan all configured dataset directories for NPZ files containing PSG data.

    Returns:
        List[str]: Sorted list of all NPZ file paths found in dataset directories
    """
    npz_files = []
    for mdir in MNC_DIRS:
        if os.path.isdir(mdir):
            npz_files.extend(find_files_with_suffix(mdir, '.npz'))
    return sorted(npz_files)


@dataclass
class Subject:
    """
    Data class representing a subject with associated PSG NPZ file paths and diagnosis label.

    Attributes:
        npz_paths (List[str]): List of NPZ file paths belonging to the subject
        diagnosis (int): Diagnosis label (e.g., 0/1/2 representing different disorders)
        subject_id (str): Unique identifier for the subject (typically folder name)
    """
    npz_paths: List[str]
    diagnosis: int  # 0/1/2
    subject_id: str


def load_subjects(merge_NT1) -> List[Subject]:
    """
    Load subjects by scanning NPZ files, extracting diagnosis labels, and grouping files by subject.

    Args:
        merge_NT1 (bool): If True, merge 'OTHER HYPERSOMNIA' (2) into 'NON-NARCOLEPSY CONTROL' (0)
                          or adjust diagnosis labels accordingly for binary classification.

    Returns:
        List[Subject]: List of Subject objects with aggregated NPZ paths and diagnosis labels
    """
    subjects: Dict[str, Subject] = {}
    files = scan_mnc_npz()
    for f in files:
        try:
            data = np.load(f)
            diag = int(data['Diagnosis'])  # Diagnosis label must exist per dataset specification

            if merge_NT1:
                # Merge diagnosis labels for simplified classification:
                # Original: 0=NON-NARCOLEPSY CONTROL, 1=T1 NARCOLEPSY, 2=OTHER HYPERSOMNIA
                # After merge: 1 remains 1 (T1 NARCOLEPSY), others become 0 (NON-T1 CONTROL)
                diag = 1 if diag == 1 else 0

            # Derive subject ID from the parent directory name of the NPZ file
            sub_id = os.path.basename(os.path.dirname(f))

            # Aggregate NPZ files and diagnosis labels by subject
            if sub_id not in subjects:
                subjects[sub_id] = Subject(npz_paths=[f], diagnosis=diag, subject_id=sub_id)
            else:
                subjects[sub_id].npz_paths.append(f)
        except Exception as e:
            # Skip files that cannot be loaded or parsed correctly, logging the error
            print(f"Skip file due to error: {f}, {e}")
    return list(subjects.values())


def load_npz_channels(npz_path: str, channels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Load specified channels from an NPZ file and stack them into a 3D numpy array.

    Args:
        npz_path (str): Path to the NPZ file containing PSG signals
        channels (List[str]): List of channel names to extract from the NPZ file

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Signal array of shape (L, cn, 3000), where L is number of epochs,
              cn is number of available channels, and 3000 is signal length per epoch
            - List of channels actually found and loaded from the NPZ file

    Raises:
        ValueError: If none of the requested channels are found in the NPZ file
    """
    npz = np.load(npz_path)
    # Filter requested channels to those available in the NPZ file
    avail = [ch for ch in channels if ch in npz.files]
    if len(avail) == 0:
        raise ValueError(f"No valid channels in {npz_path}")

    # Load each available channel as float32 numpy array with shape (L, 3000)
    mats = [npz[ch].astype(np.float32) for ch in avail]

    # Stack channels along a new axis to shape (L, 3000, cn), then transpose to (L, cn, 3000)
    sig = np.stack(mats, axis=2).transpose(0, 2, 1)
    return sig, avail
