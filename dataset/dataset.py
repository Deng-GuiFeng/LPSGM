# -*- coding: utf-8 -*-
"""
dataset.py

This module provides utilities for handling PSG (Polysomnography) dataset subjects and domain information
for the LPSGM project. It defines channel mappings relevant to PSG data processing and includes functions
to retrieve subject paths and their corresponding domain labels from specified dataset domains.

The file plays a foundational role in dataset management, enabling downstream data loading and preprocessing
for sleep staging and mental disorder diagnosis tasks within the LPSGM framework.
"""

from sklearn.model_selection import StratifiedKFold, train_test_split
import os

# Mapping from PSG channel names to their corresponding spatial embedding indices
CHANNEL_TO_INDEX = {
    'F3': 0,
    'F4': 1,
    'C3': 2,
    'C4': 3,
    'O1': 4,
    'O2': 5,
    'E1': 6,
    'E2': 7,

    'Chin': 8,
}

# Set of all possible channel names used for validation and channel configuration
ALL_CHANNELS = set(CHANNEL_TO_INDEX.keys())

# Root directory containing all public PSG datasets
source_datasets_root = r"./data/"

def get_datasets_subjects(domains: set):
    """
    Retrieve full paths to subject data folders and their associated domain labels 
    for the specified dataset domains.

    Args:
        domains (set): A set of domain names (dataset identifiers) to include.

    Returns:
        tuple: 
            Subjects (list of str): List of full paths to each subject's data directory.
            Domains (list of str): Corresponding list of domain labels for each subject.
    """
    Subjects, Domains = [], []
    for domain in domains:
        domain_root = os.path.join(source_datasets_root, domain)  # Path to the domain dataset folder
        subjects = os.listdir(domain_root)  # List all subject directories within the domain
        # Extend Subjects list with full paths to each subject directory
        Subjects.extend(os.path.join(domain_root, sub_id) for sub_id in subjects)
        # Extend Domains list with the domain label repeated for each subject
        Domains.extend([domain] * len(subjects))
    return Subjects, Domains
