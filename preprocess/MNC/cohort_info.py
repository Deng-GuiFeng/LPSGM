"""
cohort_info.py

This module provides metadata and cohort information for the MNC dataset used in the LPSGM project.
It defines diagnosis mappings and reads subject-to-diagnosis associations for multiple cohorts within the MNC dataset
from the cohort table ``cohorts_deid.xlsx`` distributed by the NSRR together with the MNC recordings.
No per-subject diagnosis is stored in this repository.
The cohorts include CNC, DHC, SSC, FHC, IHC, and KHC, each representing different subject groups.

The file also includes functionality to count diagnosis occurrences and filter cohort dictionaries
based on the actual subject directories present in the dataset folders.

This information is essential for dataset management, subject diagnosis labeling, and cohort-specific analyses
within the Large Polysomnography Model (LPSGM) framework, which focuses on sleep staging and downstream clinical screening.
"""

import re

import pandas as pd


# Path to the MNC cohort table downloaded from the NSRR with the dataset
DEFAULT_COHORTS_XLSX = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/cohorts_deid.xlsx"

# Mapping of diagnosis labels to integer codes for classification tasks
Diagnosis_Mapping = {
    'NON-NARCOLEPSY CONTROL': 0,
    'T1 NARCOLEPSY': 1,
    'OTHER HYPERSOMNIA': 2,
}


def _dhc_subject_id(table_id):
    """
    Match a DHC table ID to the subject ID derived from the EDF file name.

    The cohort table pads the numbers of "Sub" IDs to three digits (e.g., "Sub001"),
    whereas the EDF files use at least two digits (e.g., "Sub01-nsrr.edf").
    """
    m = re.match(r'^Sub0*(\d+)$', table_id)
    return f'Sub{int(m.group(1)):02d}' if m else table_id


# Per-cohort conversion from table IDs to the subject IDs used by the preprocessing scripts
# (IHC EDF file names are lower case)
ID_MAPS = {
    'DHC': _dhc_subject_id,
    'IHC': str.lower,
}


def load_cohort(cohort, xlsx_path=DEFAULT_COHORTS_XLSX):
    """
    Read the subject-to-diagnosis mapping of one MNC cohort from the NSRR cohort table.

    Args:
        cohort (str): Cohort name as given in the table's "Cohort" column (e.g., "CNC").
        xlsx_path (str): Path to ``cohorts_deid.xlsx``.

    Returns:
        dict: Subject ID -> diagnosis (a key of ``Diagnosis_Mapping``).
    """
    df = pd.read_excel(xlsx_path, dtype={'ID': str})
    df = df[df['Cohort'] == cohort]
    ids = df['ID'].str.strip()
    if cohort in ID_MAPS:
        ids = ids.map(ID_MAPS[cohort])
    # Diagnoses are stored in quotes, e.g. "'T1 NARCOLEPSY'"
    diagnoses = df['Diagnosis'].astype(str).str.strip().str.strip("'").str.strip()
    return dict(zip(ids, diagnoses))


def get_value_counts(dct):
    """
    Count the occurrences of each unique value in the given dictionary.

    Args:
        dct (dict): Dictionary with values to count occurrences for.

    Returns:
        dict: A dictionary mapping each unique value to its count.
    """
    value_counts = {}
    # Iterate over all values in the dictionary and count occurrences
    for value in dct.values():
        value_counts[value] = value_counts.get(value, 0) + 1

    return value_counts


if __name__ == '__main__':
    import os

    for cohort in ['CNC', 'DHC', 'FHC', 'IHC', 'KHC', 'SSC']:
        # Root directory of the preprocessed cohort dataset
        cohort_root = f"./data/MNC-{cohort}/"
        # List all subject directories present in the cohort dataset folder
        subjects = os.listdir(cohort_root)
        # Filter the cohort mapping to include only subjects present in the dataset folder
        cohort_filtered = {k: v for k, v in load_cohort(cohort).items() if k in subjects}
        # Print the number of filtered subjects and their diagnosis distribution
        print(f'Number of subjects in {cohort}:', len(cohort_filtered), get_value_counts(cohort_filtered))
