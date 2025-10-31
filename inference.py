# -*- coding: utf-8 -*-
"""
inference.py

This script performs sleep stage prediction inference using the LPSGM (Large Polysomnography Model) framework.
It processes raw PSG EDF files by loading and preprocessing signals, then generates hypnograms (sleep stage sequences)
using the trained model. The script supports batch processing of EDF files in a directory and saves the predicted
hypnograms as text files. This inference pipeline is a key component for applying LPSGM to new PSG recordings
for sleep staging and mental disorder diagnosis.
"""

from web_demo.inference_backend import inference_recodings
from web_demo.utils import load_sig, pre_process, get_ch_names, calculate_hypnogram_metrics

from tqdm import tqdm
import os, shutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def save_txt(hypnogram, save_path):
    """
    Save the predicted hypnogram (sleep stages) to a text file.

    Args:
        hypnogram (list or iterable): Sequence of predicted sleep stages
        save_path (str): File path to save the hypnogram text file

    Returns:
        None
    """
    with open(save_path, 'w') as f:
        for stage in hypnogram:
            f.write(f"{stage}\n")  # Write each sleep stage on a new line


def inference_edf(edf_file_path):
    """
    Perform inference on a single EDF file to generate a predicted hypnogram.

    Args:
        edf_file_path (str): Path to the raw PSG EDF file

    Returns:
        list: Predicted sequence of sleep stages (hypnogram)
    """
    # Load raw signals from EDF file using the predefined channel mapping
    start_time, sig_dict = load_sig(edf_file_path, channel_map_for_load_sig)

    # Preprocess signals: resample to target frequency and apply filtering if needed
    processed_signals = pre_process(sig_dict, resample_rate=RESAMPLE_RATE, notch=False)

    # Run the LPSGM inference backend to obtain sleep stage predictions
    hypnogram = inference_recodings(processed_signals)
    return hypnogram


if __name__ == "__main__":
    # Directory containing raw EDF files for inference
    edf_dir = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/is-rc/"
    # Directory to save predicted hypnogram text files
    hypnogram_dir = r"/nvme1/denggf/PSG_datasets/public_datasets/mnc/IS-RC-hypnogram_LPSGM/"

    # Clear existing hypnogram output directory and recreate it
    shutil.rmtree(hypnogram_dir, ignore_errors=True)
    os.makedirs(hypnogram_dir, exist_ok=True)

    # Define channel mapping for loading signals from EDF files
    # Keys are target channel names; values are tuples of possible EDF channel names
    channel_map_for_load_sig = {
        # 'F3': ('F3', ),
        # 'F4': ('F4', ),
        'C3': ('C3', ),
        'C4': ('C4', ),
        'O1': ('O1', ),
        'O2': ('O2', ),
        'E1': ('E1', ),
        'E2': ('E2', ),
        'Chin': ('cchin_l', ),  # Chin EMG channel
    }
    RESAMPLE_RATE = 100  # Target sampling rate in Hz for preprocessing

    # Iterate over all EDF files in the input directory with progress bar
    for edf_file in tqdm(os.listdir(edf_dir), desc="Processing EDF files"):
        if not edf_file.endswith('.edf'):
            continue  # Skip non-EDF files

        # Extract subject ID from filename (assumes format: ID-...)
        sub_id = edf_file.split('-')[0].upper()

        edf_file_path = os.path.join(edf_dir, edf_file)
        hypnogram_save_path = os.path.join(hypnogram_dir, f"{sub_id}.STA")

        # Perform inference to obtain predicted hypnogram
        hypnogram = inference_edf(edf_file_path)

        # Save predicted hypnogram to text file
        save_txt(hypnogram, hypnogram_save_path)
