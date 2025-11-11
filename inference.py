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

    # ==================================================================================
    # CHANNEL MAPPING CONFIGURATION - IMPORTANT FOR INFERENCE
    # ==================================================================================
    # 
    # LPSGM supports flexible channel configurations with 9 standard channels:
    #   EEG: F3, F4, C3, C4, O1, O2
    #   EOG: E1, E2
    #   EMG: Chin
    # 
    # The model can handle missing channels through a padding and masking mechanism,
    # allowing inference with 1 to 9 channels. However, using more channels (especially
    # C3, C4, E1, E2) generally improves sleep staging performance.
    # 
    # HOW TO CONFIGURE channel_map_for_load_sig:
    # -------------------------------------------
    # This dictionary maps LPSGM's standard channel names (keys) to your EDF channel
    # names (values). Each value is a tuple of options tried in order of priority.
    # 
    # Two types of channel mapping are supported:
    # 
    # 1. SINGLE CHANNEL MAPPING:
    #    If your EDF already contains pre-referenced channels (e.g., 'F3-M2', 'C3-A2'),
    #    map directly to the channel name as a string:
    #    
    #    'F3': ('F3-M2', 'F3-A2', 'EEG F3-M2'),  # Try 'F3-M2' first, then 'F3-A2', etc.
    #    'C3': ('C3-M2',),                        # Only one option
    # 
    # 2. DIFFERENTIAL CHANNEL MAPPING:
    #    If your EDF contains individual electrodes (e.g., 'F3', 'M2'), use a tuple
    #    of two channel names to compute the difference (first minus second):
    #    
    #    'F3': (('F3', 'M2'), ('F3', 'A2')),     # Try F3-M2 first, then F3-A2
    #    'C3': (('C3', 'M2'),),                   # Compute C3-M2 as differential
    # 
    # 3. MIXED MAPPING (RECOMMENDED):
    #    You can mix both types to handle different EDF formats:
    #    
    #    'F3': ('F3-M2', ('F3', 'M2'), 'EEG F3'),  # Try single channel first, then differential
    # 
    # STEPS TO CONFIGURE FOR YOUR EDF FILES:
    # --------------------------------------
    # 1. Check your EDF channel names using: get_ch_names(your_edf_path)
    # 2. Identify which channels correspond to F3, F4, C3, C4, O1, O2, E1, E2, Chin
    # 3. For each LPSGM channel you want to use:
    #    - If the EDF has a pre-referenced channel (e.g., 'C3-M2'), use single mapping
    #    - If the EDF has separate electrodes (e.g., 'C3' and 'M2'), use differential mapping
    # 4. Comment out channels not available in your EDF (model handles missing channels)
    # 5. Provide multiple options for each channel if your EDF naming varies
    # 
    # EXAMPLE CONFIGURATIONS:
    # -----------------------
    # Example 1: EDF with pre-referenced channels
    # channel_map_for_load_sig = {
    #     'C3': ('C3-M2', 'C3-A2'),
    #     'C4': ('C4-M1', 'C4-A1'),
    #     'E1': ('E1-M2', 'LOC-M2'),
    #     'E2': ('E2-M1', 'ROC-M1'),
    # }
    # 
    # Example 2: EDF with individual electrodes (differential)
    # channel_map_for_load_sig = {
    #     'C3': (('C3', 'M2'), ('C3', 'A2')),
    #     'C4': (('C4', 'M1'), ('C4', 'A1')),
    #     'E1': (('E1', 'M2'),),
    #     'E2': (('E2', 'M1'),),
    # }
    # 
    # Example 3: Mixed mapping (handles multiple EDF formats)
    # channel_map_for_load_sig = {
    #     'C3': ('C3-M2', ('C3', 'M2'), 'EEG C3-M2'),
    #     'C4': ('C4-M1', ('C4', 'M1'), 'EEG C4-M1'),
    #     'E1': ('E1-M2', ('E1', 'M2'), 'EOG-L'),
    #     'E2': ('E2-M1', ('E2', 'M1'), 'EOG-R'),
    # }
    # 
    # NOTE: The load_sig() function (in web_demo/utils.py) processes this mapping.
    # It tries each option in order until a match is found in the EDF file.
    # ==================================================================================
    
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
