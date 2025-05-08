# -*- coding: utf-8 -*-
import numpy as np
import os
from tqdm import tqdm
import shutil
import pandas as pd
from multiprocessing import Pool

from utils import *


def process_recording(sub_id):
    # print(sub_id)

    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}.csv")

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)    # (TN, C)
    
    # read labels from CSV
    labels = pd.read_csv(ano_path, usecols=['Start Time', 'Event'])

    # convert time str to datetime object
    labels['Start Time'] = pd.to_datetime(labels['Start Time'], format='%H:%M:%S').apply(lambda dt: dt.replace(year=start_time.year, month=start_time.month, day=start_time.day))

    # check timestamp, add 1 to day if it less than the previous one
    for i in range(1, len(labels)):
        if labels.loc[i, 'Start Time'] < labels.loc[i - 1, 'Start Time']:
            labels.loc[i, 'Start Time'] += pd.Timedelta(days=1)

    labels = labels[labels['Event'].isin(SleepStages.keys())]

    labels['start'] = labels['Start Time']
    labels['end'] = labels['Start Time'] + pd.Timedelta(seconds=30)

    SigEpoch, AnoEpoch = [], []
    StartIndex, EndIndex = [], []
    for i, row in labels.iterrows():
        start_seconds = (row['start'] - start_time).total_seconds()
        end_seconds = (row['end'] - start_time).total_seconds()

        start_index, end_index = int(start_seconds *sample_rate), int(end_seconds *sample_rate)
        sig_epoch = sig[start_index : end_index]
        ano_epoch = SleepStages[row['Event']]

        StartIndex.append(start_index)
        EndIndex.append(end_index)

        SigEpoch.append(sig_epoch)
        AnoEpoch.append(ano_epoch)

    if len(SigEpoch) == len(AnoEpoch) == 0:
        # without any data
        print(f"Empty data for {sub_id}")
        return
    
    sig = np.concatenate(SigEpoch, axis=0)
    sig = pre_process(sig, sample_rate, resample_rate)

    ano = np.array(AnoEpoch, dtype=np.int64)
    ano = ano[:len(sig)]

    sig_list, ano_list = [], []
    last_idx = 0
    for i in range(1, len(StartIndex)):
        if StartIndex[i] != EndIndex[i-1]:
            sig_list.append(sig[last_idx : i])
            ano_list.append(ano[last_idx : i])
            last_idx = i
    if last_idx < len(sig):
        sig_list.append(sig[last_idx:])
        ano_list.append(ano[last_idx:])

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set([os.path.splitext(f)[0] for f in os.listdir(src_root) if f.endswith(".csv")]) - set(SUB_REMOVE)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects)):
            pass
   

def test():
    pass


if __name__ == "__main__":

    channel_id = {
        'F3M2': 0, 'EEG_F3-A2': 0, 'F3': 0, 'EEG_F3-A1': 0, 'F3-M2': 0, 
        'F4M1': 1, 'EEG_F4-A1': 1, 'F4': 1, 'EEG_F4-A2': 1, 'F4-M1': 1, 
        'C3M2': 2, 'EEG_C3-A2': 2, 'C3': 2, 'EEG_C3-A1': 2, 'C3-M2': 2, 
        'C4M1': 3, 'EEG_C4-A1': 3, 'C4': 3, 'EEG_C4-A2': 3, 'C4-M1': 3, 
        'O1M2': 4, 'EEG_O1-A2': 4, 'O1': 4, 'EEG_O1-A1': 4, 'O1-M2': 4, 
        'O2M1': 5, 'EEG_O2-A1': 5, 'O2': 5, 'EEG_O2-A2': 5, 'O2-M1': 5,
        'E1M2': 6, 'EOG_LOC-A2': 6, 'E1': 6, 'EOG1': 6, 'L-EOG': 6, 'LOC': 6, 'E1_(LEOG)': 6, 
        'E2M2': 7, 'EOG_ROC-A2': 7, 'E2': 7, 'EOG2': 7, 'R-EOG': 7, 'EOG_ROC-A1': 7, 'ROC': 7, 'E2_(REOG)': 7, 
        'M1': -1, 'A1': -1, 
        'M2': -2, 'A2': -2, 
    }

    resample_rate = 100
    SleepStages = {' Wake': 0, ' Stage1': 1, ' Stage2': 2, ' Stage3': 3, ' REM': 4}


    ### 1. BOGN ###
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/BOGN/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/STAGES-BOGN/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = ['BOGN00043', ]
    run(100)
    formatting_check(dst_root)


    ### 2. STNF ###
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/STAGES/STNF/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/STAGES-STNF/"
    shutil.rmtree(dst_root, ignore_errors=True)
    SUB_REMOVE = ["STNF00373", ]
    run(100)
    formatting_check(dst_root)

