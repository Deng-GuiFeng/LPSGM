# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from multiprocessing import Pool

from utils import *


def load_ano(ano_path):
    stage_dict = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}

    sleep_stages, start_seconds, end_seconds = [], [], []
    with open(ano_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split('\t')
        stage, onset, duration = line[0], float(line[1]), float(line[2])
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])
            start_seconds.append(onset)
            end_seconds.append(onset + duration)
    
    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages, start_seconds, end_seconds


def continue_split(sig, ano, start, end):
    sig_list, ano_list = [], []

    last = 0
    for i in range(1, len(start)):
        if start[i] != end[i-1]:
            sig_list.append(sig[last : i])
            ano_list.append(ano[last : i])
            last = i
    if last < len(sig):
        sig_list.append(sig[last:])
        ano_list.append(ano[last:])
    
    return sig_list, ano_list

    
def process_recording(sub_id):
    # print(sub_id)

    edf_path = os.path.join(src_root, sub_id + '.edf')
    annot_path = os.path.join(src_root, sub_id + '.annot')

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(edf_path, channel_id)
    ano, start_seconds, end_seconds = load_ano(annot_path)

    start_index = [round(start*sample_rate) for start in start_seconds]
    end_index = [round(end*sample_rate) for end in end_seconds]

    SigEpoch = []
    for start, end in zip(start_index, end_index):
        SigEpoch.append(sig[start:end])
    sig = np.concatenate(SigEpoch, axis=0)

    sig = pre_process(sig, sample_rate, resample_rate)

    ano = ano[:len(sig)]

    sig_list, ano_list = continue_split(sig, ano, start_seconds, end_seconds)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing NCHSDB Dataset"):
            pass


def test():
    pass


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/nchsdb/sleep_data/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/NCHSDB/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = ['12334_19465', '1795_24829']    # NaN

    channel_id = {
        'EEG F3-M2': 0, 'EEG F3': 0, 'F3': 0, 
        'EEG F4-M1': 1, 'EEG F4': 1, 'EEG F4-M2': 1, 'F4': 1, 
        'EEG C3-M2': 2, 'EEG C3': 2, 'C3': 2, 
        'EEG C4-M1': 3, 'EEG C4': 3, 'EEG C4-M2': 3, 'C4': 3, 
        'EEG O1-M2': 4, 'EEG O1': 4, 'O1': 4, 
        'EEG O2-M1': 5, 'EEG O2': 5, 'O2': 5, 
        'EOG LOC-M2': 6, 'EEG LOC-M2': 6, 'EEG E1': 6, 'LOC': 6, 
        'EOG ROC-M1': 7, 'EEG ROC-M1': 7, 'EEG E2': 7, 'EEG ROC-M2': 7, 'ROC': 7,
        'EEG M1': -1, 'EEG M2': -2,
    }

    run(300)

    formatting_check(dst_root)

    # test()
