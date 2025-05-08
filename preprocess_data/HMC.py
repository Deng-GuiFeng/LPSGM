# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from multiprocessing import Pool

from utils import *


def load_ano(txt_path):
    stage_dict = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}

    sleep_stages, start_seconds, end_seconds = [], [], []
    with open(txt_path, 'r')as f:
        lines = f.readlines()
    
    for line in lines[1:]:
        line = line.strip().split(',')
        stage, onset, duration = line[4].strip(), line[2].strip(), line[3].strip()
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])
            start_seconds.append(int(onset))
            end_seconds.append(int(onset) + int(duration))

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

    sig_path = os.path.join(src_root, f"{sub_id}.edf")
    ano_path = os.path.join(src_root, f"{sub_id}_sleepscoring.txt")
    
    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)
    ano, start_seconds, end_seconds = load_ano(ano_path)
    
    SigEpoch = []
    for start, end in zip(start_seconds, end_seconds):
        SigEpoch.append(sig[int(start*sample_rate):int(end*sample_rate)])
    sig = np.concatenate(SigEpoch, axis=0)

    sig = pre_process(sig, sample_rate, resample_rate)
    
    sig_list, ano_list = continue_split(sig, ano, start_seconds, end_seconds)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set([f.split('.')[0] for f in os.listdir(src_root) if '_' not in f]) - set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(single_process, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing HMC Dataset"):
            pass


def test():
    subjects = set([f.split('.')[0] for f in os.listdir(src_root) if '_' not in f]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/HMC/recordings/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/HMC/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {'EEG F4-M1': 1, 'EEG C3-M2': 2, 'EEG C4-M1': 3, 'EEG O2-M1': 5, 'EOG E1-M2': 6, 'EOG E2-M2': 7}

    run(100)
    
    formatting_check(dst_root)

    # test()