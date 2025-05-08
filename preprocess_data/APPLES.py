# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from multiprocessing import Pool
from datetime import datetime, timedelta   

from utils import *


def load_ano(ano_path):
    stage_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}

    sleep_stages, start_times, end_times = [], [], []
    with open(ano_path, 'r')as f:
        lines = f.readlines()

    for line in lines:
        line = line.split('\t')
        stage, start, end = line[0], line[3], line[4]
        if stage in stage_dict.keys():
            sleep_stages.append(stage_dict[stage])
            start_times.append(datetime.strptime(start, "%H:%M:%S"))
            end_times.append(datetime.strptime(end, "%H:%M:%S"))
    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages, start_times, end_times

    
def process_recording(sub_id):
    # print(sub_id)

    edf_path = os.path.join(src_root, sub_id + '.edf')
    annot_path = os.path.join(src_root, sub_id + '.annot')

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(edf_path, channel_id)

    ano, start_times, end_times = load_ano(annot_path)
    start_times = [st.replace(year=start_time.year, month=start_time.month, day=start_time.day) for st in start_times]
    end_times = [et.replace(year=start_time.year, month=start_time.month, day=start_time.day) for et in end_times]

    if start_times[0] > start_time and start_times[0].hour > 12 and start_times[0].hour < 12:
        start_times[0] += timedelta(days=1)
    if end_times[0] < start_times[0]:
        end_times[0] += timedelta(days=1)
    for i in range(1, len(start_times)):
        if start_times[i] < start_times[i-1]:
            start_times[i] += timedelta(days=1)
        if end_times[i] < end_times[i-1]:
            end_times[i] += timedelta(days=1)
        
    SigEpoch = []
    StartIndex, EndIndex = [], []
    for st, et in zip(start_times, end_times):
        start_seconds = (st - start_time).total_seconds()
        end_seconds = (et - start_time).total_seconds()

        start_index, end_index = int(start_seconds *sample_rate), int(end_seconds *sample_rate)
        sig_epoch = sig[start_index : end_index]
        
        StartIndex.append(start_index)
        EndIndex.append(end_index)

        SigEpoch.append(sig_epoch)

    sig = np.concatenate(SigEpoch, axis=0)
    sig = pre_process(sig, sample_rate, resample_rate)

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
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(single_process, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing APPLES Dataset"):
            pass


def test():
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/APPLES/polysomnography/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/APPLES/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = ['apples-240095', ]

    channel_id = {
        'C3_M2': 2, 'C4_M1': 3, 'O1_M2': 4, 'O2_M1': 5, 'LOC': 6, 'ROC': 7
    }

    run(100)

    formatting_check(dst_root)

    # test()







