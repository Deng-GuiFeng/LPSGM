# -*- coding: utf-8 -*-
import numpy as np
import os
from mne.io import read_raw_edf
from scipy import signal
from scipy.interpolate import interp1d
from multiprocessing import Pool
import shutil

from utils import *


def load_ano(ano_path):
    stage_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

    sleep_stages = []
    with open(ano_path, 'r')as f:
        lines = f.readlines()

    for line in lines:
        s = line.replace('\n','').split(',')
        # onset = int(s[0])
        duration = int(s[1])
        stage = s[2]
        sleep_stages.extend([stage_dict[stage]]*int(duration/30))
    sleep_stages = np.array(sleep_stages, dtype=np.int32)

    return sleep_stages


def remove_wake(sig, ano):
    # sig: numpy.ndarray with shape (EpochN, 3000, C)
    # ano: numpy.ndarray with shape (EpochN, ) and value in {0, 1, 2, 3, 4}
    
    EpochN = ano.shape[0]
    W_C = []
    w_duration, w_onset = 0, 0
    for i, ano_i in enumerate(ano):
        if ano_i == 0:
            if w_duration == 0:
                w_onset = i
            w_duration += 1
        else:
            if (w_onset ==0 and w_duration > 60) or (w_onset !=0 and w_duration > 120):
                W_C.append({'onset': w_onset, 'duration': w_duration, 'end': w_onset+w_duration})
            w_duration, w_onset = 0, 0
    if w_duration > 60:
        W_C.append({'onset': w_onset, 'duration': w_duration, 'end': w_onset+w_duration})

    Session_Start, Session_End = [], []
    for wc in W_C:
        onset = wc['onset']
        # duration = wc['duration']
        end = wc['end']

        if onset == 0:
            Session_Start.append(end-60)
        elif end == EpochN:
            Session_End.append(onset+60)
        else:
            Session_End.append(onset+60)
            Session_Start.append(end-60)
    if W_C[0]['onset'] != 0:
        Session_Start.insert(0, 0)
    if len(Session_Start) > len(Session_End):
        Session_End.append(EpochN)
    
    sig_list = [sig[s:e] for s,e in zip(Session_Start, Session_End)]
    ano_list = [ano[s:e] for s,e in zip(Session_Start, Session_End)]

    return sig_list, ano_list


def process_recording(sub_id):
    # print(sub_id)

    edf_path = os.path.join(src_root, sub_id, 'psg.edf')
    ids_path = os.path.join(src_root, sub_id, 'hypnogram.ids')

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(edf_path, channel_id)
    ano = load_ano(ids_path)

    sig = pre_process(sig, sample_rate, resample_rate)

    sig_list, ano_list = remove_wake(sig, ano)
    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(single_process, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing DCSM Dataset"):
            pass


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/DCSM/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/DCSM/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        'F3-M2': 0, 'F4-M1': 1, 'C3-M2': 2, 'C4-M1': 3, 'O1-M2': 4, 'O2-M1': 5, 'E1-M2': 6, 'E2-M2': 7,
    }

    run(100)

    formatting_check(dst_root)

    # test()

