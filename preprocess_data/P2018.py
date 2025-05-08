# -*- coding: utf-8 -*-
import scipy.io as scio
import hdf5storage
import numpy as np
import os
import shutil
from scipy.stats import mode
from multiprocessing import Pool

from utils import *


def process_recording(sub_id):
    # print(sub_id)

    sig_path = os.path.join(src_root, sub_id, f"{sub_id}.mat")
    ano_path = os.path.join(src_root, sub_id, f"{sub_id}-arousal.mat")

    # load sequence data
    mat_data=scio.loadmat(sig_path)
    sig_raw = mat_data['val']
    sig = np.stack([sig_raw[ch] for ch in channel_id.keys()], axis=1)
    
    # load label data
    label_data_all = hdf5storage.loadmat(ano_path)['data']
    sleep_stages = label_data_all[0]['sleep_stages']
    wake = sleep_stages[0]['wake']
    nonrem1 = sleep_stages[0]['nonrem1']
    nonrem2 = sleep_stages[0]['nonrem2']
    nonrem3 = sleep_stages[0]['nonrem3']
    rem = sleep_stages[0]['rem']
    undefined = sleep_stages[0]['undefined']
    label = np.concatenate([wake, nonrem1, nonrem2, nonrem3, rem, undefined], 1) # (N, 6)
    label = label@np.array([0,1,2,3,4, 9])

    sig = pre_process(sig, sample_rate, resample_rate)
    EpochN = sig.shape[0]
    ano = np.reshape(label[:EpochN*sample_rate*30], (EpochN, sample_rate*30))
    ano = mode(ano, axis=1)[0]  # majority voting

    sig_list, ano_list = rm_unknown_label(sig, ano)
    
    ch_id = np.array(list(channel_id.values()), dtype=np.int32)
    save(dst_root, sub_id, sig_list, ano_list, ch_id)
        

def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set(os.listdir(src_root))- set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(single_process, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing P2018 Dataset"):
            pass


def test():
    subjects = set(os.listdir(src_root))- set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/P2018/training/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/P2018/"
    shutil.rmtree(dst_root, ignore_errors=True) 

    sample_rate = 200
    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        0: 0, # F3-M2
        1: 1, # F4-M1
        2: 2, # C3-M2
        3: 3, # C4-M1
        4: 4, # O1-M2
        5: 5, # O2-M1
        6: 6, # E1-M2
    }

    run(100)
    
    formatting_check(dst_root)

    # test()  

