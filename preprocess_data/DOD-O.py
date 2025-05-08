# -*- coding: utf-8 -*-
import numpy as np
import os
import h5py
import shutil
from multiprocessing import Pool

from utils import *


def read_h5(h5_path):
    h5_file = h5py.File(h5_path, 'r')

    signals = [h5_file[p][:] for p in channel_id.keys()]
    sig = np.stack(signals, 0).transpose(1, 0)
    ano = h5_file["/hypnogram"][:]
    sample_rate = int(sig.shape[0] / ano.shape[0] / 30)
    return sig, ano, sample_rate


def process_recording(sub_id):
    # print(sub_id)

    h5_path = os.path.join(src_root, f"{sub_id}.h5")
    sig, ano, sample_rate = read_h5(h5_path)

    sig = pre_process(sig, sample_rate, resample_rate)

    sig_list, ano_list = rm_unknown_label(sig, ano)

    ch_id = np.array(list(channel_id.values()), dtype=np.int32)
    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(process_recording, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing DOD-O Dataset"):
            pass


def test():
    subjects = set([f.split('.')[0] for f in os.listdir(src_root)]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/DOD/DOD-O/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/DOD-O/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        '/signals/eeg/F3_M2': 0, 
        '/signals/eeg/C3_M2': 2, 
        '/signals/eeg/C4_M1': 3, 
        '/signals/eeg/O1_M2': 4, 
        '/signals/eeg/O2_M1': 5, 
        '/signals/eog/EOG1': 6, 
        '/signals/eog/EOG2': 7, 
    }

    run(100)
    
    formatting_check(dst_root)

    # test()

