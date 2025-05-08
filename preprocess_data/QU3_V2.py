# -*- coding: utf-8 -*-
import numpy as np
import os
import warnings
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool

from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


def read_xml(xml_path):
    id_stages = BeautifulSoup(open(xml_path), features="xml").find_all('SleepStage')
    for i in range(len(id_stages)):
        ss = float(id_stages[i].get_text())
        if ss == 5:
            ss = 4
        if not(ss>=0 and ss<=4):
            ss = 9
        id_stages[i] = ss
    id_stages = np.array(id_stages)
    return id_stages


def read_txt(txt_path):
    mapping = {
        'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4,
        'WK': 0, 'NN1': 1, 'NN2': 2, 'NN3': 3, 'REM': 4,
        '0': 0, '1': 1, '2': 2, '3': 3, '5': 4,
        '?' : 9, 'M': 9,
    }
    str_stages = np.loadtxt(txt_path, dtype=str)
    vectorized_map = np.vectorize(lambda s: mapping[s])
    id_stages = vectorized_map(str_stages)
    return id_stages


def process_recording(sub_id):
    # print(sub_id)

    edf_path = os.path.join(src_root, sub_id, sub_id+".edf")
    txt_path = os.path.join(src_root, sub_id, sub_id+".txt")

    sig, ch_id, meas_date, sample_rate, select_ch_names = load_sig(edf_path, channel_id)  # (TN, cn)
    ano = read_txt(txt_path)

    sig = pre_process(sig, sample_rate, resample_rate, norch=True)  # (N, 3000, cn)

    if sig.shape[0] != ano.shape[0]:
        print(f"Warning: {sub_id} sig.shape[0] != ano.shape[0], sig.shape[0]: {sig.shape[0]}, ano.shape[0]: {ano.shape[0]}, minimal epochN is used.")
        epochN = min(sig.shape[0], ano.shape[0])
        sig = sig[:epochN]
        ano = ano[:epochN]

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing QU3 Dataset"):
            pass


def test():
    subjects = set(os.listdir(src_root)) - set(SUB_REMOVE)
    
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == '__main__':
    src_root = r"/nvme1/denggf/PSG_datasets/private_datasets/QS_V2/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/QU3_V2/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        'EEG F3-M2': 0, 'F3': 0, 
        'EEG F4-M1': 1, 'F4': 1, 
        'EEG C3-M2': 2, 'C3': 2, 
        'EEG C4-M1': 3, 'C4': 3, 
        'EEG O1-M2': 4, 'O1': 4, 
        'EEG O2-M1': 5, 'O2': 5, 
        'EOG E1-M2': 6, 'E1': 6, 
        'EOG E2-M2': 7, 'E2': 7, 
        'M1': -1, 'M2': -2,        
    }

    run(100)

    formatting_check(dst_root)

    # test()