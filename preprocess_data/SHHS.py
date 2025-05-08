# -*- coding: utf-8 -*-
import numpy as np
import os
from bs4 import BeautifulSoup
from multiprocessing import Pool
import shutil

from utils import *

"""
The PSG in this dataset is collected at patients' home.
Lots of epochs of stage 'wake' at the final of sequence.
May be we should remove some of them.
"""

def load_ano(ano_path):
    stage_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4}

    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')
    sleep_stages = [stage_dict.get(sleep_stage.get_text(), 9) for sleep_stage in sleep_stages]

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id):
    # print(sub_id)

    sig_path = os.path.join(sig_root, f"{sub_id}.edf")
    ano_path = os.path.join(ano_root, f"{sub_id}-profusion.xml")

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)    # (TN, C)
    ano = load_ano(ano_path)   

    sig = pre_process(sig, sample_rate, resample_rate)
    sig, ano = remove_wake_start_end(sig, ano)

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)
    

def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")

 
def run(num_processes):
    subjects = set([f.split('.')[0] for f in os.listdir(sig_root)]) - set(SUB_REMOVE)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing SHHS Dataset"):
            pass


def test():
    pass


if __name__ == "__main__":
    ### SHHS-1 ###
    sig_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/polysomnography/edfs/shhs1/"
    ano_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/polysomnography/annotations-events-profusion/shhs1/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/SHHS-1/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = [] 

    channel_id = {
        'EEG': 2, 
        'EEG(sec)': 3, 'EEG2': 3, 'EEG sec': 3, 'EEG 2': 3, 'EEG(SEC)': 3, 
        'EOG(L)': 6, 
        'EOG(R)': 7, 
    }
    
    run(200)

    formatting_check(dst_root)


    ### SHHS-2 ###
    sig_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/polysomnography/edfs/shhs2/"
    ano_root = r"/nvme1/denggf/PSG_datasets/public_datasets/shhs/polysomnography/annotations-events-profusion/shhs2/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/SHHS-2/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        'EEG': 2, 
        'EEG(sec)': 3, 'EEG2': 3, 'EEG sec': 3, 'EEG 2': 3, 'EEG(SEC)': 3, 
        'EOG(L)': 6, 
        'EOG(R)': 7, 
    }
    
    run(200)

    formatting_check(dst_root)