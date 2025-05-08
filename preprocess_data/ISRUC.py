# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from mne.io import read_raw_edf
import shutil
from multiprocessing import Pool

from utils import *


'LOC-A2', 'ROC-A1', 'C3-A2', 'O1-A2', 'C4-A1', 'O2-A1'  # 1 (Subgroup1-8)
'ROC', 'A1', 'LOC', 'A2', 'C4', 'O2', 'C3', 'O1', 'F4', 'F3'    # 1 (Subgroup1-40)
'LOC-A2', 'ROC-A1', 'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'  # 28
'E1-M2',  'E2-M1',  'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'  # 96
   

def load_ano(ano_path):
    with open(ano_path, 'r')as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    sleep_stages = [line.replace('5', '4') for line in lines]

    ano = np.array(sleep_stages, dtype=np.int32)
    return ano


def process_recording(sub_id, sig_path, ano_path):
    # print(sub_id)
  
    shutil.copyfile(sig_path, sig_path.replace('.rec', '.edf'))
    sig_path = sig_path.replace('.rec', '.edf')
    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)
    os.remove(sig_path)

    ano = load_ano(ano_path)

    sig = pre_process(sig, sample_rate, resample_rate)

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    task_inputs = []

    for i in [1,3]:
        subgroup_dir = os.path.join(src_root, f"Subgroup_{i}")
        for sub_id in os.listdir(subgroup_dir):
            sig_path = os.path.join(subgroup_dir, sub_id, f"{sub_id}.rec")
            ano_path = os.path.join(subgroup_dir, sub_id, f"{sub_id}_1.txt") # Anotation from 1st expert
            sub_id = f"Subgroup_{i}-{sub_id}"
            task_inputs.append((sub_id, sig_path, ano_path))

    subgroup_dir = os.path.join(src_root, f"Subgroup_2")
    for sub_id in os.listdir(subgroup_dir):        
        # Session 1
        sig_path_1 = os.path.join(subgroup_dir, sub_id, "1", "1.rec")
        ano_path_1 = os.path.join(subgroup_dir, sub_id, "1", "1_1.txt") # Anotation from 1st expert
        sub_id_1 = f"Subgroup_2-{sub_id}-S1"
        task_inputs.append((sub_id_1, sig_path_1, ano_path_1))
        # Session 2
        sig_path_2 = os.path.join(subgroup_dir, sub_id, "2", "2.rec")
        ano_path_2 = os.path.join(subgroup_dir, sub_id, "2", "2_1.txt") # Anotation from 1st expert
        sub_id_2 = f"Subgroup_2-{sub_id}-S2"
        task_inputs.append((sub_id_2, sig_path_2, ano_path_2))

    task_inputs = [task_input for task_input in task_inputs if task_input[0] not in SUB_REMOVE]

    # with Pool(num_processes) as p:
    #     p.starmap(single_process, task_inputs)

    with Pool(num_processes) as pool:
        with tqdm(total=len(task_inputs), desc="Processing ISRUC Dataset") as pbar:
            for args in task_inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def pro_special():
    # print('Subgroup_1-40')

    sig_path = os.path.join(src_root, "Subgroup_1", "40/40.rec")
    ano_path = os.path.join(src_root, "Subgroup_1", "40/40_1.txt") # Anotation from 1st expert
    sub_id = "Subgroup_1-40"
  
    shutil.copyfile(sig_path, sig_path.replace('.rec', '.edf'))
    sig_path = sig_path.replace('.rec', '.edf')
    sig_raw = read_raw_edf(sig_path, include=('ROC', 'A1', 'LOC', 'A2', 'C4', 'O2', 'C3', 'O1', 'F4', 'F3'), verbose=False)
    sig_data = sig_raw.to_data_frame().to_numpy()   # (TN, C)
    sample_rate = 1/(sig_data[1,0]-sig_data[0,0])
    sig = sig_data[:, 1:]
    F3A2 = sig[:, 9] - sig[:, 3]
    F4A1 = sig[:, 8] - sig[:, 1]
    C3A2 = sig[:, 6] - sig[:, 3]
    C4A1 = sig[:, 4] - sig[:, 1]
    O1A2 = sig[:, 7] - sig[:, 3]
    O2A1 = sig[:, 5] - sig[:, 1]
    E1A2 = sig[:, 2] - sig[:, 3]
    E2A1 = sig[:, 0] - sig[:, 1]
    sig = np.stack([F3A2, F4A1, C3A2, C4A1, O1A2, O2A1, E1A2, E2A1], axis=-1)
    ch_id = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    os.remove(sig_path)

    ano = load_ano(ano_path)

    sig = pre_process(sig, sample_rate, resample_rate)

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/ISRUC/unrar/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/ISRUC/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = ['Subgroup_1-40', ]

    channel_id = {
        'F3-M2': 0, 'F4-M1': 1, 'C3-M2': 2, 'C4-M1': 3, 'O1-M2': 4, 'O2-M1': 5, 'E1-M2': 6,  'E2-M1': 7,
        'F3-A2': 0, 'F4-A1': 1, 'C3-A2': 2, 'C4-A1': 3, 'O1-A2': 4, 'O2-A1': 5, 'LOC-A2': 6, 'ROC-A1': 7,
    }

    run(100)

    pro_special()

    formatting_check(dst_root)
