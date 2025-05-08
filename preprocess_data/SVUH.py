# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from multiprocessing import Pool

from utils import *


def load_ano(ano_path):
    with open(ano_path, "r")as f:
        lines = f.readlines()
    sleep_stages = [int(l.strip()) for l in lines]
    for i, ss in enumerate(sleep_stages):
        if  ss == 4:
            sleep_stages[i] = 3
        if ss == 5:
            sleep_stages[i] = 4
    ano = np.array(sleep_stages, dtype=np.int32)
    return ano


def process_recording(sub_id):
    # print(sub_id)

    rec_path = os.path.join(src_root, sub_id+".rec")
    edf_path = rec_path.replace('.rec', '.edf')
    shutil.copyfile(rec_path, edf_path)
    sig, ch_id, start_time, sample_rate, ch_names = load_sig(edf_path, channel_id)
    os.remove(edf_path)
      
    txt_path = os.path.join(src_root, sub_id+"_stage.txt")

    ano = load_ano(txt_path)

    sig = pre_process(sig, sample_rate, resample_rate)

    sig = sig[: len(ano)]   

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    subjects = set([f.split(".")[0] for f in os.listdir(src_root) if f.endswith('.rec')]) - set(SUB_REMOVE)

    # with Pool(num_processes) as p:
    #     p.map(single_process, subjects)

    with Pool(num_processes) as p:
        for _ in tqdm(p.imap_unordered(single_process, subjects), total=len(subjects), desc="Processing SVUH Dataset"):
            pass


def test():
    subjects = set([f.split(".")[0] for f in os.listdir(src_root) if f.endswith('.rec')]) - set(SUB_REMOVE)
    for sub_id in subjects:
        process_recording(sub_id)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/SVUH/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/SVUH/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {'C3A2': 2, 'C4A1': 3, 'Lefteye': 6, 'RightEye': 7}
    
    run(100)

    formatting_check(dst_root)

    # test()
