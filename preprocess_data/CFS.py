# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool

from utils import *


def load_ano(ano_path):
    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')

    for i in range(len(sleep_stages)):
        ss = float(sleep_stages[i].get_text())
        if ss == 4:
            ss = 3
        elif ss == 5:
            ss = 4
        sleep_stages[i] = ss
    ano = np.array(sleep_stages)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    # print(sub_id)

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)   # (TN, C)
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

    Inputs = []

    for sub_id in os.listdir(edf_root):
        sub_id = sub_id.split('.')[0]

        if sub_id in SUB_REMOVE:
                continue
        
        edf_path = os.path.join(edf_root, sub_id+".edf")
        ano_path = os.path.join(ano_root, sub_id+"-profusion.xml")
        Inputs.append((sub_id, edf_path, ano_path))
    
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing CFS Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    pass
        

if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/cfs/polysomnography/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/CFS/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        'C3': 2, 'C4': 3, 'LOC': 6, 'ROC': 7, 'M1': -1, 'M2': -2,
    }

    edf_root = os.path.join(src_root, "edfs/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/")

    run(100)

    formatting_check(dst_root)

    # test()