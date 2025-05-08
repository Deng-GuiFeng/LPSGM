# -*- coding: utf-8 -*-
import numpy as np
import os
from mne.io import read_raw_edf
from scipy import signal
from scipy.interpolate import interp1d
import shutil
from bs4 import BeautifulSoup
from multiprocessing import Pool

from utils import *


def load_ano(ano_path):
    stage_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4, '6': 6}

    sleep_stages = BeautifulSoup(open(ano_path), features="xml").find_all('SleepStage')
    sleep_stages = [stage_dict[sleep_stage.get_text()] for sleep_stage in sleep_stages]

    sleep_stages = np.array(sleep_stages, dtype=np.int32)
    return sleep_stages


def process_recording(sub_id, sig_path, ano_path):
    # print(sub_id)

    sig, ch_id, start_time, sample_rate, ch_names = load_sig(sig_path, channel_id)
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
    for group in os.listdir(edf_root):
        edf_group_dir = os.path.join(edf_root, group)
        ano_group_dir = os.path.join(ano_root, group)

        for file in os.listdir(edf_group_dir):
            sub_id = file.split('.')[0]
            edf_path = os.path.join(edf_group_dir, sub_id+".edf")
            ano_path = os.path.join(ano_group_dir, sub_id+"-profusion.xml")
            Inputs.append((sub_id, edf_path, ano_path))

    Inputs = [subject for subject in Inputs if subject[0] not in SUB_REMOVE]

    # with Pool(num_processes) as p:
    #     p.starmap(single_process, subjects)
    
    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing HOMEPAP Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    pass

if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/homepap/polysomnography/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/HOMEPAP/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = ["homepap-lab-full-1600047", "homepap-lab-full-1600197"]

    channel_id = {
        'F3': 0, 'F3-M2': 0, 
        'F4': 1, 'F4-M1': 1, 
        'C3-M2': 2, 'C3': 2, 
        'C4': 3, 'C4-M1': 3, 
        'O1': 4, 'O1-M2': 4, 
        'O2-M1': 5, 'O2': 5, 
        'E1-E2': 6, 'E-1': 6, 'L-EOG': 6, 'LOC': 6, 'E1': 6, 'E1-M2': 6, 
        'R-EOG': 7, 'E-2': 7, 'E2': 7, 'E2-M1': 7, 'ROC': 7,
        'M1': -1, 'A1': -1, 
        'M2': -2, 'A2': -2, 
    }
    
    edf_root = os.path.join(src_root, "edfs/lab/")
    ano_root = os.path.join(src_root, "annotations-events-profusion/lab/")

    run(100)

    formatting_check(dst_root)

    # test()

