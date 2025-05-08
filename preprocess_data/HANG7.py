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


def process_recording(sub_id, edf_path, xml_path, txt_path):
    # print(sub_id)

    sig, ch_id, meas_date, sample_rate, select_ch_names = load_sig(edf_path, channel_id)  # (TN, cn)
    # ano = read_xml(xml_path)  # error in sub_id: "20220128-T2-韦又丹"
    ano = read_txt(txt_path)

    sig = pre_process(sig, sample_rate, resample_rate)  # (N, 3000, cn)

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
    Inputs = []
    for group in groups:
        group_dir = os.path.join(src_root, group)
        for sub_id in os.listdir(group_dir):
            if sub_id in SUB_REMOVE:
                continue
            edf_path = os.path.join(group_dir, sub_id, sub_id+".edf")
            xml_path = os.path.join(group_dir, sub_id, sub_id+".edf.XML")
            txt_path = os.path.join(group_dir, sub_id, sub_id+".txt")
            Inputs.append((sub_id, edf_path, xml_path, txt_path))

    # with Pool(num_processes) as task_pool:
    #     task_pool.starmap(single_process, Inputs)

    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing HANG7 Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    subjects = []
    for group in groups:
        group_dir = os.path.join(src_root, group)
        for sub_id in os.listdir(group_dir):
            if sub_id in SUB_REMOVE:
                continue
            edf_path = os.path.join(group_dir, sub_id, sub_id+".edf")
            xml_path = os.path.join(group_dir, sub_id, sub_id+".edf.XML")
            txt_path = os.path.join(group_dir, sub_id, sub_id+".txt")
            subjects.append((sub_id, edf_path, xml_path, txt_path))
    
    for (sub_id, edf_path, xml_path, txt_path) in subjects:
        process_recording(sub_id, edf_path, xml_path, txt_path)


if __name__ == '__main__':
    src_root = r"/nvme1/denggf/PSG_datasets/private_datasets/HQ_collation/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/HANG7/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    groups = (
        "depression",
        "healthy_control",
        "narcolepsy",
    )

    channel_id = {
        'F3-M2': 0, 'F4-M1': 1, 'C3-M2': 2, 'C4-M1': 3,
        'O1-M2': 4, 'O2-M1': 5, 'E1-M2': 6, 'E2-M2': 7,
        'F3': 0, 'F4': 1, 'C3': 2, 'C4': 3,
        'O1': 4, 'O2': 5, 'E1': 6, 'E2': 7,
        'M1': -1, 'M2': -2,
    }

    run(127)

    formatting_check(dst_root)

    # test()