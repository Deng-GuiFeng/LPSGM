# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
import pandas as pd
from multiprocessing import Pool

from utils import *


def process_recording(sub_id, eeg_edf_path, eog_edf_path, ano_xlsx_path, start_epoch, end_epoch):
    # print(sub_id)

    sig_eeg, ch_id_eeg, _, sample_rate, _ = load_sig(eeg_edf_path, channel_id)
    sig_eog, ch_id_eog, _, sample_rate, _ = load_sig(eog_edf_path, channel_id)
    sig = np.concatenate((sig_eeg, sig_eog), axis=1)
    ch_id = np.concatenate((ch_id_eeg, ch_id_eog))

    ano = pd.read_excel(ano_xlsx_path, header=None).iloc[:, 0].tolist()

    EpochTN = sample_rate * 30
    EpochN = sig.shape[0] // EpochTN
    sig = sig[:EpochN*EpochTN, :].reshape((EpochN, EpochTN, 8))

    sig = sig[start_epoch +1:end_epoch]    # Fix the mismatch error in original data
    ano = ano[start_epoch:end_epoch -1]    

    sig = pre_process(sig.reshape(-1, 8), sample_rate, resample_rate)  # (EpochN, 3000, ChN)
    ano = np.array([sleepstage[a] for a in ano], dtype=np.int64)     # (EpochN, )

    sig_list, ano_list = rm_unknown_label(sig, ano)

    save(dst_root, sub_id, sig_list, ano_list, ch_id)


def single_process(*args):
    try:
        process_recording(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")


def run(num_processes):
    Inputs = []

    healthy_xlsx = os.path.join(src_root, "健康被试数据-edf", "健康被试-整夜起止epoch统计.xlsx")
    for healthy_idx in range(80):
        sub_id = f"健康被试-{healthy_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(healthy_xlsx, header=None).iloc[healthy_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "健康被试数据-edf", "EEG-健康-edf", f"{healthy_idx+1}.edf")
        eog_path = os.path.join(src_root, "健康被试数据-edf", "EOG-健康-edf", f"{healthy_idx+1}.edf")
        ano_path = os.path.join(src_root, "健康被试数据-edf", "标签-健康", f"{healthy_idx+1}.xlsx")

        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    depressed_xlsx = os.path.join(src_root, "抑郁患者数据-edf", "抑郁-开关灯时间统计.xlsx")
    for depressed_idx in range(24):
        sub_id = f"抑郁患者-{depressed_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(depressed_xlsx, header=None).iloc[depressed_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "抑郁患者数据-edf", "EEG-抑郁-edf", f"{depressed_idx+1}.edf")
        eog_path = os.path.join(src_root, "抑郁患者数据-edf", "EOG-抑郁-edf", f"{depressed_idx+1}.edf")
        ano_path = os.path.join(src_root, "抑郁患者数据-edf", "标签-抑郁", f"{depressed_idx+1}.xlsx")
        
        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    # multi process
    # with Pool(num_processes) as task_pool:
    #     task_pool.starmap(single_process, Inputs)

    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing SYSU Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()



def test():

    Inputs = []

    healthy_xlsx = os.path.join(src_root, "健康被试数据-edf", "健康被试-整夜起止epoch统计.xlsx")
    for healthy_idx in range(80):
        sub_id = f"健康被试-{healthy_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(healthy_xlsx, header=None).iloc[healthy_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "健康被试数据-edf", "EEG-健康-edf", f"{healthy_idx+1}.edf")
        eog_path = os.path.join(src_root, "健康被试数据-edf", "EOG-健康-edf", f"{healthy_idx+1}.edf")
        ano_path = os.path.join(src_root, "健康被试数据-edf", "标签-健康", f"{healthy_idx+1}.xlsx")

        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))


    depressed_xlsx = os.path.join(src_root, "抑郁患者数据-edf", "抑郁-开关灯时间统计.xlsx")
    for depressed_idx in range(24):
        sub_id = f"抑郁患者-{depressed_idx+1}"
        (start_epoch, end_epoch) = pd.read_excel(depressed_xlsx, header=None).iloc[depressed_idx + 2, 1:3].tolist()
        eeg_path = os.path.join(src_root, "抑郁患者数据-edf", "EEG-抑郁-edf", f"{depressed_idx+1}.edf")
        eog_path = os.path.join(src_root, "抑郁患者数据-edf", "EOG-抑郁-edf", f"{depressed_idx+1}.edf")
        ano_path = os.path.join(src_root, "抑郁患者数据-edf", "标签-抑郁", f"{depressed_idx+1}.xlsx")
        
        Inputs.append((sub_id, eeg_path, eog_path, ano_path, start_epoch, end_epoch))

    for args in Inputs:
        process_recording(*args)


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/private_datasets/SYSU/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/SYSU/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    sleepstage = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'U': 9, '?': 9}

    channel_id = {
        'F3-M2': 0, 'F4-M1': 1, 'C3-M2': 2, 'C4-M1': 3, 'O1-M2': 4, 'O2-M1': 5, 'E1-M2': 6, 'E2-M1': 7,
        'F3-M1': 0, 'F4-M2': 1, 'C3-M1': 2, 'C4-M2': 3, 'O1-M1': 4, 'O2-M2': 5, 'E1-M1': 6, 'E2-M2': 7,
    }
       
    run(104)

    formatting_check(dst_root)

    # test()

