import os
import numpy as np
from bs4 import BeautifulSoup
import shutil
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
    ano = np.array(sleep_stages, dtype=np.int32)

    return ano


def process_recording(sub_id, sig_path, ano_path):
    # print(sub_id)

    sig, ch_id, meas_date, sample_rate, select_ch_names = load_sig(sig_path, channel_id)  # (TN, cn)
    ano = load_ano(ano_path)    

    sig = pre_process(sig, sample_rate, resample_rate)  # (N, 3000, cn)

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
        edf_group = os.path.join(edf_root, group)
        xml_group = os.path.join(xml_root, group)

        for sub_id in os.listdir(edf_group):
            sub_id = sub_id.split(".")[0]
            if sub_id in SUB_REMOVE:
                continue
            edf_path = os.path.join(edf_group, sub_id+".edf")
            xml_path = os.path.join(xml_group, sub_id+"-profusion.xml")

            Inputs.append((sub_id, edf_path, xml_path))

    with Pool(num_processes) as pool:
        with tqdm(total=len(Inputs), desc="Processing ABC Dataset") as pbar:
            for args in Inputs:
                pool.apply_async(single_process, args=args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()


def test():
    pass


if __name__ == "__main__":
    src_root = r"/nvme1/denggf/PSG_datasets/public_datasets/abc/"
    dst_root = r"/nvme1/denggf/SleepStagingProcessedData/ABC/"
    shutil.rmtree(dst_root, ignore_errors=True)

    resample_rate = 100

    SUB_REMOVE = []

    channel_id = {
        'F3': 0, 'F4': 1, 'C3': 2, 'C4': 3, 'O1': 4, 'O2': 5, 'E1': 6, 'E2': 7, 'M1': -1, 'M2': -2, 
    }

    edf_root = os.path.join(src_root, "polysomnography/edfs/")
    xml_root = os.path.join(src_root, "polysomnography/annotations-events-profusion/")

    run(100)

    formatting_check(dst_root)

    # test()



