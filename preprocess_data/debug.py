import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from utils import *


root = r"/nvme1/denggf/SleepStagingProcessedData/NCHSDB/"

def single_process(npz_file):
    npz = np.load(npz_file)
    sig = npz["sig"]
    # ano = npz["ano"]
    # ch_id = npz["ch_id"]
    has_nan, has_inf = check_nan_and_inf(sig)
    if has_nan or has_inf:
        print(npz_file)


npz_files = find_files_suffix(root, ".npz")
with Pool(200) as p:
    for _ in tqdm(p.imap_unordered(single_process, npz_files), total=len(npz_files), desc=""):
        pass
