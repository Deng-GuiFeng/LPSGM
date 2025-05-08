import os
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from mne.io import read_raw_edf


def find_files_suffix(root, suffix):
    files = []
    for root, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(suffix):
                files.append(os.path.join(root, f))
    return files


def check_nan_and_inf(data):
    arr = np.array(data)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    return has_nan, has_inf


def formatting_check(dataset_root):
    CH_COUNT = {}
    ClassNum = [0]*5
    SUM = 0
    subjects = os.listdir(dataset_root)
    print(f"Total Subjects: {len(subjects)}")

    for sub_id in tqdm(os.listdir(dataset_root)):
        sub_dir = os.path.join(dataset_root, sub_id)
        
        if len(os.listdir(sub_dir)) == 0:
            print(sub_id, "empty error")
            # shutil.rmtree(sub_dir)
            continue

        for seq_id in os.listdir(sub_dir):
            seq_path = os.path.join(sub_dir, seq_id)

            npz = np.load(seq_path)
            sig = npz['sig']
            ano = npz['ano']
            ch_id = npz['ch_id']

            has_nan, has_inf = check_nan_and_inf(sig)
            if has_nan or has_inf:
                print(sub_id, seq_id, "nan or inf error")
            
            # check shape
            EpochN1, TN, ChN1 = sig.shape
            EpochN2 = ano.shape[0]
            ChN2 = len(ch_id)

            if any([EpochN1!=EpochN2, TN!=3000, ChN1!=ChN2]):
                print(sub_id, seq_id, "shape error", sig.shape, ano.shape, ch_id.shape)

            SUM += EpochN2
            for i in range(5):
                ClassNum[i] += np.sum(ano == i)
            
            # check classes
            ClassSet = set(ano.tolist())
            if  not ClassSet.issubset({0,1,2,3,4}):
                print(sub_id, "classes error", ClassSet)

            # log channel count
            CH_COUNT[ChN1] = CH_COUNT.get(ChN1, 0) + 1

    ClassRatio = [round(100 *ClassNum[c]/SUM, 1) for c in range(5)]
    print(f"Epochs Count: {SUM}, {ClassNum}")
    print(f"Epochs Ratio: {ClassRatio}")

    r_CalssNum = [1 / ClassNum[c] for c in range(5)]
    ClassWeight = r_CalssNum / np.sum(r_CalssNum)
    print(f"Classes Rebalance Weights: {ClassWeight}")

    print("Channel Count:", CH_COUNT)


def info(*mats):
    for mat in mats:
        print(mat.shape, mat.dtype, mat.min(), mat.max())


def pre_process(sig, sample_rate, resample_rate=100, norch=False):
    TN, ChN = sig.shape
    # 0.3-35Hz bandpass filter
    b, a = signal.butter(N=4, Wn=[0.3 * 2 / sample_rate, 35 * 2 / sample_rate], btype='bandpass')
    for c in range(ChN):
        sig[:, c] = signal.filtfilt(b, a, sig[:, c])
    
    if norch:
        # 50Hz notch filter
        b_notch, a_notch = signal.iirnotch(w0=50, Q=20, fs=sample_rate)
        for c in range(ChN):
            sig[:, c] = signal.filtfilt(b_notch, a_notch, sig[:, c])

    if resample_rate != sample_rate:
        scaled_TN = round(resample_rate / sample_rate * TN)
        sig_r = np.zeros((scaled_TN, ChN))
        for c in range(ChN):
            sig_r[:, c] = interp1d(np.linspace(0, TN - 1, TN), sig[:, c], kind='linear')(
                np.linspace(0, TN - 1, scaled_TN))
    else:
        scaled_TN = TN
        sig_r = sig
    # Z-Score
    sig_r = (sig_r - np.mean(sig_r, axis=0)) / np.std(sig_r, axis=0)
    EpochN = scaled_TN // 3000
    sig_r = np.reshape(sig_r[:EpochN * 3000, :], (EpochN, 3000, ChN))
    return sig_r


def rm_unknown_label(sig, ano):
    # sig: (N, 3000, cn)    ano: (N, )
    indices = np.where((ano > 4) | (ano < 0))[0]

    sig_list, ano_list = [], []
    start = 0

    for i in indices:
        if i > start:
            sig_list.append(sig[start:i])
            ano_list.append(ano[start:i])
        start = i + 1

    if start < len(ano):
        sig_list.append(sig[start:])
        ano_list.append(ano[start:])

    return sig_list, ano_list


def save(dst_root, sub_id, sig_list, ano_list, ch_id):
    ch_id = ch_id.astype(np.int32)

    dst_sub_dir = os.path.join(dst_root, sub_id)
    os.makedirs(dst_sub_dir, exist_ok=True)

    for i, (sig, ano) in enumerate(zip(sig_list, ano_list)):
        seq_path = os.path.join(dst_sub_dir, f"{sub_id}-s{i}.npz")
        sig, ano = sig.astype(np.float32), ano.astype(np.int32)
        np.savez(seq_path, sig=sig, ano=ano, ch_id=ch_id)
        # info(sig, ano, ch_id)


def load_sig(sig_path, channel_id):
    sig_raw = read_raw_edf(sig_path, include=channel_id.keys(), verbose=False)
    ch_names = sig_raw.ch_names
    sig_data = sig_raw.to_data_frame().to_numpy()  # (TN, C)
    sample_rate = round(1 / (sig_data[1, 0] - sig_data[0, 0]))
    sig = sig_data[:, 1:]
    ch_id = np.array([channel_id[ch] for ch in ch_names], dtype=np.int32)

    # subtract M1 and M2
    sig, ch_id = subtract(sig, ch_id)

    # get start time
    start_time = sig_raw.info['meas_date']
    start_time = start_time.replace(tzinfo=None)    # remove timezone

    return sig, ch_id, start_time, sample_rate, ch_names


def subtract(sig, ch_id):
    # sig: (TN, C)  ch_id: (C, )
    if -1 in ch_id and -2 in ch_id:
        M1_idx = int(np.where(ch_id == -1)[0])
        M2_idx = int(np.where(ch_id == -2)[0])
        for i in range(sig.shape[1]):
            if ch_id[i] % 2 == 0:  # right
                sig[:, i] -= sig[:, M1_idx]
            else:  # left
                sig[:, i] -= sig[:, M2_idx]
        # remove M1 and M2
        sig = np.delete(sig, [M1_idx, M2_idx], axis=1)
        ch_id = np.delete(ch_id, [M1_idx, M2_idx])
    return sig, ch_id


def remove_wake_start_end(sig, ano, duration=30):
    non_wake_indices = np.where(ano != 0)[0]
    
    if len(non_wake_indices) == 0:
        return None, None
    
    first_non_wake = non_wake_indices[0]
    last_non_wake = non_wake_indices[-1]
    
    start = max(0, first_non_wake - duration*2)
    end = min(len(ano), last_non_wake + duration*2 + 1) 
    
    if start >= end:
        return None, None
    
    return sig[start:end], ano[start:end]

