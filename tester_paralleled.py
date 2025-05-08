import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import sys
import time
import re

from model.model import LPSGM
from utils import *
from data.dataset import *


warnings.filterwarnings("ignore", category=UserWarning)



class TestDataset(Dataset):
    def __init__(self,
                 x: list,
                 channels: tuple,):
        super(TestDataset).__init__()

        self.x = x
        self.channels = channels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        npz = np.load(self.x[index])
        # seq
        seq = npz['seq'].astype(np.float32)   # (seql, cn, 3000)
        seql, cn = seq.shape[:2]
        # seq_idx
        seq_idx = np.arange(seql).reshape(seql, 1)
        seq_idx = np.tile(seq_idx, cn)  # (seql, cn)
        # ch_idx
        ch_idx = npz['ch_id'].reshape(1, cn)
        ch_idx = np.tile(ch_idx, (seql, 1)) # (seql, cn)
        
        seq = np.reshape(seq, (seql*cn, -1))
        seq_idx = np.reshape(seq_idx, (seql*cn, ))
        ch_idx = np.reshape(ch_idx, (seql*cn, ))

        return seq, seq_idx, ch_idx


class TestDataLoader(DataLoader):
    def __init__(self,
                 subjects: list,
                 batch_size: int,
                 num_workers: int,
                 num_processes: int,
                 cache_root: str,
                 channels: dict,
                 seq_len: int,
                 ):
        
        self.subjects = subjects
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_processes = num_processes
        self.cache_root = cache_root
        self.channels = channels
        self.seq_len = seq_len

        self.retrive_logits = []


    def get_dataloader(self):
        samples_path = self.cache_data()
        dataset = TestDataset(x=samples_path,
                                   channels=self.channels,
                                   )
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                collate_fn = self.collate_func,
                                drop_last=False,
                                )
        
        return dataloader
    

    def collate_func(self, batch):
        # Separate sequences and labels
        sequences, seq_idxs, ch_idxs = zip(*batch)

        # Pad sequences
        padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
        # Pad sequence's index
        padded_seq_idxs = pad_sequence([torch.tensor(seq_idx) for seq_idx in seq_idxs], batch_first=True, padding_value=0)
        # Pad channel's index
        padded_ch_idxs = pad_sequence([torch.tensor(ch_idx) for ch_idx in ch_idxs], batch_first=True, padding_value=0)
        
        # Generate padding mask
        padding_masks = [torch.zeros((seq.shape[0],), dtype=torch.bool) for seq in sequences]
        padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=True)

        ori_lens = [seq.shape[0] for seq in sequences]

        return padded_sequences, padded_seq_idxs, padded_ch_idxs, padding_masks, ori_lens


    def clear_cache(self):
        shutil.rmtree(self.cache_root, 
                      ignore_errors=True)
        
    
    def cache_subject(self, subject: str):

        cache_subject_seq_dir = os.path.join(self.cache_root, os.path.split(subject)[-1], 'seq')
        cache_subject_label_dir = os.path.join(self.cache_root, os.path.split(subject)[-1], 'labels')
        os.makedirs(cache_subject_seq_dir, exist_ok=True)
        os.makedirs(cache_subject_label_dir, exist_ok=True)

        data_pairs = []
        for seq_name in os.listdir(subject):
            seq_path = os.path.join(subject, seq_name)
            npz = np.load(seq_path)
            seq_data, labels_data, ch_id = npz['sig'].astype(np.float32), npz['ano'].astype(np.int64), npz['ch_id'].astype(np.int64)
            # (L, 3000, cn), (L, ), (cn, )
            channel_indices = np.where(np.isin(ch_id, self.channels))[0]
            seq_data = seq_data[:, :, channel_indices]  # (L, 3000, cn')
            ch_id = ch_id[channel_indices]  # (cn', )
            data_pairs.append((seq_data, labels_data))

        sample_x = []
        for i, (seq_data, labels_data) in enumerate(data_pairs):
            if seq_data.shape[0] < self.seq_len:
                continue
            seq_data = seq_data.transpose(0, 2, 1)  # (L, cn', 3000)
            seq_batch = np.stack([seq_data[i:i+self.seq_len] for i in range(len(seq_data)-self.seq_len+1)], axis=0)  # (SeqN, seq_len, cn', 3000)
            splited_seq = np.split(seq_batch, len(seq_batch), axis=0)
            sample_x.extend(splited_seq)
            np.save(os.path.join(cache_subject_label_dir, f"{i}.npy"), 
                    labels_data)

        for i, x in enumerate(sample_x):
            np.savez(os.path.join(cache_subject_seq_dir, f"{i}.npz"), 
                    seq=x.squeeze(0), ch_id=ch_id)
            
    
    def cache_data(self):
        self.clear_cache()

        with Pool(self.num_processes) as p:
            p.map(self.cache_subject, self.subjects)
               
        # return samples path
        samples_path = []
        for subject in self.subjects:
            cache_subject_seq_dir = os.path.join(self.cache_root, os.path.split(subject)[-1], 'seq')
            if os.path.exists(cache_subject_seq_dir) == False:
                continue
            for i in range(len(os.listdir(cache_subject_seq_dir))):
                seq_path = os.path.join(cache_subject_seq_dir, f"{i}.npz")
                samples_path.append(seq_path)
        return samples_path


    def retrive_data(self, logit):
        self.retrive_logits.append(logit)


    def get_metrics(self, verbose=False, save_pred=False, save_dir=None):

        def sequence_voting(logits, N):
            # logits: (seqN=N-seql+1, seql, 5)
            voted_logits = np.zeros((N, 5))
            for i in range(len(logits)):
                logits_i = logits[i]
                for j in range(self.seq_len):
                    # if i+j < N:
                    voted_logits[i+j] += logits_i[j]
            voted_predicts = np.argmax(voted_logits, axis=-1)   # (N, )
            return voted_predicts.tolist()

      
        PREDICTS, LABELS = [], []
        idx = 0
        self.retrive_logits = np.concatenate(self.retrive_logits, axis=0)
        for subject in self.subjects:
            cache_subject_label_dir = os.path.join(self.cache_root, os.path.split(subject)[-1], 'labels')
            if os.path.exists(cache_subject_label_dir) == False:
                continue
            predicts, labels = [], []
            label_files = {os.path.basename(f).split('.')[0]: f for f in os.listdir(cache_subject_label_dir)}
            label_files = sorted(label_files.items(), key=lambda x: int(x[0]))
            for (_, label_name) in label_files:
                label_path = os.path.join(cache_subject_label_dir, label_name)
                if os.path.exists(label_path) == False:
                    continue
                label = np.load(label_path)
                
                N_label = label.shape[0]
                N_logit = N_label - self.seq_len + 1
                logit = self.retrive_logits[idx:idx+N_logit]
                idx += N_logit

                predict = sequence_voting(logit, N_label)

                predicts.extend(predict)
                labels.extend(label.tolist())

            if verbose:
                predicts = np.array(predicts).squeeze()
                labels = np.array(labels).squeeze()

                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = get_metric(labels, predicts)

                if logger_type == 'txt':
                    # txt logger
                    print("="*50)
                    print(f"{subject}:\nacc: {acc}, f1: {f1} \ncm:\n{cm} \nwake_f1: {wake_f1}, n1_f1: {n1_f1}, n2_f1: {n2_f1}, n3_f1: {n3_f1}, rem_f1: {rem_f1}, kappa: {kappa}")
                else:
                    # csv logger
                    print(f"{subject.split('/')[-1]}, {acc}, {f1}, {kappa}, {wake_f1}, {n1_f1}, {n2_f1}, {n3_f1}, {rem_f1}")

            if save_pred:
                predicts = np.array(predicts).squeeze()
                labels = np.array(labels).squeeze()
                save_sub_dir = os.path.join(save_dir, os.path.basename(subject).split('.')[0])
                os.makedirs(save_sub_dir, exist_ok=True)
                np.save(os.path.join(save_sub_dir, "predicts.npy"), predicts)
                np.save(os.path.join(save_sub_dir, "labels.npy"), labels)

            PREDICTS.extend(predicts)
            LABELS.extend(labels)

        assert idx == len(self.retrive_logits), "Error: idx != len(self.retrive_logits)"

        PREDICTS = np.array(PREDICTS).squeeze()
        LABELS = np.array(LABELS).squeeze()
        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = get_metric(LABELS, PREDICTS)
        return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa


@torch.no_grad()
def Tester(args, model, channels=None, logger_type='txt', verbose=False):
    if channels is None:
        channels = range(args.ch_num)

    # log test subjects
    print(f"Test Subjects: {len(args.test_subjects)}")
    print("\n"*10, "Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    start_time = time.time()
    
    model.eval()

    dataloader = TestDataLoader(
        subjects = [sub for (sub, _) in args.test_subjects],
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        num_processes = args.num_processes,
        cache_root = args.cache_root,
        channels = channels,
        seq_len = args.seq_len,
    )
    dataset = dataloader.get_dataloader()

    for x, seq_idx, ch_idx, mask, ori_len in tqdm(dataset):
        logits = model(x, mask, ch_idx, seq_idx, ori_len)   # (SeqN, seql, 5)
        logits = torch.softmax(logits, dim=-1)              # (SeqN, seql, 5)
        logits = logits.cpu().numpy()

        dataloader.retrive_data(logits)

    acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = \
        dataloader.get_metrics(verbose, args.save_pred, args.save_dir)
    
    dataloader.clear_cache()

    end_time = time.time()

    if logger_type == 'txt':
        # txt logger
        print('\n'*3, "="*30, "Inference", "="*30)
        print(f"Channels: {channels}")
        print("acc: {:.5f}, f1: {:.5f}, kappa: {:.5f}".format(acc, f1, kappa))
        print("cm: \n{}".format(cm))
        print("wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
            wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,
        ))
        print(f"Time: {(end_time-start_time)/len(args.test_subjects):.2f} s")
        print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(acc, f1, kappa, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,))

    else:
        # csv logger
        print(f"\nOverall, {acc}, {f1}, {kappa}, {wake_f1}, {n1_f1}, {n2_f1}, {n3_f1}, {rem_f1}")

    return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa


def pt_test(center: str, 
            weights_file: str,
            channels: tuple,
            verbose=False,
            logger_type='txt',
        ):
    
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.cache_root=os.path.join(args.cache_root, current_time)

    if center == "HANG7":
        subjects, subtypes = get_HANG7_subjects()
    elif center == "SYSU":
        subjects, subtypes = get_SYSU_datasets()

    args.test_subjects = [(sub, center) for sub in subjects]

    model = nn.DataParallel(LPSGM(args)).cuda()
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])

    print(f"Model Weights: {weights_file}")

    Tester(args, model, 
        channels, 
        logger_type=logger_type,
        verbose = verbose,
    )


def ft_test(center: str, kfolds: int, n_fold: int, seed: int,
            weights_file: str,
            channels: tuple,
            verbose=False,
            logger_type='txt',
        ):
    
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.cache_root=os.path.join(args.cache_root, current_time)

    if center == "HANG7":
        subjects, subtypes = get_HANG7_subjects()
    elif center == "SYSU":
        subjects, subtypes = SYSU_StratifiedKFold()

    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    for i, (_, test_idx) in enumerate(skf.split(subjects, subtypes)):
        if i == n_fold:
            break

    args.test_subjects = [(subjects[idx], center) for idx in test_idx]

    model = nn.DataParallel(LPSGM(args)).cuda()
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])

    print(f"Model Weights: {weights_file}")

    Tester(args, model, 
        channels, 
        logger_type=logger_type,
        verbose = verbose,
    )


if __name__ == "__main__":

    class args(object):
        architecture = 'cat_cls'
        epoch_encoder_dropout = 0
        transformer_num_heads = 8
        transformer_dropout = 0
        transformer_attn_dropout = 0
        ch_num = 8
        seq_len = 20
        ch_emb_dim =32
        seq_emb_dim = 64
        num_transformer_blocks = 4
        clamp_value = 10

        batch_size = 128
        num_workers = 128
        num_processes = 100
        cache_root = r"/nvme2/denggf/cache/"

        save_pred = True
        save_dir = r"/home/denggf/Desktop/LPSGM-main/run/Mar05_13-38-06/predicts/SYSU"
    

################################ Pretrained Test ################################
    logger_type = 'txt'
    center = "SYSU"
    channels = (0, 1, 2, 3, 4, 5, 6, 7)

    run_name = r"/home/denggf/Desktop/LPSGM-main/run/Mar05_13-38-06"
    weights_file = os.path.join(run_name, 'model_dir/eval_best.pth') 
    sys.stdout = open( os.path.join(run_name, f'{center}_metrics'), 'w')

    pt_test(center = center,
            weights_file = weights_file,
            channels = channels,
            verbose = True,
            logger_type=logger_type,
        )




# ################################ Finetuned Test ################################
#     logger_type = 'txt'
#     center = "HANG7"
#     channels = (0, 1, 2, 3, 4 ,5, 6, 7)

#     kfolds = 5
#     seed = 666
#     task_dir = r"/home/denggf/Desktop/LPSGM/run/HQFT"

#     run_name_list = [os.path.join(task_dir, f"fold{i}", run_name) for i in range(kfolds) for run_name in os.listdir(os.path.join(task_dir, f"fold{i}"))]
    
#     for run_name in run_name_list:

#         n_fold = int(re.search(r'fold(\d+)', run_name).group(1))
#         weights_file = os.path.join(run_name, 'model_dir/eval_best.pth') 
#         sys.stdout = open( os.path.join(run_name, f'{center}_metrics'), 'w')

#         ft_test(center = center,
#                 kfolds = kfolds,
#                 n_fold = n_fold,
#                 seed = seed,
#                 weights_file = weights_file,
#                 channels = channels,
#                 verbose = True,
#                 logger_type=logger_type,
#             )



