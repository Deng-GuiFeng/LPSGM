import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import shutil
from multiprocessing import Pool

from data.augmentation import *
from data.dataset import *


def cache_subject(sub_dir: str, domain: str, set_id: str, set_root:str, seq_len: int, shift_len: int):
    
    data_pairs = []
    for seq_name in os.listdir(sub_dir):
        seq_path = os.path.join(sub_dir, seq_name)
        npz = np.load(seq_path)
        sig = npz['sig'].astype(np.float32) # (L, 3000, cn)
        ano = npz['ano'].astype(np.int64) # (L, )
        ch_id = npz['ch_id'].astype(np.int64) # (cn, )
        data_pairs.append((sig, ano, ch_id))
    
    SEQ, LABELS, CH_ID = [], [], []
    for (seq_data, labels_data, ch_id) in data_pairs:
        if set_id == 'train':
            seq_data = random_temporal_shift(seq_data, shift_len).transpose(0, 2, 1) # (L, cn, 3000)
            splited_seq, splited_labels = random_split_sample(seq_data, labels_data, random=True, seq_len = seq_len)
        elif set_id == 'eval' or set_id == 'test':
            seq_data = seq_data.transpose(0, 2, 1)
            splited_seq, splited_labels = random_split_sample(seq_data, labels_data, random=False, seq_len = seq_len)
        SEQ.extend(splited_seq)
        LABELS.extend(splited_labels)
        CH_ID.extend([ch_id]*len(splited_seq))

    cache_sub_dir = os.path.join(set_root, domain, os.path.split(sub_dir)[-1])
    os.makedirs(cache_sub_dir, exist_ok=True)

    for i, (sig, ano, ch_id) in enumerate(zip(SEQ, LABELS, CH_ID)):
        sample_path = os.path.join(cache_sub_dir, f"{i}.npz")
        np.savez(sample_path, sig=sig, ano=ano, ch_id=ch_id)


def cache_data(train_subjects: list, 
               eval_subjects: list,
               test_subjects: list,
               cache_root: str,
               num_processes: int,
               seq_len: int,
               shift_len: int):
    
    cache_train_root = os.path.join(cache_root, 'train')
    cache_eval_root = os.path.join(cache_root, 'eval')
    cache_test_root = os.path.join(cache_root, 'test')

    if len(train_subjects) > 0:
        train_subjects = [(subject_path, subject_domain, 'train', cache_train_root, seq_len, shift_len) for subject_path, subject_domain in train_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, train_subjects)

    if len(eval_subjects) > 0:
        eval_subjects = [(subject_path, subject_domain, 'eval', cache_eval_root, seq_len, 0) for subject_path, subject_domain in eval_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, eval_subjects)

    if len(test_subjects) > 0:
        test_subjects = [(subject_path, subject_domain, 'test', cache_test_root, seq_len, 0) for subject_path, subject_domain in test_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, test_subjects)

        
class TrainDataset(Dataset):
    def __init__(self, data_pairs: list):
        super(TrainDataset, self).__init__()
        
        self.sample_paths, self.sample_domains = zip(*data_pairs)

    def __len__(self):
        assert len(self.sample_paths) == len(self.sample_domains),\
            'The length of path and domain are not equal'
        return len(self.sample_paths)

    def __getitem__(self, index):
        npz = np.load(self.sample_paths[index])
        # seq
        seq = npz['sig'].astype(np.float32)   # (seql, cn, 3000)
        seql, cn = seq.shape[:2]
        # label
        label = npz['ano'].astype(np.int64)  # (seql, )
        # seq_idx
        seq_idx = np.arange(seql).reshape(seql, 1)
        seq_idx = np.tile(seq_idx, cn)  # (seql, cn)
        # ch_idx
        ch_id = npz['ch_id'].astype(np.int64)  # (cn, )
        ch_idx = ch_id.reshape(1, cn)
        ch_idx = np.tile(ch_idx, (seql, 1)) # (seql, cn)

        seq, _, seq_idx, ch_idx, _ = \
            random_channel_crop(seq, None, seq_idx, ch_idx, None)
        
        seql, cn = seq.shape[:2]
        seq = np.reshape(seq, (seql*cn, -1))
        seq_idx = np.reshape(seq_idx, (seql*cn, ))
        ch_idx = np.reshape(ch_idx, (seql*cn, ))

        return seq, label, seq_idx, ch_idx


class TestDataset(Dataset):
    def __init__(self, data_pairs: list):
        super(TestDataset, self).__init__()
        
        self.sample_paths, self.sample_domains = zip(*data_pairs)

    def __len__(self):
        assert len(self.sample_paths) == len(self.sample_domains),\
            'The length of path and domain are not equal'
        return len(self.sample_paths)

    def __getitem__(self, index):
        npz = np.load(self.sample_paths[index])
        # seq
        seq = npz['sig'].astype(np.float32)   # (seql, cn, 3000)
        seql, cn = seq.shape[:2]
        # label
        label = npz['ano'].astype(np.int64)  # (seql, )
        # seq_idx
        seq_idx = np.arange(seql).reshape(seql, 1)
        seq_idx = np.tile(seq_idx, cn)  # (seql, cn)
        # ch_idx
        ch_id = npz['ch_id'].astype(np.int64)  # (cn, )
        ch_idx = ch_id.reshape(1, cn)
        ch_idx = np.tile(ch_idx, (seql, 1)) # (seql, cn)

        seq = np.reshape(seq, (seql*cn, -1))
        seq_idx = np.reshape(seq_idx, (seql*cn, ))
        ch_idx = np.reshape(ch_idx, (seql*cn, ))

        return seq, label, seq_idx, ch_idx


class ClassifyDataLoader():
    def __init__(self, 
                 train_subjects: list,
                 eval_subjects: list,
                 test_subjects: list,
    
                 seq_len: int,
                 batch_size: int,
                 num_workers: int,
                 num_processes: int,
                 cache_root: str,
                 shift_len: int,
                 ):
        self.train_subjects = train_subjects
        self.eval_subjects = eval_subjects
        self.test_subjects = test_subjects

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_processes = num_processes
        self.cache_root = cache_root
        self.shift_len = shift_len
        
        cache_data(
            [],
            self.eval_subjects,
            self.test_subjects,
            self.cache_root,
            self.num_processes,
            self.seq_len,
            self.shift_len,
        )
        # self.train_pairs = self.get_samples('train')
        self.eval_pairs = self.get_samples('eval')
        self.test_pairs = self.get_samples('test')
        
        
    def get_train_data_loader(self):

        cache_data(self.train_subjects,
                [],
                [],
                self.cache_root,
                self.num_processes,
                self.seq_len,
                self.shift_len)
        self.train_pairs = self.get_samples('train')

        train_set = TrainDataset(self.train_pairs)

        data_loader = DataLoader(
                train_set,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                collate_fn = self.collate_func,
                drop_last=True
            )
        
        return data_loader
    
    def get_eval_data_loader(self):
        eval_set = TestDataset(self.eval_pairs)

        data_loader = DataLoader(
                eval_set,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                collate_fn = self.collate_func,
                drop_last=False
            )
        
        return data_loader
    
    def get_test_data_loader(self):
        test_set = TestDataset(self.test_pairs)

        data_loader = DataLoader(
                test_set,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                collate_fn = self.collate_func,
                drop_last=False,
            )
        
        return data_loader

    def collate_func(self, batch):
        # Separate sequences and labels
        sequences, labels, seq_idxs, ch_idxs = zip(*batch)

        # Pad sequences
        padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
        # Pad labels
        padded_labels = pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=0)
        # Pad sequence's index
        padded_seq_idxs = pad_sequence([torch.tensor(seq_idx) for seq_idx in seq_idxs], batch_first=True, padding_value=0)
        # Pad channel's index
        padded_ch_idxs = pad_sequence([torch.tensor(ch_idx) for ch_idx in ch_idxs], batch_first=True, padding_value=0)
        
        # Generate padding mask
        padding_masks = [torch.zeros((seq.shape[0],), dtype=torch.bool) for seq in sequences]
        padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=True)

        ori_lens = [seq.shape[0] for seq in sequences]

        return padded_sequences, padded_labels, padded_seq_idxs, padded_ch_idxs, padding_masks, ori_lens
    

    def get_samples(self, set_id: str):
        if set_id == 'train':
            set_root = os.path.join(self.cache_root, 'train')
        elif set_id == 'eval':
            set_root = os.path.join(self.cache_root, 'eval')
        elif set_id == 'test':
            set_root = os.path.join(self.cache_root, 'test')

        if not os.path.exists(set_root):
            return []
        
        data_pairs = []
        for domain in os.listdir(set_root):
            for sub_id in os.listdir(os.path.join(set_root, domain)):
                sub_dir = os.path.join(set_root, domain, sub_id)
                for seq_name in os.listdir(sub_dir):
                    sample_path = os.path.join(sub_dir, seq_name)
                    data_pairs.append((sample_path, domain))

        return data_pairs
    

    def clear_cache(self):
        shutil.rmtree(self.cache_root, 
                    ignore_errors=True)








