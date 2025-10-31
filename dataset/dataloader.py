# -*- coding: utf-8 -*-
"""
dataloader.py

This module provides a custom data loading pipeline for the LPSGM project, which focuses on large-scale polysomnography (PSG) data processing for sleep staging and mental disorder diagnosis. It includes functions and classes to cache raw PSG data into processed segments, handle data augmentation and splitting, and efficiently load data batches for training, evaluation, and testing.

Key functionalities:
- Cache raw PSG recordings into fixed-length sequences with optional temporal shifts and channel cropping.
- Support multiprocessing for parallel caching of multiple subjects.
- Define PyTorch Dataset subclasses for training and testing datasets with appropriate data formatting.
- Provide a unified DataLoader wrapper class to generate PyTorch DataLoader instances with custom collation and padding.
- Manage domain information and channel indexing for flexible multi-channel PSG data handling.

This module is critical for preparing and feeding PSG data into the LPSGM model during different phases of model development and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import shutil
from multiprocessing import Pool

from dataset.augmentation import *
from dataset.dataset import *


def cache_subject(sub_dir: str, domain: str, set_id: str, set_root:str, seq_len: int, shift_len: int):
    """
    Process and cache PSG data sequences from a single subject directory.

    Loads raw NPZ files containing multi-channel PSG signals and hypnogram labels, applies temporal shifting and splitting,
    and saves processed fixed-length sequences into a cache directory for efficient loading.

    Args:
        sub_dir (str): Path to the subject's raw data directory containing NPZ sequence files.
        domain (str): Domain or dataset identifier for the subject.
        set_id (str): Dataset split identifier ('train', 'eval', or 'test').
        set_root (str): Root directory where cached data will be stored.
        seq_len (int): Length of each sequence segment in epochs.
        shift_len (int): Maximum temporal shift applied for data augmentation (only for training).
    """
    
    data_pairs = []
    for seq_name in os.listdir(sub_dir):
        seq_path = os.path.join(sub_dir, seq_name)
        npz = np.load(seq_path)
        
        # Load hypnogram labels (sleep stages) as int64 array of shape (L,)
        ano = npz['Hypnogram'].astype(np.int64)
        
        # Collect available PSG channels and their data
        available_channels = []
        channel_data_list = []
        channel_indices = []
        
        for ch_name in ALL_CHANNELS:
            if ch_name in npz.files:
                available_channels.append(ch_name)
                # Load channel data as float32 array of shape (L, 3000)
                channel_data_list.append(npz[ch_name].astype(np.float32))
                channel_indices.append(CHANNEL_TO_INDEX[ch_name])
        
        if len(available_channels) == 0:
            print(f"Warning: No valid channels found in {seq_path}")
            continue
            
        # Stack channel data along new axis to form (L, 3000, cn)
        sig = np.stack(channel_data_list, axis=2)
        # Channel indices as int64 array of shape (cn,)
        ch_id = np.array(channel_indices, dtype=np.int64)
        
        data_pairs.append((sig, ano, ch_id))
    
    SEQ, LABELS, CH_ID = [], [], []
    for (seq_data, labels_data, ch_id) in data_pairs:
        if set_id == 'train':
            # Apply random temporal shift for data augmentation during training
            seq_data = random_temporal_shift(seq_data, shift_len).transpose(0, 2, 1)  # (L, cn, 3000)
            # Randomly split sequences into fixed-length segments with labels
            splited_seq, splited_labels = random_split_sample(seq_data, labels_data, random=True, seq_len=seq_len)
        elif set_id == 'eval' or set_id == 'test':
            # For evaluation and testing, no temporal shift; transpose to (L, cn, 3000)
            seq_data = seq_data.transpose(0, 2, 1)
            # Split sequences deterministically without randomness
            splited_seq, splited_labels = random_split_sample(seq_data, labels_data, random=False, seq_len=seq_len)
        SEQ.extend(splited_seq)
        LABELS.extend(splited_labels)
        CH_ID.extend([ch_id]*len(splited_seq))

    # Create cache directory for this subject under the appropriate split and domain
    cache_sub_dir = os.path.join(set_root, domain, os.path.split(sub_dir)[-1])
    os.makedirs(cache_sub_dir, exist_ok=True)

    # Save each processed sequence segment as a separate NPZ file
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
    """
    Cache PSG data for multiple subjects across training, evaluation, and testing splits using multiprocessing.

    Args:
        train_subjects (list): List of tuples (subject_path, domain) for training subjects.
        eval_subjects (list): List of tuples (subject_path, domain) for evaluation subjects.
        test_subjects (list): List of tuples (subject_path, domain) for testing subjects.
        cache_root (str): Root directory for cached data storage.
        num_processes (int): Number of parallel processes for caching.
        seq_len (int): Length of each sequence segment in epochs.
        shift_len (int): Maximum temporal shift for training data augmentation.
    """
    
    cache_train_root = os.path.join(cache_root, 'train')
    cache_eval_root = os.path.join(cache_root, 'eval')
    cache_test_root = os.path.join(cache_root, 'test')

    if len(train_subjects) > 0:
        # Prepare arguments for multiprocessing caching of training subjects
        train_subjects = [(subject_path, subject_domain, 'train', cache_train_root, seq_len, shift_len) for subject_path, subject_domain in train_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, train_subjects)

    if len(eval_subjects) > 0:
        # Evaluation subjects use zero shift (no augmentation)
        eval_subjects = [(subject_path, subject_domain, 'eval', cache_eval_root, seq_len, 0) for subject_path, subject_domain in eval_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, eval_subjects)

    if len(test_subjects) > 0:
        # Testing subjects use zero shift (no augmentation)
        test_subjects = [(subject_path, subject_domain, 'test', cache_test_root, seq_len, 0) for subject_path, subject_domain in test_subjects]
        with Pool(num_processes) as p:
            p.starmap(cache_subject, test_subjects)

        
class TrainDataset(Dataset):
    """
    PyTorch Dataset for training data, loading cached PSG sequences with data augmentation.

    Each sample consists of reshaped PSG signals, corresponding labels, and indices for sequence and channel.
    """
    def __init__(self, data_pairs: list):
        """
        Initialize the training dataset.

        Args:
            data_pairs (list): List of tuples (sample_path, domain) for cached training samples.
        """
        super(TrainDataset, self).__init__()
        
        self.sample_paths, self.sample_domains = zip(*data_pairs)

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        assert len(self.sample_paths) == len(self.sample_domains),\
            'The length of path and domain are not equal'
        return len(self.sample_paths)

    def __getitem__(self, index):
        """
        Load and process a single training sample.

        Applies random channel cropping augmentation and reshapes data for model input.

        Args:
            index (int): Index of the sample to load.

        Returns:
            tuple: (seq, label, seq_idx, ch_idx)
                seq (np.ndarray): PSG signal data reshaped to (seql*cn, 3000).
                label (np.ndarray): Corresponding labels of shape (seql,).
                seq_idx (np.ndarray): Sequence indices repeated for each channel, shape (seql*cn,).
                ch_idx (np.ndarray): Channel indices repeated for each sequence step, shape (seql*cn,).
        """
        npz = np.load(self.sample_paths[index])
        # Load PSG signals of shape (seql, cn, 3000)
        seq = npz['sig'].astype(np.float32)
        seql, cn = seq.shape[:2]
        # Load labels of shape (seql,)
        label = npz['ano'].astype(np.int64)
        # Generate sequence indices for each epoch, shape (seql, cn)
        seq_idx = np.arange(seql).reshape(seql, 1)
        seq_idx = np.tile(seq_idx, cn)
        # Load channel indices and tile for each sequence step, shape (seql, cn)
        ch_id = npz['ch_id'].astype(np.int64)
        ch_idx = ch_id.reshape(1, cn)
        ch_idx = np.tile(ch_idx, (seql, 1))

        # Apply random channel cropping augmentation; other inputs set to None as unused
        seq, _, seq_idx, ch_idx, _ = random_channel_crop(seq, None, seq_idx, ch_idx, None)
        
        seql, cn = seq.shape[:2]
        # Flatten sequence and channel dimensions for model input
        seq = np.reshape(seq, (seql*cn, -1))
        seq_idx = np.reshape(seq_idx, (seql*cn, ))
        ch_idx = np.reshape(ch_idx, (seql*cn, ))

        return seq, label, seq_idx, ch_idx


class TestDataset(Dataset):
    """
    PyTorch Dataset for evaluation and testing data, loading cached PSG sequences without augmentation.

    Each sample consists of reshaped PSG signals, corresponding labels, and indices for sequence and channel.
    """
    def __init__(self, data_pairs: list):
        """
        Initialize the test dataset.

        Args:
            data_pairs (list): List of tuples (sample_path, domain) for cached evaluation or test samples.
        """
        super(TestDataset, self).__init__()
        
        self.sample_paths, self.sample_domains = zip(*data_pairs)

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        assert len(self.sample_paths) == len(self.sample_domains),\
            'The length of path and domain are not equal'
        return len(self.sample_paths)

    def __getitem__(self, index):
        """
        Load and process a single evaluation or test sample.

        Args:
            index (int): Index of the sample to load.

        Returns:
            tuple: (seq, label, seq_idx, ch_idx)
                seq (np.ndarray): PSG signal data reshaped to (seql*cn, 3000).
                label (np.ndarray): Corresponding labels of shape (seql,).
                seq_idx (np.ndarray): Sequence indices repeated for each channel, shape (seql*cn,).
                ch_idx (np.ndarray): Channel indices repeated for each sequence step, shape (seql*cn,).
        """
        npz = np.load(self.sample_paths[index])
        # Load PSG signals of shape (seql, cn, 3000)
        seq = npz['sig'].astype(np.float32)
        seql, cn = seq.shape[:2]
        # Load labels of shape (seql,)
        label = npz['ano'].astype(np.int64)
        # Generate sequence indices for each epoch, shape (seql, cn)
        seq_idx = np.arange(seql).reshape(seql, 1)
        seq_idx = np.tile(seq_idx, cn)
        # Load channel indices and tile for each sequence step, shape (seql, cn)
        ch_id = npz['ch_id'].astype(np.int64)
        ch_idx = ch_id.reshape(1, cn)
        ch_idx = np.tile(ch_idx, (seql, 1))

        # Flatten sequence and channel dimensions for model input
        seq = np.reshape(seq, (seql*cn, -1))
        seq_idx = np.reshape(seq_idx, (seql*cn, ))
        ch_idx = np.reshape(ch_idx, (seql*cn, ))

        return seq, label, seq_idx, ch_idx


class ClassifyDataLoader():
    """
    Wrapper class to manage data loading for training, evaluation, and testing phases.

    Handles caching of raw data, loading cached samples, and creating PyTorch DataLoader instances with custom collation.
    """
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
        """
        Initialize the data loader with dataset splits and loading parameters.

        Args:
            train_subjects (list): List of tuples (subject_path, domain) for training subjects.
            eval_subjects (list): List of tuples (subject_path, domain) for evaluation subjects.
            test_subjects (list): List of tuples (subject_path, domain) for testing subjects.
            seq_len (int): Length of each sequence segment in epochs.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of worker threads for PyTorch DataLoader.
            num_processes (int): Number of parallel processes for caching.
            cache_root (str): Root directory for cached data storage.
            shift_len (int): Maximum temporal shift for training data augmentation.
        """
        self.train_subjects = train_subjects
        self.eval_subjects = eval_subjects
        self.test_subjects = test_subjects

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_processes = num_processes
        self.cache_root = cache_root
        self.shift_len = shift_len
        
        # Cache evaluation and test data initially (training cached on demand)
        cache_data(
            [],
            self.eval_subjects,
            self.test_subjects,
            self.cache_root,
            self.num_processes,
            self.seq_len,
            self.shift_len,
        )
        # Load cached evaluation and test sample paths
        self.eval_pairs = self.get_samples('eval')
        self.test_pairs = self.get_samples('test')
        
        
    def get_train_data_loader(self):
        """
        Cache training data and return a DataLoader for training.

        Returns:
            DataLoader: PyTorch DataLoader for training dataset with shuffling and augmentation.
        """
        # Cache training data with augmentation
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
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_func,
                drop_last=True
            )
        
        return data_loader
    
    def get_eval_data_loader(self):
        """
        Return a DataLoader for evaluation data without shuffling.

        Returns:
            DataLoader: PyTorch DataLoader for evaluation dataset.
        """
        eval_set = TestDataset(self.eval_pairs)

        data_loader = DataLoader(
                eval_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_func,
                drop_last=False
            )
        
        return data_loader
    
    def get_test_data_loader(self):
        """
        Return a DataLoader for test data without shuffling.

        Returns:
            DataLoader: PyTorch DataLoader for test dataset.
        """
        test_set = TestDataset(self.test_pairs)

        data_loader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_func,
                drop_last=False,
            )
        
        return data_loader

    def collate_func(self, batch):
        """
        Custom collate function to pad sequences, labels, and indices for batch processing.

        Pads variable-length sequences and generates padding masks for model input.

        Args:
            batch (list): List of tuples (seq, label, seq_idx, ch_idx) from Dataset __getitem__.

        Returns:
            tuple: (padded_sequences, padded_labels, padded_seq_idxs, padded_ch_idxs, padding_masks, ori_lens)
                padded_sequences (Tensor): Padded PSG sequences tensor of shape (batch, max_seq_len, 3000).
                padded_labels (Tensor): Padded label tensor of shape (batch, max_seq_len).
                padded_seq_idxs (Tensor): Padded sequence indices tensor of shape (batch, max_seq_len).
                padded_ch_idxs (Tensor): Padded channel indices tensor of shape (batch, max_seq_len).
                padding_masks (Tensor): Boolean mask tensor indicating padding positions (True for padding).
                ori_lens (list): Original lengths of sequences before padding.
        """
        # Unpack batch elements
        sequences, labels, seq_idxs, ch_idxs = zip(*batch)

        # Pad sequences to the maximum length in batch with zeros
        padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
        # Pad labels similarly
        padded_labels = pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=0)
        # Pad sequence indices
        padded_seq_idxs = pad_sequence([torch.tensor(seq_idx) for seq_idx in seq_idxs], batch_first=True, padding_value=0)
        # Pad channel indices
        padded_ch_idxs = pad_sequence([torch.tensor(ch_idx) for ch_idx in ch_idxs], batch_first=True, padding_value=0)
        
        # Create padding mask: False for valid data, True for padding
        padding_masks = [torch.zeros((seq.shape[0],), dtype=torch.bool) for seq in sequences]
        padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=True)

        # Store original sequence lengths before padding
        ori_lens = [seq.shape[0] for seq in sequences]

        return padded_sequences, padded_labels, padded_seq_idxs, padded_ch_idxs, padding_masks, ori_lens
    

    def get_samples(self, set_id: str):
        """
        Retrieve cached sample file paths and their domains for a given dataset split.

        Args:
            set_id (str): Dataset split identifier ('train', 'eval', or 'test').

        Returns:
            list: List of tuples (sample_path, domain) for all cached samples in the split.
        """
        if set_id == 'train':
            set_root = os.path.join(self.cache_root, 'train')
        elif set_id == 'eval':
            set_root = os.path.join(self.cache_root, 'eval')
        elif set_id == 'test':
            set_root = os.path.join(self.cache_root, 'test')
        else:
            return []
        
        if not os.path.exists(set_root):
            return []
        
        data_pairs = []
        # Iterate over domains and subjects to collect sample paths
        for domain in os.listdir(set_root):
            for sub_id in os.listdir(os.path.join(set_root, domain)):
                sub_dir = os.path.join(set_root, domain, sub_id)
                for seq_name in os.listdir(sub_dir):
                    sample_path = os.path.join(sub_dir, seq_name)
                    data_pairs.append((sample_path, domain))

        return data_pairs
    

    def clear_cache(self):
        """
        Remove all cached data from the cache root directory.
        """
        shutil.rmtree(self.cache_root, 
                    ignore_errors=True)
