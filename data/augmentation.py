import numpy as np


def random_channel_crop(seq, label, seq_idx, ch_idx, spg):
    # seq: (seql, cn, 3000)
    # label: (seql, cn) or None
    # seq_idx: (seql, cn)
    # ch_idx: (seql, cn)
    # spg: (seql, cn, 129, 29)
    orig_cn = seq.shape[1]
    new_cn = np.random.randint(1, orig_cn + 1)
    ch_indices = np.random.choice(orig_cn, new_cn, replace=False) 

    new_seq = seq[:, ch_indices, :]
    new_seq_idx = seq_idx[:, ch_indices]
    new_ch_idx = ch_idx[:, ch_indices]
    
    if type(label) == np.ndarray:
        new_label = label[:, ch_indices]
    else:
        new_label = None

    if type(spg) == np.ndarray:
        new_spg = spg[:, ch_indices, :, :]
    else:
        new_spg = None

    return new_seq, new_label, new_seq_idx, new_ch_idx, new_spg


def random_temporal_shift(seq, shift_len=100):
    # seq: (L, 3000, cn)
    seq = seq.reshape(-1, seq.shape[-1]) # (L*3000, cn)
    shift_len = round(np.random.normal(0, shift_len, 1)[0])
    shifted_seq = np.roll(seq, shift_len, axis=0).reshape(-1, 3000, seq.shape[-1])
    return shifted_seq  # (L, 3000, cn)


def random_split_sample(seq, labels, random: bool, seq_len):
    # seq: (L, cn, 3000)
    # labels: (L,)
    if random == True:
        rn = np.random.randint(0, seq_len)
    else:
        rn = 0

    seqn = (seq.shape[0] - rn) // seq_len

    if not seqn > 0:
        if type(seq) == type(labels):   # labels != None
            return [], []
        else:
            return []
    
    seq = seq[rn : rn+seqn*seq_len]
    seq = seq.reshape(seqn, seq_len, seq.shape[1], seq.shape[2]) # (seqn, seql, cn, 3000)
    splited_seq = [np.squeeze(arr) for arr in np.split(seq, seqn, axis=0)]

    try:
        labels = labels[rn : rn+seqn*seq_len]
        labels = labels.reshape(seqn, seq_len) # (seqn, seql)
        splited_labels = [np.squeeze(arr) for arr in np.split(labels, seqn, axis=0)]
    except:
        return splited_seq
    else:
        return splited_seq, splited_labels
