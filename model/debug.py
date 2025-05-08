

import torch
import torch.nn as nn

from model import LPSGM
from utils import *
from data.dataset import *



class args(object):
    epoch_encoder_dropout = 0
    transformer_num_heads = 8
    transformer_mlp_dim = 512
    transformer_dropout = 0
    transformer_attn_dropout = 0
    decoder_dropout = 0
    ch_num = 8
    seq_len = 20
    ch_emb_dim =32
    seq_emb_dim = 64
    
    
weights_file = r"/home/denggf/Desktop/UFSB_cls/run/Aug13_15-21-47/model_dir/test_best.pth"
model = nn.DataParallel(LPSGM(args)).cuda()
model.load_state_dict(torch.load(weights_file)['model_state_dict'])







