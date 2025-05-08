import torch
import torch.nn as nn

from model.classifier import Classifier
from model.epoch_encoder import EpochEncoder


class LPSGM(nn.Module):
    def __init__(self, args):
        super(LPSGM, self).__init__()

        self.args = args

        self.epoch_encoder = EpochEncoder(args.epoch_encoder_dropout)

        if args.architecture == 'cat_cls':
            from model.seq_encoder.seq_encoder_cat_cls import TransformerEncoder
            feat_dim = 512 + args.ch_emb_dim + args.seq_emb_dim
        elif args.architecture == 'add_cls':
            from model.seq_encoder.seq_encoder_add_cls import TransformerEncoder
            feat_dim = 512
        elif args.architecture == 'cat_avg':
            from model.seq_encoder.seq_encoder_cat_avg import TransformerEncoder
            feat_dim = 512 + args.ch_emb_dim + args.seq_emb_dim
        elif args.architecture == 'none_cls':
            from model.seq_encoder.seq_encoder_none_cls import TransformerEncoder
            feat_dim = 512
        else:
            raise NotImplementedError

        self.seq_encoder = TransformerEncoder(
            ch_num = args.ch_num,
            seq_len = args.seq_len,
            num_heads = args.transformer_num_heads,
            hidden_dim = 512,
            dropout = args.transformer_dropout,
            attention_dropout = args.transformer_attn_dropout,
            ch_emb_dim = args.ch_emb_dim,
            seq_emb_dim = args.seq_emb_dim,
            num_transformer_blocks = args.num_transformer_blocks,
        )

        self.classifier = Classifier(
            feat_dim = feat_dim, 
            num_classes = 5,
            )
        

    def forward(self, x, mask, ch_idx, seq_idx, ori_len):
        # x: (bz, seql*cn, 3000)
        # mask, ch_idx, seq_idx: (bz, seql*cn, )
        # ori_len: (bz, )

        x = torch.clamp(x, -self.args.clamp_value, self.args.clamp_value)
        bz, seql_cn, _ = x.shape

        x = x.view(bz*seql_cn, 1, -1)   # (bz*seql*cn, 1, 3000)
        x = self.epoch_encoder(x)       # (bz*seql*cn, hidden_dim)
        x_epoch = x.view(bz, seql_cn, -1)   # (bz, seql*cn, hidden_dim)
        feat = self.seq_encoder(x_epoch, mask, ch_idx, seq_idx, ori_len)   # (bz, seql, hidden_dim)

        out = self.classifier(feat)    # (bz, seql, 5)

        return out
        

   



