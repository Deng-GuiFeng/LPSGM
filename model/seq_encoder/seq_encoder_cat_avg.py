# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.seq_encoder.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(self,
                 ch_num: int,
                 seq_len: int,
                 num_heads: int,
                 hidden_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 ch_emb_dim: int,
                 seq_emb_dim: int,
                 num_transformer_blocks: int,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, ch_emb_dim))  # from ViT
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, seq_emb_dim))
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim + ch_emb_dim))

        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim + ch_emb_dim + seq_emb_dim,
                hidden_dim + ch_emb_dim + seq_emb_dim,
                dropout,
                attention_dropout,
            ) for _ in range(num_transformer_blocks)
        ])
        
        self.ln = nn.LayerNorm(hidden_dim + ch_emb_dim + seq_emb_dim)


    def forward(self, input, mask, ch_idx, seq_idx, ori_len):
        # input: (bz, seql*cn, hidden_dim)
        # mask, ch_idx, seq_idx: (bz, seql*cn, )
        # ori_len: (bz, )

        bz, seql_cn = input.shape[:2]
        input = input.view(bz*seql_cn, -1)  # (bz*seql*cn, hidden_dim)
        ch_idx = ch_idx.view(bz*seql_cn, )  # (bz*seql*cn, )
        seq_idx = seq_idx.view(bz*seql_cn, )    # (bz*seql*cn, )

        # Use advanced indexing instead of loop
        SE = self.spatial_embedding[ch_idx]  # (bz*seql*cn, 32)
        TE = self.temporal_embedding[seq_idx]  # (bz*seql*cn, 64)

        input = torch.cat([input, SE, TE], dim=1)   # (bz*seql*cn, hidden_dim+ch_emb_dim+seq_emb_dim)
        input = input.view(bz, seql_cn, -1) # (bz, seql*cn, hidden_dim+ch_emb_dim+seq_emb_dim)

        input = self.dropout(input)

        for transformer_block in self.transformer_blocks:
            input = transformer_block(input, mask, attn_mask=None)

        feat = self.ln(input)   # (bz, seql+seql*cn, hidden_dim+ch_emb_dim+seq_emb_dim)

        feat_list = []
        for i, ft in enumerate(feat): # (seql*cn, hidden_dim+ch_emb_dim+seq_emb_dim)
            ft = ft[:ori_len[i]] # (seql*cn', hidden_dim+ch_emb_dim+seq_emb_dim)
            seql_cn = ft.shape[0]
            cn = seql_cn // self.seq_len
            ft = ft.view(self.seq_len, cn, -1)  # (seql, cn', hidden_dim+ch_emb_dim+seq_emb_dim)
            ft = ft.mean(1)  # (seql, hidden_dim+ch_emb_dim+seq_emb_dim)
            feat_list.append(ft.unsqueeze(0))     # (1, seql, hidden_dim+ch_emb_dim+seq_emb_dim)
        feat = torch.concat(feat_list, dim=0) # (bz, seql, hidden_dim+ch_emb_dim+seq_emb_dim)

        return feat












