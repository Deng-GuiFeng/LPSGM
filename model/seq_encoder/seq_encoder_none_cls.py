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
                 ch_emb_dim: None,
                 seq_emb_dim: None,
                 num_transformer_blocks: int,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.spatial_embedding = nn.Parameter(torch.randn(ch_num, hidden_dim))  # from ViT
        self.temporal_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(seq_len, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim,
                hidden_dim,
                dropout,
                attention_dropout,
            ) for _ in range(num_transformer_blocks)
        ])
        
        self.ln = nn.LayerNorm(hidden_dim)


    def forward(self, input, mask, ch_idx, seq_idx, ori_len):
        # input: (bz, seql*cn, hidden_dim)
        # mask, ch_idx, seq_idx: (bz, seql*cn, )
        # ori_len: (bz, )

        bz, seql_cn = input.shape[:2]
        input = input.view(bz*seql_cn, -1)  # (bz*seql*cn, hidden_dim)
        # ch_idx = ch_idx.view(bz*seql_cn, )  # (bz*seql*cn, )
        # seq_idx = seq_idx.view(bz*seql_cn, )    # (bz*seql*cn, )

        input = input.view(bz, seql_cn, -1) # (bz, seql*cn, hidden_dim)

        cls_tokens = self.cls_token.unsqueeze(0).expand(bz, -1, -1) + self.temporal_embedding.unsqueeze(0).expand(bz, -1, -1)
        input = torch.cat([cls_tokens, input], dim=1)   # (bz, seql*cn+seql, hidden_dim)

        padding_mask = torch.zeros(bz, self.seq_len).bool().to(mask.device)
        mask = torch.cat([padding_mask, mask], dim=1)

        input = self.dropout(input)

        for transformer_block in self.transformer_blocks:
            input = transformer_block(input, mask, attn_mask=None)

        feat = self.ln(input)   # (bz, seql*cn+seql, hidden_dim)

        feat = feat[:,:self.seq_len,:]   # (bz, seql, hidden_dim)

        return feat












