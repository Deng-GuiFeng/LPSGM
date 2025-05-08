# -*- coding: utf-8 -*-
# Modify from DeepSleepNet
import torch
import torch.nn as nn


class EpochEncoder(nn.Module):
    def __init__(self, dropout):
        super(EpochEncoder, self).__init__()
        self.encoder_branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, padding=24), # 3000->500
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4), # 500->63

            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same'), # 63->63
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'), # 63->63
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding='same'), # 63->63
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),# 63->16
            nn.AdaptiveAvgPool1d(1),# 16->1
        )

        self.encoder_branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50),# 3000->53
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4),# 53->13
            
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding='same'),# 13->13
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding='same'),# 13->13
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding='same'),# 13->13
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2),# 13->6
            nn.AdaptiveAvgPool1d(1),# 6->1
        )

    def forward(self, x):
        # x: (bz*seql*cn, 1, 3000)
        x1 = self.encoder_branch1(x).squeeze()
        x2 = self.encoder_branch2(x).squeeze()
        x = torch.concat([x1,x2], dim=-1)
        return x


if __name__ == "__main__":
    bz, seql, cn = 64, 20, 8
    x = torch.randn((bz, seql, cn, 3000))
    x = x.view((bz*seql*cn, 1, -1))
    print(x.size())

    model = EpochEncoder(0.5)
    print(sum(p.numel() for p in model.parameters()))

    y = model(x)
    print(y.size())




