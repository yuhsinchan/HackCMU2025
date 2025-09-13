import os, math, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_tcn import TCN   # pip install pytorch-tcn

class RepMistakeNet(nn.Module):
    """
    Input:  x (B, D, T)  where D=num_inputs channels (features per frame)
    Output: clip_logits (B, 6)  multi-label logits
    """
    def __init__(self, num_inputs: int, num_classes: int = 7,
                 channels=(64, 64, 64), kernel_size=4, dropout=0.1, causal=True):
        super().__init__()
        # TCN expects NCL: (B, C, L)
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NCL',
            use_skip_connections=False
        )
        out_C = channels[-1]

        # heads
        self.clip_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # pool over time
            nn.Flatten(),
            nn.Linear(out_C, num_classes)  # 7 logits
        )
        # self.frame_head = nn.Conv1d(out_C, 1, kernel_size=1)  # per-frame score

    def forward(self, x):           # x: (B, T, D)
        h = self.tcn(x)             # -> (B, C, T), C=channels[-1]
        clip_logits = self.clip_head(h)              # (B, 7)
        return clip_logits

# example
if __name__ == "__main__":
    B, T, D = 8, 200, 97
    model = RepMistakeNet(num_inputs=D, num_classes=7, channels=(64,64,64), kernel_size=4, dropout=0.1)
    x = torch.randn(B, T, D)
    clip_logits, frame_score = model(x)
    print(clip_logits.shape, frame_score.shape)  # torch.Size([8, 6]) torch.Size([8, 200])
