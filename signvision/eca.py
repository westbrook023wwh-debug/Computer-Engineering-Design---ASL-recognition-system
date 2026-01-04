from __future__ import annotations

import math

import torch
from torch import nn


class ECA1d(nn.Module):
    """
    Efficient Channel Attention (ECA) for 1D features.
    Expects x shaped (B, C, T). Produces channel weights from pooled temporal stats.
    """

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0) -> None:
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 == 1 else k + 1
        self._kernel_size = max(1, k)
        self._conv = nn.Conv1d(1, 1, kernel_size=self._kernel_size, padding=(self._kernel_size - 1) // 2, bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ECA1d expects (B,C,T), got {tuple(x.shape)}")
        # (B,C,1)
        avg = x.mean(dim=2, keepdim=True)
        mx = x.amax(dim=2, keepdim=True)
        pooled = avg + mx
        # Conv over channel axis: (B,1,C) -> (B,1,C) -> (B,C,1)
        y = pooled.transpose(1, 2)
        y = self._conv(y)
        y = self._sigmoid(y).transpose(1, 2)
        return x * y

