from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .config import SignVisionConfig
from .eca import ECA1d
from .positional_encoding import SinusoidalPositionalEncoding


class _ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.eca = ECA1d(out_ch)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.eca(x)
        return x + residual


class SignVisionModel(nn.Module):
    """
    Hybrid: 1D CNN (temporal) + ECA channel attention + Transformer encoder.

    Input:
      - x: (B, T, F)
      - mask (optional): (B, T) bool, True for valid steps
    Output:
      - logits: (B, num_classes)
    """

    def __init__(self, config: SignVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.in_proj = nn.Conv1d(config.feature_dim, config.cnn_dim, kernel_size=1, bias=False)
        self.cnn = nn.Sequential(
            *[
                _ConvBlock1d(
                    in_ch=config.cnn_dim,
                    out_ch=config.cnn_dim,
                    kernel_size=config.cnn_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.cnn_blocks)
            ]
        )

        self.pos = SinusoidalPositionalEncoding(config.cnn_dim, max_len=max(2048, config.seq_len * 8))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.cnn_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.cnn_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.cnn_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x as (B,T,F), got {tuple(x.shape)}")
        if x.size(-1) != self.config.feature_dim:
            raise ValueError(f"feature_dim mismatch: expected {self.config.feature_dim}, got {x.size(-1)}")

        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.cnn(x)
        # (B,C,T) -> (B,T,C)
        x = x.transpose(1, 2)

        x = self.pos(x)

        key_padding_mask = None
        if mask is not None:
            if mask.ndim != 2 or mask.shape[:2] != x.shape[:2]:
                raise ValueError(f"mask must be (B,T) matching x; got mask={tuple(mask.shape)} x={tuple(x.shape)}")
            key_padding_mask = ~mask  # Transformer expects True for padding

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        return self.head(pooled)

    def to_checkpoint(self, label_map: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"config": asdict(self.config), "state_dict": self.state_dict()}
        if label_map is not None:
            payload["label_map"] = list(label_map)
        return payload

