from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignVisionConfig:
    seq_len: int = 32
    feature_dim: int = 126
    num_classes: int = 250

    # 1D CNN stage (Conv over time; channels after projection)
    cnn_dim: int = 256
    cnn_blocks: int = 3
    cnn_kernel_size: int = 5
    dropout: float = 0.2

    # Transformer encoder
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 512

