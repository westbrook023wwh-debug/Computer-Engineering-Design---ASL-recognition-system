from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def pad_or_truncate_sequence(
    seq: Iterable[np.ndarray], seq_len: int, feature_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    xs = [np.asarray(x, dtype=np.float32).reshape(feature_dim) for x in seq]
    t = len(xs)
    x = np.zeros((seq_len, feature_dim), dtype=np.float32)
    mask = np.zeros((seq_len,), dtype=bool)
    n = min(seq_len, t)
    if n > 0:
        x[:n] = np.stack(xs[:n], axis=0)
        mask[:n] = True
    return x, mask

