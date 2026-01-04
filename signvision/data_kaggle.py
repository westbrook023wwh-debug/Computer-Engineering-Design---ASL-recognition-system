from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .feature_sets import feature_dim_for_set, select_feature_set
from .landmarks import pad_or_truncate_sequence


@dataclass(frozen=True)
class KagglePaths:
    root: Path

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def sign_map_json(self) -> Path:
        return self.root / "sign_to_prediction_index_map.json"


def resolve_kaggle_root(root: str | Path) -> Path:
    """
    Kaggle downloads/extractions sometimes introduce an extra nested folder.
    This resolves the dataset root by searching for `train.csv` + label map.
    """
    root = Path(root)
    required = ["train.csv", "sign_to_prediction_index_map.json"]
    if all((root / r).exists() for r in required):
        return root

    # Search shallowly for a nested dataset folder
    candidates = []
    for depth in (1, 2):
        for p in root.glob("*/" * depth + "train.csv"):
            cand = p.parent
            if all((cand / r).exists() for r in required):
                candidates.append(cand)
        if candidates:
            break

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find Kaggle dataset files under '{root}'. Expected: {', '.join(required)}. "
        "Make sure you downloaded the Kaggle 'asl-signs' competition data and extracted the zip(s) into this folder."
    )


def _is_zip_by_sig(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _read_single_from_zip(path: Path, expected_suffix: str | None = None) -> bytes:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if expected_suffix:
            names = [n for n in names if n.lower().endswith(expected_suffix.lower())]
        if not names:
            raise FileNotFoundError(f"No matching files inside zip: {path}")
        with zf.open(names[0], "r") as f:
            return f.read()


def _read_text_maybe_zipped(path: Path, encoding: str = "utf-8") -> str:
    if _is_zip_by_sig(path):
        data = _read_single_from_zip(path, expected_suffix=None)
        return data.decode(encoding, errors="strict")
    return path.read_text(encoding=encoding)


def _read_csv_maybe_zipped(path: Path) -> pd.DataFrame:
    if _is_zip_by_sig(path):
        data = _read_single_from_zip(path, expected_suffix=".csv")
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def load_kaggle_label_map(root: str | Path) -> list[str]:
    root = resolve_kaggle_root(root)
    mapping_path = Path(root) / "sign_to_prediction_index_map.json"
    mapping = json.loads(_read_text_maybe_zipped(mapping_path, encoding="utf-8"))
    max_idx = max(int(v) for v in mapping.values())
    labels = [""] * (max_idx + 1)
    for sign, idx in mapping.items():
        labels[int(idx)] = sign
    return labels


def _read_parquet_sequence(path: Path) -> np.ndarray:
    """
    Best-effort loader for asl-signs parquet files.
    Returns float array shaped (T, F) where F is inferred from parquet.

    Handles common formats:
      A) long: columns include frame, landmark_index, x, y, z (optionally type/row_id)
      B) wide: x_0,y_0,z_0...
      C) fallback: numeric columns only
    """
    if _is_zip_by_sig(path):
        data = _read_single_from_zip(path, expected_suffix=".parquet")
        table = pq.read_table(io.BytesIO(data))
    else:
        table = pq.read_table(path)
    df = table.to_pandas()

    cols = set(df.columns)
    if {"frame", "landmark_index", "x", "y", "z"}.issubset(cols):
        frames = int(df["frame"].max()) + 1
        landmarks = int(df["landmark_index"].max()) + 1
        arr = np.zeros((frames, landmarks, 3), dtype=np.float32)
        x = df["x"].to_numpy(np.float32, copy=False)
        y = df["y"].to_numpy(np.float32, copy=False)
        z = df["z"].to_numpy(np.float32, copy=False)
        f = df["frame"].to_numpy(np.int32, copy=False)
        li = df["landmark_index"].to_numpy(np.int32, copy=False)
        arr[f, li, 0] = np.nan_to_num(x, nan=0.0)
        arr[f, li, 1] = np.nan_to_num(y, nan=0.0)
        arr[f, li, 2] = np.nan_to_num(z, nan=0.0)
        return arr.reshape(frames, -1)

    if any(c.startswith("x_") for c in cols) and any(c.startswith("y_") for c in cols) and any(c.startswith("z_") for c in cols):
        xs = sorted([c for c in df.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))
        ys = sorted([c for c in df.columns if c.startswith("y_")], key=lambda s: int(s.split("_")[1]))
        zs = sorted([c for c in df.columns if c.startswith("z_")], key=lambda s: int(s.split("_")[1]))
        feat = np.stack(
            [df[xs].to_numpy(np.float32), df[ys].to_numpy(np.float32), df[zs].to_numpy(np.float32)], axis=-1
        )
        return np.nan_to_num(feat, nan=0.0).reshape(feat.shape[0], -1)

    num = df.select_dtypes(include=["number"]).to_numpy(np.float32)
    return np.nan_to_num(num, nan=0.0)


class KaggleASLSignsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Loads Kaggle asl-signs sequences from parquet and returns:
      x: (seq_len, feature_dim)
      mask: (seq_len,)
      y: ()
    """

    def __init__(
        self,
        root: str | Path,
        seq_len: int,
        feature_set: str = "hands",
        indices: np.ndarray | None = None,
    ) -> None:
        resolved_root = resolve_kaggle_root(root)
        self.paths = KagglePaths(Path(resolved_root))
        self.df = _read_csv_maybe_zipped(self.paths.train_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.sign_map: dict[str, int] = json.loads(_read_text_maybe_zipped(self.paths.sign_map_json, encoding="utf-8"))
        self.seq_len = int(seq_len)
        self.feature_set = feature_set
        self.feature_dim = feature_dim_for_set(feature_set)

        # If train.csv references per-row file paths, drop rows whose parquet isn't available locally.
        # This prevents training from crashing when a partial subset has missing downloads.
        if "path" in self.df.columns:
            root_dir = self.paths.root
            exists = self.df["path"].apply(lambda p: (root_dir / str(p)).exists())
            self.df = self.df[exists].reset_index(drop=True)
            if len(self.df) == 0:
                raise FileNotFoundError(
                    f"No parquet files found under '{root_dir}'. "
                    "Make sure `train_landmark_files/` exists for the full dataset, "
                    "or use `python -m signvision.kaggle_subset` to create a local subset."
                )

    def __len__(self) -> int:
        return len(self.df)

    def _seq_path(self, row: Any) -> Path:
        if "path" in self.df.columns:
            return self.paths.root / str(row["path"])
        sequence_id = row["sequence_id"]
        return self.paths.root / "train_landmarks" / f"{int(sequence_id)}.parquet"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        sign = row["sign"]
        y = int(self.sign_map[sign])

        seq = _read_parquet_sequence(self._seq_path(row))  # (T, F*)
        # Expect full landmarks (543*3) and then slice by feature_set
        if seq.shape[1] >= 543 * 3:
            seq = seq[:, : 543 * 3]
        else:
            pad = np.zeros((seq.shape[0], 543 * 3 - seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)

        seq = seq.reshape(seq.shape[0], 543, 3)
        seq_feats = [select_feature_set(seq[t], self.feature_set) for t in range(seq.shape[0])]
        x, mask = pad_or_truncate_sequence(seq_feats, seq_len=self.seq_len, feature_dim=self.feature_dim)

        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.tensor(y, dtype=torch.long),
        )
