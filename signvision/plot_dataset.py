from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_kaggle import resolve_kaggle_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot dataset sign distribution for PPT/report.")
    p.add_argument("--data", type=str, required=True, help="Kaggle dataset root (e.g., data/asl-signs-subset)")
    p.add_argument("--topn", type=int, default=20, help="How many top classes to show in bar chart")
    p.add_argument("--out", type=str, default="runs/dataset_distribution.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = resolve_kaggle_root(Path(args.data))
    train_csv = data_root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv under: {data_root}")

    df = pd.read_csv(train_csv)
    if "sign" not in df.columns:
        raise RuntimeError("train.csv has no 'sign' column.")

    vc = df["sign"].value_counts()
    counts = vc.to_numpy()
    n_rows = int(len(df))
    n_classes = int(vc.size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(counts, bins=min(30, max(5, int(counts.max()))), color="tab:blue", alpha=0.85)
    ax1.set_title("Class Count Histogram")
    ax1.set_xlabel("Samples per class")
    ax1.set_ylabel("Number of classes")
    ax1.grid(True, alpha=0.25)

    topn = max(1, int(args.topn))
    top = vc.head(topn)[::-1]
    ax2.barh(top.index.astype(str), top.values, color="tab:orange", alpha=0.9)
    ax2.set_title(f"Top-{topn} Most Frequent Signs")
    ax2.set_xlabel("Samples")
    ax2.grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"Dataset Distribution: rows={n_rows}, classes={n_classes}", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()

