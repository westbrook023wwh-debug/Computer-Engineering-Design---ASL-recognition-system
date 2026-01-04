from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot realtime benchmark CSV into figures for report.")
    p.add_argument("--csv", type=str, default="runs/bench_realtime.csv")
    p.add_argument("--out-dir", type=str, default="runs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    # FPS over time (using total_ms)
    df["fps"] = df["total_ms"].apply(lambda x: 1000.0 / x if x > 0 else 0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["frame_idx"], df["fps"], linewidth=1.0)
    ax.set_title("Realtime FPS over Frames")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("FPS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bench_fps.png", dpi=200)
    plt.close(fig)

    # Timing breakdown boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["capture_ms", "mp_ms", "infer_ms", "total_ms"]
    ax.boxplot([df[c].values for c in cols], tick_labels=cols, showfliers=False)
    ax.set_title("Latency Breakdown (ms)")
    ax.set_ylabel("Milliseconds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bench_latency_box.png", dpi=200)
    plt.close(fig)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
