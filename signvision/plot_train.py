from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot JSONL training logs produced by signvision.train.")
    p.add_argument("--log", type=str, required=True, help="Path to JSONL log (one dict per epoch).")
    p.add_argument("--out", type=str, default="runs/train_curves.png")
    return p.parse_args()


def _read_text_guess_encoding(path: Path) -> str:
    data = path.read_bytes()
    # BOM detection
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        # Likely written by PowerShell Tee-Object default (UTF-16 LE with BOM)
        return data.decode("utf-16")
    # Try utf-8 first, then common Windows fallback
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return data.decode("gbk", errors="replace")


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs: list[int] = []
    tr_loss: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []

    for line in _read_text_guess_encoding(log_path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in obj:
            epochs.append(int(obj["epoch"]))
            tr_loss.append(float(obj.get("train_loss", "nan")))
            val_loss.append(float(obj.get("val_loss", "nan")))
            val_acc.append(float(obj.get("val_acc", "nan")))

    if not epochs:
        raise RuntimeError(f"No epoch entries found in log: {log_path}")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, tr_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, color="tab:green", label="val_acc")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
