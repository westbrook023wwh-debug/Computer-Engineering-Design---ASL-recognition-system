from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint, save_label_map
from .config import SignVisionConfig
from .data_kaggle import KaggleASLSignsDataset, load_kaggle_label_map, resolve_kaggle_root
from .feature_sets import feature_dim_for_set
from .model import SignVisionModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Kaggle asl-signs dataset root")
    p.add_argument("--out", type=str, default="checkpoints", help="Output dir")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--feature-set", choices=["hands", "full"], default="hands")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    data_root = resolve_kaggle_root(Path(args.data))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_kaggle_label_map(data_root)
    save_label_map(out_dir / "label_map.json", label_map)

    feature_dim = feature_dim_for_set(args.feature_set)

    full_ds = KaggleASLSignsDataset(data_root, seq_len=args.seq_len, feature_set=args.feature_set)
    n = len(full_ds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * args.val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_ds = KaggleASLSignsDataset(data_root, seq_len=args.seq_len, feature_set=args.feature_set, indices=tr_idx)
    val_ds = KaggleASLSignsDataset(data_root, seq_len=args.seq_len, feature_set=args.feature_set, indices=val_idx)

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    cfg = SignVisionConfig(seq_len=args.seq_len, feature_dim=feature_dim, num_classes=len(label_map))
    model = SignVisionModel(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(tr_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, mask, y in tqdm(tr_loader, desc=f"train {epoch}/{args.epochs}"):
            x = x.to(args.device)
            mask = mask.to(args.device)
            y = y.to(args.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask=mask)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += float(loss.item()) * x.size(0)
        tr_loss /= max(1, len(tr_ds))

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, mask, y in tqdm(val_loader, desc=f"val {epoch}/{args.epochs}"):
                x = x.to(args.device)
                mask = mask.to(args.device)
                y = y.to(args.device)
                logits = model(x, mask=mask)
                loss = loss_fn(logits, y)
                val_loss += float(loss.item()) * x.size(0)
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
        val_loss /= max(1, len(val_ds))
        acc = correct / max(1, len(val_ds))
        print(json.dumps({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": acc}))

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, label_map=label_map)

    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
