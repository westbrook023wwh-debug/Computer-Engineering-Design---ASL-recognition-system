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
    p.add_argument(
        "--compact-label-map",
        action="store_true",
        help="Remap labels to contiguous ids for only the signs used in training (recommended for small subsets).",
    )
    p.add_argument(
        "--min-class-count",
        type=int,
        default=1,
        help="Drop signs with fewer than this many samples before train/val split (applies to compact label map).",
    )
    p.add_argument(
        "--max-classes",
        type=int,
        default=0,
        help="Keep only the top-N most frequent signs after filtering (0 = no limit; applies to compact label map).",
    )
    p.add_argument("--class-weighted", action="store_true", help="Use class-balanced weights for CE loss")
    p.add_argument("--augment-noise-std", type=float, default=0.0, help="Add Gaussian noise std to features during train")
    p.add_argument("--log-jsonl", type=str, default="", help="Optional path to write JSONL logs (train/val metrics per epoch)")
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

    feature_dim = feature_dim_for_set(args.feature_set)

    full_ds = KaggleASLSignsDataset(data_root, seq_len=args.seq_len, feature_set=args.feature_set)

    label_map: list[str]
    sign_map_override = None
    eligible_idx = np.arange(len(full_ds))

    if args.compact_label_map:
        vc = full_ds.df["sign"].value_counts()
        min_count = max(1, int(args.min_class_count))
        kept = vc[vc >= min_count]
        if int(args.max_classes) > 0:
            kept = kept.iloc[: int(args.max_classes)]
        kept_signs = list(kept.index.astype(str))
        if not kept_signs:
            raise RuntimeError("No signs left after filtering. Lower --min-class-count / --max-classes or download more data.")

        # Remap to contiguous ids.
        kept_signs_sorted = sorted(kept_signs)
        sign_map_override = {s: i for i, s in enumerate(kept_signs_sorted)}
        label_map = kept_signs_sorted
        save_label_map(out_dir / "label_map.json", label_map)

        eligible_mask = full_ds.df["sign"].isin(set(kept_signs_sorted)).to_numpy()
        eligible_idx = np.nonzero(eligible_mask)[0]
        if eligible_idx.size == 0:
            raise RuntimeError("No samples left after filtering. Lower --min-class-count or download more data.")
        print(
            json.dumps(
                {
                    "dataset_rows": int(len(full_ds)),
                    "eligible_rows": int(eligible_idx.size),
                    "num_classes": int(len(label_map)),
                    "min_class_count": int(min_count),
                    "max_classes": int(args.max_classes),
                },
                ensure_ascii=False,
            )
        )
    else:
        label_map = load_kaggle_label_map(data_root)
        save_label_map(out_dir / "label_map.json", label_map)
        print(
            json.dumps(
                {
                    "dataset_rows": int(len(full_ds)),
                    "eligible_rows": int(len(full_ds)),
                    "num_classes": int(len(label_map)),
                    "note": "Using Kaggle label map; consider --compact-label-map for subsets.",
                },
                ensure_ascii=False,
            )
        )

    # Split eligible indices
    idx = eligible_idx.copy()
    np.random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if tr_idx.size == 0:
        raise RuntimeError("Training split is empty. Reduce --val-split or download more data.")

    tr_ds = KaggleASLSignsDataset(
        data_root,
        seq_len=args.seq_len,
        feature_set=args.feature_set,
        indices=tr_idx,
        sign_map_override=sign_map_override,
    )
    val_ds = KaggleASLSignsDataset(
        data_root,
        seq_len=args.seq_len,
        feature_set=args.feature_set,
        indices=val_idx,
        sign_map_override=sign_map_override,
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    cfg = SignVisionConfig(seq_len=args.seq_len, feature_dim=feature_dim, num_classes=len(label_map))
    model = SignVisionModel(cfg).to(args.device)

    class_weights = None
    if args.class_weighted:
        counts = np.bincount(tr_ds.df["sign"].apply(lambda s: tr_ds.sign_map[s]).to_numpy(), minlength=len(label_map))
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=args.device)

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
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    jsonl_path = Path(args.log_jsonl) if args.log_jsonl else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, mask, y in tqdm(tr_loader, desc=f"train {epoch}/{args.epochs}"):
            x = x.to(args.device)
            mask = mask.to(args.device)
            y = y.to(args.device)

            if args.augment_noise_std > 0.0:
                noise = torch.randn_like(x) * float(args.augment_noise_std)
                x = x + noise

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

        if jsonl_path:
            _append = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": acc}
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_append, ensure_ascii=False) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, label_map=label_map)

    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
