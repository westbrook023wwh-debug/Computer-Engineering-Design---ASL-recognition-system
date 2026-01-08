from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export realtime event logs to a CSV for manual labeling.")
    p.add_argument("--events", type=str, default="runs/realtime_events/events.jsonl", help="Path to events.jsonl")
    p.add_argument("--out", type=str, default="runs/realtime_events/events_for_labeling.csv")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    events_path = Path(args.events)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)

    if not rows:
        print(f"No events found in {events_path}")
        return

    df = pd.DataFrame(rows)
    # Keep key columns for labeling
    keep_cols = [
        "timestamp_ms",
        "event_type",
        "reason",
        "pred_label",
        "max_prob",
        "topk",
        "jpg",
        "npz",
        "mp4",
        "feature_set",
        "seq_len",
        "camera",
        "backend",
        "device",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[keep_cols]
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()

