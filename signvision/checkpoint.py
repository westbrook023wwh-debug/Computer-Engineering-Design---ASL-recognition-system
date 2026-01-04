from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .config import SignVisionConfig
from .model import SignVisionModel


def save_checkpoint(path: str | Path, model: SignVisionModel, label_map: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to_checkpoint(label_map=label_map), path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[SignVisionModel, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=device)
    cfg = SignVisionConfig(**ckpt["config"])
    label_map = ckpt.get("label_map")
    if isinstance(label_map, list) and len(label_map) > 0:
        cfg = replace(cfg, num_classes=len(label_map))
    model = SignVisionModel(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    return model, ckpt


def save_label_map(path: str | Path, label_map: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")


def load_label_map(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

