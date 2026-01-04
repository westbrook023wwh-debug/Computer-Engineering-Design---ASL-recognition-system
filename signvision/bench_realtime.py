from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import mediapipe as mp

from .checkpoint import load_checkpoint, load_label_map
from .feature_sets import feature_dim_for_set, select_feature_set
from .mediapipe_extract import holistic_to_543
from .mediapipe_tasks import create_detectors, default_model_dir, detect_543_tasks, resolve_models


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark realtime pipeline (camera + mediapipe + model) and save CSV.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--label-map", type=str, required=True)
    p.add_argument("--out", type=str, default="runs/bench_realtime.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--feature-set", choices=["hands", "full"], default="hands")
    p.add_argument("--predict-every", type=int, default=2)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--warmup-seconds", type=float, default=2.0)
    p.add_argument("--mp-model-dir", type=str, default=str(default_model_dir()))
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    return p.parse_args()


def _open_camera(index: int, backend: str) -> cv2.VideoCapture:
    cap_dshow = getattr(cv2, "CAP_DSHOW", None)
    cap_msmf = getattr(cv2, "CAP_MSMF", None)
    if backend == "dshow":
        candidates = [cap_dshow]
    elif backend == "msmf":
        candidates = [cap_msmf]
    elif backend == "any":
        candidates = [None]
    else:
        candidates = [cap_dshow, cap_msmf, None]

    def warmup(cap: cv2.VideoCapture, seconds: float = 2.0) -> bool:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t0 = time.time()
        while (time.time() - t0) < seconds:
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.05)
        return False

    for b in candidates:
        cap = cv2.VideoCapture(index) if b is None else cv2.VideoCapture(index, b)
        if not cap.isOpened():
            cap.release()
            continue
        if warmup(cap, seconds=2.0):
            return cap
        cap.release()
    return cv2.VideoCapture()


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    arr = np.array(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    model, _ = load_checkpoint(Path(args.checkpoint), device=device)
    label_map = load_label_map(Path(args.label_map))
    model.eval()

    expected = getattr(model, "config").feature_dim
    actual = feature_dim_for_set(args.feature_set)
    if expected != actual:
        raise ValueError(f"Model feature_dim={expected} but --feature-set {args.feature_set} gives {actual}.")

    cap = _open_camera(args.camera, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    use_tasks = not hasattr(mp, "solutions")
    detectors = None
    mp_holistic = None
    if use_tasks:
        models = resolve_models(args.mp_model_dir, use_face=not args.no_face, use_pose=not args.no_pose)
        detectors = create_detectors(models=models)
    else:
        mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]

    seq: deque[np.ndarray] = deque(maxlen=args.seq_len)
    prob_hist: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    frame_idx = 0
    pred_idx = -1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_ms_list: list[float] = []
    mp_ms_list: list[float] = []
    infer_ms_list: list[float] = []

    t_start = time.time()
    t_end = t_start + float(args.seconds)

    # warmup period (not recorded)
    while time.time() < (t_start + float(args.warmup_seconds)):
        ok, _frame = cap.read()
        if not ok:
            break

    try:
        while time.time() < t_end:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok or frame is None:
                break
            frame_idx += 1

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t_mp0 = time.perf_counter()
            if use_tasks:
                assert detectors is not None
                ts = int(time.time() * 1000)
                lms_543, _face, _left, _right, _pose = detect_543_tasks(detectors, rgb, ts)
            else:
                assert mp_holistic is not None
                with mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    refine_face_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ) as holistic:
                    res = holistic.process(rgb)
                lms_543 = holistic_to_543(res)
            t_mp1 = time.perf_counter()

            feat = select_feature_set(lms_543, args.feature_set)
            seq.append(feat)

            t_inf0 = time.perf_counter()
            ran_infer = False
            if len(seq) == args.seq_len and (frame_idx % max(1, args.predict_every) == 0):
                x = torch.from_numpy(np.stack(seq, axis=0)).unsqueeze(0).to(device)
                mask = torch.ones((1, args.seq_len), dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits = model(x, mask=mask)
                    prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                prob_hist.append(prob)
                avg_prob = np.mean(np.stack(list(prob_hist), axis=0), axis=0)
                pred_idx = int(avg_prob.argmax())
                ran_infer = True
            t_inf1 = time.perf_counter()

            t2 = time.perf_counter()

            capture_ms = (t1 - t0) * 1000.0
            mp_ms = (t_mp1 - t_mp0) * 1000.0
            infer_ms = (t_inf1 - t_inf0) * 1000.0 if ran_infer else 0.0
            total_ms = (t2 - t0) * 1000.0

            total_ms_list.append(total_ms)
            mp_ms_list.append(mp_ms)
            if ran_infer:
                infer_ms_list.append(infer_ms)

            rows.append(
                {
                    "t_wall_s": time.time(),
                    "frame_idx": frame_idx,
                    "capture_ms": round(capture_ms, 4),
                    "mp_ms": round(mp_ms, 4),
                    "infer_ms": round(infer_ms, 4),
                    "total_ms": round(total_ms, 4),
                    "ran_infer": int(ran_infer),
                    "pred_idx": pred_idx,
                    "pred_label": label_map[pred_idx] if 0 <= pred_idx < len(label_map) else "",
                    "backend": args.backend,
                    "feature_set": args.feature_set,
                    "use_tasks": int(use_tasks),
                    "device": str(device),
                }
            )
    finally:
        cap.release()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    fps_list = [1000.0 / ms for ms in total_ms_list if ms > 0]
    print("Saved:", out_path)
    print("Frames:", len(total_ms_list))
    print("FPS:", _summary(fps_list))
    print("Total ms:", _summary(total_ms_list))
    print("MP ms:", _summary(mp_ms_list))
    print("Infer ms:", _summary(infer_ms_list))


if __name__ == "__main__":
    main()

