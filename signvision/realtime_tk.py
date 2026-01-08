from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageTk
from tkinter import BOTH, LEFT, TOP, X, ttk
import tkinter as tk

from .checkpoint import load_checkpoint, load_label_map
from .feature_sets import feature_dim_for_set, select_feature_set
from .mediapipe_extract import holistic_to_543
from .mediapipe_tasks import create_detectors, default_model_dir, detect_543_tasks, draw_overlays_tasks, resolve_models


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--label-map", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")

    # MediaPipe Tasks support
    p.add_argument("--mp-model-dir", type=str, default=str(default_model_dir()))
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-pose", action="store_true")

    # Model input
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--feature-set", choices=["hands", "full"], default="hands")

    # Inference cadence / smoothing
    p.add_argument("--predict-every", type=int, default=2)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument("--fps-limit", type=float, default=30.0)

    # UI output
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--unknown-threshold", type=float, default=0.50)

    # Logging suspicious segments to runs/ (for later relabeling / retraining)
    p.add_argument("--log-dir", type=str, default="runs/realtime_events")
    p.add_argument("--log-cooldown-s", type=float, default=8.0)
    p.add_argument("--log-lowconf-threshold", type=float, default=0.50)
    p.add_argument("--log-jitter-window", type=int, default=8)
    p.add_argument("--log-jitter-changes", type=int, default=3)
    p.add_argument("--save-clip", action="store_true")
    p.add_argument("--clip-frames", type=int, default=48)
    return p.parse_args()


@dataclass
class _SharedState:
    lock: threading.Lock
    stop: threading.Event
    frame_rgb: np.ndarray | None = None
    pred_text: str = ""
    pred_detail: str = ""
    status: str = "Starting..."


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
        if warmup(cap):
            return cap
        cap.release()
    return cv2.VideoCapture()


def _format_topk(label_map: list[str], probs: np.ndarray, k: int) -> tuple[str, int, float]:
    k = max(1, int(k))
    idx = np.argsort(probs)[-k:][::-1]
    best_i = int(idx[0])
    best_p = float(probs[best_i])
    lines: list[str] = []
    for i in idx:
        ii = int(i)
        p = float(probs[ii])
        name = label_map[ii] if 0 <= ii < len(label_map) else str(ii)
        lines.append(f"{name}: {p:.2f}")
    return "\n".join(lines), best_i, best_p


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _save_event(
    *,
    log_dir: Path,
    ts_ms: int,
    event_type: str,
    reason: str,
    args: argparse.Namespace,
    feature_set: str,
    pred_label: str,
    pred_idx: int,
    max_prob: float,
    topk_text: str,
    seq_feat: np.ndarray,
    seq_lms_543: np.ndarray,
    frame_bgr: np.ndarray,
    frame_buf: deque[np.ndarray] | None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    base = f"{ts_ms}_{event_type}"
    npz_path = log_dir / f"{base}.npz"
    jpg_path = log_dir / f"{base}.jpg"
    mp4_path = log_dir / f"{base}.mp4"

    cv2.imwrite(str(jpg_path), frame_bgr)
    np.savez_compressed(
        npz_path,
        seq_feat=seq_feat.astype(np.float32, copy=False),
        seq_lms_543=seq_lms_543.astype(np.float32, copy=False),
        feature_set=feature_set,
        pred_label=pred_label,
        pred_idx=int(pred_idx),
        max_prob=float(max_prob),
        topk=topk_text,
        reason=reason,
        timestamp_ms=int(ts_ms),
    )

    if getattr(args, "save_clip", False) and frame_buf is not None and len(frame_buf) > 1:
        h, w = frame_buf[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, 15.0, (w, h))
        try:
            for fr in list(frame_buf):
                writer.write(fr)
        finally:
            writer.release()

    _append_jsonl(
        log_dir / "events.jsonl",
        {
            "timestamp_ms": int(ts_ms),
            "event_type": event_type,
            "reason": reason,
            "pred_label": pred_label,
            "pred_idx": int(pred_idx),
            "max_prob": float(max_prob),
            "topk": topk_text,
            "npz": str(npz_path),
            "jpg": str(jpg_path),
            "mp4": str(mp4_path) if mp4_path.exists() else "",
            "feature_set": feature_set,
            "seq_len": int(args.seq_len),
            "camera": int(args.camera),
            "backend": str(args.backend),
            "device": str(args.device),
        },
    )


def _maybe_log_event(
    *,
    args: argparse.Namespace,
    log_dir: Path,
    last_log_ts: float,
    pred_hist: deque[int],
    max_prob: float,
    topk_text: str,
    pred_label: str,
    pred_idx: int,
    seq: deque[np.ndarray],
    seq_lms: deque[np.ndarray],
    frame_bgr: np.ndarray,
    frame_buf: deque[np.ndarray],
) -> float:
    now_ts = time.time()
    if (now_ts - last_log_ts) < float(args.log_cooldown_s):
        return last_log_ts

    reason = ""
    event_type = ""
    if max_prob < float(args.log_lowconf_threshold):
        event_type = "lowconf"
        reason = f"max_prob={max_prob:.3f} < {float(args.log_lowconf_threshold):.3f}"
    else:
        if len(pred_hist) >= 2:
            changes = sum(1 for i in range(1, len(pred_hist)) if pred_hist[i] != pred_hist[i - 1])
            if changes >= int(args.log_jitter_changes):
                event_type = "jitter"
                reason = f"changes={changes} in window={len(pred_hist)}"

    if not event_type:
        return last_log_ts

    ts_ms = int(time.time() * 1000)
    _save_event(
        log_dir=log_dir,
        ts_ms=ts_ms,
        event_type=event_type,
        reason=reason,
        args=args,
        feature_set=args.feature_set,
        pred_label=pred_label,
        pred_idx=pred_idx,
        max_prob=max_prob,
        topk_text=topk_text,
        seq_feat=np.stack(seq, axis=0),
        seq_lms_543=np.stack(seq_lms, axis=0),
        frame_bgr=frame_bgr,
        frame_buf=frame_buf if getattr(args, "save_clip", False) else None,
    )
    return now_ts


def _worker(args: argparse.Namespace, state: _SharedState) -> None:
    cap: cv2.VideoCapture | None = None
    try:
        with state.lock:
            state.status = f"Opening camera (index={args.camera}, backend={args.backend})..."
        cap = _open_camera(args.camera, args.backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera} (try --backend msmf/dshow/any)")

        device = torch.device(args.device)
        with state.lock:
            state.status = "Loading model..."
        model, _ = load_checkpoint(Path(args.checkpoint), device=device)
        label_map = load_label_map(Path(args.label_map))
        model.eval()

        expected = getattr(model, "config").feature_dim
        actual = feature_dim_for_set(args.feature_set)
        if expected != actual:
            raise ValueError(f"Model feature_dim={expected} but --feature-set {args.feature_set} gives {actual}.")

        with state.lock:
            state.status = "Initializing MediaPipe..."

        use_solutions = hasattr(mp, "solutions")
        log_dir = Path(args.log_dir)

        seq: deque[np.ndarray] = deque(maxlen=int(args.seq_len))
        seq_lms: deque[np.ndarray] = deque(maxlen=int(args.seq_len))  # flattened 543*3
        frame_buf: deque[np.ndarray] = deque(maxlen=max(1, int(args.clip_frames)))
        prob_hist: deque[np.ndarray] = deque(maxlen=max(1, int(args.smooth)))
        pred_hist: deque[int] = deque(maxlen=max(2, int(args.log_jitter_window)))

        last_pred_display = "..."
        last_conf = 0.0
        last_topk = ""
        last_log_ts = 0.0

        min_dt = 1.0 / max(1e-6, float(args.fps_limit))
        last_time = 0.0
        frame_idx = 0

        if use_solutions:
            mp_holistic = mp.solutions.holistic
            mp_draw = mp.solutions.drawing_utils
            mp_draw_styles = mp.solutions.drawing_styles
            holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detectors = None
        else:
            holistic = None
            models = resolve_models(args.mp_model_dir, use_face=not args.no_face, use_pose=not args.no_pose)
            detectors = create_detectors(models=models)

        with state.lock:
            state.status = "Running"

        try:
            while not state.stop.is_set():
                now = time.time()
                if last_time != 0.0 and (now - last_time) < min_dt:
                    time.sleep(0.001)
                    continue
                last_time = now

                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed")
                frame_idx += 1

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if use_solutions:
                    assert holistic is not None
                    res = holistic.process(rgb)
                    lms_543 = holistic_to_543(res)

                    # overlays (optional but nice for UI)
                    if res.face_landmarks is not None:
                        mp_draw.draw_landmarks(
                            image=frame,
                            landmark_list=res.face_landmarks,
                            connections=mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style(),
                        )
                    if res.left_hand_landmarks is not None:
                        mp_draw.draw_landmarks(
                            image=frame,
                            landmark_list=res.left_hand_landmarks,
                            connections=mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_draw_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_draw_styles.get_default_hand_connections_style(),
                        )
                    if res.right_hand_landmarks is not None:
                        mp_draw.draw_landmarks(
                            image=frame,
                            landmark_list=res.right_hand_landmarks,
                            connections=mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_draw_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_draw_styles.get_default_hand_connections_style(),
                        )
                    if res.pose_landmarks is not None:
                        mp_draw.draw_landmarks(
                            image=frame,
                            landmark_list=res.pose_landmarks,
                            connections=mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                            connection_drawing_spec=mp_draw_styles.get_default_pose_connections_style(),
                        )
                else:
                    assert detectors is not None
                    ts_ms = int(time.time() * 1000)
                    lms_543, face_pts, left, right, pose_pts = detect_543_tasks(detectors, rgb, ts_ms)
                    draw_overlays_tasks(frame, face_pts, left, right, pose_pts)

                feat = select_feature_set(lms_543, args.feature_set)
                seq.append(feat)
                seq_lms.append(lms_543.reshape(-1))
                frame_buf.append(frame.copy())

                # Only predict when we have a full window
                if len(seq) == int(args.seq_len) and (frame_idx % max(1, int(args.predict_every)) == 0):
                    x = torch.from_numpy(np.stack(seq, axis=0)).unsqueeze(0).to(device)
                    mask = torch.ones((1, int(args.seq_len)), dtype=torch.bool, device=device)
                    with torch.no_grad():
                        logits = model(x, mask=mask)
                        prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                    prob_hist.append(prob)
                    avg_prob = np.mean(np.stack(list(prob_hist), axis=0), axis=0)

                    topk_text, pred_idx, max_prob = _format_topk(label_map, avg_prob, args.topk)
                    pred_raw = label_map[pred_idx] if 0 <= pred_idx < len(label_map) else str(pred_idx)
                    pred_hist.append(pred_idx)

                    pred_display = pred_raw if max_prob >= float(args.unknown_threshold) else "Unknown"

                    last_pred_display = pred_display
                    last_conf = float(max_prob)
                    last_topk = topk_text

                    last_log_ts = _maybe_log_event(
                        args=args,
                        log_dir=log_dir,
                        last_log_ts=last_log_ts,
                        pred_hist=pred_hist,
                        max_prob=float(max_prob),
                        topk_text=topk_text,
                        pred_label=pred_raw,
                        pred_idx=int(pred_idx),
                        seq=seq,
                        seq_lms=seq_lms,
                        frame_bgr=frame.copy(),
                        frame_buf=frame_buf,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with state.lock:
                    state.frame_rgb = frame_rgb
                    state.pred_text = last_pred_display
                    state.pred_detail = f"conf={last_conf:.2f}\n{last_topk}" if last_topk else ""
        finally:
            if holistic is not None:
                holistic.close()

    except BaseException as e:
        with state.lock:
            state.status = f"Error: {type(e).__name__}: {e}"
        print(traceback.format_exc(), flush=True)
    finally:
        if cap is not None:
            cap.release()


class _App(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.title("SignVision - ASL Recognition")
        self.geometry("980x720")
        self.minsize(720, 540)

        self.state = _SharedState(lock=threading.Lock(), stop=threading.Event())
        self.worker = threading.Thread(target=_worker, args=(args, self.state), daemon=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._photo: ImageTk.PhotoImage | None = None
        self.worker.start()
        self.after(10, self._tick)

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=TOP, fill=X)

        self.pred_var = tk.StringVar(value="...")
        self.detail_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Starting...")

        title = ttk.Label(top, text="SignVision", font=("Segoe UI", 18, "bold"), foreground="#d7261e")
        title.pack(side=LEFT, padx=12, pady=10)

        pred = ttk.Label(top, textvariable=self.pred_var, font=("Segoe UI", 18))
        pred.pack(side=LEFT, padx=12)

        detail = ttk.Label(top, textvariable=self.detail_var, font=("Segoe UI", 10))
        detail.pack(side=LEFT, padx=12)

        status = ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10))
        status.pack(side=LEFT, padx=12)

        body = ttk.Frame(self)
        body.pack(fill=BOTH, expand=True)

        self.video = ttk.Label(body)
        self.video.pack(fill=BOTH, expand=True, padx=12, pady=12)

        hint = ttk.Label(self, text="Close window to quit.", font=("Segoe UI", 10))
        hint.pack(side=TOP, pady=(0, 10))

    def _tick(self) -> None:
        with self.state.lock:
            frame_rgb = None if self.state.frame_rgb is None else self.state.frame_rgb.copy()
            pred_text = self.state.pred_text
            pred_detail = self.state.pred_detail
            status_text = self.state.status

        if pred_text:
            self.pred_var.set(pred_text)
        self.detail_var.set(pred_detail)
        self.status_var.set(status_text)

        if frame_rgb is not None:
            h, w = frame_rgb.shape[:2]
            target_w = max(1, self.video.winfo_width())
            target_h = max(1, self.video.winfo_height())
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = Image.fromarray(frame_rgb).resize((new_w, new_h), resample=Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(image=img)
            self.video.configure(image=self._photo)

        self.after(15, self._tick)

    def _on_close(self) -> None:
        self.state.stop.set()
        self.destroy()


def main() -> None:
    args = _parse_args()
    _App(args).mainloop()


if __name__ == "__main__":
    main()
