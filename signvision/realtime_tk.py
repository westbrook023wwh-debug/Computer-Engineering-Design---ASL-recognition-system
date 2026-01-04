from __future__ import annotations

import argparse
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
    p.add_argument("--mp-model-dir", type=str, default=str(default_model_dir()))
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--feature-set", choices=["hands", "full"], default="hands")
    p.add_argument("--predict-every", type=int, default=2)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument("--fps-limit", type=float, default=30.0)
    return p.parse_args()


@dataclass
class _SharedState:
    lock: threading.Lock
    stop: threading.Event
    frame_rgb: np.ndarray | None = None
    pred_text: str = ""
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
        # auto: try common Windows backends first, then default
        candidates = [cap_dshow, cap_msmf, None]

    def warmup(cap: cv2.VideoCapture, seconds: float = 2.0) -> bool:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Some cameras need a short warm-up; don't fail on the very first frame.
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

    # Return unopened capture to signal failure.
    return cv2.VideoCapture()


def _worker(args: argparse.Namespace, state: _SharedState) -> None:
    cap: cv2.VideoCapture | None = None
    try:
        t0 = time.time()
        with state.lock:
            state.status = f"Opening camera (index={args.camera}, backend={args.backend})..."
        print(f"[signvision] opening camera index={args.camera} backend={args.backend}", flush=True)
        cap = _open_camera(args.camera, args.backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera} (try --camera 1/2 or --backend msmf/dshow)")
        print(f"[signvision] camera opened in {time.time()-t0:.2f}s", flush=True)

        device = torch.device(args.device)
        with state.lock:
            state.status = "Loading model..."
        print(f"[signvision] loading checkpoint={args.checkpoint} on device={device}", flush=True)
        t1 = time.time()
        model, _ = load_checkpoint(Path(args.checkpoint), device=device)
        label_map = load_label_map(Path(args.label_map))
        model.eval()
        print(f"[signvision] model loaded in {time.time()-t1:.2f}s (classes={len(label_map)})", flush=True)

        expected = getattr(model, "config").feature_dim
        actual = feature_dim_for_set(args.feature_set)
        if expected != actual:
            raise ValueError(f"Model feature_dim={expected} but --feature-set {args.feature_set} gives {actual}.")

        with state.lock:
            state.status = "Initializing MediaPipe..."
        if hasattr(mp, "solutions"):
            print("[signvision] using mediapipe.solutions (holistic)", flush=True)
        else:
            print("[signvision] using mediapipe.tasks (landmarkers)", flush=True)

        seq: deque[np.ndarray] = deque(maxlen=args.seq_len)
        prob_hist: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
        last_pred = ""
        frame_idx = 0

        min_dt = 1.0 / max(1e-6, float(args.fps_limit))
        last_time = 0.0

        if hasattr(mp, "solutions"):
            mp_holistic = mp.solutions.holistic
            mp_draw = mp.solutions.drawing_utils
            mp_draw_styles = mp.solutions.drawing_styles

            with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                with state.lock:
                    state.status = "Running"

                while not state.stop.is_set():
                    now = time.time()
                    if last_time != 0.0 and (now - last_time) < min_dt:
                        time.sleep(0.001)
                        continue
                    last_time = now

                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera read failed")
                    frame_idx += 1

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(rgb)

                    lms_543 = holistic_to_543(res)
                    feat = select_feature_set(lms_543, args.feature_set)
                    seq.append(feat)

                    if len(seq) == args.seq_len and (frame_idx % max(1, args.predict_every) == 0):
                        x = torch.from_numpy(np.stack(seq, axis=0)).unsqueeze(0).to(device)
                        mask = torch.ones((1, args.seq_len), dtype=torch.bool, device=device)
                        with torch.no_grad():
                            logits = model(x, mask=mask)
                            prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                        prob_hist.append(prob)
                        avg_prob = np.mean(np.stack(list(prob_hist), axis=0), axis=0)
                        pred_idx = int(avg_prob.argmax())
                        last_pred = label_map[pred_idx] if 0 <= pred_idx < len(label_map) else str(pred_idx)

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

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with state.lock:
                        state.frame_rgb = frame_rgb
                        state.pred_text = last_pred
        else:
            models = resolve_models(args.mp_model_dir, use_face=not args.no_face, use_pose=not args.no_pose)
            detectors = create_detectors(models=models)
            with state.lock:
                state.status = "Running"

            while not state.stop.is_set():
                now = time.time()
                if last_time != 0.0 and (now - last_time) < min_dt:
                    time.sleep(0.001)
                    continue
                last_time = now

                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera read failed")
                frame_idx += 1

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int(time.time() * 1000)
                lms_543, face_pts, left, right, pose_pts = detect_543_tasks(detectors, rgb, ts)

                feat = select_feature_set(lms_543, args.feature_set)
                seq.append(feat)

                if len(seq) == args.seq_len and (frame_idx % max(1, args.predict_every) == 0):
                    x = torch.from_numpy(np.stack(seq, axis=0)).unsqueeze(0).to(device)
                    mask = torch.ones((1, args.seq_len), dtype=torch.bool, device=device)
                    with torch.no_grad():
                        logits = model(x, mask=mask)
                        prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                    prob_hist.append(prob)
                    avg_prob = np.mean(np.stack(list(prob_hist), axis=0), axis=0)
                    pred_idx = int(avg_prob.argmax())
                    last_pred = label_map[pred_idx] if 0 <= pred_idx < len(label_map) else str(pred_idx)

                draw_overlays_tasks(frame, face_pts, left, right, pose_pts)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with state.lock:
                    state.frame_rgb = frame_rgb
                    state.pred_text = last_pred

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
        self.status_var = tk.StringVar(value="Starting...")

        title = ttk.Label(top, text="SignVision", font=("Segoe UI", 18, "bold"), foreground="#d7261e")
        title.pack(side=LEFT, padx=12, pady=10)

        pred = ttk.Label(top, textvariable=self.pred_var, font=("Segoe UI", 18))
        pred.pack(side=LEFT, padx=12)

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
            status_text = self.state.status

        if pred_text:
            self.pred_var.set(pred_text)
        self.status_var.set(status_text)

        if frame_rgb is not None:
            # Resize to fit label keeping aspect ratio
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
