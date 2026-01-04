from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import time

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
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--feature-set", choices=["hands", "full"], default="hands")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    p.add_argument("--mp-model-dir", type=str, default=str(default_model_dir()))
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--predict-every", type=int, default=2)
    p.add_argument("--smooth", type=int, default=8)
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
        if warmup(cap):
            return cap
        cap.release()
    return cv2.VideoCapture()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model, ckpt = load_checkpoint(Path(args.checkpoint), device=device)
    label_map = load_label_map(Path(args.label_map))
    model.eval()

    # Basic guard: feature dims should match.
    expected = getattr(model, "config").feature_dim
    actual = feature_dim_for_set(args.feature_set)
    if expected != actual:
        raise ValueError(f"Model feature_dim={expected} but --feature-set {args.feature_set} gives {actual}.")

    seq: deque[np.ndarray] = deque(maxlen=args.seq_len)
    prob_hist: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    last_pred = ""
    frame_idx = 0

    cap = _open_camera(args.camera, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera} (try --camera 1/2 or --backend dshow)")

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
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
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

                cv2.putText(frame, last_pred, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow("SignVision - ASL Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    else:
        models = resolve_models(args.mp_model_dir, use_face=not args.no_face, use_pose=not args.no_pose)
        detectors = create_detectors(models=models)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
            cv2.putText(frame, last_pred, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("SignVision - ASL Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
