from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mediapipe as mp


_HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def ensure_model(url: str, dst: str | Path) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted model URL)
    os.replace(tmp, dst)
    return dst


def default_model_dir() -> Path:
    return Path("assets") / "mediapipe"


@dataclass(frozen=True)
class TaskModels:
    hand: Path
    face: Path | None
    pose: Path | None


def resolve_models(model_dir: str | Path, use_face: bool = True, use_pose: bool = True) -> TaskModels:
    model_dir = Path(model_dir)
    hand = ensure_model(_HAND_URL, model_dir / "hand_landmarker.task")
    face = ensure_model(_FACE_URL, model_dir / "face_landmarker.task") if use_face else None
    pose = ensure_model(_POSE_URL, model_dir / "pose_landmarker_lite.task") if use_pose else None
    return TaskModels(hand=hand, face=face, pose=pose)


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# Common pose skeleton (MediaPipe Pose)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
]


def _draw_points(frame_bgr: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], r: int) -> None:
    import cv2

    h, w = frame_bgr.shape[:2]
    for x, y, _z in pts:
        px = int(x * w)
        py = int(y * h)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(frame_bgr, (px, py), r, color, -1, lineType=cv2.LINE_AA)


def _draw_edges(
    frame_bgr: np.ndarray, pts: np.ndarray, edges: list[tuple[int, int]], color: tuple[int, int, int], t: int
) -> None:
    import cv2

    h, w = frame_bgr.shape[:2]
    for a, b in edges:
        if a >= len(pts) or b >= len(pts):
            continue
        x1, y1, _ = pts[a]
        x2, y2, _ = pts[b]
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.line(frame_bgr, p1, p2, color, t, lineType=cv2.LINE_AA)


def draw_overlays_tasks(
    frame_bgr: np.ndarray,
    face: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose: np.ndarray | None,
) -> None:
    # face: draw points only (mesh is huge)
    if face is not None and face.size > 0:
        _draw_points(frame_bgr, face, (255, 200, 200), r=1)
    if left_hand is not None and left_hand.size > 0:
        _draw_edges(frame_bgr, left_hand, HAND_CONNECTIONS, (0, 255, 255), t=2)
        _draw_points(frame_bgr, left_hand, (0, 255, 255), r=2)
    if right_hand is not None and right_hand.size > 0:
        _draw_edges(frame_bgr, right_hand, HAND_CONNECTIONS, (0, 255, 0), t=2)
        _draw_points(frame_bgr, right_hand, (0, 255, 0), r=2)
    if pose is not None and pose.size > 0:
        _draw_edges(frame_bgr, pose, POSE_CONNECTIONS, (255, 255, 0), t=2)
        _draw_points(frame_bgr, pose, (255, 255, 0), r=2)


def _landmarks_to_np(lms: object, count: int | None = None) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
    if count is not None:
        if pts.shape[0] < count:
            pad = np.zeros((count - pts.shape[0], 3), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=0)
        else:
            pts = pts[:count]
    return pts


@dataclass
class TaskDetectors:
    hand: object
    face: object | None
    pose: object | None


def create_detectors(models: TaskModels, num_hands: int = 2) -> TaskDetectors:
    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions

    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(models.hand)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    face = None
    if models.face is not None:
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(models.face)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    pose = None
    if models.pose is not None:
        pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(models.pose)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    return TaskDetectors(hand=hand, face=face, pose=pose)


def detect_543_tasks(
    detectors: TaskDetectors, rgb: np.ndarray, timestamp_ms: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Returns:
      - lms_543: (543,3)
      - face_468: (468,3) or None
      - left_hand_21: (21,3) or None
      - right_hand_21: (21,3) or None
      - pose_33: (33,3) or None
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_pts = None
    if detectors.face is not None:
        fr = detectors.face.detect_for_video(mp_image, timestamp_ms)
        if getattr(fr, "face_landmarks", None):
            face_pts = _landmarks_to_np(fr.face_landmarks[0], count=468)

    pose_pts = None
    if detectors.pose is not None:
        pr = detectors.pose.detect_for_video(mp_image, timestamp_ms)
        if getattr(pr, "pose_landmarks", None):
            pose_pts = _landmarks_to_np(pr.pose_landmarks[0], count=33)

    left = None
    right = None
    hr = detectors.hand.detect_for_video(mp_image, timestamp_ms)
    hands = getattr(hr, "hand_landmarks", None) or []
    handed = getattr(hr, "handedness", None) or []
    for i, lms in enumerate(hands):
        label = None
        if i < len(handed) and handed[i]:
            label = handed[i][0].category_name  # "Left"/"Right"
        pts = _landmarks_to_np(lms, count=21)
        if label == "Left":
            left = pts
        elif label == "Right":
            right = pts
        else:
            if left is None:
                left = pts
            elif right is None:
                right = pts

    out = np.zeros((543, 3), dtype=np.float32)
    if face_pts is not None:
        out[0:468] = face_pts
    if left is not None:
        out[468:489] = left
    if pose_pts is not None:
        out[489:522] = pose_pts
    if right is not None:
        out[522:543] = right

    return out, face_pts, left, right, pose_pts

