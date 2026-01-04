from __future__ import annotations

import numpy as np


def _lm_to_np(landmarks: object, n: int) -> np.ndarray | None:
    if landmarks is None:
        return None
    pts = getattr(landmarks, "landmark", None)
    if pts is None:
        return None
    if len(pts) != n:
        return None
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def holistic_to_543(frame_result: object) -> np.ndarray:
    """
    Returns (543,3) float32 with ordering: face(468), left hand(21), pose(33), right hand(21).
    Missing parts are zeros.
    """
    out = np.zeros((543, 3), dtype=np.float32)

    face = _lm_to_np(getattr(frame_result, "face_landmarks", None), 468)
    if face is not None:
        out[0:468] = face

    left = _lm_to_np(getattr(frame_result, "left_hand_landmarks", None), 21)
    if left is not None:
        out[468:489] = left

    pose = _lm_to_np(getattr(frame_result, "pose_landmarks", None), 33)
    if pose is not None:
        out[489:522] = pose

    right = _lm_to_np(getattr(frame_result, "right_hand_landmarks", None), 21)
    if right is not None:
        out[522:543] = right

    return out

