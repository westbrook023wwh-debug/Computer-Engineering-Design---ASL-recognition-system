from __future__ import annotations

import numpy as np


# Kaggle ASL Signs (commonly used ordering):
# 468 face + 21 left hand + 33 pose + 21 right hand = 543
FACE = np.arange(0, 468)
LEFT_HAND = np.arange(468, 489)
POSE = np.arange(489, 522)
RIGHT_HAND = np.arange(522, 543)


def select_feature_set(frame_landmarks: np.ndarray, feature_set: str) -> np.ndarray:
    """
    frame_landmarks: (543,3) float
    feature_set: "full" or "hands"
    returns flattened (F,) float32
    """
    if frame_landmarks.shape != (543, 3):
        raise ValueError(f"Expected (543,3), got {frame_landmarks.shape}")
    if feature_set == "full":
        out = frame_landmarks
    elif feature_set == "hands":
        out = np.concatenate([frame_landmarks[LEFT_HAND], frame_landmarks[RIGHT_HAND]], axis=0)
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    return out.astype(np.float32, copy=False).reshape(-1)


def feature_dim_for_set(feature_set: str) -> int:
    if feature_set == "hands":
        return 21 * 3 * 2
    if feature_set == "full":
        return 543 * 3
    raise ValueError(f"Unknown feature_set={feature_set}")

