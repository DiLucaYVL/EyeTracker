"""Turn MediaPipe face landmarks into a compact gaze feature vector."""
from __future__ import annotations

import math

import numpy as np

# Feature layout: [uR, vR, uL, vL, lid, yaw, pitch, roll, tx, ty, tz]
N_FEATURES = 11

# MediaPipe FaceMesh indices (with iris refinement, 478 points).
# For each eye: (image-left corner, image-right corner, upper lid, lower lid, iris centre)
RIGHT_EYE = (33, 133, 159, 145, 468)   # subject's right eye (image left)
LEFT_EYE = (362, 263, 386, 374, 473)   # subject's left eye (image right)


def eye_features(pts: np.ndarray, idx: tuple[int, int, int, int, int]) -> tuple[float, float, float] | None:
    """Iris position in eye-local coordinates (u along the eye axis, v perpendicular), plus lid opening.

    Everything is normalised by the eye width so it is invariant to face distance.
    u > 0: iris towards image right. v > 0: iris towards image bottom.
    """
    left, right, upper, lower, iris = idx
    a, b = pts[left, :2], pts[right, :2]
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    ax = axis / width
    perp = np.array([-ax[1], ax[0]])
    d = pts[iris, :2] - (a + b) / 2.0
    opening = float(np.linalg.norm(pts[upper, :2] - pts[lower, :2])) / width
    return float(d @ ax) / width, float(d @ perp) / width, opening


def head_pose(matrix: np.ndarray | None) -> np.ndarray:
    """Yaw, pitch, roll (rad) and translation (cm) from the facial transformation matrix."""
    if matrix is None:
        return np.zeros(6)
    m = np.asarray(matrix, dtype=np.float64)
    rot = m[:3, :3]
    scale = np.linalg.norm(rot, axis=0)
    if np.any(scale < 1e-9):
        return np.zeros(6)
    rot = rot / scale
    yaw = math.atan2(rot[0, 2], rot[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -rot[1, 2])))
    roll = math.atan2(rot[1, 0], rot[1, 1])
    return np.array([yaw, pitch, roll, m[0, 3], m[1, 3], m[2, 3]])


def extract_features(norm_pts: np.ndarray, width: int, height: int, matrix: np.ndarray | None) -> np.ndarray | None:
    """norm_pts: (478, 3) landmarks normalised to the image. Returns the 11-D feature vector."""
    if norm_pts.shape[0] < 478:
        return None
    pts = norm_pts[:, :2] * np.array([width, height], dtype=np.float64)
    r = eye_features(pts, RIGHT_EYE)
    l = eye_features(pts, LEFT_EYE)
    if r is None or l is None:
        return None
    lid = (r[2] + l[2]) / 2.0
    return np.concatenate([[r[0], r[1], l[0], l[1], lid], head_pose(matrix)])


def face_box(norm_pts: np.ndarray) -> tuple[float, float, float]:
    """(centre x, centre y, width) of the face in normalised image coordinates."""
    xs, ys = norm_pts[:, 0], norm_pts[:, 1]
    return float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2), float(xs.max() - xs.min())


def eye_region_box(norm_pts: np.ndarray, width: int, height: int, aspect: float = 3.6) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) around both eyes with a fixed width/height aspect, clipped to the image."""
    idx = [33, 133, 159, 145, 362, 263, 386, 374]
    xs = norm_pts[idx, 0] * width
    ys = norm_pts[idx, 1] * height
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w = (xs.max() - xs.min()) * 1.35
    h = max((ys.max() - ys.min()) * 1.8, w / aspect)
    w = max(w, h * aspect)
    x0, x1 = int(max(cx - w / 2, 0)), int(min(cx + w / 2, width))
    y0, y1 = int(max(cy - h / 2, 0)), int(min(cy + h / 2, height))
    return x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)
