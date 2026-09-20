"""MediaPipe wrappers: face landmarks (iris + blendshapes + head pose) and hand landmarks."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FACE_MODEL_PATH, HAND_MODEL_PATH

_BASE = "https://storage.googleapis.com/mediapipe-models"
FACE_MODEL_URL = f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_models() -> None:
    """Download the MediaPipe .task files on first run."""
    for path, url in ((FACE_MODEL_PATH, FACE_MODEL_URL), (HAND_MODEL_PATH, HAND_MODEL_URL)):
        if path.exists() and path.stat().st_size > 500_000:
            continue
        print(f"Baixando {path.name} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)


@dataclass
class FaceData:
    pts: np.ndarray            # (478, 3) landmarks normalised to the image
    blink_left: float
    blink_right: float
    matrix: np.ndarray | None  # 4x4 facial transformation matrix


@dataclass
class HandData:
    pts: np.ndarray    # (21, 3) landmarks normalised to the image
    world: np.ndarray  # (21, 3) metric landmarks (metres), hand-centred


class FaceHandTracker:
    """Runs FaceLandmarker and HandLandmarker on the same frame."""

    def __init__(self, want_hands: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        ))
        self._hand = None
        if want_hands:
            self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            ))

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> tuple[FaceData | None, list[HandData]]:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._detect_face(image, ts_ms), self._detect_hands(image, ts_ms)

    def _detect_face(self, image, ts_ms: int) -> FaceData | None:
        res = self._face.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            return None
        pts = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)
        scores = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        matrix = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        return FaceData(pts, scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0), matrix)

    def _detect_hands(self, image, ts_ms: int) -> list[HandData]:
        if self._hand is None:
            return []
        res = self._hand.detect_for_video(image, ts_ms)
        hands = []
        for lm, wl in zip(res.hand_landmarks, res.hand_world_landmarks):
            hands.append(HandData(
                np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32),
                np.array([(p.x, p.y, p.z) for p in wl], dtype=np.float32),
            ))
        return hands

    def close(self) -> None:
        self._face.close()
        if self._hand is not None:
            self._hand.close()
