"""Eye tracking and gaze estimation using MediaPipe Face Landmarker."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from model_loader import get_model_path

# 3D model points for head pose (generic face model)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


@dataclass
class GazeResult:
    face_found: bool
    looking_away: bool
    yaw: float = 0.0
    pitch: float = 0.0
    message: str = ""


class GazeTracker:
    def __init__(self) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=get_model_path("face_landmarker")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def _landmark_xy(self, landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
        lm = landmarks[idx]
        return lm.x * w, lm.y * h

    def _estimate_head_pose(
        self, landmarks, w: int, h: int
    ) -> Optional[Tuple[float, float]]:
        image_points = []
        for key in (
            "nose_tip",
            "chin",
            "left_eye_outer",
            "right_eye_outer",
            "left_mouth",
            "right_mouth",
        ):
            image_points.append(self._landmark_xy(landmarks, _LANDMARK_IDS[key], w, h))
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
        pitch, yaw, _roll = angles
        return float(yaw), float(pitch)

    def analyze(self, frame: np.ndarray) -> GazeResult:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.face_landmarks:
            return GazeResult(face_found=False, looking_away=False, message="No face for gaze")

        landmarks = results.face_landmarks[0]
        pose = self._estimate_head_pose(landmarks, w, h)

        if pose is None:
            return GazeResult(face_found=True, looking_away=False, message="Pose unavailable")

        yaw, pitch = pose
        looking_away = (
            abs(yaw) > config.GAZE_YAW_THRESHOLD or abs(pitch) > config.GAZE_PITCH_THRESHOLD
        )
        msg = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f}"
        if looking_away:
            msg += " [LOOKING AWAY]"

        return GazeResult(
            face_found=True,
            looking_away=looking_away,
            yaw=yaw,
            pitch=pitch,
            message=msg,
        )

    def draw(self, frame: np.ndarray, result: GazeResult) -> None:
        if not result.face_found:
            return
        color = (0, 165, 255) if result.looking_away else (200, 200, 0)
        cv2.putText(frame, result.message, (10, 58), config.FONT, 0.55, color, 2)

    def close(self) -> None:
        self._landmarker.close()
