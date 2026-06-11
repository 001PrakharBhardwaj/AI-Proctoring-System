"""Face detection module using MediaPipe Tasks API."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from model_loader import get_model_path


@dataclass
class FaceResult:
    count: int
    boxes: List[Tuple[int, int, int, int]]  # x, y, w, h
    confidences: List[float]


class FaceDetector:
    def __init__(self) -> None:
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=get_model_path("face_detector")),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=config.FACE_MIN_CONFIDENCE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> FaceResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        results = self._detector.detect_for_video(mp_image, self._timestamp_ms)

        boxes: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []

        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.bounding_box
                x = max(0, int(bbox.origin_x))
                y = max(0, int(bbox.origin_y))
                width = min(int(bbox.width), w - x)
                height = min(int(bbox.height), h - y)
                boxes.append((x, y, width, height))
                confidences.append(float(detection.categories[0].score))

        return FaceResult(count=len(boxes), boxes=boxes, confidences=confidences)

    def draw(self, frame: np.ndarray, result: FaceResult, multiple: bool = False) -> None:
        color = (0, 0, 255) if multiple else (0, 255, 0)
        for (x, y, w, h) in result.boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"Faces: {result.count}"
        if result.count > config.MAX_ALLOWED_FACES:
            label += " [VIOLATION]"
        cv2.putText(frame, label, (10, 30), config.FONT, 0.8, color, 2)

    def close(self) -> None:
        self._detector.close()


def run_standalone() -> None:
    """Standalone demo (original script behavior)."""
    detector = FaceDetector()
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)

    while True:
        success, frame = cap.read()
        if not success:
            break

        result = detector.detect(frame)
        detector.draw(frame, result, multiple=result.count > config.MAX_ALLOWED_FACES)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_standalone()
