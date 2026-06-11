"""Mobile phone detection module using YOLO."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config


@dataclass
class PhoneDetection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class PhoneResult:
    detected: bool
    detections: List[PhoneDetection]


class PhoneDetector:
    def __init__(self, model_path: str = config.YOLO_MODEL) -> None:
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> PhoneResult:
        results = self.model(frame, verbose=False)
        detections: List[PhoneDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                if label in config.PHONE_CLASSES and conf >= config.PHONE_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        PhoneDetection(label=label, confidence=conf, box=(x1, y1, x2, y2))
                    )

        return PhoneResult(detected=len(detections) > 0, detections=detections)

    def draw(self, frame: np.ndarray, result: PhoneResult) -> None:
        for det in result.detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.0%}",
                (x1, max(y1 - 8, 20)),
                config.FONT,
                0.6,
                (0, 140, 255),
                2,
            )


def run_standalone() -> None:
    """Standalone demo (original script behavior)."""
    detector = PhoneDetector()
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        if result.detected:
            print("Phone Detected!")
        detector.draw(frame, result)
        cv2.imshow("Phone Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_standalone()
