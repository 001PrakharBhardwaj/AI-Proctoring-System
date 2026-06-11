"""Download and cache MediaPipe task model files."""

import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODELS = {
    "face_detector": {
        "filename": "blaze_face_short_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite"
        ),
    },
    "face_landmarker": {
        "filename": "face_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task"
        ),
    },
}


def get_model_path(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    info = MODELS[name]
    path = MODEL_DIR / info["filename"]

    if not path.exists():
        print(f"Downloading {info['filename']}...")
        urllib.request.urlretrieve(info["url"], path)
        print(f"Saved to {path}")

    return str(path)
