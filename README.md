# AI Proctoring System

Real-time exam monitoring using **OpenCV**, **YOLO**, and **MediaPipe**.

## Detections

| Feature | Technology |
|---------|------------|
| Face detection | MediaPipe Face Detection |
| Multiple people | Face count > 1 |
| Mobile phone | YOLOv8 (`cell phone` class) |
| Looking away | MediaPipe Face Mesh + head pose |
| No face | Consecutive frames without face |
| Tab switching | Foreground window monitor (Windows) |
| Live alerts | On-screen banner + log file |

## Setup

```bash
pip install -r requirements.txt
```

On first run, YOLO downloads `yolov8n.pt` automatically.

## Run

```bash
python main.py
```

- **ESC** — exit  
- **SPACE** — pause / resume monitoring  

Alerts are written to `proctoring_alerts.log`.

### Standalone demos

```bash
python face_detection.py
python phone_detection.py
```

### Browser tab monitoring

Open `web/tab_monitor.html` during a web-based exam for tab-switch detection via the Page Visibility API.

## Project structure

```
main.py              # Integrated proctoring app
face_detection.py    # MediaPipe face module
phone_detection.py   # YOLO phone module
eye_tracking.py      # Gaze / head pose module
tab_monitor.py       # Window focus monitor
alerts.py            # Alert manager
config.py            # Thresholds and settings
web/tab_monitor.html # Browser tab detection
```

## Configuration

Edit `config.py` to tune thresholds (gaze angles, alert cooldown, YOLO confidence, etc.).
