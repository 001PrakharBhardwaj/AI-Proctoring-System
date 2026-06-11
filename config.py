"""Configuration for the AI Proctoring System."""

WEBCAM_INDEX = 0
WINDOW_TITLE = "AI Proctoring System"

# Face detection (MediaPipe)
FACE_MIN_CONFIDENCE = 0.5
MAX_ALLOWED_FACES = 1

# Phone detection (YOLO)
YOLO_MODEL = "yolov8n.pt"
PHONE_CONFIDENCE = 0.45
PHONE_CLASSES = ("cell phone", "remote")  # remote often picked up as phone-like

# Gaze / head pose (MediaPipe Face Mesh)
GAZE_YAW_THRESHOLD = 25.0   # degrees left/right
GAZE_PITCH_THRESHOLD = 20.0  # degrees up/down
NO_FACE_FRAMES_THRESHOLD = 15  # consecutive frames before alert
LOOKING_AWAY_FRAMES_THRESHOLD = 10

# Tab / window focus
TAB_SWITCH_CHECK_INTERVAL = 0.5  # seconds
EXAM_WINDOW_KEYWORDS = ("proctoring", "exam")

# Alerts
ALERT_COOLDOWN_SEC = 4.0
ALERT_LOG_FILE = "proctoring_alerts.log"
SHOW_ALERT_BANNER_SEC = 3.0

# Display
PANEL_WIDTH = 320
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
