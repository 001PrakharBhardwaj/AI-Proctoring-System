"""
AI Proctoring System — unified exam monitoring.

Detects:
  - Multiple persons in frame
  - Mobile phone usage (YOLO)
  - Looking away from screen (MediaPipe gaze)
  - No face detected
  - Tab/window switching
  - Live suspicious-activity alerts

Press ESC to exit. Press SPACE to pause/resume monitoring.
"""

import time

import cv2
import numpy as np

import config
from alerts import AlertManager, ViolationType
from eye_tracking import GazeTracker
from face_detection import FaceDetector
from phone_detection import PhoneDetector
from tab_monitor import TabMonitor


class ProctoringSystem:
    def __init__(self) -> None:
        self.face = FaceDetector()
        self.phone = PhoneDetector()
        self.gaze = GazeTracker()
        self.tab = TabMonitor()
        self.alerts = AlertManager()

        self._no_face_streak = 0
        self._away_streak = 0
        self._monitoring = True
        self._frame_count = 0
        self._start_time = time.time()

    def _process_violations(
        self,
        face_result,
        phone_result,
        gaze_result,
        focus,
    ) -> dict:
        statuses = {
            "Single person": True,
            "Face visible": True,
            "Eyes on screen": True,
            "No phone": True,
            "Window focused": focus.focused,
        }

        if not self._monitoring:
            return statuses

        # No face
        if face_result.count == 0:
            self._no_face_streak += 1
            statuses["Face visible"] = False
            if self._no_face_streak >= config.NO_FACE_FRAMES_THRESHOLD:
                self.alerts.trigger(ViolationType.NO_FACE)
        else:
            self._no_face_streak = 0

        # Multiple people
        if face_result.count > config.MAX_ALLOWED_FACES:
            statuses["Single person"] = False
            self.alerts.trigger(
                ViolationType.MULTIPLE_PEOPLE,
                detail=f"{face_result.count} faces",
            )

        # Phone
        if phone_result.detected:
            statuses["No phone"] = False
            labels = ", ".join(d.label for d in phone_result.detections)
            self.alerts.trigger(ViolationType.PHONE_DETECTED, detail=labels)

        # Gaze (only when exactly one face in frame)
        if face_result.count == 1 and gaze_result.face_found:
            if gaze_result.looking_away:
                self._away_streak += 1
                statuses["Eyes on screen"] = False
                if self._away_streak >= config.LOOKING_AWAY_FRAMES_THRESHOLD:
                    self.alerts.trigger(
                        ViolationType.LOOKING_AWAY,
                        detail=f"yaw={gaze_result.yaw:.0f} pitch={gaze_result.pitch:.0f}",
                    )
            else:
                self._away_streak = 0

        # Tab / window switch
        if not focus.focused:
            self.alerts.trigger(
                ViolationType.TAB_SWITCH,
                detail=focus.active_title[:40] or "unknown window",
            )

        # Flag when multiple violation types occur at once
        failed = [k for k, v in statuses.items() if not v]
        if len(failed) >= 2:
            self.alerts.trigger(
                ViolationType.SUSPICIOUS,
                detail=" + ".join(failed),
            )

        return statuses

    def _build_display(self, frame: np.ndarray, statuses: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = config.PANEL_WIDTH
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        # Border color from worst status
        border_color = (0, 180, 0)
        if any(not v for v in statuses.values()):
            border_color = (0, 0, 255)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_color, 4)

        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        state = "MONITORING" if self._monitoring else "PAUSED"
        cv2.putText(
            canvas[:h, :w],
            f"{state} | {mins:02d}:{secs:02d} | Frame {self._frame_count}",
            (10, h - 15),
            config.FONT,
            0.5,
            (200, 200, 200),
            1,
        )

        self.alerts.draw_status_badges(canvas[:h, :w], statuses)
        self.alerts.draw_overlay(canvas, panel_x=w)

        # Panel legend
        cv2.putText(
            canvas,
            "Controls:",
            (w + 15, h - 80),
            config.FONT,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.putText(canvas, "ESC - Exit", (w + 15, h - 55), config.FONT, 0.45, (150, 150, 150), 1)
        cv2.putText(canvas, "SPACE - Pause", (w + 15, h - 30), config.FONT, 0.45, (150, 150, 150), 1)

        return canvas

    def run(self) -> None:
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)

        print("AI Proctoring System started.")
        print("Monitoring: face, phone, gaze, window focus.")
        print("Alerts logged to:", config.ALERT_LOG_FILE)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Webcam read failed.")
                    break

                self._frame_count += 1
                frame = cv2.flip(frame, 1)

                face_result = self.face.detect(frame)
                phone_result = self.phone.detect(frame)
                gaze_result = self.gaze.analyze(frame)
                focus = self.tab.check()

                self.face.draw(
                    frame,
                    face_result,
                    multiple=face_result.count > config.MAX_ALLOWED_FACES,
                )
                self.phone.draw(frame, phone_result)
                self.gaze.draw(frame, gaze_result)

                statuses = self._process_violations(
                    face_result, phone_result, gaze_result, focus
                )
                display = self._build_display(frame, statuses)
                cv2.imshow(config.WINDOW_TITLE, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    self._monitoring = not self._monitoring
                    print("Monitoring", "resumed" if self._monitoring else "paused")

        finally:
            cap.release()
            self.face.close()
            self.gaze.close()
            cv2.destroyAllWindows()
            print(f"Session ended. Total alerts: {len(self.alerts.history)}")


def main() -> None:
    system = ProctoringSystem()
    system.run()


if __name__ == "__main__":
    main()
