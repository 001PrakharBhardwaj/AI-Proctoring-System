"""Live alert manager for proctoring violations."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import cv2

import config


class ViolationType(Enum):
    MULTIPLE_PEOPLE = "Multiple people detected"
    PHONE_DETECTED = "Mobile phone detected"
    LOOKING_AWAY = "Looking away from screen"
    NO_FACE = "No face detected"
    TAB_SWITCH = "Tab/window switch detected"
    SUSPICIOUS = "Suspicious activity"


@dataclass
class Alert:
    violation: ViolationType
    message: str
    timestamp: float = field(default_factory=time.time)

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


class AlertManager:
    def __init__(self) -> None:
        self.history: List[Alert] = []
        self._last_fired: dict[ViolationType, float] = {}
        self._active_banner: Optional[Alert] = None
        self._banner_until: float = 0.0

    def trigger(
        self,
        violation: ViolationType,
        detail: str = "",
        force: bool = False,
    ) -> bool:
        now = time.time()
        last = self._last_fired.get(violation, 0.0)
        if not force and (now - last) < config.ALERT_COOLDOWN_SEC:
            return False

        message = violation.value
        if detail:
            message = f"{message}: {detail}"

        alert = Alert(violation=violation, message=message, timestamp=now)
        self.history.append(alert)
        self._last_fired[violation] = now
        self._active_banner = alert
        self._banner_until = now + config.SHOW_ALERT_BANNER_SEC

        self._log(alert)
        print(f"[ALERT {alert.time_str}] {alert.message}")
        return True

    def _log(self, alert: Alert) -> None:
        try:
            with open(config.ALERT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.fromtimestamp(alert.timestamp).isoformat()} | {alert.message}\n")
        except OSError:
            pass

    def recent(self, limit: int = 8) -> List[Alert]:
        return self.history[-limit:]

    def draw_overlay(self, frame, panel_x: int) -> None:
        h, w = frame.shape[:2]
        now = time.time()

        # Side panel background
        cv2.rectangle(frame, (panel_x, 0), (w, h), (30, 30, 30), -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h), (80, 80, 80), 2)

        y = 30
        cv2.putText(frame, "LIVE ALERTS", (panel_x + 15, y), config.FONT, 0.7, (255, 255, 255), 2)
        y += 35

        for alert in reversed(self.recent(8)):
            color = self._color_for(alert.violation)
            text = f"{alert.time_str} {alert.violation.name[:12]}"
            cv2.putText(frame, text, (panel_x + 10, y), config.FONT, 0.45, color, 1)
            y += 22
            if y > h - 40:
                break

        # Active violation banner on video area
        if self._active_banner and now < self._banner_until:
            banner = self._active_banner
            color = self._color_for(banner.violation)
            cv2.rectangle(frame, (0, 0), (panel_x, 50), color, -1)
            cv2.putText(
                frame,
                banner.message[:55],
                (10, 35),
                config.FONT,
                0.65,
                (255, 255, 255),
                2,
            )

    @staticmethod
    def _color_for(violation: ViolationType):
        colors = {
            ViolationType.MULTIPLE_PEOPLE: (0, 0, 255),
            ViolationType.PHONE_DETECTED: (0, 140, 255),
            ViolationType.LOOKING_AWAY: (0, 165, 255),
            ViolationType.NO_FACE: (0, 0, 200),
            ViolationType.TAB_SWITCH: (255, 0, 255),
            ViolationType.SUSPICIOUS: (0, 0, 255),
        }
        return colors.get(violation, (0, 0, 255))

    def draw_status_badges(self, frame, statuses: dict) -> None:
        x, y = 10, 70
        for label, active in statuses.items():
            color = (0, 0, 255) if active else (0, 180, 0)
            symbol = "!" if active else "OK"
            cv2.putText(frame, f"{symbol} {label}", (x, y), config.FONT, 0.55, color, 2)
            y += 28
