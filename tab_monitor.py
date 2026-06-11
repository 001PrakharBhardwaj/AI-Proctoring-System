"""Detect when the user switches away from the proctoring window."""

import ctypes
import sys
import time
from dataclasses import dataclass

import config


@dataclass
class FocusState:
    focused: bool
    active_title: str


class TabMonitor:
    """Monitors foreground window on Windows; falls back to always-focused elsewhere."""

    def __init__(self, window_title: str = config.WINDOW_TITLE) -> None:
        self.window_title = window_title.lower()
        self._last_check = 0.0
        self._cached = FocusState(focused=True, active_title="")

    def _get_foreground_title(self) -> str:
        if sys.platform != "win32":
            return self.window_title

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value or ""

    def check(self) -> FocusState:
        now = time.time()
        if now - self._last_check < config.TAB_SWITCH_CHECK_INTERVAL:
            return self._cached

        self._last_check = now
        title = self._get_foreground_title()
        title_lower = title.lower()

        focused = any(kw in title_lower for kw in config.EXAM_WINDOW_KEYWORDS) or (
            self.window_title in title_lower
        )
        if sys.platform != "win32":
            focused = True

        self._cached = FocusState(focused=focused, active_title=title)
        return self._cached
