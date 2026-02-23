"""Screen capture and visual processing."""

import numpy as np
import mss
import cv2
from utils.image_utils import numpy_to_base64, compare_frames


class Vision:
    def __init__(self, monitor_index: int = 1, region: dict = None):
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.region = region  # Optional: capture specific region
        self.last_frame = None
        self.current_frame = None

    def capture(self) -> np.ndarray:
        """Capture current screen."""
        if self.region:
            monitor = self.region
        else:
            monitor = self.sct.monitors[self.monitor_index]

        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        self.last_frame = self.current_frame
        self.current_frame = frame
        return frame

    def get_base64_frame(self) -> str:
        """Capture and return base64 encoded frame."""
        frame = self.capture()
        return numpy_to_base64(frame)

    def has_changed(self, threshold: float = 0.95) -> bool:
        """Check if screen has changed significantly."""
        similarity = compare_frames(self.last_frame, self.current_frame)
        return similarity < threshold

    def set_region(self, left: int, top: int, width: int, height: int):
        """Set capture region (for windowed games)."""
        self.region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

    def get_full_monitor(self):
        """Reset to full monitor capture."""
        self.region = None