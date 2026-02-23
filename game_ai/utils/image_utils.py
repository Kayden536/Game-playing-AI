import base64
import io
import numpy as np
from PIL import Image
import cv2


def numpy_to_base64(frame: np.ndarray) -> str:
    """Convert numpy array to base64 string for Ollama."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Resize to reduce token usage
    img = img.resize((512, 384), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def highlight_regions(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw bounding boxes on frame for debugging."""
    debug_frame = frame.copy()
    for region in regions:
        x, y, w, h = region
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return debug_frame


def compare_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Return similarity score between two frames (0-1)."""
    if frame1 is None or frame2 is None:
        return 0.0
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (64, 48))
    gray2 = cv2.resize(gray2, (64, 48))
    score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    return float(score[0][0])