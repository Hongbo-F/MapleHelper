"""YOLO11n pattern detections for the Lie Detector mini-game, in shadow mode only.

The detector suggests the centre of every pattern it sees. It does not decide which pattern is the
fading target, move the pointer, press a key, or call any other input API. Ultralytics and torch are
loaded lazily so the ordinary template spotter and its tests stay lightweight.
"""

from pathlib import Path

import cv2
import numpy as np

from ..paths import ASSETS


MODEL = ASSETS / "lie_detector_patterns.pt"
FORM = "yolo patterns"
CONFIDENCE = 0.5
MIN_PATTERNS = 4  # active boards have about 13-17; one stray detection isn't the mini-game
IMAGE_SIZE = 640
NATIVE_FRAME = (1366, 768)
BOARD_AT_NATIVE = (750, 500)  # the annotation videos' mini-game ROI, centred in a full game frame
FULL_FRAME_WIDTH = 1000  # smaller inputs are already board crops, as in the training/validation set


class YoloPatternShadow:
    """Runs the local YOLO checkpoint on RGB game frames and returns scores and pointer suggestions.

    The score is the number of detections divided by MIN_PATTERNS, so >= 1 means a likely active
    pattern board. Each suggestion has a bounding box and centre in original frame pixels, plus its
    confidence. A model can be injected for tests; otherwise it is loaded on the first check.
    """

    def __init__(self, model_path: Path = MODEL, confidence: float = CONFIDENCE,
                 min_patterns: int = MIN_PATTERNS, device=None, model=None):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.min_patterns = min_patterns
        self.device = device
        self._model = model

    def _load(self):
        if self._model is None:
            import torch
            from ultralytics import YOLO

            if self.device is None:
                self.device = 0 if torch.cuda.is_available() else "cpu"
            self._model = YOLO(str(self.model_path))
        return self._model

    def check(self, frame: np.ndarray) -> tuple[float, list[dict]]:
        """Return (normalized pattern-board score, pattern-centre suggestions); never sends input."""
        model = self._load()
        device = "cpu" if self.device is None else self.device
        h, w = frame.shape[:2]
        x0 = y0 = 0
        board = frame
        if w >= FULL_FRAME_WIDTH:
            # Whole-screen inference shrank the board and missed most patterns. Use its native-size ROI instead;
            # scale that ROI with the game window and translate detections back to original frame pixels below.
            bw = min(w, round(w * BOARD_AT_NATIVE[0] / NATIVE_FRAME[0]))
            bh = min(h, round(h * BOARD_AT_NATIVE[1] / NATIVE_FRAME[1]))
            x0, y0 = (w - bw) // 2, (h - bh) // 2
            board = frame[y0:y0 + bh, x0:x0 + bw]
        # Ultralytics treats ndarray sources as BGR, while the watchdog captures RGB.
        bgr = cv2.cvtColor(np.ascontiguousarray(board), cv2.COLOR_RGB2BGR)
        results = model.predict(source=bgr, imgsz=IMAGE_SIZE, conf=self.confidence,
                                device=device, verbose=False)
        suggestions = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for (x1, y1, x2, y2), confidence in zip(boxes.xyxy.cpu().tolist(),
                                                   boxes.conf.cpu().tolist()):
                x1, x2, y1, y2 = x1 + x0, x2 + x0, y1 + y0, y2 + y0
                suggestions.append({
                    "class": "pattern",
                    "box": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                    "center": [round((x1 + x2) / 2, 3), round((y1 + y2) / 2, 3)],
                    "confidence": round(confidence, 5),
                })
        return len(suggestions) / self.min_patterns, suggestions
