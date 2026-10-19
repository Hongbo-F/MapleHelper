"""Spotting MapleStory's Lie Detector (its anti-bot check) on screen, so gamelab stops at once.

It comes in more than one form, and each is recognized:
- a countdown window titled "LIE DETECTOR" in yellow letters; its title bar is see-through, so the
  yellow letter shapes are matched rather than raw pixels;
- "The Lie Detector Mini-game" ("Use the mouse to follow the fading figure"): a window with the
  Maple Administrator's portrait and a blue header, matched at half size. On 2026-10-18 this one
  wasn't recognized yet, went unanswered, and she was sent to the Truth Booth;
- the Truth Booth itself (where a failed test sends her), by its "Time Left" timer.
Never try to answer it automatically: a person has to.

The key player, which runs as administrator, loads this module: it imports only gamelab.paths.
"""

import cv2
import numpy as np
from PIL import Image

from ..paths import ASSETS


TITLE = ASSETS / "lie_detector_title.png"
THRESHOLD = 0.6  # yellow-letter match; the real window scores ~1.0, game scenes stay well below
# Half-size grey templates and their thresholds. The real screens score 1.0; across ~190 saved game
# screenshots and recordings nothing else scored above 0.53 (mini-game) or 0.69 (timer).
HALF_SIZE = {
    "mini-game": (ASSETS / "lie_detector_minigame.png", 0.8),
    "truth booth": (ASSETS / "truth_booth_timer.png", 0.85),
}


def yellow(rgb: np.ndarray) -> np.ndarray:
    """1.0 where a pixel is the Lie Detector title's yellow, else 0.0."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r > 180) & (g > 140) & (b < 110) & (r - b > 110)).astype(np.float32)


class LieDetectorSpotter:
    """Works on full-size (unscaled) RGB frames of the game's client area."""

    def __init__(self):
        self.template = yellow(np.asarray(Image.open(TITLE).convert("RGB")))
        self.half = {name: (cv2.cvtColor(np.asarray(Image.open(png).convert("RGB")), cv2.COLOR_RGB2GRAY), t)
                     for name, (png, t) in HALF_SIZE.items()}

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """How well each form of the Lie Detector matches, as a fraction of its threshold (>= 1: seen)."""
        out = {"countdown": self._yellow_title(frame) / THRESHOLD}
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(np.ascontiguousarray(frame), (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        for name, (template, threshold) in self.half.items():
            if gray.shape[0] >= template.shape[0] and gray.shape[1] >= template.shape[1]:
                score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
                out[name] = score / threshold
        return out

    def score(self, frame: np.ndarray) -> float:
        """The best match over all forms, scaled so that THRESHOLD means seen."""
        return max(self.scores(frame).values()) * THRESHOLD

    def _yellow_title(self, frame: np.ndarray) -> float:
        """How well the yellow letters of the countdown window's title match (0 when there's too little yellow)."""
        mask = yellow(frame)
        if mask.sum() < 0.5 * self.template.sum():  # not even enough yellow for the title
            return 0.0
        return float(cv2.matchTemplate(mask, self.template, cv2.TM_CCOEFF_NORMED).max())

    def seen(self, frame: np.ndarray) -> bool:
        """Whether any form of the Lie Detector is on screen."""
        return self.score(frame) >= THRESHOLD
