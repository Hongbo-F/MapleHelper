"""Spotting MapleStory's Lie Detector (its anti-bot check) on screen, so gamelab stops at once.

It comes in more than one form, and each is recognized:
- a countdown window titled "LIE DETECTOR" in yellow letters; its title bar is see-through, so the
  yellow letter shapes are matched rather than raw pixels;
- the mouse mini-game's notice, 10 s before it: a box at the bottom right with the Maple
  Administrator's portrait, "The Lie Detector Mini-game", "The mini-game will start soon. Go to a
  safe place before the game begins.";
- "The Lie Detector Mini-game" ("Use the mouse to follow the fading figure"): the board itself, a
  window with the same portrait and a blue header, matched at half size. On 2026-10-18 this one
  wasn't recognized yet, went unanswered, and she was sent to the Truth Booth;
- the image quiz's warning, 30 s before it: a see-through blue box, "Question in N seconds",
  "Answer within 10 seconds";
- the image quiz: "Click all matching images", three pictures and a Confirm button, 10 s to answer;
- the Truth Booth itself (where a failed test sends her), by its "Time Left" timer.
The notice and the quiz's warning are matched so that the hunt stops, and she retreats, before the
test begins. Never try to answer it automatically: a person has to.

The key player, which runs as administrator, loads this module: it imports only gamelab.paths.
"""

import cv2
import numpy as np
from PIL import Image

from ..paths import ASSETS


TITLE = ASSETS / "lie_detector_title.png"
THRESHOLD = 0.6  # yellow-letter match; the real window scores ~1.0, game scenes stay well below
# The templates were cut from the game stretched to a 1920x1080 window. Since 2026-10-23 the game runs at its own
# 1366x768 (pixel for pixel), where at that size none of them matched (the six saved Lie Detector screens shrunk to
# 1366x768 scored 0.32-0.64 of the threshold, every form): a whole frame of that size is stretched to this size first,
# as the game itself stretched its 1366x768 view before. With the templates' own frames shrunk to 1366x768 and
# stretched back, each scores as it did (tests/test_alarm.py).
TEMPLATE_FRAME = (1920, 1080)
NATIVE_FRAME = (1366, 768)  # (only a whole frame of that size: crops are matched as they are)
# Half-size grey templates and their thresholds. The real screens score 1.0; across ~190 saved game
# screenshots and recordings nothing else scored above 0.53 (mini-game) or 0.69 (timer).
# The notice's and the quiz's were cut on 2026-10-25 from the rolling recorder's frames of that night (their fixtures
# in tests/fixtures/lie_detector), as `scores` sees a 1366x768 frame: stretched, halved, grey. The notice's is its
# portrait, title and first lines (00:55:01); the quiz's two are the line that stays the same, "Answer within 10
# seconds", and the quiz's title, each the median of its frames (the box is see-through: the scene behind averages
# out). On the 12,552 saved frames of 2026-10-19 to 25 (D:/MapleHelper-scratch/report-1025): the notice scored 0.998
# or more on every notice, the 1920x1080 ones of 2026-10-19 and 21 too, 0.80 on the board (the same portrait and
# title) and at most 0.34 on anything else; the warning 0.82 or more on its 296 frames and at most 0.50 elsewhere; the
# quiz 0.89 or more on its 78 frames, SUCCESS over it included, and at most 0.52 elsewhere.
HALF_SIZE = {
    "mini-game notice": (ASSETS / "lie_detector_notice.png", 0.9),
    "mini-game": (ASSETS / "lie_detector_minigame.png", 0.8),
    "quiz warning": (ASSETS / "lie_detector_quiz_warning.png", 0.7),
    "quiz": (ASSETS / "lie_detector_quiz.png", 0.7),
    "truth booth": (ASSETS / "truth_booth_timer.png", 0.85),
}


def to_template(frame: np.ndarray) -> np.ndarray:
    """A whole frame of the game's client (RGB) at the size `scores` matches it at: the game's own (NATIVE_FRAME, which
    `scores` stretches itself) and the templates' (TEMPLATE_FRAME) as they are, any other brought to TEMPLATE_FRAME
    (shrunk by area, stretched with Lanczos): a client full screen at 3840x2160, or resized. As they were, the saved
    screens stretched to such a client scored 0.53-0.78 of their thresholds (D:/MapleHelper-scratch/winfix/
    review-safety/watchdog_sizes.out); brought so, every form but the countdown is seen at every size tried, the
    countdown at 3840x2160, 1600x900 and 1396x1952 (control.window_size.lie_detector_sighting). The player's watchdog
    and the perception service's picture of a client of another size (perception.frames.glimpse) bring it so."""
    size = (frame.shape[1], frame.shape[0])
    if size in (NATIVE_FRAME, TEMPLATE_FRAME):
        return frame
    shrink = size[0] >= TEMPLATE_FRAME[0] and size[1] >= TEMPLATE_FRAME[1]
    return cv2.resize(np.ascontiguousarray(frame), TEMPLATE_FRAME,
                      interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)


def yellow(rgb: np.ndarray) -> np.ndarray:
    """1.0 where a pixel is the Lie Detector title's yellow, else 0.0: red over 180, green over 140, blue
    under 110, and red over blue by more than 110. Tested in OpenCV (red minus blue saturates at 0, which
    fails as a negative difference would): the same pixels as numpy's int16 tests, 6.9 ms against 11.8."""
    r, _, b = cv2.split(rgb)
    colour = cv2.inRange(rgb, (181, 141, 0), (255, 255, 109))
    out = np.zeros(rgb.shape[:2], np.float32)
    out[cv2.bitwise_and(colour, cv2.inRange(cv2.subtract(r, b), 111, 255)) > 0] = 1.0
    return out


class LieDetectorSpotter:
    """Works on full-size (unscaled) RGB frames of the game's client area."""

    def __init__(self):
        self.template = yellow(np.asarray(Image.open(TITLE).convert("RGB")))
        self.half = {name: (cv2.cvtColor(np.asarray(Image.open(png).convert("RGB")), cv2.COLOR_RGB2GRAY), t)
                     for name, (png, t) in HALF_SIZE.items()}

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """How well each form of the Lie Detector matches, as a fraction of its threshold (>= 1: seen). A frame that
        the game's own size (NATIVE_FRAME) is stretched to TEMPLATE_FRAME first."""
        if frame.shape[:2] == NATIVE_FRAME[::-1]:
            frame = cv2.resize(np.ascontiguousarray(frame), TEMPLATE_FRAME, interpolation=cv2.INTER_LINEAR)
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
