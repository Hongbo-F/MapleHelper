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

Each form has always shown at one place (PLACES), and the player's watchdog looks 4 times a second: most of its looks
match each form only around its places (`near`, 11 ms of CPU at 1366x768), every WHOLE_EVERY-th the whole frame
(`scores`, 151 ms), for a window anywhere else; and every look the whole frame on a frame whose places aren't known (of
another size, or the game's picture drawn off its client: control.player.Watchdog).

The key player, which runs as administrator, loads this module: it imports only gamelab.paths.
"""

import threading

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
# Where each form has shown, its template's top-left in TEMPLATE_FRAME px (the countdown's two: its window, then the
# second window with the yellow title): the same pixel on every recording and fixture of 2026-10-18 to 26 that showed
# it (3,758 saved frames, 1,502 with a form: D:/MapleHelper-scratch/cpu/ldwatch/scan.jsonl, locs.py), the 1920x1080
# ones and the 1366x768 ones stretched alike. `near` matches each form within NEAR_MARGIN px of these, on the very
# pictures `scores` matches (the frame stretched, halved, grey): on those frames every form scored within 1e-4 of
# `scores`, none missed, and nothing else scored higher than with `scores` (its boxes are inside the frame).
PLACES = {
    "countdown": ((910, 323), (1648, 589)),
    "mini-game notice": ((1458, 680),),
    "mini-game": ((432, 124),),
    "quiz warning": ((598, 566),),
    "quiz": ((604, 470),),
    "truth booth": ((776, 40),),
}
NEAR_MARGIN = 48  # TEMPLATE_FRAME px around each place (24 at half size)
# The forms whose window lies over the play area, where she is, each at the one of its PLACES where it does: the
# countdown's window (its yellow title at the first; at the second its box at the right, which comes before it as the
# mini-game's notice does), the mini-game's board and the quiz. Not the boxes that come before them (the mini-game's
# notice, the quiz's warning, the countdown's box: she retreats then, the hunt stopped), nor the Truth Booth's timer
# (another map). For her place by the scene counted "covered" under them (localization.locate.Cover, `over_play`). On
# every 2nd frame of the 15 Lie Detector clips of 2026-10-26 to 30 (lab/alarm_clips, 1366x768): the board on every
# frame `near` saw it on (11 clips), the countdown's window from its window on (4: not its box before it), none of the
# notices; on every 3rd frame of the live clip of 17:54:33, nothing (D:/MapleHelper-scratch/carried/ad/fix/scan_ld.py).
OVER_PLAY = {"countdown": PLACES["countdown"][0], "mini-game": PLACES["mini-game"][0], "quiz": PLACES["quiz"][0]}
# The watchdog's every this many looks is of the whole frame (`scores`): a form somewhere else is seen within that many
# looks (2 s at 4 a second; the notice comes 10 s before the mini-game, the quiz's warning 30 s before the quiz). At 4
# looks a second on the saved hunt frames, 35 ms of CPU a look against 151 for the whole frame's every time: 0.14 cores
# against 0.61 (D:/MapleHelper-scratch/cpu/ldwatch/roi_eval.py).
WHOLE_EVERY = 8


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


def _box(place: tuple[int, int], size: tuple[int, int], frame: tuple[int, int], margin: int) -> tuple[slice, slice]:
    """The rows and columns of a frame of size `frame` (w, h) within `margin` px of a picture of size `size` (w, h)
    whose top-left is at `place`."""
    (x, y), (w, h) = place, size
    return (slice(max(0, y - margin), min(frame[1], y + h + margin)),
            slice(max(0, x - margin), min(frame[0], x + w + margin)))


class LieDetectorSpotter:
    """Works on full-size (unscaled) RGB frames of the game's client area."""

    def __init__(self):
        self.template = yellow(np.asarray(Image.open(TITLE).convert("RGB")))
        self.template_yellow = cv2.countNonZero(self.template)
        self.half = {name: (cv2.cvtColor(np.asarray(Image.open(png).convert("RGB")), cv2.COLOR_RGB2GRAY), t)
                     for name, (png, t) in HALF_SIZE.items()}
        self._pictures = threading.local()  # each thread's pictures a look is made in (`_kept`)

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """How well each form of the Lie Detector matches, as a fraction of its threshold (>= 1: seen). A frame that
        the game's own size (NATIVE_FRAME) is stretched to TEMPLATE_FRAME first."""
        frame = self._stretched(frame)
        out = {"countdown": self._yellow_title(frame) / THRESHOLD}
        gray = self._halved(frame)
        for name, (template, threshold) in self.half.items():
            if gray.shape[0] >= template.shape[0] and gray.shape[1] >= template.shape[1]:
                score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
                out[name] = score / threshold
        return out

    def near(self, frame: np.ndarray) -> dict[str, float]:
        """`scores`, each form matched only within NEAR_MARGIN of its PLACES (one elsewhere scores low here: the whole
        frame's look, `scores`, sees it). A frame of another size than NATIVE_FRAME and TEMPLATE_FRAME, where the
        places aren't known, is matched whole (`scores`)."""
        if (frame.shape[1], frame.shape[0]) not in (NATIVE_FRAME, TEMPLATE_FRAME):
            return self.scores(frame)
        frame = self._stretched(frame)
        th, tw = self.template.shape
        best = 0.0
        for place in PLACES["countdown"]:
            rows, cols = _box(place, (tw, th), TEMPLATE_FRAME, NEAR_MARGIN)
            best = max(best, self._yellow_title(frame[rows, cols]))
        out = {"countdown": best / THRESHOLD}
        gray = self._halved(frame)
        half = (TEMPLATE_FRAME[0] // 2, TEMPLATE_FRAME[1] // 2)
        for name, (template, threshold) in self.half.items():
            best = 0.0
            for x, y in PLACES[name]:
                rows, cols = _box((x // 2, y // 2), template.shape[::-1], half, NEAR_MARGIN // 2)
                best = max(best, float(cv2.matchTemplate(gray[rows, cols], template, cv2.TM_CCOEFF_NORMED).max()))
            out[name] = best / threshold
        return out

    def over_play(self, frame: np.ndarray) -> str | None:
        """The first form whose window over the play area (OVER_PLAY) is seen at its place on this frame of the game's
        own size (NATIVE_FRAME) or the templates', as `near` matches it there; else None, and None at any other size
        (its places not known there)."""
        if (frame.shape[1], frame.shape[0]) not in (NATIVE_FRAME, TEMPLATE_FRAME):
            return None
        frame = self._stretched(frame)
        th, tw = self.template.shape
        rows, cols = _box(OVER_PLAY["countdown"], (tw, th), TEMPLATE_FRAME, NEAR_MARGIN)
        if self._yellow_title(frame[rows, cols]) >= THRESHOLD:
            return "countdown"
        gray = self._halved(frame)
        half = (TEMPLATE_FRAME[0] // 2, TEMPLATE_FRAME[1] // 2)
        for name in ("mini-game", "quiz"):
            template, threshold = self.half[name]
            x, y = OVER_PLAY[name]
            rows, cols = _box((x // 2, y // 2), template.shape[::-1], half, NEAR_MARGIN // 2)
            if float(cv2.matchTemplate(gray[rows, cols], template, cv2.TM_CCOEFF_NORMED).max()) >= threshold:
                return name
        return None

    def _stretched(self, frame: np.ndarray) -> np.ndarray:
        """A frame of the game's own size (NATIVE_FRAME) stretched to TEMPLATE_FRAME (into `_kept`'s picture); any
        other as it is."""
        if frame.shape[:2] == NATIVE_FRAME[::-1]:
            out = self._kept("stretched", (TEMPLATE_FRAME[1], TEMPLATE_FRAME[0]) + frame.shape[2:], frame.dtype)
            return cv2.resize(np.ascontiguousarray(frame), TEMPLATE_FRAME, dst=out, interpolation=cv2.INTER_LINEAR)
        return frame

    def _halved(self, frame: np.ndarray) -> np.ndarray:
        """The frame at half size, grey: what the half-size templates are matched on (into `_kept`'s pictures)."""
        h, w = frame.shape[:2]
        half = cv2.resize(np.ascontiguousarray(frame), (w // 2, h // 2), interpolation=cv2.INTER_AREA,
                          dst=self._kept("half", (h // 2, w // 2) + frame.shape[2:], frame.dtype))
        return cv2.cvtColor(half, cv2.COLOR_RGB2GRAY, dst=self._kept("grey", (h // 2, w // 2), frame.dtype))

    def _kept(self, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        """This thread's picture `name` of that shape, kept from look to look (made anew only when a frame's size
        changes): the stretched frame, its half and its grey were 8 MB of fresh memory a look at 1366x768, 2,000 page
        faults that Windows' kernel zeroes (its System process: about a core for 100,000 a second, D:/MapleHelper-
        scratch/cpu/review-numbers/zerotest.py). What's in it is a look's own: nothing of it is handed out."""
        kept = self._pictures.__dict__
        out = kept.get(name)
        if out is None or out.shape != shape or out.dtype != dtype:
            out = kept[name] = np.empty(shape, dtype)
        return out

    def score(self, frame: np.ndarray) -> float:
        """The best match over all forms, scaled so that THRESHOLD means seen."""
        return max(self.scores(frame).values()) * THRESHOLD

    def _yellow_title(self, frame: np.ndarray) -> float:
        """How well the yellow letters of the countdown window's title match (0 when there's too little yellow)."""
        mask = yellow(np.ascontiguousarray(frame))
        if cv2.countNonZero(mask) < 0.5 * self.template_yellow:  # not even enough yellow for the title (counted:
            return 0.0                                            # 0.26 ms at 1920x1080, the sum 1.3)
        return float(cv2.matchTemplate(mask, self.template, cv2.TM_CCOEFF_NORMED).max())

    def seen(self, frame: np.ndarray) -> bool:
        """Whether any form of the Lie Detector is on screen."""
        return self.score(frame) >= THRESHOLD
