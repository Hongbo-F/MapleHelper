"""Finding her on screen by her name tag.

The tag is matched at full resolution: at half size the score swings between ~0.6 and ~0.9 depending on
which pixels the downscale keeps, and her name in the minimap scored ~0.6 too. The full-size template is
cut from the screen the first time, where the half-size one from calibration finds her, and kept.
"""

from pathlib import Path

import cv2
import numpy as np

from ..screen import SCALE


PLAY_BOTTOM = 0.9      # the bottom bar starts below this share of the screen's height: not searched
MINIMAP_WIDTH = 0.17   # the minimap's share of the screen's width ...
MINIMAP_HEIGHT = 0.3   # ... and height, at the top left: not searched
MATCH = 0.7            # a full-size match at least this good is one of the places her name shows
CUT_MATCH = 0.75       # the half-size template must match this well to cut the full-size one there
CANDIDATES = 4         # at most this many good matches are compared to pick hers
START_HEIGHT = 0.6     # until she's found, the match nearest the screen's middle, this share of the way down, wins
SAME_NAME = (20, 60)   # full-size px around a match (rows, columns) where other matches are the same name


def _hide_minimap(res: np.ndarray, shape: tuple[int, ...]) -> None:
    """Rule out matches in the minimap of a screen of this shape (in place)."""
    height, width = shape[:2]
    res[:int(height * MINIMAP_HEIGHT), :int(width * MINIMAP_WIDTH)] = -1


class NameTag:
    """Her name tag: where it is in each look."""

    def __init__(self, small: np.ndarray, path: Path):
        self.small = small    # half-size template from calibration, used to find her once
        self.path = path      # where the full-size template is kept
        self.full = np.load(path) if path.exists() else None
        self._last: tuple[int, int] | None = None

    def find(self, full: np.ndarray) -> tuple[int, int] | None:
        """Centre of her name tag in a full-size frame, in half-size frame pixels (None: not on screen).
        Her name also shows in the minimap, the bottom bar and the party list, so of the good matches in
        the play area, the one nearest where she was (or the screen centre, which the camera keeps her
        near) wins."""
        if self.full is None:
            self.full = self._cut(full)
        h, w = full.shape[:2]
        gray = cv2.cvtColor(full[:int(h * PLAY_BOTTOM)], cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(gray, self.full, cv2.TM_CCOEFF_NORMED)
        _hide_minimap(res, full.shape)
        th, tw = self.full.shape
        ref = self._last or (w // 2 // SCALE, int(h * START_HEIGHT) // SCALE)
        best = None
        for _ in range(CANDIDATES):
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < MATCH:
                break
            centre = ((x + tw // 2) // SCALE, (y + th // 2) // SCALE)
            dist = abs(centre[0] - ref[0]) + abs(centre[1] - ref[1])
            if best is None or dist < best[0]:
                best = (dist, centre)
            dy, dx = SAME_NAME
            res[max(0, y - dy):y + dy, max(0, x - dx):x + dx] = -1
        if best is None:
            return None
        self._last = best[1]
        return best[1]

    def _cut(self, full: np.ndarray) -> np.ndarray:
        """Find her with the half-size template (trying all four ways of halving the frame), then cut
        a full-resolution template of her name tag there and keep it for next time."""
        best = None
        for oy in (0, 1):
            for ox in (0, 1):
                half = cv2.cvtColor(np.ascontiguousarray(full[oy::SCALE, ox::SCALE]), cv2.COLOR_RGB2GRAY)
                res = cv2.matchTemplate(half[:int(half.shape[0] * PLAY_BOTTOM)], self.small, cv2.TM_CCOEFF_NORMED)
                _hide_minimap(res, half.shape)
                _, score, _, (x, y) = cv2.minMaxLoc(res)
                if best is None or score > best[0]:
                    best = (score, SCALE * x + ox, SCALE * y + oy)
        if best is None or best[0] < CUT_MATCH:
            raise RuntimeError("can't find her name tag to build a full-size template; run `calibrate --fresh`")
        # Cut exactly where the half-size template matched, so a name next to hers can't get in.
        th, tw = (SCALE * n for n in self.small.shape)
        tag = cv2.cvtColor(np.ascontiguousarray(full[best[2]:best[2] + th, best[1]:best[1] + tw]), cv2.COLOR_RGB2GRAY)
        np.save(self.path, tag)
        return tag
