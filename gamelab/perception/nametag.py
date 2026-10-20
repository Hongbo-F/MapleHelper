"""Finding her on screen by her name tag.

The tag is matched at full resolution: at half size the score swings between ~0.6 and ~0.9 depending on
which pixels the downscale keeps, and her name in the minimap scored ~0.6 too. The full-size template is
cut from the screen the first time, where the half-size one from calibration finds her, and kept.

A look searches near where she was found last first (3.4 ms; the whole play area takes 35): the camera
keeps her at one spot on screen except near the map's edges, so her tag is seldom far from where it was.
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
# Full-size px across and down from where she was found last that a look searches first. Where the camera
# stops at the map's edges she moves on screen: 7 px a frame walking, up to about 31 falling. When she isn't
# in there (a warp, a long walk by an edge, the tag hidden), the whole play area is searched.
NEAR = (240, 120)
FULL_EVERY = 30        # ... and every this many looks the whole play area anyway, so a wrong match can't stick


def _hide_minimap(res: np.ndarray, shape: tuple[int, ...], x0: int = 0, y0: int = 0) -> None:
    """Rule out matches in the minimap of a screen of this shape (in place). `res` holds the matches of a
    box of the screen whose top-left corner is (x0, y0)."""
    height, width = shape[:2]
    res[:max(0, int(height * MINIMAP_HEIGHT) - y0), :max(0, int(width * MINIMAP_WIDTH) - x0)] = -1


class NameTag:
    """Her name tag: where it is in each look."""

    def __init__(self, small: np.ndarray, path: Path):
        self.small = small    # half-size template from calibration, used to find her once
        self.path = path      # where the full-size template is kept
        self.full = np.load(path) if path.exists() else None
        self._last: tuple[int, int] | None = None
        self._looks = 0

    def find(self, full: np.ndarray) -> tuple[int, int] | None:
        """Centre of her name tag in a full-size frame, in half-size frame pixels (None: not on screen).
        Her name also shows in the minimap, the bottom bar and the party list, so of the good matches in
        the play area, the one nearest where she was (or the screen centre, which the camera keeps her
        near) wins. It's looked for within NEAR of where she was first, and in the whole play area if
        it isn't there (and every FULL_EVERY looks)."""
        if self.full is None:
            self.full = self._cut(full)
        self._looks += 1
        found = None
        if self._last is not None and self._looks % FULL_EVERY:
            found = self.search(full, self._near(full.shape))
        if found is None:
            found = self.search(full)
        if found is not None:
            self._last = found
        return found

    def _near(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """The box of the play area (x0, y0, x1, y1; full-size px) a tag within NEAR of where she was
        found last lies in."""
        th, tw = self.full.shape
        x, y = SCALE * self._last[0] - tw // 2, SCALE * self._last[1] - th // 2  # its top-left corner then
        return (max(0, x - NEAR[0]), max(0, y - NEAR[1]),
                min(shape[1], x + NEAR[0] + tw), min(int(shape[0] * PLAY_BOTTOM), y + NEAR[1] + th))

    def search(self, full: np.ndarray, box: tuple[int, int, int, int] | None = None) -> tuple[int, int] | None:
        """Her tag's centre (half-size px) in a box of the frame (x0, y0, x1, y1; default: the whole play
        area): of the good matches there, the one nearest where she was. None if there's none."""
        h, w = full.shape[:2]
        x0, y0, x1, y1 = box or (0, 0, w, int(h * PLAY_BOTTOM))
        th, tw = self.full.shape
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        gray = cv2.cvtColor(full[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(gray, self.full, cv2.TM_CCOEFF_NORMED)
        _hide_minimap(res, full.shape, x0, y0)
        ref = self._last or (w // 2 // SCALE, int(h * START_HEIGHT) // SCALE)
        best = None
        for _ in range(CANDIDATES):
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < MATCH:
                break
            centre = ((x0 + x + tw // 2) // SCALE, (y0 + y + th // 2) // SCALE)
            dist = abs(centre[0] - ref[0]) + abs(centre[1] - ref[1])
            if best is None or dist < best[0]:
                best = (dist, centre)
            dy, dx = SAME_NAME
            res[max(0, y - dy):y + dy, max(0, x - dx):x + dx] = -1
        return None if best is None else best[1]

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
