"""Where the game draws the text we read, as fractions of its picture, and black-bar trimming.

The game's interface scales with its window, so a region given as fractions fits any window size. The two
presets are layout measurements of MapleStory Classic World published with Maple Helper (measured at
1368x800, 1922x1112 and 2734x1568).
"""

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]  # x, y, width, height in pixels


@dataclass(frozen=True)
class Region:
    """A box as fractions (0..1) of the game picture: left, top, right, bottom."""
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError(f"need 0 <= left < right <= 1 and 0 <= top < bottom <= 1, got {self}")

    def box(self, area: Box) -> Box:
        """This region in pixels of `area` (x, y, w, h), never smaller than 1x1."""
        x, y, w, h = area
        x0, y0 = round(self.left * w), round(self.top * h)
        x1, y1 = round(self.right * w), round(self.bottom * h)
        return x + x0, y + y0, max(1, x1 - x0), max(1, y1 - y0)

    def crop(self, picture: np.ndarray) -> np.ndarray:
        """This region cut out of a whole game picture."""
        h, w = picture.shape[:2]
        x, y, cw, ch = self.box((0, 0, w, h))
        return picture[y:y + ch, x:x + cw]


@dataclass(frozen=True)
class Preset:
    name: str
    region: Region
    min_height: int  # crops shorter than this (px) are enlarged before OCR; 0 = never


# The bottom bar: level, job, name, HP, MP, EXP. Its text is big enough as it is.
HUD = Preset("hud", Region(0.0, 0.94, 1.0, 1.0), min_height=0)
# The pickup messages ("You received EXP (+21)") stacked bottom right of the play area, above the
# skill bar and below the chat bubbles. Small text: enlarged to 450 px tall.
LOOT = Preset("loot", Region(0.78, 0.57, 1.0, 0.86), min_height=450)

PRESETS = {p.name: p for p in (HUD, LOOT)}

_DARK = 12      # a bar pixel's brightest channel is at most this
_MIN_BAR = 4    # px: thinner dark edges are just the scene
_BAR_SLACK = 2  # px: the two bars of a centred picture may differ by this much


def content_box(picture: np.ndarray) -> Box:
    """The part of `picture` without the black bars a game adds to keep its aspect ratio (a 16:9 game on an
    ultrawide screen, or a 4:3 one on 16:9). Only real bars count: nearly black and the same width on both
    sides; a dark scene touching one edge is left alone."""
    bright = picture.max(axis=2) if picture.ndim == 3 else picture
    x, cw = _trim(bright.max(axis=0))
    y, ch = _trim(bright.max(axis=1))
    return x, y, cw, ch


def _trim(profile: np.ndarray) -> tuple[int, int]:
    # profile: the brightest value of each column (or row). Returns (start, length) of the content.
    n = len(profile)
    lit = np.flatnonzero(profile > _DARK)
    if len(lit) == 0:
        return 0, n
    lead, trail = int(lit[0]), int(n - 1 - lit[-1])
    if min(lead, trail) < _MIN_BAR or abs(lead - trail) > _BAR_SLACK:
        return 0, n
    return lead, n - lead - trail
