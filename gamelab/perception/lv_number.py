"""Her level in the bottom bar, read: the orange digits after "LV.", glyph by glyph as her EXP and HP numbers are read
(perception.exp_number: the same exact best path through cells, the same FIT and LEAD checks: its Layout). A wrong
value must never come out; no value is fine. The safety stop's touches are worked out at it (behavior.bootstrap: her
level from her HUD, not from config.toml), and her touches are kept by it.

Where (the game data, gamedata.canvas): the digits are StatusBar.img's lvNumber pictures (UI/_Canvas/Basic.img/LevelNo:
11 x 13 px, an orange box with the digit in white, opaque but for its four corners), drawn one after another with a
blank column between (12 px apart) on the status bar's panel (canvas:backgrnd1, 570 x 64, drawn at the bar's top left
(279, 695) less its origin (-7, -9): at (286, 704); the "LV." label is in it). On every lossless 1366x768 frame saved the
panel is the picture to the level, and a two-digit level's first digit has its top left at game px (329, 744): the
bar's posNumberLv (56, 49) puts the number's row at 744; how one or three digits are placed isn't seen yet (her levels
19-28 have two), so the box read (`read_box`) takes the panel from x 317 to 362, room for three digits anywhere in it,
and the reading may start at any column of it (blank cells before the number). The background is the panel's picture
itself (Layout.panel), not the box's darkest columns: it isn't the same along a row (its left edge's gradient), and
the digits' boxes cover half of the box. The grammar: a level of 1 to 3 digits, no leading 0, blank cells around it.

Measured on 2026-10-28 (D:/MapleHelper-scratch/safety3/stats: lv_frames.py): the lossless frames of lab/ and lab/shots
(calibrate_1366x768.png, the Lie Detectors' frames, the menu shots, the hp0_* frames) all read exactly (every cell 0
levels from its glyph) as the level their HP number's max says (752 at 24, 774 at 25, 796 at 26, 840 at 28: 22 x
level + 224), 37 of 40, the other 3 garbage (hp0_*, refused); the rolling recorder's JPEG frames don't pass FIT (its
ringing), as with her HP number: read with LEAD and the grammar alone, all 5,297 of every 10th frame of the 17
recordings read their recording's level (24, 25, 26, 28, as their HP maxes), no frame another (lv_frames.py's JPEG
pass: the design's data, not the live reader's). A read takes 7-8 ms at 1366x768, 13-15 at 1920x1080 (600 ms and
1 s the first, its cells made: `warm`); `Follow` reads it only when its box's pixels change (at the first frame and a
level-up).
"""

import re
from dataclasses import dataclass

import numpy as np

from ..screen import Geometry
from .exp_number import FIT, LEAD, Layout, box_of, cells_of, draw_text, geometry_of, read_text

NUMBER_AT = (318, 744)  # game px: the first column a digit's box may start at (x), and the digits' top row (y)
NUMBER_END = 364        # game px: the cells end before this (the panel's last opaque column is 362)
GLYPHS = "0123456789"
LV = Layout(NUMBER_AT, NUMBER_END, GLYPHS, glyph_set="lv", rows=13, panel=True)
NUMBER = re.compile(r" *([1-9]\d{0,2}) *")
TOP = 200               # the highest level the game has (a reading over it isn't hers)


@dataclass(frozen=True)
class LvReading:
    """Her level as read: `level` (None: not read, `why` says why). `text`: the best path's glyphs, read or not; `fit`
    and `lead`: its worst cell's (levels)."""
    level: int | None
    text: str = ""
    fit: float = 0.0
    lead: float = 0.0
    why: str = ""


def read_box(geo: Geometry) -> tuple[int, int, int, int]:
    """The full-size screen box read (x0, y0, x1, y1) in a frame of geometry `geo`: (317, 742, 363, 760) at 1366x768."""
    return box_of(LV, geo)


def read(box: np.ndarray | None, geo: Geometry | None = None) -> LvReading:
    """Her level in `box`, the screen's `read_box` (full size, RGB, as Eyes.look_at cuts it) in a frame of geometry
    `geo` (by default the client size whose box is the size of this one)."""
    geo = geo or geometry_of(LV, box)
    if box is None or geo is None:
        return LvReading(None, why="no level box")
    x0, y0, x1, y1 = read_box(geo)
    if box.shape[:2] != (y1 - y0, x1 - x0):
        return LvReading(None, why="no level box")
    text, fit, lead = read_text(LV, box, geo)
    if fit > FIT:
        return LvReading(None, text, fit, lead, why=f"a cell unlike any glyph ({fit:.1f} levels)")
    if lead < LEAD:
        return LvReading(None, text, fit, lead, why=f"a cell that could be another ({lead:.1f} levels)")
    m = NUMBER.fullmatch(text)
    if m is None:
        return LvReading(None, text, fit, lead, why=f"not a level: {text.strip()!r}")
    level = int(m.group(1))
    if level > TOP:
        return LvReading(None, text, fit, lead, why=f"level {level} is over the game's {TOP}")
    return LvReading(level, text, fit, lead)


def shown(level: int) -> str:
    """What the bar shows at `level`, as the reader's text: its digits from where a two-digit level's start (329: 11
    blank cells after the box's first column, 318), as every saved frame has them (one or three digits: placed the
    same way, the first at 329, for the simulator's drawing)."""
    return " " * (329 - NUMBER_AT[0]) + str(int(level))


def draw(level: int, geo: Geometry, under: np.ndarray | None = None) -> np.ndarray:
    """Her level drawn over `read_box(geo)` as the game draws it (the reader's own model of it: the simulator's), over
    the box `under` (RGB; by default the panel's picture)."""
    return draw_text(LV, shown(level), under, geo)


def warm(geo: Geometry) -> None:
    """Its cells made now for frames of geometry `geo` (not at the first read, mid-hunt)."""
    cells_of(LV, geo)


class Follow:
    """Her level frame after frame (the perception service's world step; a hunter's own looks without a feed): read
    whenever its box's pixels change (and at the first frame), what it said kept while they don't (`now`)."""

    def __init__(self):
        self._box: np.ndarray | None = None
        self.now: LvReading | None = None  # the last change's reading, if it read (else None)
        self.reads = 0
        self.failed = 0

    def see(self, box: np.ndarray | None, geo: Geometry | None = None) -> LvReading | None:
        """A frame's box (None: none, nothing known then): if its pixels changed, its reading (read or not); else None
        (`now` as it was)."""
        if box is None:
            self._box, self.now = None, None
            return None
        if self._box is not None and box.shape == self._box.shape and np.array_equal(box, self._box):
            return None
        self._box = np.array(box, copy=True)
        reading = read(self._box, geo)
        self.reads += 1
        self.failed += reading.level is None
        self.now = reading if reading.level is not None else None
        return reading

    @property
    def level(self) -> int | None:
        """Her level as the digits say it now, if they were read at their last change."""
        return None if self.now is None else self.now.level
