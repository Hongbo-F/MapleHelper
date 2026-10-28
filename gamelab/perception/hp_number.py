"""Her HP number in the bottom bar, read: "[624/840]", her HP now and when full, glyph by glyph as her EXP number is
read (perception.exp_number: the same status bar pictures, the same exact best path through cells, the same FIT and
LEAD checks: its Layout). A wrong value must never come out; no value is fine. The safety stop measures her HP by it
(behavior.safety: a reserve of touches in HP, not a share of the bar) and her touches' sizes, and her max HP is read
off it rather than taken from config.toml.

Where: StatusBar.img's gauge/posNumberHp (239, 45) from the bar's left (279) puts its first glyph's top left at game px
(518, 740), on the same rows as her EXP number; its cells end before game x 610 (the "MP" label's outline at 611).
The box read is x 517-608, rows 738-749 at 1366x768, and its columns sampling game x 517-518 are taken for the
background (`masked_to`): the HP label's "P" and its dark outline reach into them, and the '[' draws nothing there.
The glyphs are the EXP number's digits and brackets and the 'slash' picture; the grammar "[cur/max]", cur <= max.

Measured on 2026-10-28 (D:/MapleHelper-scratch/safety3/design: hp_number.py, the prototype, and read_recordings.py):
- the 40 lossless 1366x768 frames in lab/ and lab/shots (calibrate_1366x768.png, the Lie Detectors' frames, the menu
  shots, the hp0_* frames): 37 read, every cell within 0.58 of its glyph (FIT 4) and 22.5 ahead of any other (LEAD 10);
  the 3 refused are hp0_* frames whose bottom bar is garbage (no value rather than a wrong one). Maxes read: 752, 774,
  796, 840 (her levels 24, 25, 26, 28). The perception service's frames are lossless (DXGI) like these;
- the rolling recorder's JPEG frames (17 recordings, 50,193 frames, 10-24 23:50 to 10-28 12:16): FIT fails on every
  one (19-20 levels: JPEG's ringing); read with the grammar and LEAD alone all 50,193 read, her max HP her level's
  throughout and no value flipping for one frame. That's the design's data, not the live reader's;
- the bar against it: her HP bar's red px (perception.hud.bars) is ceil(104 cur / max) on 70-76% of the frames and one
  more on 22-30% (its animation), so (share - 1/104) x max never overstates her HP outside the bar's animation
  (behavior.safety.BAR_SLACK); the bar lags the number by 0.07-0.25 s after a change and settles within a px by 0.37 s;
- a read takes 2.9 ms (median; p90 5 ms; 190 ms the first, its cells): `Follow` reads it only when the box's pixels
  change, about 5.4 times a minute hunting (16 ms of CPU a minute; every frame would be 0.09 of a core).
"""

import re
from dataclasses import dataclass

import numpy as np

from ..screen import Geometry
from .exp_number import FIT, LEAD, Layout, box_of, cells_of, draw_text, geometry_of, read_text

NUMBER_AT = (518, 740)  # game px where its first glyph has its top left (StatusBar.img gauge/posNumberHp + the bar's 279)
NUMBER_END = 610        # game px: its cells end before this (the "MP" label's outline at 611)
GLYPHS = "0123456789[]/"
HP = Layout(NUMBER_AT, NUMBER_END, GLYPHS, (("/", "slash"),), masked_to=NUMBER_AT[0] + 0.5)
NUMBER = re.compile(r"\[(0|[1-9]\d{0,4})/([1-9]\d{0,4})\] *")


@dataclass(frozen=True)
class HpReading:
    """Her HP number as read: `cur` and `max` (None: not read, `why` says why). `text`: the best path's glyphs, read or
    not; `fit` and `lead`: its worst cell's (levels)."""
    cur: int | None
    max: int | None = None
    text: str = ""
    fit: float = 0.0
    lead: float = 0.0
    why: str = ""

    @property
    def of(self) -> tuple[int, int] | None:
        """(cur, max), if read."""
        return None if self.cur is None else (self.cur, self.max)


def read_box(geo: Geometry) -> tuple[int, int, int, int]:
    """The full-size screen box read (x0, y0, x1, y1) in a frame of geometry `geo`: (517, 738, 609, 750) at 1366x768."""
    return box_of(HP, geo)


def read(box: np.ndarray | None, geo: Geometry | None = None) -> HpReading:
    """Her HP number in `box`, the screen's `read_box` (full size, RGB, as Eyes.look_at cuts it) in a frame of
    geometry `geo` (by default the client size whose box is the size of this one)."""
    geo = geo or geometry_of(HP, box)
    if box is None or geo is None:
        return HpReading(None, why="no HP number box")
    x0, y0, x1, y1 = read_box(geo)
    if box.shape[:2] != (y1 - y0, x1 - x0):
        return HpReading(None, why="no HP number box")
    text, fit, lead = read_text(HP, box, geo)
    if fit > FIT:
        return HpReading(None, text=text, fit=fit, lead=lead, why=f"a cell unlike any glyph ({fit:.1f} levels)")
    if lead < LEAD:
        return HpReading(None, text=text, fit=fit, lead=lead, why=f"a cell that could be another ({lead:.1f} levels)")
    m = NUMBER.fullmatch(text)
    if m is None:
        return HpReading(None, text=text, fit=fit, lead=lead, why=f"not an HP number: {text.strip()!r}")
    cur, most = int(m.group(1)), int(m.group(2))
    if cur > most:
        return HpReading(None, text=text, fit=fit, lead=lead, why=f"{cur} over its max {most}")
    return HpReading(cur, most, text, fit, lead)


def shown(cur: int, most: int) -> str:
    """What the bar shows with HP `cur` of `most`: "[624/840]"."""
    return f"[{cur}/{most}]"


def draw(text: str, geo: Geometry, under: np.ndarray | None = None) -> np.ndarray:
    """`text` drawn over `read_box(geo)` as the game draws her HP number (the reader's own model of it), over the box
    `under` (RGB; by default the bar's own colours, exp_number.BAR)."""
    return draw_text(HP, text, under, geo)


def warm(geo: Geometry) -> None:
    """Its cells made now for frames of geometry `geo` (190 ms: not at the first read, mid-hunt)."""
    cells_of(HP, geo)


class Follow:
    """Her HP number frame after frame (the perception service's world step; a hunter's own looks without a feed): read
    whenever its box's pixels change (and at the first frame), what it said kept while they don't (`now`)."""

    def __init__(self):
        self._box: np.ndarray | None = None  # the last frame's box
        self.now: HpReading | None = None    # what it says now: the last change's reading, if it read (else None)
        self.reads = 0                       # reads made (the box changed) ...
        self.failed = 0                      # ... and those that didn't read

    def see(self, box: np.ndarray | None, geo: Geometry | None = None) -> tuple[HpReading, int | None] | None:
        """A frame's box (None: none, nothing known then): if its pixels changed, its reading (read or not) and the HP
        the number said just before (None: it wasn't read at the change before, so what came between isn't known); else
        None (`now` as it was)."""
        if box is None:
            self._box, self.now = None, None
            return None
        if self._box is not None and box.shape == self._box.shape and np.array_equal(box, self._box):
            return None
        before = None if self.now is None else self.now.cur
        self._box = np.array(box, copy=True)
        reading = read(self._box, geo)
        self.reads += 1
        self.failed += reading.cur is None
        self.now = reading if reading.cur is not None else None
        return reading, before

    @property
    def of(self) -> tuple[int, int] | None:
        """(cur, max) as the number says it now, if it was read at its last change."""
        return None if self.now is None else self.now.of
