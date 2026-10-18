"""Reading the pickup messages from the loot box, and only them.

The box is a fixed part of the screen, so other interface can sit in it: the party panel (member names
with HP bars) does, for one. Such text isn't a pickup message and stays put, while a message scrolls up as
new ones arrive and fades within ~12 s. So text that doesn't parse as a pickup and stays in one place for
STATIC_S is taken for interface and blacked out of the box before OCR: it's neither detected nor read again.
Every RECHECK_S one read goes without the blackout, and spots whose text is gone are freed.
"""

import difflib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .engine import OcrEngine, Row
from .parse import LootLine, parse_loot
from .regions import LOOT, Preset


def similar_text(a: str, b: str, ratio: float = 0.6) -> bool:
    """Whether two OCR reads are likely the same text: OCR drops or swaps a character or two between reads
    ("PlayerOne", "PlayerOn", "* PlayerOne"). Compares letters and digits only."""
    a, b = _letters(a), _letters(b)
    if not a or not b:
        return a == b
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def same_line(a: Row, b: Row) -> bool:
    """Whether two rows from different reads are the same line on screen: about the same height, overlapping
    left to right, and similar text."""
    return (abs(a.y - b.y) <= 0.5 * max(a.height, b.height)
            and min(a.right, b.right) >= max(a.left, b.left) and similar_text(a.text, b.text))


def _near(a: Row, b: Row) -> bool:
    # Whether `a` overlaps `b` or sits within a line height of it. Unmasked, OCR doesn't always read what it
    # read with the rest blacked out (a party leader's icon beside the name): text right next to a spot
    # shows the interface it belongs to is still there.
    gap = max(a.height, b.height)
    return (a.left - gap <= b.right and b.left - gap <= a.right
            and abs(a.y - b.y) <= (a.height + b.height) / 2 + gap)


def _letters(text: str) -> str:
    return re.sub(r"\W", "", text.lower())


# Words of the pickup messages, as OCR may read them.
_PICKUP_WORDS = re.compile(r"recei|ceiv|gain|e[ae]?r?(?:rn|[nm])e?d|\bexp\b|(?:m|rn)es|\(\s*[+t4]", re.I)


@dataclass
class _Spot:
    anchor: Row     # where the text first showed: it must stay there, not drift (a message scrolling up)
    row: Row        # the latest read of it
    since: float    # first seen here
    seen: float     # last seen here
    masked: bool = False


@dataclass
class StaticText:
    """Finds interface text that stays put in the box and blacks it out before OCR."""

    STATIC_S = 15.0   # longer than a pickup message stays on screen, so a message never gets blacked out
    RECHECK_S = 30.0  # how often one read goes without the blackout, to free spots whose text is gone
    FORGET_S = 3.0    # a candidate not seen for this long (and not blacked out) is dropped
    PAD = 4           # px around the text that are blacked out too

    _spots: list[_Spot] = field(default_factory=list)
    _rechecked: float = 0.0

    def masked_rows(self) -> list[Row]:
        return [s.row for s in self._spots if s.masked]

    def prepare(self, crop: np.ndarray, now: float) -> tuple[np.ndarray, bool]:
        """`crop` with the known interface text blacked out, and whether this read is a recheck (unmasked)."""
        masked = [s for s in self._spots if s.masked]
        if not masked:
            return crop, False
        if now - self._rechecked >= self.RECHECK_S:
            self._rechecked = now
            return crop, True
        out = crop.copy()
        h, w = out.shape[:2]
        for s in masked:
            r = s.row
            y0, y1 = max(0, int(r.y - r.height / 2) - self.PAD), min(h, int(r.y + r.height / 2) + self.PAD + 1)
            x0, x1 = max(0, int(r.left) - self.PAD), min(w, int(r.right) + self.PAD + 1)
            out[y0:y1, x0:x1] = _surround(crop, x0, y0, x1, y1)
        return out, False

    def learn(self, ignored: list[Row], now: float, recheck: bool) -> None:
        """Updates the spots from this read's rows that aren't pickup messages."""
        # A message read only in part (fading out) is never interface.
        ignored = [r for r in ignored if not _PICKUP_WORDS.search(r.text)]
        matched = set()
        for row in ignored:
            spot = next((s for s in self._spots if id(s) not in matched and same_line(s.anchor, row)), None)
            if spot is None:
                self._spots.append(spot := _Spot(row, row, now, now))
            spot.row, spot.seen = row, now
            matched.add(id(spot))
            if not spot.masked and now - spot.since >= self.STATIC_S:
                if not any(s.masked for s in self._spots):
                    self._rechecked = now  # the first recheck comes RECHECK_S after the first blackout
                spot.masked = True
        kept = []
        for s in self._spots:
            if s.masked and recheck and id(s) not in matched and not any(_near(r, s.row) for r in ignored):
                continue  # the recheck found no text there: the interface is gone (left the party, closed a window)
            if not s.masked and id(s) not in matched and now - s.seen > self.FORGET_S:
                continue
            kept.append(s)
        self._spots = kept


def _surround(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, ring: int = 3) -> np.ndarray:
    # The usual colour just outside a box. Painting the box with it hides the text without leaving edges
    # (a black patch on a grey panel has corners OCR reads as characters).
    h, w = image.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - ring), max(0, y0 - ring), min(w, x1 + ring), min(h, y1 + ring)
    outer = image[Y0:Y1, X0:X1].reshape(-1, image.shape[2])
    inside = np.zeros((Y1 - Y0, X1 - X0), bool)
    inside[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
    border = outer[~inside.ravel()]
    return np.median(border, axis=0).astype(image.dtype) if len(border) else np.zeros(image.shape[2], image.dtype)


@dataclass(frozen=True)
class LootRead:
    rows: list[Row]                       # everything OCR read in the box, in reading order
    messages: list[tuple[Row, LootLine]]  # the pickup messages among them
    ignored: list[Row]                    # the other text (not counted)
    interface: list[Row]                  # text known to be interface, blacked out of the box before OCR
    recheck: bool                         # this read went without the blackout


class LootReader:
    """OCR of the loot box that returns its pickup messages and learns which text in it is interface."""

    def __init__(self, engine: OcrEngine, preset: Preset = LOOT, item_names=None):
        self.engine, self.preset, self.item_names = engine, preset, item_names
        self.static = StaticText()

    def read(self, crop: np.ndarray, now: float | None = None) -> LootRead | None:
        """The pickup messages in `crop` (the loot box's pixels); None when OCR is unavailable."""
        now = time.monotonic() if now is None else now
        image, recheck = self.static.prepare(crop, now)
        rows = self.engine.read(image, min_height=self.preset.min_height)
        if rows is None:
            return None
        messages, ignored = [], []
        for row in rows:
            line = parse_loot(row.text, self.item_names)
            if line:
                messages.append((row, line))
            else:
                ignored.append(row)
        self.static.learn(ignored, now, recheck)
        return LootRead(rows, messages, ignored, self.static.masked_rows(), recheck)
