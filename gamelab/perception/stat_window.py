"""Her W.DEF off her stat window, read: what her touches are worked out at (behavior.bootstrap: her stat window's DEF
over a fit to her touches, over the prior). Prepared on 2026-10-28 like the key bindings' read was (the window opened
by its key, found by the game data's picture, its field cut where the game data puts it); not read live yet: no saved
frame shows the window (the 4,085 lossless frames under lab/ and every 20th frame of the rolling recordings searched:
D:/MapleHelper-scratch/safety3/stats/statwin, find_stat.py and find_stat_jpeg.py), so the reader is checked on frames
made from the game data's own pictures (tests/test_stat_window.py) and needs the live game to confirm (README, Her
stats from perception: To check in game).

How it's opened: her key bindings (lab/shots/keyconfig.png, 2026-10-28) have S on ABILITY, the stat window (KEY; the
bottom bar's stat button, the third of the five over CASH SHOP, opens it too). It opens as CHARACTER STAT
(UI CharacterStat.img Main/backgrnd, 185 x 281: NAME, JOB, LEVEL, HP, MP, EXP., FAME, the four stats, ABILITY POINT);
its DETAIL button (Main/button:detail) opens the panel beside it (Detail/backgrnd, 187 x 223) with ATTACK, WEAPON DEF.,
MAGIC, MAGIC DEF., ACCURACY, EVASION, CRIT. RATE, CRIT. DAMAGE, SPEED, JUMP: her W.DEF is WEAPON DEF.'s value, drawn
from the panel's (78, 28) (Detail/vector:pdd), in the light cell right of its label. The values are text in the game's
own font (UI/Font: NotoSansKRMedium), not the bar's glyph pictures, so the cell is read by the OCR (maplehelper.ocr's
shared RapidOCR, as the message bar is: perception.messages), its first whole number taken.

What it does: `find` places a picture of the game data's on a frame (masked squared difference over its opaque px,
FIND_FIT a channel at the most: 0 for the picture pasted in, 32 on the same frame as a JPEG at quality 85, 20,000 for
the best place on a frame without it); `read_wdef` finds the Detail panel and reads W.DEF's cell, None with why when it
isn't there. Wanted live (not built: it sends keys): S, a click on DETAIL (the panel's button, where `find` puts the
Main window), the read, S again to close it; at a hunt's start, between fights.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

KEY = "s"           # ABILITY in her key bindings (lab/shots/keyconfig.png, 2026-10-28): opens CHARACTER STAT
FIND_FIT = 400.0    # levels^2 a channel over the picture's opaque px at the most, for it to be there
WDEF_AT = (78, 28)  # game px in the Detail panel: where WEAPON DEF.'s value is drawn from (Detail/vector:pdd)
CELL = (76, 28, 177, 43)  # ... and the light cell it's drawn in (x0, y0, x1, y1, panel px: the picture's 225, 223, 213)
NUMBER = re.compile(r"\d{1,4}")


@dataclass(frozen=True)
class StatReading:
    """W.DEF as read off the stat window: `wdef` (None: not read, `why` says why), `text` the OCR's, `at` the Detail
    panel's top left on the frame."""
    wdef: int | None
    text: str = ""
    at: tuple[int, int] | None = None
    why: str = ""


@lru_cache(maxsize=4)
def picture(part: str) -> np.ndarray:
    """A picture of the stat window from the game data (UI CharacterStat.img: "Main/backgrnd", "Detail/backgrnd"),
    RGBA."""
    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    tree = Package(COPY / "UI" / "UI_000.wz").image("CharacterStat.img")
    a, b = part.split("/")
    out = pixels(source(tree[a][b], tree, COPY))
    out.flags.writeable = False
    return out


def find(frame: np.ndarray, part: str = "Detail/backgrnd") -> tuple[tuple[int, int] | None, float]:
    """Where `part`'s picture is on `frame` (full size, RGB, at the game's own 1366x768: its UI px are the screen's), its
    top left, and how well it fits there (levels^2 a channel); None for the place if it fits worse than FIND_FIT."""
    import cv2
    rgba = picture(part)
    h, w = rgba.shape[:2]
    if frame is None or frame.shape[0] < h or frame.shape[1] < w:
        return None, float("inf")
    mask = (rgba[..., 3] > 250).astype(np.uint8)
    r = cv2.matchTemplate(np.ascontiguousarray(frame[..., :3]), np.ascontiguousarray(rgba[..., :3]), cv2.TM_SQDIFF,
                          mask=mask)
    v, _, loc, _ = cv2.minMaxLoc(r)
    fit = float(v) / max(1, int(mask.sum())) / 3
    return (tuple(int(c) for c in loc) if fit <= FIND_FIT else None), fit


def cell(at: tuple[int, int]) -> tuple[int, int, int, int]:
    """W.DEF's cell on the frame (x0, y0, x1, y1), the Detail panel at `at`."""
    return at[0] + CELL[0], at[1] + CELL[1], at[0] + CELL[2], at[1] + CELL[3]


def _ocr(bgr: np.ndarray) -> str:
    from .messages import _shared_ocr
    return _shared_ocr(bgr)


def read_wdef(frame: np.ndarray, ocr=None) -> StatReading:
    """Her W.DEF off `frame` (full size, RGB) if the stat window's Detail panel is open on it: its cell read by `ocr`
    (a BGR crop -> text; None: maplehelper.ocr's shared engine), enlarged x2 as the message bar's is, the first whole
    number taken."""
    at, fit = find(frame)
    if at is None:
        return StatReading(None, why=f"no stat window's Detail panel on the frame (its picture fits {fit:.0f} at best)")
    x0, y0, x1, y1 = cell(at)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1, 2::-1])
    big = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
    text = (ocr or _ocr)(big)
    m = NUMBER.search(text or "")
    if m is None:
        return StatReading(None, text or "", at, why=f"no number in W.DEF's cell ({text!r})")
    return StatReading(int(m.group(0)), text, at)
