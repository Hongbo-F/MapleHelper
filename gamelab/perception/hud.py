"""Bars on screen: her HP and MP in the bottom bar, her EXP number (it changes only when she
gains EXP), the purple text when a monster touches her, and a monster's own HP bar.
"""

import numpy as np


# Her EXP number in the bottom bar ("1,385 [8.30%]"), full-size px: it only changes when she gains
# EXP, i.e. when a monster she hit dies. It sits on the bar's plain dark background, so its text pixels
# are identical from look to look otherwise. (The "You received EXP" message at the bottom right was
# tried first, but it's faint and was missed over a bright background.)
EXP_NUMBER = (1060, 1030, 1200, 1056)  # x0, y0, x1, y1
# HP and MP bars in the bottom bar, as fractions of the game window (measured at 1920x1080).
HP_BAR = (707 / 1920, 854 / 1920)
MP_BAR = (859 / 1920, 1006 / 1920)
BAR_ROW = 1064 / 1080
BAR_SEARCH = 45   # half-size px either side of a monster's centre searched for its own HP bar
BAR_ABOVE = 30    # ... and how far above its top
BAR_MIN_RUN = 22  # half-size px of bar colours in a row that make a monster's HP bar


def bars(full: np.ndarray) -> tuple[float, float]:
    """HP and MP as fractions (0-1), read from the bar colours in a full-size game frame."""
    h, w = full.shape[:2]
    row = full[int(BAR_ROW * h)].astype(np.int16)

    def span(bar: tuple[float, float]) -> np.ndarray:
        return row[int(bar[0] * w):int(bar[1] * w)]

    return hp_fill(span(HP_BAR)), mp_fill(span(MP_BAR))


def hp_fill(row: np.ndarray) -> float:
    """The share of a row of the HP bar (RGB, int16) that's red: her HP left."""
    on = (row[:, 0] > 150) & (row[:, 1] < 90) & (row[:, 2] < 90)
    return float(on.sum()) / max(1, len(row))


def mp_fill(row: np.ndarray) -> float:
    """The share of a row of the MP bar (RGB, int16) that's blue: her MP left."""
    on = (row[:, 2] > 150) & (row[:, 0] < 90)
    return float(on.sum()) / max(1, len(row))


def touch_popup(frame: np.ndarray, tag: tuple[int, int]) -> int:
    """Purple pixels above her head (half-size frame; `tag`: her name tag's centre). The "MISS" a
    thief's evasion shows when a monster touches her without damage, and the numbers of damage she
    takes, are both purple, and nothing else on screen is. A count of 40+ means a monster touched her.
    (Used to count touches when hops over monsters were timed; nothing calls it during a hunt.)"""
    x, y = tag
    box = frame[max(0, y - 120):max(0, y - 15), max(0, x - 70):x + 70].astype(np.int16)
    if box.size == 0:
        return 0
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    return int(((r > 120) & (b > 150) & (g < 110) & (b - g > 80) & (r - g > 50)).sum())


def hp_bar(frame: np.ndarray, x: int, top: int) -> float | None:
    """A monster's HP bar, shown above it once it has been hit: the fraction still green, or None
    if there's no bar. `frame` is a half-size RGB game frame; (x, top) the monster's centre and top.

    The bar's inside is pure colours, rare in the scenery: green (0, g, 0) for the HP left, black for
    HP lost, and for a moment dark red (r, 0, 0) for the latest damage. It sits ~18 px above the
    monster (half size) and is ~35 px wide."""
    y0, x0 = max(0, top - BAR_ABOVE), max(0, x - BAR_SEARCH)
    box = frame[y0:max(y0, top + 2), x0:x + BAR_SEARCH].astype(np.int16)
    if box.size == 0:
        return None
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    green = (g > 120) & (r < 15) & (b < 15)
    # Any shade of pure green (the half-size frame blends green and black where they meet), black,
    # or the dark red of the latest damage.
    inside = (b < 15) & ((r < 15) | ((r > 90) & (g < 50)))
    # The longest run of bar colours in any row (the first of equally long ones).
    best = None
    for y in range(inside.shape[0]):
        run, start = 0, 0
        for i, on in enumerate(inside[y]):
            run = run + 1 if on else 0
            if run == 1:
                start = i
            if run >= BAR_MIN_RUN and (best is None or run > best[0]):
                best = (run, y, start)
    if best is None:
        return None
    run, y, start = best
    return float(green[y, start:start + run].sum()) / run
