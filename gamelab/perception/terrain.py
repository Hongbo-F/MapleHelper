"""The ground and ropes on screen: how far the ground under her goes (half-size frame px), and ropes
and ladders (full-size px).
"""

from dataclasses import dataclass

import cv2
import numpy as np


ROPE_ROWS = 0.9        # ropes are looked for in this top share of the screen
MIN_ROPE_LENGTH = 90   # px of a rope that must show for it to count
MAX_ROPE_WIDTH = 12    # px wide at most ...
MIN_ROPE_FILL = 1.5    # ... and on average at least (its area over its length)
ROPE_GAP = (6, 40)     # pieces of one rope: at most this far apart across (centres) and down (px)


def ground_extent(frame: np.ndarray, x: int, floor_y: int) -> tuple[int, int]:
    """Screen x range of the ground she stands on: from her x outwards along the row just under
    her feet, until sky or sea shows (3 pixels in a row). Walls of dirt count as ground, so on a
    floor between cliffs it reaches the walls; on a floating platform it stops at the edges."""
    rows = frame[max(0, floor_y + 2):floor_y + 7].astype(np.int16)
    if rows.size == 0:
        return x, x
    r, g, b = rows[..., 0], rows[..., 1], rows[..., 2]
    sky = ((b > r + 30) & (b >= g - 20)) | ((r > 190) & (g > 190) & (b > 190))
    open_air = sky.mean(axis=0) > 0.5

    def scan(step):
        pos, run = x, 0
        while 0 <= pos + step < frame.shape[1]:
            pos += step
            run = run + 1 if open_air[pos] else 0
            if run >= 3:
                return pos - step * 3
        return pos

    return scan(-1), scan(1)


@dataclass(frozen=True)
class ScreenRope:
    """A rope or ladder seen on screen (full-size px)."""
    x: int        # centre
    top: int
    bottom: int


def rope_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels with the colour of a rope's light strands (greyish beige)."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r >= 140) & (r <= 235) & (g >= 125) & (g <= 215) & (b >= 95) & (b <= 180)
            & (r - b >= 20) & (r - b <= 70) & (np.abs(r - g) <= 30))


def find_ropes(full: np.ndarray) -> list[ScreenRope]:
    """Ropes and ladders: long, thin, vertical runs of rope colour (strands may break up a little)."""
    h = int(full.shape[0] * ROPE_ROWS)
    mask = rope_mask(full[:h]).astype(np.uint8)
    # Join the twisted strands vertically, then keep only tall, narrow shapes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((31, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    ropes = []
    for x, y, w, rh, area in stats[1:count]:
        if rh >= MIN_ROPE_LENGTH and w <= MAX_ROPE_WIDTH and area >= rh * MIN_ROPE_FILL:
            ropes.append(ScreenRope(int(x + w // 2), int(y), int(y + rh)))
    # One rope can split into pieces where something passes in front: merge pieces in one column.
    ropes.sort(key=lambda r: (r.x, r.top))
    merged: list[ScreenRope] = []
    for rope in ropes:
        last = merged[-1] if merged else None
        if last and abs(last.x - rope.x) <= ROPE_GAP[0] and rope.top - last.bottom <= ROPE_GAP[1]:
            merged[-1] = ScreenRope((last.x + rope.x) // 2, last.top, max(last.bottom, rope.bottom))
        else:
            merged.append(rope)
    return merged
