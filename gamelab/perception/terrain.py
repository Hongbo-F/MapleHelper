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
# A rope's light strands (greyish beige): red, green and blue each in these ranges (from, to) ...
ROPE_RGB = ((140, 235), (125, 215), (95, 180))
ROPE_RED_OVER_BLUE = (20, 70)  # ... red this much above blue ...
ROPE_RED_GREEN = 30            # ... and red and green at most this far apart
# Px (across, down) around the rope colour's pixels that the shapes are worked on in (_ropes_in). Closing
# (9 rows by 3) and opening (31 by 1) can't reach more than 4 + 15 rows and 1 column past the colour's own
# pixels, and an erosion next to the box's edge must see only what the whole strip has there, nothing: 2
# columns and 19 rows would do.
CROP_PAD = (4, 32)


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
    return _rope_pixels(rgb) > 0


def _rope_pixels(rgb: np.ndarray) -> np.ndarray:
    """rope_mask as OpenCV makes it, 255 where it's set (uint8): 6 ms over the screen against 16 with numpy's
    int16 tests, and the same pixels. Red minus blue saturates at 0 below zero, which fails its test as a
    negative difference would."""
    r, g, b = cv2.split(rgb)
    (r0, r1), (g0, g1), (b0, b1) = ROPE_RGB
    colour = cv2.inRange(rgb, (r0, g0, b0), (r1, g1, b1))
    red_over_blue = cv2.inRange(cv2.subtract(r, b), *ROPE_RED_OVER_BLUE)
    red_green = cv2.inRange(cv2.absdiff(r, g), 0, ROPE_RED_GREEN)
    return cv2.bitwise_and(cv2.bitwise_and(colour, red_over_blue), red_green)


def find_ropes(full: np.ndarray, columns: list[tuple[int, int]] | None = None) -> list[ScreenRope]:
    """Ropes and ladders: long, thin, vertical runs of rope colour (strands may break up a little).
    `columns`: only these ranges of x (from, to; full-size px) are searched; None: the whole screen."""
    h, width = int(full.shape[0] * ROPE_ROWS), full.shape[1]
    ropes = []
    for x0, x1 in [(0, width)] if columns is None else _spans(columns, width):
        ropes += _ropes_in(full[:h, x0:x1], x0)
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


def _ropes_in(strip: np.ndarray, x0: int) -> list[ScreenRope]:
    """The ropes in a strip of the screen whose left edge is at x0. The strip is copied out of the screen
    first: OpenCV split a strip viewed in place 8 times slower (2.9 ms a look against 0.4). After the colour
    test the shapes are made only in the box around its pixels (CROP_PAD; on the saved frames 58% of a
    strip), and connected only in the box around what's left of them (9%): the same shapes as in the whole
    strip, connected in 0.1 ms a strip instead of 0.4."""
    mask = _rope_pixels(np.ascontiguousarray(strip))
    bx, by, bw, bh = cv2.boundingRect(mask)
    if not bw:
        return []
    h, w = mask.shape
    px, py = CROP_PAD
    left, top = _even(bx - px), _even(by - py)
    mask = mask[top:min(h, by + bh + py), left:min(w, bx + bw + px)]
    # Join the twisted strands vertically, then keep only tall, narrow shapes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((31, 1), np.uint8))
    sx, sy, sw, sh = cv2.boundingRect(mask)  # the shapes are connected in their own box: the same shapes
    if not sw:
        return []
    sw, sh, sx, sy = sw + sx % 2, sh + sy % 2, _even(sx), _even(sy)
    left, top = left + sx, top + sy
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask[sy:sy + sh, sx:sx + sw])
    return [ScreenRope(int(x0 + left + x + w // 2), int(top + y), int(top + y + rh))
            for x, y, w, rh, area in stats[1:count]
            if rh >= MIN_ROPE_LENGTH and w <= MAX_ROPE_WIDTH and area >= rh * MIN_ROPE_FILL]


def _even(n: int) -> int:
    """n rounded down to even, at least 0: a box cut from the strip starting on an even row and column, the
    shapes are connected in the same order as in the whole strip (OpenCV labels 2 by 2 pixel blocks)."""
    return max(0, n - n % 2)


def _spans(columns: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    """Ranges of x on a screen this wide, overlapping ones joined (so a rope in two is found once)."""
    spans: list[tuple[int, int]] = []
    for x0, x1 in sorted((max(0, int(a)), min(width, int(b))) for a, b in columns):
        if x1 <= x0:
            continue
        if spans and x0 <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], x1))
        else:
            spans.append((x0, x1))
    return spans
