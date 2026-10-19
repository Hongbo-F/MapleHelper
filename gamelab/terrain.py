"""Where she is on the map (minimap dot) and the ropes on screen.

All positions are in full-size game-window pixels unless noted.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# The minimap window sits at the top left; its map picture is inside this box (full-size px).
MINIMAP_BOX = (0, 95, 300, 300)  # x0, y0, x1, y1


def her_dot(full: np.ndarray) -> tuple[float, float] | None:
    """Her yellow dot on the minimap, in game-window pixels (full size), or None if hidden."""
    x0, y0, x1, y1 = MINIMAP_BOX
    return dot_in_box(full[y0:y1, x0:x1])


def dot_in_box(box: np.ndarray) -> tuple[float, float] | None:
    """Her dot in a capture of just MINIMAP_BOX, in game-window pixels."""
    x0, y0 = MINIMAP_BOX[:2]
    area = box.astype(np.int16)
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    yellow = ((r > 220) & (g > 190) & (b < 90)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(yellow)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, count)
             if 12 <= stats[i, cv2.CC_STAT_AREA] <= 200
             and stats[i, cv2.CC_STAT_WIDTH] <= 16 and stats[i, cv2.CC_STAT_HEIGHT] <= 16]
    if not blobs:
        return None
    _, (cx, cy) = max(blobs, key=lambda b: b[0])
    return float(cx) + x0, float(cy) + y0


@dataclass(frozen=True)
class Rope:
    x: int        # centre
    top: int
    bottom: int

    @property
    def length(self) -> int:
        return self.bottom - self.top


def rope_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels with the colour of a rope's light strands (greyish beige)."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r >= 140) & (r <= 235) & (g >= 125) & (g <= 215) & (b >= 95) & (b <= 180) & \
           (r - b >= 20) & (r - b <= 70) & (np.abs(r - g) <= 30)


def find_ropes(full: np.ndarray, min_length: int = 90, play_bottom: float = 0.9) -> list[Rope]:
    """Ropes and ladders: long, thin, vertical runs of rope colour (strands may break up a little)."""
    h = int(full.shape[0] * play_bottom)
    mask = rope_mask(full[:h]).astype(np.uint8)
    # Join the twisted strands vertically, then keep only tall, narrow shapes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((31, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    ropes = []
    for x, y, w, rh, area in stats[1:count]:
        if rh >= min_length and w <= 12 and area >= rh * 1.5:
            ropes.append(Rope(int(x + w // 2), int(y), int(y + rh)))
    # One rope can split into pieces where something passes in front: merge pieces in one column.
    ropes.sort(key=lambda r: (r.x, r.top))
    merged: list[Rope] = []
    for rope in ropes:
        last = merged[-1] if merged else None
        if last and abs(last.x - rope.x) <= 6 and rope.top - last.bottom <= 40:
            merged[-1] = Rope((last.x + rope.x) // 2, last.top, max(last.bottom, rope.bottom))
        else:
            merged.append(rope)
    return merged
