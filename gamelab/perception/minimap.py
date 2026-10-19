"""Her yellow dot on the minimap (top left), in full-size game-window pixels.
"""

import cv2
import numpy as np


# The minimap window sits at the top left; its map picture is inside this box (full-size px).
MINIMAP_BOX = (0, 95, 300, 300)  # x0, y0, x1, y1
DOT_AREA = (12, 200)  # px a yellow blob covers to be her dot (smallest, largest) ...
DOT_SIZE = 16         # ... and its widest and tallest


def her_dot(full: np.ndarray) -> tuple[float, float] | None:
    """Her yellow dot on the minimap, in game-window pixels (full size), or None if hidden."""
    x0, y0, x1, y1 = MINIMAP_BOX
    return dot_in_box(full[y0:y1, x0:x1])


def dot_in_box(box: np.ndarray) -> tuple[float, float] | None:
    """Her dot in a capture of just MINIMAP_BOX (RGB), in game-window pixels: the biggest yellow blob of
    a dot's size."""
    x0, y0 = MINIMAP_BOX[:2]
    area = box.astype(np.int16)
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    yellow = ((r > 220) & (g > 190) & (b < 90)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(yellow)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, count)
             if DOT_AREA[0] <= stats[i, cv2.CC_STAT_AREA] <= DOT_AREA[1]
             and stats[i, cv2.CC_STAT_WIDTH] <= DOT_SIZE and stats[i, cv2.CC_STAT_HEIGHT] <= DOT_SIZE]
    if not blobs:
        return None
    _, (cx, cy) = max(blobs, key=lambda blob: blob[0])
    return float(cx) + x0, float(cy) + y0
