"""How far the camera scrolled, so her movement can be measured in the world, not on screen.

The camera follows her, so on-screen movement shrinks or even reverses mid-chain. The platform she
stands on scrolls 1:1 with the camera; tracking it gives the camera's position.
"""

import cv2
import numpy as np


def camera_x(frames, rows, band=(12, 30)) -> list[float]:
    """Cumulative camera x per frame, from the textured strip `band` px below her resting name tag."""
    rest_y = rows[0][2]
    y0, y1 = rest_y + band[0], rest_y + band[1]

    def strip(f):
        return cv2.cvtColor(np.ascontiguousarray(f[y0:y1]), cv2.COLOR_RGB2GRAY).astype(np.float32)

    prev = strip(frames[0])
    window = cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F)
    total, out = 0.0, [0.0]
    for f in frames[1:]:
        cur = strip(f)
        (dx, _), _ = cv2.phaseCorrelate(prev, cur, window)
        # The game scrolls in whole pixels (half pixels at scale 2); rounding drops sub-pixel noise
        # that would otherwise add up to a drift.
        total -= round(dx * 2) / 2  # scenery moving left on screen means the camera moved right
        out.append(total)
        prev = cur
    return out


def world_x(frames, rows) -> list[float]:
    """Her name tag's x in world terms (frame pixels, relative to where the camera started)."""
    return [r[1] + c for r, c in zip(rows, camera_x(frames, rows))]
