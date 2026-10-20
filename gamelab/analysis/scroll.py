"""How far the camera scrolled, so her movement can be measured in the world, not on screen.

The camera follows her, so on-screen movement shrinks or even reverses mid-chain. The platform she
stands on scrolls 1:1 with the camera; tracking it gives the camera's position (camera_x, from the
strip under her), or the platforms anywhere in the frame (camera, in x and y).
"""

import cv2
import numpy as np

from .track import peak


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


# The camera from the foreground: square patches of platform (grass or dirt: this map's colours), each
# matched in the next frame within a few px. The strip under her (camera_x) fails where her own body or a
# drifting cloud is in it: the clouds move on their own, so scenery-wide estimates drifted too.
PATCH, SEARCH = 24, 10        # px: patch size, and how far it is looked for in the next frame
PATCH_MIN_STD = 15.0          # a patch needs this much texture to be matched
PATCH_MIN_SCORE = 0.85
PATCH_MIN_FOREGROUND = 0.25   # share of a patch that must be grass or dirt
HER_BOX = (40, 75)            # px either side of her, and above her name tag: patches there are her


def foreground(rgb: np.ndarray) -> np.ndarray:
    """Where a frame shows platform: grass green or dirt brown (this map's)."""
    r, g, b = [rgb[..., i].astype(np.int16) for i in range(3)]
    grass = (g > 120) & (g > b + 50) & (g > r + 5)
    dirt = (r > 70) & (r < 200) & (g > 50) & (b < 0.6 * g) & (r >= g - 25) & (abs(r - g) < 60)
    return grass | dirt


def _patch_shifts(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, her: tuple[float, float]) -> np.ndarray:
    """(dx, dy) of each textured foreground patch of `prev` (away from her) in `cur`, to a fraction of a px."""
    h, w = prev.shape
    hx, hy = her
    out = []
    for y in range(SEARCH, h - PATCH - SEARCH, PATCH // 2):
        for x in range(SEARCH, w - PATCH - SEARCH, PATCH // 2):
            if abs(x + PATCH / 2 - hx) < HER_BOX[0] and -HER_BOX[1] < y + PATCH / 2 - hy < 22:
                continue
            if mask[y:y + PATCH, x:x + PATCH].mean() < PATCH_MIN_FOREGROUND:
                continue
            p = prev[y:y + PATCH, x:x + PATCH]
            if p.std() < PATCH_MIN_STD:
                continue
            r = cv2.matchTemplate(cur[y - SEARCH:y + PATCH + SEARCH, x - SEARCH:x + PATCH + SEARCH], p,
                                  cv2.TM_CCOEFF_NORMED)
            _, score, _, (mx, my) = cv2.minMaxLoc(r)
            if score < PATCH_MIN_SCORE:
                continue
            fx = peak(r[my, mx - 1], r[my, mx], r[my, mx + 1]) if 0 < mx < r.shape[1] - 1 else 0
            fy = peak(r[my - 1, mx], r[my, mx], r[my + 1, mx]) if 0 < my < r.shape[0] - 1 else 0
            out.append((mx + fx - SEARCH, my + fy - SEARCH))
    return np.array(out) if out else np.zeros((0, 2))


def camera(frames, her) -> np.ndarray:
    """Cumulative camera (x, y) per frame in frame px (positive: the camera moved right or down), from
    the foreground patches; `her`: where she is in each frame (patches there are skipped). Each step is
    the median shift of the patches that agree with the most common one (to half a px): moving scenery
    and her own body make the others."""
    gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    out = [np.zeros(2)]
    for i in range(1, len(frames)):
        d = _patch_shifts(gray[i - 1], gray[i], foreground(frames[i - 1]), her[i - 1])
        if len(d) < 2:
            out.append(out[-1].copy())
            continue
        bins = np.round(d[:, 0] * 2) / 2
        values, counts = np.unique(bins, return_counts=True)
        agree = np.abs(d[:, 0] - values[np.argmax(counts)]) <= 0.75
        out.append(out[-1] - np.median(d[agree], axis=0))  # scenery moving left: the camera moved right
    return np.array(out)
