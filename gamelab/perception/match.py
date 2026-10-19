"""Template matching shared by the monster and loot detectors (mobs.py, loot.py): the masked colour
match, and keeping the best of overlapping detections.
"""

from typing import TypeVar

import cv2
import numpy as np


Hit = TypeVar("Hit")  # a detection (mobs.Mob, loot.Item): anything with a centre x, y, a size w, h and a score


def masked_match(image: np.ndarray, template: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Masked, normalised colour match of `template` at every place in `image`. A place where it's
    undefined (a flat window) scores -1."""
    return np.nan_to_num(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask3), nan=-1.0,
                         posinf=-1.0, neginf=-1.0)


def suppress_overlaps(hits: list[Hit]) -> list[Hit]:
    """The best of overlapping detections: each, best score first, is kept unless it overlaps one kept."""
    kept: list[Hit] = []
    for hit in sorted(hits, key=lambda m: -m.score):
        if all(abs(hit.x - k.x) > max(hit.w, k.w) // 2 or abs(hit.y - k.y) > max(hit.h, k.h) // 2 for k in kept):
            kept.append(hit)
    return kept
