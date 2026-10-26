"""Template matching shared by the monster and loot detectors (mobs.py, loot.py): the masked colour
match, and keeping the best of overlapping detections.

The masked match is in C where it's built (native.py, native.c's mm_*: OpenCV's score worked out exactly, in
integers, within 1e-6 of OpenCV's own in colour and 3e-5 on brightness), else OpenCV's. In C each template is made
ready once (`prepared`: its sums, its mask's runs), and a few templates over one window (a monster re-found) go in one
call (`best_matches`). On the masked matches of 60 frames of a crowd (2026-10-26 01:01, the frame loop's and its
searches': D:/MapleHelper-scratch/pconst/pnative/cmp_calls.py, one thread, best of 5) a re-find's window took 0.45 ms
against OpenCV's 1.57, a step-2 check 0.05 against 0.58, a drop's strip 0.28 against 0.68 (4.5x over all of them), the
best place and the 0.55, 0.6 and 0.9 thresholds' decisions the same on all 2,894 calls.
"""

import math
import weakref
from collections.abc import Iterator
from typing import TypeVar

import cv2
import numpy as np

from .. import native


Hit = TypeVar("Hit")  # a detection (mobs.Mob, loot.Item): anything with a centre x, y, a size w, h and a score

# (id(template), id(mask3)) -> (a weak reference to each, the pair made ready for the C: native.Masked, or None if the
# C doesn't take it). An entry goes when either array does; the references tell a reused id from its array.
_PREPARED: dict[tuple[int, int], tuple] = {}


def prepared(template: np.ndarray, mask3: np.ndarray) -> "native.Masked | None":
    """`template` and its mask made ready for the masked match in C (native.prepare_masked), once for each pair of
    arrays while they're kept (a detector's templates: made ready as it loads them); None without the C, or for one it
    doesn't take."""
    if native.lib() is None:
        return None
    key = (id(template), id(mask3))
    entry = _PREPARED.get(key)
    if entry is not None and entry[0]() is template and entry[1]() is mask3:
        return entry[2]
    ready = native.prepare_masked(template, mask3)
    try:
        gone = lambda _, key=key: _PREPARED.pop(key, None)  # noqa: E731
        _PREPARED[key] = (weakref.ref(template, gone), weakref.ref(mask3, gone), ready)
    except TypeError:  # (not an array that takes a weak reference: made ready each time)
        pass
    return ready


def masked_match(image: np.ndarray, template: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Masked, normalised colour match of `template` at every place in `image`. A place where it's
    undefined (a flat window) scores -1."""
    ready = prepared(template, mask3)
    if ready is not None:
        out = native.masked_match(image, ready)
        if out is not None:
            return out
    return _opencv(image, template, mask3)


def _opencv(image: np.ndarray, template: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    return np.nan_to_num(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask3), nan=-1.0,
                         posinf=-1.0, neginf=-1.0)


def best_match(image: np.ndarray, template: np.ndarray, mask3: np.ndarray) -> tuple[float, tuple[int, int]]:
    """The best score of `masked_match` over `image` and where it is (x, y), as cv2.minMaxLoc gives them: the first
    highest, row by row."""
    ready = prepared(template, mask3)
    if ready is not None:
        found = native.best_matches(image, [ready])
        if found is not None:
            score, x, y = found[0]
            return score, (x, y)
    _, score, _, at = cv2.minMaxLoc(_opencv(image, template, mask3))
    return score, at


def best_matches(image: np.ndarray, pictures: list[tuple[np.ndarray, np.ndarray]],
                 stop: float = math.inf) -> Iterator[tuple[float, tuple[int, int]]]:
    """`best_match` of each (template, mask3) in turn over one image, as the caller goes through them. In C they're
    matched in one call, the image loaded once, up to the first whose best is `stop` or more; the rest only if the
    caller goes on past it (a monster re-found stops at its first picture that scores REFIND_GOOD: mobs.refind)."""
    ready = [prepared(t, m) for t, m in pictures]
    i = 0
    while i < len(pictures):
        j = i
        while j < len(pictures) and ready[j] is not None:
            j += 1
        found = native.best_matches(image, ready[i:j], stop) if j > i else None
        if not found:  # (OpenCV's: one at a time)
            yield best_match(image, *pictures[i])
            i += 1
            continue
        for score, x, y in found:
            yield score, (x, y)
        i += len(found)


def suppress_overlaps(hits: list[Hit]) -> list[Hit]:
    """The best of overlapping detections: each, best score first, is kept unless it overlaps one kept."""
    kept: list[Hit] = []
    for hit in sorted(hits, key=lambda m: -m.score):
        if all(abs(hit.x - k.x) > max(hit.w, k.w) // 2 or abs(hit.y - k.y) > max(hit.h, k.h) // 2 for k in kept):
            kept.append(hit)
    return kept
