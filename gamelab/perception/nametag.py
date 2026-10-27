"""Finding her on screen by her name tag.

The tag is matched at full resolution: at half size the score swings between ~0.6 and ~0.9 depending on
which pixels the downscale keeps, and her name in the minimap scored ~0.6 too. The full-size template is
cut from the screen the first time, where the half-size one from calibration finds her, and kept.

A look searches near where she was found last first (1.6 ms within NEAR_FRAME, 3.6 within NEAR; the whole play area
takes 35, matched in row bands at once: BANDS): the camera keeps her at one spot on screen except near the map's
edges, so her tag is seldom far from where it was.

Her tag is drawn at the game's size: its templates are made for one client size (screen.Geometry; the files told
apart by its `suffix`), and the px below were measured on the 1920x1080 window and go by the geometry.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..screen import STRETCHED, Geometry


PLAY_BOTTOM = 0.9      # the bottom bar starts below this share of the screen's height: not searched
MINIMAP_WIDTH = 0.17   # the minimap's share of the screen's width ...
MINIMAP_HEIGHT = 0.3   # ... and height, at the top left: not searched
MATCH = 0.7            # a full-size match at least this good is one of the places her name shows
CUT_MATCH = 0.75       # the half-size template must match this well to cut the full-size one there
CANDIDATES = 4         # at most this many good matches are compared to pick hers
START_HEIGHT = 0.6     # until she's found, the match nearest the screen's middle, this share of the way down, wins
SAME_NAME = (20, 60)   # full-size px around a match (rows, columns) where other matches are the same name
# Full-size px across and down from where she was found last that a look searches first. Where the camera
# stops at the map's edges she moves on screen: 7 px a frame walking, up to about 31 falling. When she isn't
# in there (a warp, a long walk by an edge, the tag hidden), the whole play area is searched.
# First within NEAR_FRAME of it (about a frame's move), then within NEAR. On the saved frames of 2026-10-26 01:01
# (3,000 finds) the answers were the same as with NEAR alone, 1.60 ms a find against 3.63
# (D:/MapleHelper-scratch/cpu/perception/tag_ab.py).
NEAR = (240, 120)
NEAR_FRAME = (80, 40)
FULL_EVERY = 30        # ... and every this many looks the whole play area anyway, so a wrong match can't stick: 1 s
                       # hunting (the service's 30 frames a second), 6 s between hunts (5 a second). At 35 ms each, 1.2
                       # ms a frame (150, tried on 2026-10-26: 5 s hunting but 30 s between hunts)
# After this many looks in a row without her tag, the full-size template is cut again from calibration's half-size one
# (`_cut`, about 35 ms), in case it no longer matches her (kept from another day); if the half-size one doesn't find her
# either, nothing changes. In the tours on Tree Dungeon on 2026-10-24 (7,646 looks, 1.8 a second) her tag was hidden 14
# looks in a row at the most.
RECUT_AFTER = 50
# The whole play area's match (`search` with no box: every FULL_EVERY-th look, and when she isn't near where she was)
# in this many row bands at once, on threads of its own (OpenCV lets go of Python's lock), each band its rows and the
# template's height less one more, the results stacked. OpenCV takes as long for it on one thread as on 12. On the
# 3,050 saved frames of 2026-10-26 01:01 (D:/MapleHelper-scratch/pconst/plan, tag_split.py: one at a time, the machine
# busy with other work) 12.5 ms against 24.3 (the median of the first 1,500; 9.9 against 17.0 of the rest), the same
# answers on every frame; matched in pieces OpenCV's scores (its correlation goes through Fourier transforms) differ by
# up to 1.6e-3, in windows near flat.
BANDS = 4
_pool: list = []  # its threads, made on the first whole search (any process's)
_pool_made = threading.Lock()


def _hide_minimap(res: np.ndarray, shape: tuple[int, ...], x0: int = 0, y0: int = 0) -> None:
    """Rule out matches in the minimap of a screen of this shape (in place). `res` holds the matches of a
    box of the screen whose top-left corner is (x0, y0)."""
    height, width = shape[:2]
    res[:max(0, int(height * MINIMAP_HEIGHT) - y0), :max(0, int(width * MINIMAP_WIDTH) - x0)] = -1


class NameTag:
    """Her name tag: where it is in each look, in frames of geometry `geo` (its templates made for that size)."""

    def __init__(self, small: np.ndarray, path: Path, geo: Geometry = STRETCHED):
        self.small = small    # half-size template from calibration, used to find her once
        self.path = path      # where the full-size template is kept
        self.geo = geo
        self.full = np.load(path) if path.exists() else None
        self._last: tuple[int, int] | None = None
        self.looks = 0   # finds so far ...
        self.missed = 0  # ... and how many of them didn't find her near where she was (the whole play area searched)
        self.unseen = 0  # finds in a row that found her nowhere
        self.recuts = 0  # times the full-size template was cut again (RECUT_AFTER)

    def lose(self) -> None:
        """Forget where she was: the next find searches the whole play area (after a look that failed its checks)."""
        self._last = None

    def find(self, full: np.ndarray) -> tuple[int, int] | None:
        """Centre of her name tag in a full-size frame, in look-frame pixels (None: not on screen).
        Her name also shows in the minimap, the bottom bar and the party list, so of the good matches in
        the play area, the one nearest where she was (or the screen centre, which the camera keeps her
        near) wins. It's looked for within NEAR_FRAME of where she was first, then within NEAR, and in the whole
        play area if it isn't there (and every FULL_EVERY looks)."""
        if self.full is None:
            self.full = self._cut(full)
        self.looks += 1
        found = None
        if self._last is not None and self.looks % FULL_EVERY:
            found = self.search(full, self._near(full.shape, NEAR_FRAME))
            if found is None:
                found = self.search(full, self._near(full.shape))
            self.missed += found is None
        if found is None:
            found = self.search(full)
        self.unseen = 0 if found is not None else self.unseen + 1
        if self.unseen >= RECUT_AFTER:
            found = self._recut(full)
        if found is not None:
            self._last = found
        return found

    def _recut(self, full: np.ndarray) -> tuple[int, int] | None:
        """Cut the full-size template again from calibration's (RECUT_AFTER) and search with it; None if the half-size
        one doesn't find her either (the template is kept then)."""
        self.unseen = 0
        try:
            self.full = self._cut(full)
        except RuntimeError:
            return None
        self.recuts += 1
        return self.search(full)

    def _near(self, shape: tuple[int, ...], near: tuple[int, int] = NEAR) -> tuple[int, int, int, int]:
        """The box of the play area (x0, y0, x1, y1; full-size px) a tag within `near` (base px: NEAR by default) of
        where she was found last lies in."""
        th, tw = self.full.shape
        k = self.geo.look_scale
        x, y = k * self._last[0] - tw // 2, k * self._last[1] - th // 2  # its top-left corner then
        nx, ny = self.geo.ipx(near[0]), self.geo.ipx(near[1])
        return (max(0, x - nx), max(0, y - ny),
                min(shape[1], x + nx + tw), min(int(shape[0] * PLAY_BOTTOM), y + ny + th))

    def search(self, full: np.ndarray, box: tuple[int, int, int, int] | None = None) -> tuple[int, int] | None:
        """Her tag's centre (look-frame px) in a box of the frame (x0, y0, x1, y1; default: the whole play
        area): of the good matches there, the one nearest where she was. None if there's none."""
        h, w = full.shape[:2]
        x0, y0, x1, y1 = box or (0, 0, w, int(h * PLAY_BOTTOM))
        th, tw = self.full.shape
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        gray = cv2.cvtColor(full[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        res = _banded(gray, self.full) if box is None else cv2.matchTemplate(gray, self.full, cv2.TM_CCOEFF_NORMED)
        _hide_minimap(res, full.shape, x0, y0)
        k = self.geo.look_scale
        ref = self._last or (w // 2 // k, int(h * START_HEIGHT) // k)
        best = None
        for _ in range(CANDIDATES):
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < MATCH:
                break
            centre = ((x0 + x + tw // 2) // k, (y0 + y + th // 2) // k)
            dist = abs(centre[0] - ref[0]) + abs(centre[1] - ref[1])
            if best is None or dist < best[0]:
                best = (dist, centre)
            dy, dx = self.geo.ipx(SAME_NAME[0]), self.geo.ipx(SAME_NAME[1])
            res[max(0, y - dy):y + dy, max(0, x - dx):x + dx] = -1
        return None if best is None else best[1]

    def _cut(self, full: np.ndarray) -> np.ndarray:
        """Find her with the half-size template (trying all four ways of halving the frame), then cut
        a full-resolution template of her name tag there and keep it for next time."""
        best = None
        k = self.geo.look_scale  # calibration's template is cut from a frame shrunk as a look's is
        for oy in range(k):
            for ox in range(k):
                half = cv2.cvtColor(np.ascontiguousarray(full[oy::k, ox::k]), cv2.COLOR_RGB2GRAY)
                res = cv2.matchTemplate(half[:int(half.shape[0] * PLAY_BOTTOM)], self.small, cv2.TM_CCOEFF_NORMED)
                _hide_minimap(res, half.shape)
                _, score, _, (x, y) = cv2.minMaxLoc(res)
                if best is None or score > best[0]:
                    best = (score, k * x + ox, k * y + oy)
        if best is None or best[0] < CUT_MATCH:
            raise RuntimeError("can't find her name tag to build a full-size template; run `calibrate --fresh`")
        # Cut exactly where the half-size template matched, so a name next to hers can't get in.
        th, tw = (k * n for n in self.small.shape)
        tag = cv2.cvtColor(np.ascontiguousarray(full[best[2]:best[2] + th, best[1]:best[1] + tw]), cv2.COLOR_RGB2GRAY)
        if not (folder := Path(self.path).parent).exists():  # (another character's, under lab/characters/: paths.of)
            folder.mkdir(parents=True, exist_ok=True)
        np.save(self.path, tag)
        return tag


def _banded(gray: np.ndarray, template: np.ndarray) -> np.ndarray:
    """cv2.matchTemplate(gray, template, TM_CCOEFF_NORMED) worked out in BANDS row bands at once (each band's rows of
    the result from its rows and the template's height less one more), stacked; at once on too few rows."""
    th = template.shape[0]
    rows = gray.shape[0] - th + 1
    if BANDS < 2 or rows < BANDS * th:
        return cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    if not _pool:
        with _pool_made:
            if not _pool:
                _pool.append(ThreadPoolExecutor(BANDS, "tag-bands"))
    cuts = np.linspace(0, rows, BANDS + 1).astype(int)
    return np.vstack(list(_pool[0].map(
        lambda i: cv2.matchTemplate(gray[cuts[i]:cuts[i + 1] + th - 1], template, cv2.TM_CCOEFF_NORMED), range(BANDS))))
