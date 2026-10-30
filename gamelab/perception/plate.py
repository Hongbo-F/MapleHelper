"""Her name plate found where her tag's gray match isn't sure: its exact colours counted, so whatever covers part of it
(a monster, the map's drip art, a drop, the bottom bar, the screen's edge) only takes pixels away.

Since 2026-10-29 her tag is an orange-ribbon plate item (a wood frame, a mushroom icon left, a maple leaf right, her
name in white on a translucent dark interior), and it is drawn bit-exact: 99.9% of its stable pixels (1,395 of its
114x33 box, in 40 colours) equal its model in 60 sure frames. The localization prototype of that day
(D:/MapleHelper-scratch/loc2/combined, robust-tag's matcher and the combined localizer's rule against other wearers)
found her this way on every held-out frame where the gray template wasn't sure but some of her plate showed (77 of
the 81 hard frames with her tag hidden or cut; NameTag found 12% of them): here it is ported, its counting in C.

The model (`PlateModel`, made at calibration from frames where her tag is sure: `PlateModel.build`, kept under her lab
folder: `model_path`): the plate's median picture around her tag's centre and which of its pixels are stable; of
those, her name's letters (white inside the interior) are hers alone, the frame and icons (less their pure black
outline, which is everywhere off the map's art) are shared by anyone wearing the same plate item (another player,
AAABattery, on 10-29: 998 of its 1,252 frame px match hers), and the interior's translucent background is never pure
white on her plate (white there is another name's letters). Her feet from the plate's centre (`feet`) are the plate's
own: (+1.5, -16) for this one, measured (the old text tag's eyes.tag_below_feet, 11, puts her 5 px low).

A window's candidates (`Plate.candidates`, the prototype's RobustTag._exact): at every place, the count of the plate's
stable pixels whose exact colour the frame has there (the prototype counted with a correlation of random palette
vectors, a little cross-talk between colours; here the counts are exact, the same peaks: the tests and the
evaluation set's frames check it); the TOPK best over FLOOR, NMS apart; the first refined +-1 px by the evidence
(frame px + W_LETTERS letters - 2 white where none should be); each one's `clean` (her letters in column bins of
BIN_WIDTH that show them all with no stray white: what of her name is surely hers). `decide` takes the best by
evidence less W_DIST a px from the prediction, among those good enough (DEFAULTS' rules) whose clean letters are at
least CLEAN_MIN (another wearer's plate showed at most 8 of the 143, hers 52 and more on every frame but 6 of 2,089,
where the body cue placed her) and that sit where the plate's pixels match best around them (`aligned`); without a
prediction, white where her name has none weighs as much as a letter of hers shown (another wearer's longer name, and
one that shares letters with hers, were taken for hers by the prototype's rules: DEFAULTS). `near` searches around a
prediction (NEAR, wider by GROW a frame she went unplaced), `verify` checks the gray template's peaks anywhere (no
prediction: her letters needed).

Made at calibration (localization.her_cues): the model (`PlateModel.build` from `pick_centres`' sure frames), her tag's
gray template cut from it (`tag_template`: her letters, as calibration cuts a tag from a frame) and the model re-centred
on that (`PlateModel.recentred`), her mirror axis by the plate's own stable columns (`PlateModel.axis`), her tag's sure
levels from its own scores (`sure_levels`; kept in the model, `PlateModel.sure`).

In C where the DLL is (gamelab/locate.c's lc_plate: the index, the counts 32 places a step on AVX2, the peaks and the
evidence), in numpy otherwise, the same candidates to the integer (tests/test_plate.py). A window around a prediction
(NEAR, 254x113 px) takes 0.2-0.4 ms in C.

Its gray stage (`gray_best`, `gray_peaks`): the full gray template (her tag as NameTag cuts it) by TM_CCOEFF_NORMED
in a window, or over the play area in row bands, its rows to the HUD's top (the prototype's ROWS_MAX, 700 at
1366x768: NameTag stops at PLAY_BOTTOM's 691, and 4 hard frames had her plate there).

Px here are a 1366x768 look frame's (the game drawn pixel for pixel, look_scale 1): the model is made for one client
size (its file's suffix), as her tag's templates are.
"""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2
import numpy as np

from .. import native
from ..screen import NATIVE, Geometry

BOX = (-56, 58, -16, 17)       # the plate's picture around her tag's centre (x0, x1, y0, y1; the centre: the gray
                               # template's top-left + (32, 7), as NameTag gives it)
INTERIOR = (-29, 37, -5, 5)    # its name's interior (x0..x1, y0..y1 inclusive, from the centre)
FEET = (1.5, -16.0)            # her feet from the plate's centre (the model's own; measured for this plate item)
STABLE_SHARE = 0.9             # a pixel of the model is stable where at least this share of its frames had its colour
SAME_COLOUR = 12               # ... within this (L1 over the channels) of the median
BIN_WIDTH = 8                  # her letters' column bins across the interior
RECALL, BAD_MAX, MIN_LETTERS = 0.85, 1, 6   # a bin is clean: this share of its letters white (one with fewer letters
                                            # doesn't count), at most this much white beside them
TOPK, SUPPRESS, FLOOR = 3, (3, 5), 40       # a window's peaks: at most this many, +-rows, +-columns apart, over this
NEAR = (60, 40)                # px across and down around the predicted centre searched (15 frames a second) ...
GROW, GROW_MAX = (25, 15), 4   # ... more per frame she went unplaced, up to this many frames' worth
PEAK_REACH = (4, 3)            # `verify`: around each gray peak
W_LETTERS, W_DIST = 4.0, 2.0
# The rules (`decide`). Without a prediction (`verify`, at the gray peaks anywhere) white where her name has none
# (`bad`) weighs as much as a letter of hers shown (w_bad_far 1.0; the prototype's 0.5 let another wearer's longer name
# through: on RiceBaby's frames of 10-29 BugBaby's model took AAABattery's plate, the same item, for hers on 30 of 483
# frames at the gray peaks, letters 84, bad 75, clean 18 of the 16 needed: 84 - 37.5 passed `verify`'s 25, 84 - 75
# doesn't; the review's D:/MapleHelper-scratch/loc2build/bugbaby/review/confuse.py). About a prediction it weighs half
# as before (w_bad): what covers her plate is white there too (a damage number, a notice: on BugBaby's train frames
# 572, 710, 711 bad 33-58 with 31-55 of her letters shown, refused at 1.0; RiceBaby's own up to 74 with 66 shown), and
# another wearer must be where she's predicted and her dot agree. And a candidate must be where the plate's own pixels
# match best around it (`align`, `aligned`): another wearer's plate whose name shares letters with hers ("Baby" of
# RiceBaby's under BugBaby's model, 6 px off its frame: 43 of the 483 frames, clean 52-80, 0.83-0.96 of the most there;
# her own plates 1.0 on every frame of BugBaby's held-out and RiceBaby's labelled ones) is placed where its letters
# line up, not where its frame does (v2/plates_check.py).
DEFAULTS = dict(min_frame=150,  # the letters' rule: frame px at least this and letters - w_bad bad at least l_min
                w_bad=0.5, l_min=15,
                near_only=280,  # the near rule: frame px at least this, within gate (1 + frames unplaced) of the
                gate=8.0,       # prediction, letters - w_bad bad at least l_near
                l_near=-10,
                l_far=25,       # `verify`'s (no prediction): letters - w_bad_far bad at least this
                w_bad_far=1.0,
                clean_min=16,   # every rule: clean letters at least this (another wearer's plate showed 8 at most)
                align=0.97)     # every rule: its stable px matched at least this share of the most within ALIGN_REACH
ALIGN_REACH = (12, 6)          # px across, down around a candidate where the plate's best match is looked for
MINIMAP = (0.3, 0.17)          # the minimap's share of the frame's height and width, top left: never her plate
GRAY_SURE = 0.85               # the gray template's match this good is her tag
GRAY_FLOOR = 0.3               # `gray_peaks`' least
GRAY_SAME = (20, 60)           # ... rows, columns around a peak where others are the same tag
GRAY_BANDS = 4                 # the play area's gray match in this many row bands on threads (NameTag's BANDS)
NONE = 255                     # a pixel of none of the plate's colours (locate.c's NONE)
USE_C = True                   # False: numpy's (the tests compare the two)


def pack(rgb: np.ndarray) -> np.ndarray:
    """RGB pixels as 24-bit codes (R << 16 | G << 8 | B), int32."""
    p = rgb.astype(np.int32)
    return (p[..., 0] << 16) | (p[..., 1] << 8) | p[..., 2]


def model_path(lab: Path, geo: Geometry = NATIVE) -> Path:
    """Where her plate's model is kept: her lab folder (paths.of().lab), named for the client size."""
    return Path(lab) / f"plate{geo.suffix}.npz"


@dataclass
class PlateModel:
    """Her plate as calibration saw it: the median picture (RGB) of the BOX around her tag's centre in `n` sure
    frames, the share of them that had each pixel's colour (`share`), its name's interior and her feet from its
    centre; `sure`: her tag's gray template's levels on her frames where calibration measured them (`sure_levels`:
    typical, picks, bank), None in a model made before (RiceBaby's of 10-30)."""
    picture: np.ndarray
    share: np.ndarray
    box: tuple[int, int, int, int] = BOX
    interior: tuple[int, int, int, int] = INTERIOR
    feet: tuple[float, float] = FEET
    n: int = 0
    sure: tuple[float, float, float] | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.sure is None else {"sure": np.array(self.sure, np.float64)}
        np.savez_compressed(path, picture=self.picture, share=self.share.astype(np.float32),
                            box=np.array(self.box), interior=np.array(self.interior), feet=np.array(self.feet),
                            n=np.array(self.n), **extra)

    @classmethod
    def load(cls, path: Path) -> PlateModel:
        z = np.load(Path(path))
        return cls(z["picture"], z["share"].astype(np.float64), tuple(int(v) for v in z["box"]),
                   tuple(int(v) for v in z["interior"]), tuple(float(v) for v in z["feet"]), int(z["n"]),
                   tuple(float(v) for v in z["sure"]) if "sure" in z.files else None)

    def recentred(self, dx: int, dy: int) -> PlateModel:
        """The same picture about another centre, (dx, dy) from this one's (a tag template cut elsewhere on it:
        `tag_template`): its box, interior and her feet's offset moved by -(dx, dy)."""
        x0, x1, y0, y1 = self.box
        ix0, ix1, iy0, iy1 = self.interior
        return PlateModel(self.picture, self.share, (x0 - dx, x1 - dx, y0 - dy, y1 - dy),
                          (ix0 - dx, ix1 - dx, iy0 - dy, iy1 - dy), (self.feet[0] - dx, self.feet[1] - dy), self.n,
                          self.sure)

    def letters(self) -> np.ndarray:
        """Her name's letters in the picture: its stable pure white inside the interior (a bool mask)."""
        x0, x1, y0, y1 = self.box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ix0, ix1, iy0, iy1 = self.interior
        inside = (xx >= ix0) & (xx <= ix1) & (yy >= iy0) & (yy <= iy1)
        return (self.share >= STABLE_SHARE) & (self.picture == 255).all(axis=2) & inside

    def axis(self) -> float | None:
        """Her mirror axis across from the centre, by the plate itself: the middle of its stable pixels' columns (those
        with 2 or more, so a stray one doesn't count). The game draws her name plate centred on her: on both plates of
        10-29/30 this is her feet's x as measured otherwise (RiceBaby's columns -46..49, +1.5; BugBaby's -50..43, -3.5:
        her poses' mirror axis, the plate's symmetry and the sloped platforms' lines all agree, D:/MapleHelper-scratch/
        loc2build/bugbaby/scripts/mirror_axis.py, feet_check.py). None where no column has 2."""
        cols = np.nonzero((self.share >= STABLE_SHARE).sum(axis=0) >= 2)[0]
        if not len(cols):
            return None
        return float((cols.min() + cols.max()) / 2 + self.box[0])

    @classmethod
    def build(cls, crops: Iterable[tuple[np.ndarray, tuple[int, int]]], box=BOX, interior=INTERIOR,
              feet=FEET) -> PlateModel:
        """From (frame (RGB), her tag's centre) pairs where her tag is sure (`sure_centres`): the median of the BOX
        around each centre, and each pixel's share of frames within SAME_COLOUR of it."""
        x0, x1, y0, y1 = box
        stack = []
        for frame, (cx, cy) in crops:
            if cy + y0 < 0 or cx + x0 < 0 or cx + x1 > frame.shape[1] or cy + y1 > frame.shape[0]:
                continue
            stack.append(frame[cy + y0:cy + y1, cx + x0:cx + x1])
        if not stack:
            raise ValueError("no frame had her whole plate in it")
        s = np.stack(stack).astype(np.int16)
        med = np.median(s, axis=0).astype(np.uint8)
        dev = np.abs(s - med[None]).sum(axis=3)
        return cls(med, (dev <= SAME_COLOUR).mean(axis=0), tuple(box), tuple(interior), tuple(feet), len(stack))


TEMPLATE_MARGIN = 2   # her tag's template: her letters and this many px around them (track.make_template's 2)


def tag_template(model: PlateModel, margin: int = TEMPLATE_MARGIN) -> tuple[np.ndarray, tuple[int, int]]:
    """Her tag's gray template cut from her plate's median picture as calibration cuts one from a frame (her name's
    white letters, `margin` px around them: track.make_template), but of her stable letters only (no sparkle's white)
    and from the median of her sure frames (the interior's translucent dark as it is on average). -> (the template
    (uint8 gray, the size NameTag and the locator use both its files at: at 1366x768 the half-size one is full size),
    where its centre (its top-left + (w // 2, h // 2), as NameTag gives it) is from the model's centre). BugBaby's of
    10-30 cut this way: 15 x 52, its centre (-1, 0) from her old text tag's; 0.993 median on her train frames' plates
    (p25 0.990, the second peak 0.51 at most) where the old template had 0.879 (0.86-0.895, never the 0.9 her_cues
    wanted: D:/MapleHelper-scratch/loc2build/bugbaby/v2/cut_probe.py)."""
    ly, lx = np.nonzero(model.letters())
    if not len(ly):
        raise ValueError("her plate's model has no letters to cut her tag from")
    h, w = model.picture.shape[:2]
    r0, r1 = max(0, int(ly.min()) - margin), min(h, int(ly.max()) + margin + 1)
    c0, c1 = max(0, int(lx.min()) - margin), min(w, int(lx.max()) + margin + 1)
    gray = cv2.cvtColor(np.ascontiguousarray(model.picture), cv2.COLOR_RGB2GRAY)
    t = np.ascontiguousarray(gray[r0:r1, c0:c1])
    th, tw = t.shape
    return t, (c0 + model.box[0] + tw // 2, r0 + model.box[2] + th // 2)


# Her tag's sure levels from its template's own scores (`sure_levels`): the frames of a calibration recording show her
# tag clear in most of them, so the best match's scores have a tight cluster at the template's typical score on her tag
# (its 75th percentile) and a tail where it's covered. `picks` (the plate's median picture: sure_centres) and `bank`
# (her pose bank, her_cues.sure_tag) go that cluster's spread (its scores within CLUSTER of the typical, their p25-p75)
# times PICKS_SPREAD and BANK_SPREAD below it, at most the levels the prototype's template had (0.95, 0.9): RiceBaby's
# template on her part1 frames typical 0.981, spread 0.006 (0.95, 0.9, as the constants were); BugBaby's old text tag's
# on her train frames 0.882, 0.006 (0.857, 0.832); the one cut from her plate 0.996, 0.005 (0.95, 0.9)
# (D:/MapleHelper-scratch/loc2build/bugbaby/v2: survey_rice.py, cut_probe.py, models/her_cues.log).
CLUSTER, SPREAD_LEAST = 0.1, 0.004
PICKS_SPREAD, BANK_SPREAD = 4.0, 8.0
PICKS_MOST, BANK_MOST = 0.95, 0.9


def sure_levels(best: Iterable[float]) -> tuple[float, float, float]:
    """(typical, picks, bank) from her frames' best gray scores (the top)."""
    b = np.asarray([float(v) for v in best], np.float64)
    if not len(b):
        raise ValueError("no scores")
    typical = float(np.percentile(b, 75))
    near = b[b >= typical - CLUSTER]
    spread = max(SPREAD_LEAST, float(np.percentile(near, 75) - np.percentile(near, 25)))
    return (round(typical, 4), round(min(PICKS_MOST, typical - PICKS_SPREAD * spread), 4),
            round(min(BANK_MOST, typical - BANK_SPREAD * spread), 4))


def sure_centres(frames: Iterable[tuple[object, np.ndarray]], template: np.ndarray, hud: int, best: float = 0.95,
                 second: float = 0.6, box=BOX, cell: int = 6, most: int = 400, seed: int = 1):
    """Her tag's centre in each frame (key, RGB) where the full gray template is sure: its best match at least `best`,
    no other peak `second` or more, the plate's box above the HUD's top `hud`; then one frame per `cell` x `cell` px of
    centres (seeded), at most `most`: [(key, (cx, cy))], for PlateModel.build (the prototype's build_plate.py)."""
    return pick_centres(((key, gray_peaks(frame, template, k=2, rows=hud)) for key, frame in frames), hud, best,
                        second, box, cell, most, seed)


def pick_centres(peaks: Iterable[tuple[object, list]], hud: int, best: float = 0.95, second: float = 0.6, box=BOX,
                 cell: int = 6, most: int = 400, seed: int = 1):
    """`sure_centres` from each frame's gray peaks already found ((key, gray_peaks' list)): the same picks."""
    cells: dict = {}
    for key, peaks_ in peaks:
        if not peaks_ or peaks_[0][0] < best or (len(peaks_) > 1 and peaks_[1][0] >= second):
            continue
        _, cx, cy = peaks_[0]
        if cy + box[3] >= hud:
            continue
        cells.setdefault((cx // cell, cy // cell), []).append((key, (cx, cy)))
    rng = np.random.default_rng(seed)
    pick = [v[int(rng.integers(len(v)))] for _, v in sorted(cells.items())]
    rng.shuffle(pick)
    return pick[:most]


class Candidate(NamedTuple):
    """A place her plate may be: its evidence score (frame + W_LETTERS letters - 2 bad), centre, the plate's frame px
    matched, her letters white, white where her name has none, her letters in clean bins."""
    score: int
    x: int
    y: int
    frame: int
    letters: int
    bad: int
    clean: int


class Hit(NamedTuple):
    """Her plate placed (`decide`): its decision score (evidence less the distance's), centre, feet (the model's
    offset), the candidate and its distance from the prediction (0 without one)."""
    score: float
    x: int
    y: int
    feet: tuple[float, float]
    candidate: Candidate
    distance: float


class _LcPlate(ctypes.Structure):
    """locate.c's LcPlate."""
    _fields_ = [("mul", ctypes.c_uint32), ("shift", ctypes.c_int), ("h_idx", ctypes.c_void_p),
                ("h_code", ctypes.c_void_p), ("th", ctypes.c_int), ("tw", ctypes.c_int), ("n", ctypes.c_int),
                ("rs", ctypes.c_void_p), ("ty", ctypes.c_void_p), ("tx", ctypes.c_void_p), ("ti", ctypes.c_void_p),
                ("ox", ctypes.c_int), ("oy", ctypes.c_int), ("bx", ctypes.c_int), ("by", ctypes.c_int),
                ("bh", ctypes.c_int), ("bw", ctypes.c_int), ("nf", ctypes.c_int), ("fy", ctypes.c_void_p),
                ("fx", ctypes.c_void_p), ("fi", ctypes.c_void_p), ("nl", ctypes.c_int), ("ly", ctypes.c_void_p),
                ("lx", ctypes.c_void_p), ("lb", ctypes.c_void_p), ("ng", ctypes.c_int), ("gy", ctypes.c_void_p),
                ("gx", ctypes.c_void_p), ("gb", ctypes.c_void_p), ("white", ctypes.c_int), ("nbins", ctypes.c_int),
                ("ltot", ctypes.c_void_p), ("min_let", ctypes.c_int), ("recall", ctypes.c_double),
                ("bad_max", ctypes.c_int)]


def _palette_hash(pal: np.ndarray) -> tuple[int, int]:
    """(mul, shift) of a multiplicative hash ((code * mul) mod 2^32 >> shift) telling the palette's codes apart: the
    fewest bits, then the first multiplier of a fixed sequence that does."""
    bits = max(4, int(np.ceil(np.log2(max(2, len(pal))))) + 1)
    while True:
        for k in range(4096):
            mul = (2654435761 + 2 * 40503 * k) & 0xFFFFFFFF | 1
            h = ((pal.astype(np.uint64) * mul) & 0xFFFFFFFF) >> (32 - bits)
            if len(np.unique(h)) == len(pal):
                return mul, 32 - bits
        bits += 1


class Plate:
    """Her plate's matcher, from its model (`PlateModel`)."""

    def __init__(self, model: PlateModel, params: dict | None = None):
        self.model = model
        self.p = dict(DEFAULTS, **(params or {}))
        med, share = model.picture, model.share
        x0, x1, y0, y1 = model.box
        self.box = model.box
        self.ph, self.pw = med.shape[:2]
        if (self.ph, self.pw) != (y1 - y0, x1 - x0):
            raise ValueError("the plate's picture isn't its box's size")
        self.feet = model.feet
        stable = share >= STABLE_SHARE
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ix0, ix1, iy0, iy1 = model.interior
        interior = (xx >= ix0) & (xx <= ix1) & (yy >= iy0) & (yy <= iy1)
        white = (med == 255).all(axis=2)
        letters = stable & white & interior
        frame_px = stable & ~letters & ~(med == 0).all(axis=2)
        background = interior & ~stable
        codes = pack(med)
        pal, inv = np.unique(codes[stable], return_inverse=True)
        if len(pal) >= NONE:
            raise ValueError("the plate has too many colours")
        self.pal = pal
        self.mul, self.shift = _palette_hash(pal)
        size = 1 << (32 - self.shift)
        h = ((pal.astype(np.uint64) * self.mul) & 0xFFFFFFFF) >> self.shift
        self.h_idx = np.full(size, NONE, np.uint8)
        self.h_code = np.full(size, -1, np.int32)
        self.h_idx[h] = np.arange(len(pal), dtype=np.uint8)
        self.h_code[h] = pal
        idx = np.full(stable.shape, NONE, np.uint8)
        idx[stable] = inv
        ys, xs = np.nonzero(stable)
        self.cy0, self.cy1, self.cx0, self.cx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        self.tidx = np.ascontiguousarray(idx[self.cy0:self.cy1, self.cx0:self.cx1])
        self.th, self.tw = self.tidx.shape
        self.ox, self.oy = x0 + self.cx0, y0 + self.cy0     # the count template's top-left from the centre
        self.n = int(stable.sum())
        ty, tx = np.nonzero(self.tidx != NONE)              # row by row (np.nonzero's order)
        self._ty, self._tx = ty.astype(np.int16), tx.astype(np.int16)
        self._ti = self.tidx[ty, tx].astype(np.uint8)
        self._rs = np.searchsorted(ty, np.arange(self.th + 1)).astype(np.int32)
        self._tcolours = sorted(set(self._ti.tolist()))
        self._tind = {c: (self.tidx == c).astype(np.float32) for c in self._tcolours}
        self.white = int(np.searchsorted(pal, 0xFFFFFF)) if 0xFFFFFF in set(pal.tolist()) else NONE
        fy, fx = np.nonzero(frame_px)
        self._fy, self._fx = fy.astype(np.int16), fx.astype(np.int16)
        self._fi = idx[fy, fx].astype(np.uint8)
        ly, lx = np.nonzero(letters)
        gy, gx = np.nonzero(background)
        self._ly, self._lx, self._gy, self._gx = (a.astype(np.int16) for a in (ly, lx, gy, gx))
        edges = list(range(ix0, ix1 + 1, BIN_WIDTH))
        bins = np.full(xx.shape, -1, np.int8)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            bins[(xx >= a) & (xx < b)] = k
        self.nbins = len(edges) - 1
        self._lb, self._gb = bins[ly, lx], bins[gy, gx]
        self._ltot = np.bincount(self._lb[self._lb >= 0], minlength=self.nbins).astype(np.int32)
        self.sizes = dict(stable=self.n, frame=int(frame_px.sum()), letters=int(letters.sum()),
                           background=int(background.sum()), colours=len(pal))
        self._c = None

    # --- C ---------------------------------------------------------------------------------------------------------
    def _c_model(self):
        if self._c is None:
            p = lambda a: a.ctypes.data  # (the arrays are this object's: they live as long as it)
            self._c = _LcPlate(self.mul, self.shift, p(self.h_idx), p(self.h_code), self.th, self.tw, self.n,
                               p(self._rs), p(self._ty), p(self._tx), p(self._ti), self.ox, self.oy, self.box[0],
                               self.box[2], self.ph, self.pw, len(self._fy), p(self._fy), p(self._fx), p(self._fi),
                               len(self._ly), p(self._ly), p(self._lx), p(self._lb), len(self._gy), p(self._gy),
                               p(self._gx), p(self._gb), self.white, self.nbins, p(self._ltot), MIN_LETTERS,
                               RECALL, BAD_MAX)
        return self._c

    @staticmethod
    def _in_c(frame: np.ndarray):
        if not USE_C or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 \
                or frame.strides[2] != 1 or frame.strides[1] != 3:
            return None
        return native.locate_lib()

    # --- the candidates -------------------------------------------------------------------------------------------
    def window(self, frame_shape, centre: tuple[float, float], reach: tuple[int, int]) -> tuple[int, int, int, int]:
        """The frame's box searched for a plate centred within `reach` of `centre` (rounded as Python rounds)."""
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        x0, x1, y0, y1 = self.box
        h, w = frame_shape[:2]
        return (max(0, cx + x0 - reach[0]), max(0, cy + y0 - reach[1]), min(w, cx + x1 + reach[0]),
                min(h, cy + y1 + reach[1]))

    def candidates(self, frame: np.ndarray, centre: tuple[float, float], reach: tuple[int, int],
                   hud: int | None = None) -> list[Candidate]:
        """Her plate's candidates centred within `reach` (px across, down) of `centre` in an RGB frame (rows from
        `hud` down, by default the HUD's top, never her plate's)."""
        h, w = frame.shape[:2]
        if hud is None:
            hud = _hud(frame)
        x0, y0, x1, y1 = self.window(frame.shape, centre, reach)
        if x1 - x0 < self.tw or y1 - y0 < self.th:
            return []
        mm = (int(h * MINIMAP[0]), int(w * MINIMAP[1]))
        lib = self._in_c(frame)
        if lib is not None:
            out = np.zeros(TOPK * 7, np.int32)
            n = lib.lc_plate(ctypes.byref(self._c_model()), frame.ctypes.data, frame.strides[0], x0, y0, x1, y1,
                             hud, mm[0], mm[1], TOPK, SUPPRESS[0], SUPPRESS[1], FLOOR, int(W_LETTERS), 1,
                             out.ctypes.data)
            if n >= 0:
                return [Candidate(*(int(v) for v in out[7 * k:7 * k + 7])) for k in range(n)]
        return self._candidates_py(frame, x0, y0, x1, y1, hud, mm)

    def index(self, rgb: np.ndarray) -> np.ndarray:
        """Each pixel's palette index (NONE: none of the plate's colours)."""
        codes = pack(rgb)
        h = (((codes.astype(np.uint64) * self.mul) & 0xFFFFFFFF) >> self.shift).astype(np.int64)
        idx = self.h_idx[h]
        idx[self.h_code[h] != codes] = NONE
        return idx

    def _window_index(self, frame, x0, y0, x1, y1, hud):
        idx = self.index(frame[y0:y1, x0:x1])
        if y1 > hud:
            idx[max(0, hud - y0):] = NONE
        return idx

    def counts(self, frame: np.ndarray, box: tuple[int, int, int, int], hud: int | None = None) -> np.ndarray | None:
        """The count map of a box (x0, y0, x1, y1) of the frame: at [ry, rx], the stable px matched with the count
        template's top-left at (x0 + rx, y0 + ry). In C where it can be (the tests compare the two)."""
        x0, y0, x1, y1 = box
        hud = _hud(frame) if hud is None else hud
        if x1 - x0 < self.tw or y1 - y0 < self.th:
            return None
        lib = self._in_c(frame)
        if lib is not None:
            res = np.zeros((y1 - y0 - self.th + 1, x1 - x0 - self.tw + 1), np.int32)
            if lib.lc_plate_counts(ctypes.byref(self._c_model()), frame.ctypes.data, frame.strides[0], x0, y0, x1, y1,
                                   hud, res.ctypes.data) == 1:
                return res
        return self._counts_py(self._window_index(frame, x0, y0, x1, y1, hud))

    def _counts_py(self, idx: np.ndarray) -> np.ndarray:
        """The counts in numpy: a colour at a time (those the window has), each its 0/1 picture's correlation
        (OpenCV's, float32: exact to well under 0.5 here, so rounded it's the count)."""
        res = np.zeros((idx.shape[0] - self.th + 1, idx.shape[1] - self.tw + 1), np.float64)
        present = set(np.unique(idx).tolist())
        for c in self._tcolours:
            if c in present:
                res += cv2.matchTemplate((idx == c).astype(np.float32), self._tind[c], cv2.TM_CCORR)
        return np.rint(res).astype(np.int32)

    def _candidates_py(self, frame, x0, y0, x1, y1, hud, mm) -> list[Candidate]:
        idx = self._window_index(frame, x0, y0, x1, y1, hud)
        res = self._counts_py(idx)
        c0x, c0y = x0 - self.ox, y0 - self.oy
        if mm[1] - c0x > 0 and mm[0] - c0y > 0:
            res[:max(0, mm[0] - c0y), :max(0, mm[1] - c0x)] = 0
        pad = np.full((idx.shape[0] + 2 * self.ph, idx.shape[1] + 2 * self.pw), NONE, np.uint8)
        pad[self.ph:self.ph + idx.shape[0], self.pw:self.pw + idx.shape[1]] = idx
        out = []
        for k in range(TOPK):
            ry, rx = np.unravel_index(int(np.argmax(res)), res.shape)
            v = int(res[ry, rx])
            if v <= FLOOR:
                break
            res[max(0, ry - SUPPRESS[0]):ry + SUPPRESS[0] + 1, max(0, rx - SUPPRESS[1]):rx + SUPPRESS[1] + 1] = -10 ** 9
            px, py = c0x + int(rx), c0y + int(ry)
            around = (-1, 0, 1) if k == 0 else (0,)
            best = None
            for dy in around:
                for dx in around:
                    f, l, b = self._evidence(pad, x0, y0, px + dx, py + dy)
                    s = f + int(W_LETTERS) * l - 2 * b
                    if best is None or s > best[0]:
                        best = (s, px + dx, py + dy, f, l, b)
            out.append(Candidate(*best, self._clean(pad, x0, y0, best[1], best[2])))
        return out

    def _block(self, pad, x0, y0, cx, cy):
        oy = cy + self.box[2] - y0 + self.ph
        ox = cx + self.box[0] - x0 + self.pw
        if oy < 0 or ox < 0 or oy + self.ph > pad.shape[0] or ox + self.pw > pad.shape[1]:
            return None
        return pad[oy:oy + self.ph, ox:ox + self.pw]

    def _evidence(self, pad, x0, y0, cx, cy) -> tuple[int, int, int]:
        blk = self._block(pad, x0, y0, cx, cy)
        if blk is None:
            return 0, 0, 0
        return (int((blk[self._fy, self._fx] == self._fi).sum()), int((blk[self._ly, self._lx] == self.white).sum()),
                int((blk[self._gy, self._gx] == self.white).sum()))

    def _clean(self, pad, x0, y0, cx, cy) -> int:
        blk = self._block(pad, x0, y0, cx, cy)
        if blk is None:
            return 0
        lm = blk[self._ly, self._lx] == self.white
        bm = blk[self._gy, self._gx] == self.white
        lmb = np.bincount(self._lb[(self._lb >= 0) & lm], minlength=self.nbins)
        bad = np.bincount(self._gb[(self._gb >= 0) & bm], minlength=self.nbins)
        return int(sum(int(lmb[k]) for k in range(self.nbins) if self._ltot[k] >= MIN_LETTERS
                       and lmb[k] >= RECALL * self._ltot[k] and bad[k] <= BAD_MAX))

    def evidence(self, frame: np.ndarray, cx: int, cy: int, hud: int | None = None) -> dict:
        """The evidence at one centre: frame px matched, letters, bad, clean (and her letters, frame px in all)."""
        hud = _hud(frame) if hud is None else hud
        x0, y0, x1, y1 = self.window(frame.shape, (cx, cy), (1, 1))
        idx = self._window_index(frame, x0, y0, x1, y1, hud)
        pad = np.full((idx.shape[0] + 2 * self.ph, idx.shape[1] + 2 * self.pw), NONE, np.uint8)
        pad[self.ph:self.ph + idx.shape[0], self.pw:self.pw + idx.shape[1]] = idx
        f, l, b = self._evidence(pad, x0, y0, cx, cy)
        return dict(frame=f, letters=l, bad=b, clean=self._clean(pad, x0, y0, cx, cy), of_letters=len(self._ly),
                    of_frame=len(self._fy))

    # --- the decision ----------------------------------------------------------------------------------------------
    def decide(self, cands: Iterable[Candidate], pred: tuple[float, float] | None, miss: int = 0,
               far: bool = False, frame: np.ndarray | None = None) -> Hit | None:
        """The candidates' best by evidence less W_DIST a px from the prediction, of those good enough: clean letters
        CLEAN_MIN or more, and the letters' rule, or (with a prediction) the near rule (`miss`: frames she went
        unplaced, widening its gate), or `far`'s (no prediction); with the `frame` they came from, also where her
        plate matches best about them (`aligned`; the best that is, the next one where it isn't). None if none is."""
        p = self.p
        good = []
        for c in cands:
            d = float(np.hypot(c.x - pred[0], c.y - pred[1])) if pred is not None else 0.0
            letters = c.letters - p["w_bad"] * c.bad
            if p["clean_min"] and c.clean < p["clean_min"]:
                continue
            if far:
                ok = c.frame >= p["min_frame"] and c.letters - p["w_bad_far"] * c.bad >= p["l_far"]
            else:
                ok = (c.frame >= p["min_frame"] and letters >= p["l_min"]) or \
                     (pred is not None and c.frame >= p["near_only"] and d <= p["gate"] * (1 + miss)
                      and letters >= p["l_near"])
            if ok:
                good.append((c.score - W_DIST * d, c, d))
        good.sort(key=lambda g: -g[0])   # (stable: of equal scores the first, as before)
        for sc, c, d in good:
            if frame is None or not p["align"] or self.aligned(frame, c) >= p["align"]:
                return Hit(sc, c.x, c.y, (c.x + self.feet[0], c.y + self.feet[1]), c, d)
        return None

    def aligned(self, frame: np.ndarray, c: Candidate, hud: int | None = None) -> float:
        """How well a candidate sits where her plate's pixels match best around it: its count of stable px matched (its
        frame, its letters, its outline) over the most within ALIGN_REACH of it (1.0: there). Her own plate, covered or
        not, matches best where it is; another wearer's is placed where her letters line up with his, its frame a few
        px away matching more (RiceBaby's under BugBaby's model: about half)."""
        hud = _hud(frame) if hud is None else hud
        x0, y0, x1, y1 = self.window(frame.shape, (c.x, c.y), ALIGN_REACH)
        res = self.counts(frame, (x0, y0, x1, y1), hud)
        if res is None:
            return 1.0
        ry, rx = c.y + self.oy - y0, c.x + self.ox - x0
        if not (0 <= ry < res.shape[0] and 0 <= rx < res.shape[1]):
            return 1.0
        most = int(res.max())
        return 1.0 if most <= 0 else float(res[ry, rx]) / most

    def near(self, frame: np.ndarray, pred: tuple[float, float], miss: int = 0) -> Hit | None:
        """Her plate around its predicted centre (NEAR, GROW wider a frame she went unplaced up to GROW_MAX)."""
        m = min(max(0, miss), GROW_MAX)
        return self.decide(self.candidates(frame, pred, (NEAR[0] + GROW[0] * m, NEAR[1] + GROW[1] * m)), pred, m,
                           frame=frame)

    def verify(self, frame: np.ndarray, peaks: Iterable[tuple[float, int, int]]) -> Hit | None:
        """The gray template's peaks (score, cx, cy; `gray_peaks`) checked by her plate's evidence, the first three:
        no prediction, so her letters are needed (`decide`'s far rule)."""
        cands = []
        for _, x, y in list(peaks)[:3]:
            cands += self.candidates(frame, (x, y), PEAK_REACH)
        return self.decide(cands, None, far=True, frame=frame)


def _hud(frame: np.ndarray) -> int:
    """The HUD's top row in a look frame of the game drawn pixel for pixel (screen.HUD_TOP_GAME's 700 at 768 rows)."""
    return int(round(frame.shape[0] * NATIVE.hud_top))


# --- the gray template ----------------------------------------------------------------------------------------------

def _hide_minimap(res: np.ndarray, shape, x0: int, y0: int, tw: int, th: int) -> None:
    """Rule out the matches whose centre is in the minimap, of a box whose top-left is (x0, y0) (in place)."""
    ry, rx = int(shape[0] * MINIMAP[0]) - y0, int(shape[1] * MINIMAP[1]) - x0
    if ry > 0 and rx > 0:
        res[:ry, :rx] = -1


def gray_best(frame: np.ndarray, template: np.ndarray, box: tuple[float, float, float, float],
              rows: int | None = None) -> tuple[float, int, int] | None:
    """The gray template's best match in a box (x0, y0, x1, y1) of an RGB frame, its rows to `rows` (the HUD's top):
    (score, centre x, centre y), or None where the box can't hold it. (The prototype's match_box: the minimap's
    corner ruled out at the template's top-left.)"""
    th, tw = template.shape
    h, w = frame.shape[:2]
    rows = _hud(frame) if rows is None else rows
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, rows, int(box[3]))
    if x1 - x0 < tw or y1 - y0 < th:
        return None
    r = cv2.matchTemplate(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), template, cv2.TM_CCOEFF_NORMED)
    _hide_minimap(r, frame.shape, x0, y0, tw, th)
    _, s, _, (x, y) = cv2.minMaxLoc(r)
    return float(s), x0 + x + tw // 2, y0 + y + th // 2


def gray_near(frame: np.ndarray, template: np.ndarray, centre: tuple[float, float], reach: tuple[float, float],
              rows: int | None = None) -> tuple[float, int, int] | None:
    """`gray_best` for a tag centred within `reach` (across, down) of `centre`."""
    th, tw = template.shape
    return gray_best(frame, template, (centre[0] - tw // 2 - reach[0], centre[1] - th // 2 - reach[1],
                                       centre[0] + tw // 2 + reach[0] + 1, centre[1] + th // 2 + reach[1] + 1), rows)


_pool: list = []


def gray_peaks(frame: np.ndarray, template: np.ndarray, k: int = 4, rows: int | None = None,
               gray: np.ndarray | None = None) -> list[tuple[float, int, int]]:
    """The gray template's best `k` matches over the play area (rows to `rows`), each at least GRAY_FLOOR, GRAY_SAME
    apart: [(score, centre x, centre y)]. Matched in GRAY_BANDS row bands on threads (OpenCV lets go of Python's lock),
    the bands overlapping by the template's height, as NameTag's whole search."""
    th, tw = template.shape
    rows = _hud(frame) if rows is None else rows
    g = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if gray is None else gray)[:rows]
    n = g.shape[0] - th + 1
    if n <= 0 or g.shape[1] < tw:
        return []
    if GRAY_BANDS > 1 and n >= GRAY_BANDS * th:
        if not _pool:
            _pool.append(ThreadPoolExecutor(GRAY_BANDS, "plate-gray"))
        edges = [int(round(n * b / GRAY_BANDS)) for b in range(GRAY_BANDS + 1)]
        r = np.vstack(list(_pool[0].map(lambda b: cv2.matchTemplate(g[edges[b]:edges[b + 1] + th - 1], template,
                                                                    cv2.TM_CCOEFF_NORMED), range(GRAY_BANDS))))
    else:
        r = cv2.matchTemplate(g, template, cv2.TM_CCOEFF_NORMED)
    _hide_minimap(r, frame.shape, 0, 0, tw, th)
    out = []
    for _ in range(k):
        _, s, _, (x, y) = cv2.minMaxLoc(r)
        if s < GRAY_FLOOR:
            break
        out.append((float(s), x + tw // 2, y + th // 2))
        r[max(0, y - GRAY_SAME[0]):y + GRAY_SAME[0], max(0, x - GRAY_SAME[1]):x + GRAY_SAME[1]] = -1
    return out
