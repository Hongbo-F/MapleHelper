"""Monsters found by the exact pixels of their pictures (`ExactFinder`), beside the whole screen's masked search
(perception.mobs): what finds the monsters near her that the search can't.

The bug it's for (the user, 2026-11-04, watching the live grind: "mobs near the player are not detected by perception"):
nothing masks her region; her body, her pet, her name plate, damage numbers, skill effects and drops cover part of a
monster's picture, and the search's masked correlation falls with the share still showing. On the lossless clip of
2026-11-02 22:49 (1806 frames; D:/MapleHelper-scratch/unseen/perception, an exact-pixel oracle over all of them, its
verdicts the eye's on 60 frames sampled at random) the search found a monster on her level 6% of the frames within 25
units across of her, 27% at 25-50, 68% at 50-75, 84% at 75-150; by the share of its picture showing, 7% at 0.2-0.4, 14%
at 0.4-0.6, 31% at 0.6-0.8, 91% at 0.8-0.95, 99% above; 700 of its 791 misses were refused at one gate of its step 2
(mobs.FINE_GATE), the true place's score 0.3-0.65. Lowering the search's thresholds gains little before it finds scenery
(the floor about 0.46 on that map; at 0.40, 252 new detections under 10% exact a thousand frames, a leaf).

The game draws a picture's opaque pixels as they are: on a frame of the game's own size, not scaled nor compressed, what
shows of a monster is its picture's pixels exactly, however little of it shows. So each picture's anchors (pairs of
side-by-side pixels rare in the picture, the first of a colour sparse on the map's background: `anchors_of`) vote for the
places they fall on the frame, and a place with MIN_VOTES votes has its pixels counted, MIN_EXACT exact ones at least a
monster (exact.c). The frame's other pixels stop at a colour table, so a frame takes about 3 ms on one thread of the C.

What it can't see: a monster fading in (the spawn watch's), one under a shade, a frame that isn't the game's own pixels
(another size, JPEG): it finds nothing on those, and the search carries on as it always did. So it's beside the search,
not instead of it: its finds are added only where no detection lies within NEAR_PX, as detections of score 1: the exact
pixels of a live picture, however little shows, which the death watch (perception.deaths.wants: a track found under
REFOUND_UNDER is looked at for its death) has no business doubting (with the share as its score, every monster partly
covered had its corpse's neighbours looked at: R3's #35 marked dead at frame 1073, still being hit at 1085). A find of a die1
picture (the map's monsters' death frames: deaths.DeathWatch) is that monster's death seen (localization.worldstate:
tracking.seen_dying), about at once, not 1.4 s on.

Measured: the README's "Monsters near her: the exact finder".
"""

import ctypes
from dataclasses import dataclass

import numpy as np

from .. import native
from . import gates
from .mobs import Mob

MIN_EXACT = 200   # exact pixels at least a find (of pictures of about 1,800-3,000 on Tree Dungeon's mushrooms): under it the
                  # kind can be wrong where only the shared body shows (at 200 or more, 1 of 192 flipped under 30% showing)
MIN_VOTES = 6     # anchors at least that vote for a place before its pixels are counted
NEAR_PX = 8.0     # px: a find this near a detection of the search is that detection, nothing added
MOST_ANCHORS = 300  # anchors a picture at most, spread over it (a visible part of any shape has some)
RARE = 3          # an anchor's pair of colours at most this many times in its picture
SPARSE = 100.0    # px a frame of the background at most in the colour bin of an anchor's first pixel: the frame's pixels
                  # stop at the first test (without the map's background, every anchor: slower, the same finds)
MAX_FINDS = 256


@dataclass(frozen=True)
class Picture:
    """One picture the finder looks for: its monster's kind, its name, which way it faces, its frame and pose in the game
    data, whether it's a die1 frame, its colours (RGB) and opaque pixels (bool), where its feet are from its top-left."""
    kind: str
    name: str
    facing: int
    frame: str
    pose: str
    die: bool
    rgb: np.ndarray
    mask: np.ndarray
    ox: float
    oy: float

    @property
    def w(self) -> int:
        return self.mask.shape[1]

    @property
    def h(self) -> int:
        return self.mask.shape[0]


@dataclass(frozen=True)
class Find:
    """A picture found on a frame: `x0`, `y0` its top-left (px), `exact` its exact pixels, `share` of the picture's."""
    picture: Picture
    x0: int
    y0: int
    exact: int
    share: float

    @property
    def foot_x(self) -> float:
        return self.x0 + self.picture.ox

    @property
    def feet_y(self) -> float:
        return self.y0 + self.picture.oy

    def mob(self) -> Mob:
        """As a detection of the search, of score 1 (see the top: `share` says how much of it showed)."""
        p = self.picture
        return Mob(p.kind, self.x0 + p.w // 2, self.y0 + p.h // 2, p.w, p.h, 1.0, self.foot_x, self.feet_y,
                   p.facing, p.pose, p.frame, p.name)


def density(background: np.ndarray | None) -> np.ndarray | None:
    """Pixels a frame (1366 x 700) in each colour bin (gates.bins), from a map's background's counts
    (gates.load_background); None without one."""
    if background is None or background.sum() <= 0:
        return None
    return np.asarray(background, np.float64) / float(background.sum()) * (1366 * 700)


def anchors_of(p: Picture, dens: np.ndarray | None, sparse: float = SPARSE,
               most: int = MOST_ANCHORS) -> list[tuple[int, int]]:
    """A picture's anchors: (x, y) with the pixel and its right neighbour opaque and not the same colour, the pair at
    most RARE times in the picture, the first pixel's colour bin SPARSE or less on the map's background (`dens`, if
    known), thinned to about `most` spread over it. Where the background leaves too few (under MIN_VOTES x 4), those of
    the picture without it."""
    code = (p.rgb[..., 0].astype(np.int64) << 16) | (p.rgb[..., 1].astype(np.int64) << 8) | p.rgb[..., 2]
    ys, xs = np.nonzero(p.mask[:, :-1] & p.mask[:, 1:] & (code[:, :-1] != code[:, 1:]))
    if not len(ys):
        return []
    if dens is not None:
        keep = dens[gates.bins(p.rgb[ys, xs])] <= sparse
        if keep.sum() >= MIN_VOTES * 4:
            ys, xs = ys[keep], xs[keep]
    pair = code[ys, xs] * (1 << 24) + code[ys, xs + 1]
    _, inv, count = np.unique(pair, return_inverse=True, return_counts=True)
    keep = count[inv] <= RARE
    ys, xs = ys[keep], xs[keep]
    if len(ys) > most:
        cell = max(1, int(np.sqrt(len(ys) / most)))
        _, one = np.unique((ys // cell) * 4096 + (xs // cell), return_index=True)
        ys, xs = ys[one], xs[one]
    return list(zip(xs.tolist(), ys.tolist()))


def pictures_of(templates: dict, deaths: dict | None = None) -> list[Picture]:
    """The pictures of a detector's templates (mobs.MobTemplate by kind; those whose feet the game data knows: `origin`)
    and of a death watch's die1 frames (deaths.DiePicture by kind, each both ways: even indices facing left)."""
    out = []
    for kind, ts in templates.items():
        for t in ts:
            if t.origin is None:  # (a cut-out from a screenshot: not the game's own pixels)
                continue
            out.append(Picture(kind, t.name, t.facing, t.frame, t.pose, False, np.ascontiguousarray(t.rgb),
                               np.ascontiguousarray(t.mask3[..., 0] > 0), float(t.origin[0]), float(t.origin[1])))
    for kind, pics in (deaths or {}).items():
        for i, d in enumerate(pics):
            out.append(Picture(kind, f"{kind}_die{d.index}", -1 if i % 2 == 0 else 1, f"die1/{d.index}", "die1", True,
                               np.ascontiguousarray(d.rgb), np.ascontiguousarray(d.mask3[..., 0] > 0), float(d.ox),
                               float(d.oy)))
    return out


class ExactFinder:
    """The pictures (`pictures_of`) made ready for a frame's search (`find`); `dens`: the map's background (`density`)."""

    def __init__(self, pictures: list[Picture], dens: np.ndarray | None = None):
        self.lib = native.exact_lib()
        if self.lib is None:
            raise RuntimeError(native.exact_why or "no exact finder")
        self.pictures = pictures
        self.sizes = [int(p.mask.sum()) for p in pictures]
        n = len(pictures)
        rgbs, masks = (ctypes.c_void_p * n)(), (ctypes.c_void_p * n)()
        dims = np.zeros(2 * n, np.int32)
        keep, anchors = [], []
        for i, p in enumerate(pictures):
            rgb, mask = np.ascontiguousarray(p.rgb).astype(np.uint8), np.ascontiguousarray(p.mask).astype(np.uint8)
            keep += [rgb, mask]
            rgbs[i], masks[i] = rgb.ctypes.data, mask.ctypes.data
            dims[2 * i], dims[2 * i + 1] = p.w, p.h
            anchors += [(i, x, y) for x, y in anchors_of(p, dens)]
        self.anchors = len(anchors)
        a = np.array(anchors, np.int32).reshape(-1, 3)
        self.ctx = self.lib.ex_open(n, rgbs, masks, dims.ctypes.data, len(anchors), a.ctypes.data)
        if self.ctx is None:
            raise RuntimeError(f"too many pictures for the exact finder: {n}")
        self._out = np.zeros(5 * MAX_FINDS, np.int32)

    def __del__(self):
        ctx, self.ctx = getattr(self, "ctx", None), None
        if ctx is not None:
            self.lib.ex_close(ctx)

    def find(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Find]:
        """The pictures on `frame` (RGB, the game's own size) between `rows` (their anchors' pixels: the play area), the
        most exact first, each frame pixel a find's only. One thread at a time on a finder."""
        h, w = frame.shape[:2]
        y0, y1 = rows if rows is not None else (0, h)
        got = self.lib.ex_find(self.ctx, frame.ctypes.data, h, w, frame.strides[0], 0, y0, w, y1, MIN_VOTES, MIN_EXACT,
                               self._out.ctypes.data, MAX_FINDS)
        out = []
        for i in range(got):
            pi, ox, oy, exact, _ = self._out[5 * i:5 * i + 5].tolist()
            out.append(Find(self.pictures[pi], ox, oy, exact, round(exact / max(1, self.sizes[pi]), 3)))
        return out


def fresh(finds: list[Find], mobs: list[Mob]) -> list[Mob]:
    """The finds of monsters alive that no detection of `mobs` (the search's and the spawn watch's) lies within NEAR_PX of
    (feet to feet), as detections (`Find.mob`), one for each monster: a find near another kept already is that one (two
    pictures on one body each with pixels of their own, a rare overlap)."""
    out: list[Mob] = []
    for f in finds:
        if f.picture.die:
            continue
        if any(abs(f.foot_x - m.foot_x) <= NEAR_PX and abs(f.feet_y - m.feet_y) <= NEAR_PX for m in [*mobs, *out]):
            continue
        out.append(f.mob())
    return out


def confirmed(finds: list[Find], mobs: list[Mob], before: bool = False) -> bool:
    """Whether a frame is of the game's own pixels, as far as the finder can tell: it found something on it; with nothing
    found and no detection of the search's either, as the frame before (`before`: an empty screen says nothing); with
    detections and no find, False (a shade, a fade, a frame not exact: the finder is blind on it)."""
    if finds:
        return True
    return before if not mobs else False
