"""Monsters fading in at their spawn points (SpawnWatch): every spawn point in view looked at on every frame, whether
or not a respawn is due there (the user, 2026-11-01: "Should probably always check instead of only checking when it's
due"), for a newborn drawn as a blend of its picture and the map's art behind it.

A monster the map spawns isn't drawn at once: it fades in from nothing to solid, linearly, in 0.817 s (alpha 1.224 a
second, p10-p90 1.201-1.256 over 223 births on Tree Dungeon's lossless clips: the golden clip of the user's own play and
14 of the Lie Detector's alarm clips, 20.9 minutes; HeavenClient's port has 0.32 s, this client is slower), its feet on
the spawn point (within 2 units across, 0-1 down), in its move/0 picture (194 of 206 births) or stand/0 (12), walking
from the start (42 units by the time it's solid, the median). Every pixel of it is the game's alpha blend F = B + a A
(S - B) of the map's art B behind it and its picture S (A the picture's own alpha), one opacity a over the whole picture
and every channel (the fit's residual 0.27-0.3 colour levels on those frames). The whole screen's search
(perception.mobs) looks for a monster's own colours, so its step 1 (the colour gate and silhouettes) finds a newborn
only at alpha 0.6-0.8: a median 7 frames, 468 ms, after it first shows, at alpha 0.63 (Horny Mushrooms 552 ms, Green
ones 400); 878 newborns an hour came into view in the perception service's records of 2026-10-28..30.

So at each point in view (its window in the play area, clear of the chat box, the minimap and the quest helper; past
the screen's top, left or right edge in part, the rest left out of the fit, `_shown`: until the review of 2026-11-01 a
point not wholly on the screen wasn't looked at, 13.8% of the point-frames on the clips, two thirds of them at the
screen's top; ONSET_COVERED) the frame is compared with the map's art there (the background is the map's own art drawn
from the game data, localization.mapart, which the camera registration holds anyway: on Tree Dungeon every pixel of it
is the fixed art, and a frame where nothing is drawn over a point is the art to the level), as the prototype of
2026-11-01 tuned it (D:/MapleHelper-scratch/spawn-fade/proto):
1. the empty test: fewer than EMPTY_COUNT pixels more than EMPTY_LEVELS off the art on any channel: nothing there;
2. the onset fit: its kind's ONSET pictures, both ways, the feet within BAND units of the point along its platform and
   ROWS up or down, every STRIDE-th pixel of each picture (the game's alpha as each one's weight): with D = F - B and
   G = A (S - B) over a picture's pixels and channels, a = D.G / G.G, the share explained R2 = (D.G)^2 / (G.G D.D) and
   the residual sqrt((D.D - (D.G)^2 / G.G) / n); the hypothesis explaining the most of D.D with A_LO <= a <= A_HI and
   MIN_USED of its pixels kept (below) is the point's, a newborn where A_MIN <= a <= A_START, 1 - R2 <= U_MAX and the
   residual <= RES_MAX;
3. each newborn found is followed from the next frame (`_follow`): every FOLLOW picture of its kind, its feet up to
   FOLLOW_SPEED the way each faces for the time since it was last found (at most FOLLOW_MOST_S) and BACK the other,
   ROWS up or down, the same rule and its alpha DROP_A under the most it's been at the least; until the whole screen
   finds a monster of its kind within HANDOVER_DX across and HANDOVER_DY down of it (`handed`: the frame's sighting is
   the whole screen's then, and it's the tracks' alone; one of another kind there has the frame's place, the newborn
   followed on), or its best fit is solid (a > A_HI at the rule's R2), or it's unfound LOST_AFTER frames.
Her box (HER_BOX about her feet) and the boxes of the monsters the frame before found (MOB_BOX about their feet: not
this watch's own newborns, nor, at a point's onset, one ON_POINT of the point) are left out of the fit, and a hypothesis
with less than MIN_USED of its pixels left isn't tried. A point within NEAR_POINT of a newborn followed on its platform
isn't looked at for another.

What it found (this module replayed as the prototype's phase 2 was, on every frame with a sure view of the 17 clips,
18,190 frames, 323 births traced: README, Spawns seen as they fade in; as fixed after the review,
D:/MapleHelper-scratch/spawn-fade/fix): no false positive, 1,796 sightings (350 onsets, none over A_START); of the 233
newborns whose start shows, 229 found at their first visible frame (the median; p90 the next), at alpha 0.065, where the
whole screen found them 7 frames (467 ms) later; of the 90 whose start was hidden (under her, players, pets, effects)
20; and 8 newborns (47 sightings) at points past the screen's top, a newborn's fade each by its alpha's line and by eye.
Against every one of the 153,474 point-frames labelled (empty, monsters, players, her, pets, effects, dying monsters,
the Lie Detector), the nearest negative to the rule was a solid, partly covered Horny Mushroom at a 0.94 (1 - R2 0.040,
residual 17.7: 11.8 times past the thresholds), over A_START now; over all the clips' 172,866 point-frames looked at,
every fit an onset could start from (a <= A_START) within 14 times of the rule is a newborn under an effect or a player
by eye (fix/margin.py); under a shade over the whole screen the nearest is a solid Green Mushroom under 20% black, 5.2
times off (the golden clip, every 4th frame, 3-25% black or white). A view one px off loses newborns but made no false
positive. Why the map's art, not a background learned: a running median of the point (a sample every 0.5 s, the last 11,
her and the monsters masked) came within 3.9 times of the rule on her passing (alpha 0.11, residual 2.4), and plates
kept from windows unchanged for 0.6 s gave a false positive under a looser rule; the art is never fooled by her standing
on a point for seconds, and it needs no warm-up. Pixels where the art isn't the map's fixed art alone (its `known`:
animated objects, moving backgrounds) are left out of the fit; the maps watched are MEASURED's only, and their kinds
(another map's art and monsters haven't been measured so).

A newborn can be hit while it fades (its HP bar shows from alpha about 0.36), but didn't touch her until it was solid
(of her 34 touches on the golden clip, 1 in the 0.82 s after the map's respawn passes, against 3.9 at random: one clip
only, two examples frame by frame): each sighting has its alpha (`Sighting.a`, Mob.fading), from which the tracks say
when it turns solid (localization.tracking: FADE_PER_S).

The sums are integers (the game's alpha 0..255, its colours as they are), worked out in C where it's built (gamelab/
spawns.c, native.spawns_lib: a run of 8 hypotheses one column apart in AVX2 lanes), else numpy: the same integers. In C
on those 18,190 frames (one thread, Idle priority on CPUs 12-23, 2 such replays at once and nothing else heavy): 1.19 ms
a frame at the median (p90 1.72, p99 3.96, the most 7.29; 9.5 points in view, 5.9 fitted; before the review's partly
shown points 0.98-1.10, 8.6 and 5.4), 1.7 ms more per newborn followed (4.6% of the frames follow one), where the
prototype's (gathered lanes, float sums) took 2.2 ms and 3.3; numpy 13.6 ms a frame and 36 per newborn. Only at the
game's own 1366x768, a map unit a px, with the view registered on the frame (localization.worldstate).
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .. import native
from .kinds import kind_of
from .mobs import Mob

FADE_PER_S = 1.224   # alpha a second a newborn fades in by (0.817 s from nothing to solid; the top)
EMPTY_LEVELS = 3     # the empty test: a pixel this many colour levels off the art on some channel is drawn over ...
EMPTY_COUNT = 20     # ... and fewer such pixels than this in a point's window: nothing there
STRIDE = 3           # every STRIDE-th pixel of a picture's (A > 0), evenly spread: 626-657 of a mushroom's, 530-538
                     # of a slime's
BAND = 4             # units either side of the point its feet are tried at, at the onset (82 of 99 caught before alpha
ROWS = (-1, 0, 1)    # 0.15 started at its x exactly, all within 3) ... and rows up or down of its platform
ONSET = ("stand/0", "move/0")  # the onset's pictures (move/0 194 births, stand/0 12; with every stand and move
                               # picture, 32 hidden starts found where 20, at about 3 times the cost)
FOLLOW = ("stand", "move", "hit1")  # a newborn followed: every picture of these actions (it walks; it can be hit)
# A newborn followed is looked for FOLLOW_SPEED units a second the way each picture faces for the time since it was last
# found, FOLLOW_MOST_S of it at most, and BACK units the other way: 12 units a frame at 15 a second, the prototype's
# (its steps a frame then: the median 5, p99 12, the most 16), 18 at the alarm clips' 10. By time, not frames:
# perception doesn't know the service's rate (localization.rate.HZ).
FOLLOW_SPEED = 180.0
FOLLOW_MOST_S = 0.134   # (two frames at 15 a second: it was unfound in the one before)
BACK = 3
A_LO, A_HI = 0.015, 0.95  # a hypothesis is tried at these alphas only (over A_HI: a solid monster, the whole screen's)
MIN_USED = 0.4       # ... with this share of its pixels left after the masks, at least
# The rule (rules2.py on every point-frame of the clips, the art as background: no false positive under any of the
# rules tried; these the nearest negative 11.8 times off): its alpha at least A_MIN, 1 - R2 at most U_MAX, the residual
# at most RES_MAX colour levels. Newborns' fits: 1 - R2 0.0002-0.001 and residuals 0.27-0.30 at alpha 0.05-0.6.
A_MIN = 0.02
U_MAX = 0.008
RES_MAX = 1.5
# A newborn is found first at its point at this alpha at most (the review of 2026-11-01): a solid monster within BAND of
# its point under a faint shade over the whole screen (the game's fade on a map change: 5% black, a 0.942) or a
# see-through patch over it (10%) fits nearly as a newborn at a = 1 - the shade, 1.15-1.30 times off the rule where
# unshaded the nearest negative is 11.8. Under A_START a shade must take 20% or more off it, and then fits 4.7 times
# off at the nearest (made-up frames, every point: 20% black; a 30% patch of other art 3.7) and 5.2 on the golden clip's
# frames. Of the 391 onsets replayed on the clips 48 were over 0.8, every one on a traced newborn (47 found at that
# point in the 20 frames before, 46 handed to the whole screen that frame): without them each birth is found as before.
A_START = 0.8
# Whether a point's window reaching under the chat box or the game's windows is looked at in what shows (the review of
# 2026-11-01: a window past the screen's edge is, in what's on the screen). Not: the rows just over the chat box are
# drawn over now and then (its border, lines of text), so newborns there fit 2.5-5 levels off (none found there on the
# clips), and a solid Slime cut by it came within 3.3 times of the rule where every other fit an onset could start
# from on the clips within 14 times of it, in play or cut by the screen's top, left or right edge, was a newborn under
# an effect or a player (D:/MapleHelper-scratch/spawn-fade/fix: margin.py, margin_final.py).
ONSET_COVERED = False
DROP_A = 0.08        # a newborn followed doesn't fade out: its alpha may seem to be this much under its most at most
HANDOVER_DX, HANDOVER_DY = 15.0, 8.0  # units: the whole screen's monster this near a newborn is it
LOST_AFTER = 2       # frames a newborn followed may go unfound (68 of 390 were lost to cover so, alpha 0.29 median)
NEAR_POINT = 20.0    # units: a point with a newborn followed this near on its platform isn't looked at for another
HER_BOX = (-40.0, -85.0, 40.0, 28.0)   # units about her feet left out of the fit, her body and name tag: x0, y0, x1, y1
MOB_BOX = (-32.0, -58.0, 32.0, 4.0)    # ... and about a monster's feet (56 x 52 pictures)
ON_POINT = (12.0, 8.0)  # units across, down: a monster the frame before found this near a point is left in at its onset
# The maps watched and their kinds: those measured, Tree Dungeon's (its 15 Horny Mushroom, 14 Green Mushroom and 2 Slime
# points); another map, its art and its monsters, not until its frames have been replayed so (SpawnWatch.of: none of
# its points), the same kinds on it too (the first map's Slimes).
MEASURED = {10002073: ("horny-mushroom", "green-mushroom", "slime")}


class Floor(Protocol):
    """The ground a spawn point's monsters stand on: a platform (localization.mapdata.Platform has these)."""
    left: float
    right: float

    def y_at(self, x: float) -> float: ...


@dataclass(frozen=True)
class Point:
    """A spawn point: its monster's kind, its x (map units), the floor its feet stand on, and that floor's id."""
    kind: str
    x: float
    floor: Floor = field(repr=False)
    platform: int | None = None

    def on(self, x: float) -> bool:
        return self.floor.left - 0.5 <= x <= self.floor.right + 0.5


@dataclass(frozen=True)
class Picture:
    """One frame of a kind's, facing one way, as the game draws it: its name ("move/0"), which way it faces (`right`:
    mirrored), its size, where its feet are in it, and its pixels sampled (STRIDE: rows `ys`, columns `xs`, colours `S`
    (k x 3) and alpha `A` (1-255))."""
    name: str
    right: bool
    w: int
    h: int
    ox: int
    oy: int
    ys: np.ndarray = field(repr=False)
    xs: np.ndarray = field(repr=False)
    S: np.ndarray = field(repr=False)
    A: np.ndarray = field(repr=False)

    @property
    def action(self) -> str:
        return self.name.split("/")[0]


class Pictures:
    """A kind's pictures of some actions, each sampled to the same count `k` (the fewest one has, over STRIDE), side by
    side for the sums: colours (pictures, k, 3) and alphas (pictures, k), uint8."""

    def __init__(self, pictures: list[Picture]):
        self.list = pictures
        self.k = len(pictures[0].A)
        self.S = np.ascontiguousarray(np.stack([p.S for p in pictures]), np.uint8)
        self.A = np.ascontiguousarray(np.stack([p.A for p in pictures]), np.uint8)
        self.ox, self.oy = np.array([p.ox for p in pictures]), np.array([p.oy for p in pictures])
        self.w, self.h = np.array([p.w for p in pictures]), np.array([p.h for p in pictures])
        self.ext = max(max(p.ox, p.w - p.ox) for p in pictures) + 1   # units left or right of the feet, at most
        self.up = max(p.oy for p in pictures)
        self.down = max(p.h - p.oy for p in pictures)
        self._offsets: dict[int, np.ndarray] = {}

    def offsets(self, w: int) -> np.ndarray:
        """Each picture's sampled pixels' offsets from its top-left in a window `w` px wide (pictures, k: int32)."""
        got = self._offsets.get(w)
        if got is None:
            got = np.ascontiguousarray(np.stack([p.ys * w + p.xs for p in self.list]), np.int32)
            self._offsets[w] = got
        return got


def sampled(frames: list[dict], actions: Iterable[str] | None = None, names: Iterable[str] | None = None,
            stride: int = STRIDE) -> Pictures | None:
    """The frames (gamedata.canvas.mob_frames' dicts: rgba facing left, origin) of these actions (or names, "move/0"),
    both ways, every `stride`-th of each one's pixels with alpha over 0 (as many of each as the one with the fewest has
    over `stride`, evenly spread over its own); None for none."""
    actions = None if actions is None else set(actions)
    names = None if names is None else set(names)
    chosen = [f for f in frames if (actions is None or f["action"] in actions)
              and (names is None or f"{f['action']}/{f['index']}" in names)]
    if not chosen:
        return None
    raw = []
    for f in chosen:
        rgba = np.asarray(f["rgba"], np.uint8)
        ox, oy = f["origin"]
        for right in (False, True):
            a = np.ascontiguousarray(rgba[:, ::-1] if right else rgba)
            ys, xs = np.nonzero(a[..., 3] > 0)
            raw.append((f"{f['action']}/{f['index']}", right, a, a.shape[1] - ox if right else ox, oy, ys, xs))
    k = min(len(r[5]) for r in raw) // stride
    if k < 1:
        return None
    out = []
    for name, right, a, ox, oy, ys, xs in raw:
        pick = np.linspace(0, len(ys) - 1, k).round().astype(int)
        y, x = ys[pick], xs[pick]
        out.append(Picture(name, right, a.shape[1], a.shape[0], int(ox), int(oy), y.astype(np.int32),
                           x.astype(np.int32), np.ascontiguousarray(a[y, x, :3]), np.ascontiguousarray(a[y, x, 3])))
    return Pictures(out)


@dataclass(frozen=True)
class Art:
    """The map's art behind the monsters (localization.mapart.MapArt's): its picture (H x W x 3, uint8 RGB) in map units
    from `origin`, and where it's the map's fixed art alone (`known`: what the fit may compare)."""
    rgb: np.ndarray = field(repr=False)
    origin: tuple[int, int]
    known: np.ndarray | None = field(default=None, repr=False)

    def inside(self, X0: int, Y0: int, X1: int, Y1: int) -> bool:
        h, w = self.rgb.shape[:2]
        x0, y0 = self.origin
        return x0 <= X0 and y0 <= Y0 and X1 <= x0 + w and Y1 <= y0 + h

    def clip(self, X0: int, Y0: int, X1: int, Y1: int) -> tuple[int, int, int, int]:
        """Map box (X0, Y0)-(X1, Y1) cut to it."""
        h, w = self.rgb.shape[:2]
        x0, y0 = self.origin
        return max(X0, x0), max(Y0, y0), min(X1, x0 + w), min(Y1, y0 + h)

    def crop(self, X0: int, Y0: int, X1: int, Y1: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Map box (X0, Y0)-(X1, Y1) of it (a view, inside it: `inside`), and where it's known there (None: all of
        it)."""
        x0, y0 = self.origin
        rgb = self.rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        if self.known is None:
            return rgb, None
        known = self.known[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        return rgb, None if known.all() else known


@dataclass(frozen=True)
class Sighting:
    """A newborn seen fading in, in one frame: its spawn point (its index in the watch's), kind, feet (map units),
    opacity `a`, the fit's share explained and residual, the picture that fit, and when it was first found (`since`:
    its onset's frame time). `onset`: found at its point in this frame (else followed)."""
    point: int
    kind: str
    x: float
    y: float
    a: float
    r2: float
    res: float
    picture: Picture = field(repr=False, compare=False)
    since: float = 0.0
    onset: bool = True

    def mob(self, view: tuple[int, int]) -> Mob:
        """It as the whole screen's detections are (perception.mobs.Mob: look-frame px of the frame with that view's
        top-left), with its alpha (`fading`), its fit's share explained as its score."""
        p = self.picture
        left, top = int(np.rint(self.x - p.ox)) - view[0], int(np.rint(self.y - p.oy)) - view[1]
        return Mob(self.kind, left + p.w // 2, top + p.h // 2, p.w, p.h, round(self.r2, 4), self.x - view[0],
                   self.y - view[1], 1 if p.right else -1, p.action, p.name, "", round(self.a, 3))


@dataclass
class _Newborn:
    """A newborn followed: its point, kind, feet, alpha and picture as last found, when it was first and last found, and
    the most its alpha has been (`top`)."""
    point: int
    kind: str
    x: float
    y: float
    a: float
    picture: Picture
    since: float
    last: float
    misses: int = 0
    top: float = 0.0


@dataclass
class _Hyps:
    """Hypotheses in a window (map box `rect`, its size `w` x `h`): their runs for the sums (spawns.c's: a picture, the
    top-left's row and column, 1-8 lanes a column apart), and each one's picture and feet (in the runs' order)."""
    rect: tuple[int, int, int, int]
    w: int
    h: int
    runs: np.ndarray
    pic: np.ndarray
    X: np.ndarray
    Y: np.ndarray


def _hypotheses(pics: Pictures, which: Iterable[int], places: list[tuple[float, float]], rect) -> _Hyps | None:
    """Pictures `which` (indices into pics.list) with their feet at `places` ((X, Y) map units), those wholly inside map
    box `rect` (X0, Y0, X1, Y1); in runs (_Hyps), by picture, row and column; None where none is."""
    X0, Y0, X1, Y1 = rect
    W, H = X1 - X0, Y1 - Y0
    which = np.asarray(list(which), np.int64)
    P = np.asarray(places, float).reshape(-1, 2)
    if not len(P) or not len(which) or W <= 0 or H <= 0:
        return None
    pw, ph = pics.w[which][:, None], pics.h[which][:, None]
    c = np.rint(P[None, :, 0] - pics.ox[which][:, None]).astype(np.int64) - X0   # (pictures, places)
    r = np.rint(P[None, :, 1] - pics.oy[which][:, None]).astype(np.int64) - Y0
    ok = (c >= 0) & (r >= 0) & (c + pw <= W) & (r + ph <= H)
    wi, pj = np.nonzero(ok)
    if not len(wi):
        return None
    J, R, C, X, Y = which[wi], r[wi, pj], c[wi, pj], P[pj, 0], P[pj, 1]
    order = np.lexsort((C, R, J))
    J, R, C, X, Y = J[order], R[order], C[order], X[order], Y[order]
    if len(J) > 1:   # (the same place twice: once)
        same = np.concatenate([[False], (J[1:] == J[:-1]) & (R[1:] == R[:-1]) & (C[1:] == C[:-1])])
        J, R, C, X, Y = J[~same], R[~same], C[~same], X[~same], Y[~same]
    cut = np.ones(len(J), bool)   # a new run where the picture or row changes, a column is skipped, or 8 lanes are full
    cut[1:] = (J[1:] != J[:-1]) | (R[1:] != R[:-1]) | (C[1:] != C[:-1] + 1)
    starts = np.flatnonzero(cut)
    cut |= (np.arange(len(J)) - starts[np.cumsum(cut) - 1]) % 8 == 0
    first = np.flatnonzero(cut)
    lanes = np.diff(np.append(first, len(J)))
    runs = np.stack([J[first], R[first], C[first], lanes], axis=1)
    return _Hyps(tuple(rect), W, H, np.ascontiguousarray(runs, np.int32), J.astype(np.int32), X, Y)


def empty_numpy(window: np.ndarray, plate: np.ndarray, levels: int = EMPTY_LEVELS) -> int:
    """spawns.c's sp_empty in numpy: the pixels of `window` whose largest channel is off `plate`'s by more than
    `levels`."""
    d = np.abs(window[..., :3].astype(np.int16) - plate.astype(np.int16)).max(axis=2)
    return int((d > levels).sum())


def sums_numpy(window: np.ndarray, plate: np.ndarray, keep: np.ndarray | None, hyps: _Hyps,
               pics: Pictures) -> np.ndarray:
    """spawns.c's sp_sums in numpy (the same integers): each hypothesis's D.G, G.G, D.D and pixels kept (n x 4 int64),
    of `window` (h x w x 3+, its RGB first) over `plate` (h x w x 3), `keep` (h x w, 1 where a pixel counts; None:
    all)."""
    w = hyps.w
    F = np.ascontiguousarray(window[..., :3]).reshape(-1, 3).astype(np.int64)
    B = np.ascontiguousarray(plate).reshape(-1, 3).astype(np.int64)
    offsets = pics.offsets(w)
    lane = np.concatenate([np.arange(n) for n in hyps.runs[:, 3]])
    first = np.repeat(hyps.runs[:, 1].astype(np.int64) * w + hyps.runs[:, 2], hyps.runs[:, 3]) + lane
    idx = first[:, None] + offsets[hyps.pic]                     # (n, k) window pixels
    D = F[idx] - B[idx]
    G = pics.A[hyps.pic].astype(np.int64)[..., None] * (pics.S[hyps.pic].astype(np.int64) - B[idx])
    if keep is not None:
        K = keep.reshape(-1)[idx].astype(np.int64)
        D *= K[..., None]
        G *= K[..., None]
        n = K.sum(axis=1)
    else:
        n = np.full(len(idx), pics.k, np.int64)
    return np.stack([(D * G).sum(axis=(1, 2)), (G * G).sum(axis=(1, 2)), (D * D).sum(axis=(1, 2)), n], axis=1)


def scores(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each hypothesis's alpha, energy explained, share explained (R2) and residual from its sums (the top: a = 255
    D.G / G.G with the alpha 0..255; the residual over its pixels kept and their channels)."""
    DG, GG, DD, n = (sums[:, i].astype(np.float64) for i in range(4))
    GG = np.maximum(GG, 1.0)
    a = 255.0 * DG / GG
    expl = DG * DG / GG
    r2 = expl / np.maximum(DD, 1e-9)
    res = np.sqrt(np.maximum(0.0, DD - expl) / np.maximum(3.0 * n, 1.0))
    return a, expl, r2, res


def kept(sums: np.ndarray, k: int, a: np.ndarray, expl: np.ndarray, allowed: np.ndarray | None = None) -> int | None:
    """The hypothesis kept (the top): the most energy explained among those with A_LO <= a <= A_HI and MIN_USED of their
    k pixels left (and `allowed`); None for none."""
    usable = (a >= A_LO) & (a <= A_HI) & (sums[:, 3] >= MIN_USED * k)
    if allowed is not None:
        usable &= allowed
    if not usable.any():
        return None
    return int(np.argmax(np.where(usable, expl, -1.0)))


def passes(a: float, r2: float, res: float) -> bool:
    """The rule: a newborn's fit."""
    return a >= A_MIN and 1.0 - r2 <= U_MAX and res <= RES_MAX


def _boxes(rect, boxes) -> list[tuple[float, float, float, float]]:
    X0, Y0, X1, Y1 = rect
    return [b for b in boxes if b[0] < X1 and X0 < b[2] and b[1] < Y1 and Y0 < b[3]]


def _keep(rect, boxes, known: np.ndarray | None, shown: np.ndarray | None = None) -> np.ndarray | None:
    """1 over map box `rect` but inside `boxes` (map units), where the art isn't known and (`shown`: 1 where its pixels
    are in the play area) off it; None where that's all of it."""
    X0, Y0, X1, Y1 = rect
    boxes = _boxes(rect, boxes)
    if not boxes and known is None and shown is None:
        return None
    k = np.ones((Y1 - Y0, X1 - X0), np.uint8) if known is None else np.array(known, np.uint8)
    if shown is not None:
        k &= shown
    for bx0, by0, bx1, by1 in boxes:
        c0, r0 = int(max(0, bx0 - X0)), int(max(0, by0 - Y0))
        c1, r1 = int(min(X1 - X0, bx1 - X0)), int(min(Y1 - Y0, by1 - Y0))
        if c0 < c1 and r0 < r1:
            k[r0:r1, c0:c1] = 0
    return k


class _Onset:
    """A point's onset: its window (map box: its onset pictures at their feet within BAND of it, ROWS up or down, and
    2-3 rows more), the art there, and its hypotheses."""

    def __init__(self, point: Point, pics: Pictures, art: Art):
        floor = point.floor
        xs = [point.x + dx for dx in range(-BAND, BAND + 1)]
        ys = [floor.y_at(x) for x in xs]
        self.rect = (int(math.floor(point.x - BAND - pics.ext)), int(math.floor(min(ys))) - pics.up - 2,
                     int(math.ceil(point.x + BAND + pics.ext)), int(math.ceil(max(ys))) + pics.down + 3)
        self.y = round(floor.y_at(point.x))
        self.pics = pics
        places = [(x, floor.y_at(x) + dy) for x in xs if point.on(x) for dy in ROWS]
        self.hyps = _hypotheses(pics, range(len(pics.list)), places, self.rect) if art.inside(*self.rect) else None
        if self.hyps is not None:
            plate, known = art.crop(*self.rect)
            self.plate = np.ascontiguousarray(plate)
            self.known = None if known is None else np.ascontiguousarray(known)


class SpawnWatch:
    """The map's spawn points (`points`), each kind's pictures (`frames`: kind -> gamedata.canvas.mob_frames' dicts of
    its stand, move and hit1 frames), the map's art (`art`), and the newborns followed (see the top). `exclude`: screen
    boxes the game's own windows cover at 1366x768 (localization.camreg.EXCLUDE), `bottom`: the play area's last row
    (the chat box's top), `size`: the frame's. `use_c`: None, spawns.c where it's built; False, numpy."""

    def __init__(self, points: list[Point], frames: dict[str, list[dict]], art: Art, exclude: tuple = (),
                 bottom: int = 700, size: tuple[int, int] = (1366, 768), use_c: bool | None = None):
        self.art, self.exclude, self.bottom, self.size = art, tuple(exclude), bottom, tuple(size)
        self.lib = None if use_c is False else native.spawns_lib()
        self.onset = {k: sampled(v, names=ONSET) for k, v in frames.items()}
        self.follow = {k: sampled(v, actions=FOLLOW) for k, v in frames.items()}
        self.points = [p for p in points if self.onset.get(p.kind) is not None and self.follow.get(p.kind) is not None]
        self._onsets = [_Onset(p, self.onset[p.kind], art) for p in self.points]
        self.newborns: list[_Newborn] = []
        self.faded: list[tuple[int | None, float, float]] = []  # newborns seen fading out: platform, x, until when
        self.counts = {"frames": 0, "points": 0, "partly": 0, "fits": 0, "follows": 0, "onsets": 0, "followed": 0,
                       "handed": 0, "passed": 0, "solid": 0, "lost": 0, "out": 0}

    @classmethod
    def of(cls, points: list[Point], ids: dict[str, int], art: Art, map_id: int, **kw) -> "SpawnWatch":
        """The watch over map `map_id`'s points (those of its MEASURED kinds: none on a map not measured) with each
        kind's pictures read from the game data (ids: kind -> its monster's id; gamedata.canvas.mob_frames: in memory,
        nothing written)."""
        from gamedata.canvas import mob_frames
        frames = {}
        for kind, mob_id in ids.items():
            if kind_of(kind) not in MEASURED.get(map_id, ()):
                continue
            try:
                frames[kind_of(kind)] = mob_frames(mob_id, actions=FOLLOW)
            except (FileNotFoundError, KeyError, OSError):
                pass
        return cls([p for p in points if p.kind in frames], frames, art, **kw)

    # --- a frame --------------------------------------------------------------------------------------------------

    def look(self, frame: np.ndarray, view: tuple[int, int], t: float, her: tuple[float, float] | None = None,
             others: Iterable[tuple[float, float]] = ()) -> list[Sighting]:
        """The newborns in `frame` (the look frame at 1366x768, RGB first: a map unit a px), whose top-left is map point
        `view` (registered on this frame), shown at t: the newborns followed found again (or let go), then every point
        in view not by one looked at for a new one. `her`: her feet (map units; None: not known), `others`: the feet of
        the monsters the frame before found (not these newborns), both left out of the fits (HER_BOX, MOB_BOX)."""
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"a {frame.shape[1]}x{frame.shape[0]} frame: the spawn watch is for "
                             f"{self.size[0]}x{self.size[1]}")
        self.counts["frames"] += 1
        vx, vy = int(view[0]), int(view[1])
        her_box = [] if her is None else [(her[0] + HER_BOX[0], her[1] + HER_BOX[1], her[0] + HER_BOX[2],
                                           her[1] + HER_BOX[3])]
        mobs = [(fx, fy, (fx + MOB_BOX[0], fy + MOB_BOX[1], fx + MOB_BOX[2], fy + MOB_BOX[3])) for fx, fy in others]
        out = []
        for nb in self.newborns:   # 1. the newborns followed
            self.counts["follows"] += 1
            got = self._follow(frame, (vx, vy), nb, t - nb.last, her_box, mobs)
            if got is not None and got[0] == "ok":
                s = got[1]
                nb.x, nb.y, nb.a, nb.picture, nb.last, nb.misses = s.x, s.y, s.a, s.picture, t, 0
                nb.top = max(nb.top, s.a)
                out.append(s)
            elif got is not None and got[0] == "solid":
                nb.misses = -1   # (done: the whole screen's)
                self.counts["solid"] += 1
            elif got is not None and got[0] == "out":
                nb.misses = -1   # (fading out: its place not looked at again until that fade's over)
                self.faded.append((self.points[nb.point].platform, nb.x, t + 1.0 / FADE_PER_S))
                self.counts["out"] += 1
            else:
                nb.misses += 1
                if nb.misses >= LOST_AFTER:
                    self.counts["lost"] += 1
        self.newborns = [nb for nb in self.newborns if 0 <= nb.misses < LOST_AFTER]
        self.faded = [f for f in self.faded if t < f[2]]
        for i, (point, on) in enumerate(zip(self.points, self._onsets)):   # 2. every point in view
            if on.hyps is None:
                continue
            X0, Y0, X1, Y1 = on.rect
            c0, r0, c1, r1 = X0 - vx, Y0 - vy, X1 - vx, Y1 - vy
            shown = self._shown(c0, r0, c1, r1, covered=ONSET_COVERED)
            if shown is False:
                continue
            self.counts["points"] += 1
            self.counts["partly"] += shown is not None
            if any(self.points[nb.point].platform == point.platform and abs(nb.x - point.x) <= NEAR_POINT
                   for nb in self.newborns) \
                    or any(pl == point.platform and abs(x - point.x) <= NEAR_POINT for pl, x, _ in self.faded):
                continue
            window = self._window(frame, on.plate, c0, r0, c1, r1, shown)
            if self._empty(window, on.plate) < EMPTY_COUNT:
                continue
            self.counts["fits"] += 1
            boxes = her_box + [b for fx, fy, b in mobs
                               if abs(fx - point.x) > ON_POINT[0] or abs(fy - on.y) > ON_POINT[1]]
            keep = _keep(on.rect, boxes, on.known, shown)
            sums = self._sums(window, on.plate, keep, on.hyps, on.pics)
            a, expl, r2, res = scores(sums)
            h = kept(sums, on.pics.k, a, expl)
            if h is None or a[h] > A_START or not passes(a[h], r2[h], res[h]):
                continue
            X, Y = float(on.hyps.X[h]), float(on.hyps.Y[h])
            if any(abs(nb.x - X) <= HANDOVER_DX and abs(nb.y - Y) <= HANDOVER_DY for nb in self.newborns):
                continue   # (a newborn followed already, walked into this point's window)
            pic = on.pics.list[int(on.hyps.pic[h])]
            s = Sighting(i, point.kind, X, Y, float(a[h]), float(r2[h]), float(res[h]), pic, t, True)
            self.newborns.append(_Newborn(i, point.kind, X, Y, s.a, pic, t, t, top=s.a))
            self.counts["onsets"] += 1
            out.append(s)
        self.counts["followed"] += sum(not s.onset for s in out)
        return out

    def handed(self, sightings: list[Sighting], found: Iterable[tuple[float, float, str]]) -> list[Sighting]:
        """This frame's sightings less those the whole screen found too (`found`: its monsters' feet, map units, and
        kind): a monster of a newborn's kind within HANDOVER_DX across and HANDOVER_DY down of it is it, the tracks'
        from now on (not followed); one of another kind there has the frame's place (the sighting left out: not two
        monsters where the search took it for another kind, as it did 3 of the 316 newborns it found on the clips, a
        Green and a Horny Mushroom one for the other), the newborn still followed (it may be another walking past)."""
        found = list(found)

        def near(nb: _Newborn, same: bool) -> bool:
            return any(abs(fx - nb.x) <= HANDOVER_DX and abs(fy - nb.y) <= HANDOVER_DY and (kind == nb.kind) == same
                       for fx, fy, kind in found)
        mine = [nb for nb in self.newborns if near(nb, True)]
        passed = [nb for nb in self.newborns if nb not in mine and near(nb, False)]
        if not mine and not passed:
            return sightings
        gone = {id(nb) for nb in mine}
        points = {(nb.point, nb.since) for nb in mine + passed}
        self.counts["handed"] += len(mine)
        self.counts["passed"] += len(passed)
        self.newborns = [nb for nb in self.newborns if id(nb) not in gone]
        return [s for s in sightings if (s.point, s.since) not in points]

    def missed(self) -> None:
        """A frame not looked at (no sure view, or not of the game as expected): every newborn followed missed in it."""
        for nb in self.newborns:
            nb.misses += 1
        self.counts["lost"] += sum(nb.misses >= LOST_AFTER for nb in self.newborns)
        self.newborns = [nb for nb in self.newborns if nb.misses < LOST_AFTER]

    # --- inside ---------------------------------------------------------------------------------------------------

    def _shown(self, c0: int, r0: int, c1: int, r1: int, covered: bool = True) -> np.ndarray | bool | None:
        """Which of screen box (c0, r0)-(c1, r1)'s pixels are in the play area and clear of the game's windows: None,
        all of them; False, none (or, `covered` False, some under the chat box or the game's windows: none looked at);
        else where (h x w, 1 where shown)."""
        W, H = self.size[0], self.bottom
        if c1 <= 0 or r1 <= 0 or c0 >= W or r0 >= H:
            return False
        under = [b for b in self.exclude if c0 < b[2] and b[0] < c1 and r0 < b[3] and b[1] < r1]
        if not covered and (r1 > H or under):
            return False
        if c0 >= 0 and r0 >= 0 and c1 <= W and r1 <= H and not under:
            return None
        shown = np.zeros((r1 - r0, c1 - c0), np.uint8)
        shown[max(0, -r0):min(r1, H) - r0, max(0, -c0):min(c1, W) - c0] = 1
        for ex0, ey0, ex1, ey1 in under:
            shown[max(0, ey0 - r0):min(r1, ey1) - r0, max(0, ex0 - c0):min(c1, ex1) - c0] = 0
        return shown if shown.any() else False

    @staticmethod
    def _window(frame: np.ndarray, plate: np.ndarray, c0: int, r0: int, c1: int, r1: int,
                shown: np.ndarray | None) -> np.ndarray:
        """Screen box (c0, r0)-(c1, r1) of `frame`: itself where it's all shown (`_shown` None), else the art there
        (`plate`) with the frame's pixels shown over it (no difference where it isn't: those left out of the fit too,
        `_keep`)."""
        if shown is None:
            return frame[r0:r1, c0:c1]
        out = np.array(plate, np.uint8)
        y0, x0, y1, x1 = max(0, r0), max(0, c0), min(r1, frame.shape[0]), min(c1, frame.shape[1])
        part = out[y0 - r0:y1 - r0, x0 - c0:x1 - c0]
        on = shown[y0 - r0:y1 - r0, x0 - c0:x1 - c0].astype(bool)
        part[on] = frame[y0:y1, x0:x1, :3][on]
        return out

    def _follow(self, frame: np.ndarray, view: tuple[int, int], nb: _Newborn, dt: float, her_box, mobs):
        """Newborn `nb`, last found `dt` s ago, looked for in this frame (the top, 3.): ("ok", its Sighting), ("solid",
        None), ("out", None) (found fainter than DROP_A under its most: fading out, not in), ("no", None) or None
        (nowhere to look)."""
        pics, point = self.follow[nb.kind], self.points[nb.point]
        vx, vy = view
        reach = max(1, int(round(FOLLOW_SPEED * min(max(dt, 0.0), FOLLOW_MOST_S))))
        xs = [nb.x + dx for dx in range(-reach, reach + 1) if point.on(nb.x + dx)]
        if not xs:
            return None
        ys = [point.floor.y_at(x) for x in xs]
        rect = self.art.clip(int(math.floor(min(xs))) - pics.ext, int(math.floor(min(ys))) - 1 - pics.up - 2,
                             int(math.ceil(max(xs))) + pics.ext + 1, int(math.ceil(max(ys))) + 1 + pics.down + 3)
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            return None
        X0, Y0, X1, Y1 = rect
        shown = self._shown(X0 - vx, Y0 - vy, X1 - vx, Y1 - vy)
        if shown is False:
            return None
        places = {False: [], True: []}
        for right in (False, True):
            lo, hi = (-BACK, reach) if right else (-reach, BACK)
            places[right] = [(nb.x + dx, point.floor.y_at(nb.x + dx) + dy) for dx in range(lo, hi + 1)
                             if point.on(nb.x + dx) for dy in ROWS]
        parts = [h for right in (False, True)
                 if (h := _hypotheses(pics, [j for j, p in enumerate(pics.list) if p.right == right], places[right],
                                      rect)) is not None]
        if not parts:
            return None
        hyps = parts[0] if len(parts) == 1 else _Hyps(
            rect, rect[2] - rect[0], rect[3] - rect[1], np.concatenate([h.runs for h in parts]),
            np.concatenate([h.pic for h in parts]), np.concatenate([h.X for h in parts]),
            np.concatenate([h.Y for h in parts]))
        plate, known = self.art.crop(*rect)
        window = self._window(frame, plate, X0 - vx, Y0 - vy, X1 - vx, Y1 - vy, shown)
        boxes = her_box + [b for fx, fy, b in mobs if abs(fx - nb.x) > HANDOVER_DX or abs(fy - nb.y) > HANDOVER_DY]
        sums = self._sums(window, plate, _keep(rect, boxes, known, shown), hyps, pics)
        a, expl, r2, res = scores(sums)
        h0 = int(np.argmax(expl))
        solid = a[h0] > A_HI and r2[h0] >= 1.0 - U_MAX
        least = max(nb.top, nb.a) - DROP_A
        h = kept(sums, pics.k, a, expl, a >= least)
        if h is None or not passes(a[h], r2[h], res[h]):
            if solid:
                return "solid", None
            o = kept(sums, pics.k, a, expl)
            if o is not None and a[o] < least and passes(a[o], r2[o], res[o]):
                return "out", None
            return None if h is None else ("no", None)
        pic = pics.list[int(hyps.pic[h])]
        return "ok", Sighting(nb.point, nb.kind, float(hyps.X[h]), float(hyps.Y[h]), float(a[h]), float(r2[h]),
                              float(res[h]), pic, nb.since, False)

    def _empty(self, window: np.ndarray, plate: np.ndarray) -> int:
        if self.lib is not None and _plain(window):
            h, w = window.shape[:2]
            return self.lib.sp_empty(window.ctypes.data, window.strides[0], window.strides[1], plate.ctypes.data,
                                     plate.strides[0], w, h, EMPTY_LEVELS)
        return empty_numpy(window, plate)

    def _sums(self, window: np.ndarray, plate: np.ndarray, keep: np.ndarray | None, hyps: _Hyps,
              pics: Pictures) -> np.ndarray:
        if self.lib is not None and _plain(window) and _plain(plate) and plate.strides[1] == 3:
            n = int(hyps.runs[:, 3].sum())
            out = np.empty((n, 4), np.int64)
            offsets = pics.offsets(hyps.w)
            kp = None if keep is None else np.ascontiguousarray(keep, np.uint8)
            got = self.lib.sp_sums(window.ctypes.data, window.strides[0], window.strides[1], plate.ctypes.data,
                                   plate.strides[0], None if kp is None else kp.ctypes.data, hyps.w, hyps.h,
                                   len(hyps.runs), hyps.runs.ctypes.data, pics.k, offsets.ctypes.data,
                                   pics.S.ctypes.data, pics.A.ctypes.data, out.ctypes.data)
            if got == n:
                return out
        return sums_numpy(window, plate, keep, hyps, pics)


def _plain(a: np.ndarray) -> bool:
    """Whether spawns.c can read `a` as it is: uint8, 3 or 4 channels a pixel side by side, RGB first."""
    return (a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] in (3, 4) and a.strides[2] == 1
            and a.strides[1] == a.shape[2] and a.strides[0] > 0)
