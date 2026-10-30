"""Her body found on screen where neither her tag's gray match nor her plate's colours place her: her exact poses
(a bank learned at calibration) checked pixel by pixel.

The localization prototype of 2026-10-29 (D:/MapleHelper-scratch/loc2/combined, camera-sprite's matcher as the combined
localizer used it) placed her by her body on every held-out frame where her plate didn't show enough (3 of the 81
hard frames with her tag hidden or cut; with her plate painted out of every frame, 97-98% of them within 3 px), in
7-80 ms of Python a search: too slow for the 30-a-second loop. Here it is ported, its pixel checks in C.

The bank (`PoseBank`, made at calibration from frames where her tag is sure: `build_bank`; kept under her lab folder,
`bank_path`): her poses, each her exact pixels (her own colours only) at offsets from her feet (her feet: the plate's
centre + the plate model's offset), how often it was seen and at how many places, whether she stood on a platform's
line in it, which way she faces, mirrored ones for the other facing; her anchors (her glasses, the top of her rope
pose, each as drawn and mirrored: the same picture in nearly every pose of hers) and where each sits in each pose.

A search (`BodyFinder.find`, the prototype's Localizer.find_her) of a box her feet may be in:
  1. her anchors matched in the box's window (masked squared difference, as drawn and as the blink after a touch
     darkens her: OpenCV's correlations), the best 2 places of each under HEAD_SQ; every pose holding that anchor
     tried where the anchor puts her feet: its SUBSET pixels' agreement +-1 px, the best VERIFY checked with all its
     pixels; an answer agreeing HEAD_OK or more is hers;
  2. else her colours' mass (a lookup of the bank's colours, as drawn and darkened, among the foreground: the frame
     where it differs from the map's art behind it, `background`) correlated with each lean group's mean mask (upright,
     leaning left, leaning right: a cast leans ahead), the best CENTRES; every pose there +-FINE_R by SCAN px, the best
     24 verified as in 1.
  A candidate's agreement: the share of its pixels within TOL on every channel (or within 2 of half its colour: the
  blink), pixels off the window not agreeing. With a prediction, the motion's weight: MOTION_W lost a px of jump
  beyond the slack (a cast draws one picture at two origins 25 px apart).
In C where the DLL is (gamelab/locate.c's lc_pack, lc_agree: every candidate's agreement at its offsets, eight across
a load on AVX2), in numpy otherwise, the same shares to the bit (tests/test_poses.py); the anchors' and the mass's
correlations are OpenCV's both ways. Its answer is the prototype's on the evaluation set's frames (README).

Px are a 1366x768 look frame's (look_scale 1), as the bank is made for one client size (its file's suffix). Colours
are RGB (the prototype's BGR turned).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Sequence

import cv2
import numpy as np

from .. import native
from ..screen import NATIVE, Geometry

TOL = 8                  # a pixel agrees within this on every channel ...
DARK = 0.5               # ... or within DARK_TOL of its colour x DARK (the blink after a touch: fitted 0.47-0.52)
DARK_TOL = 2
SUBSET = 160             # a pose's pixels a candidate is first checked by (seeded by its id: `subset_of`)
SCAN = 48                # ... and the mass path's scan by (the first of those)
FINE_R = 3               # the mass path's scan: every pose +-this around where the mass puts her
HEAD_SQ = 400.0          # an anchor's match: its mean squared gray difference at most this (RMS 20)
HEAD_OK = 0.8            # an anchor's answer agreeing this much ends the search
HEAD_PLACES = 2          # an anchor's best places tried
ANCHOR_OK = 0.7          # a pose holds an anchor where this share of the anchor's pixels sit in it, its colours
ANCHOR_ROWS = 20         # the rope anchor: the top rows of her commonest narrow pose off the ground
VERIFY = 8               # the candidates checked with all their pixels
CENTRES = 3              # the mass path's places
MASS_PEAKS = 2           # ... each lean group's best, over MASS_LEAST, MASS_APART apart
MASS_LEAST = 0.15
MASS_APART = 6
LEAN = 9                 # a pose leans when its mass's centre is this far across from her feet
FG = 12                  # the foreground: the frame this far (on a channel) from the map's art
FG_SCORE = (1.0, -0.5, -0.25)   # the mass's picture: her colour in the foreground, other foreground, other (no art)
LUT_GAINS = (1.0, 0.44, 0.48, 0.52, 0.56)  # her colours' lookup: as drawn and as the blink darkens them
MOTION_W = 0.015         # agreement lost a px of jump from the prediction beyond the slack (the combined localizer's)
WINDOW = (65, 66, 105, 10)   # the window around the feet box: px left, right, up, down
REACH = (70, 20)         # the pixels' box around the window (across, down), the poses' reach
FEET_ROWS = (30, 30)     # her feet at least this far down the frame, at most this far under the HUD's top
MIN_WINDOW = 60
USE_C = True             # False: numpy's (the tests compare the two)


def bank_path(lab: Path, geo: Geometry = NATIVE) -> Path:
    """Where her pose bank is kept: her lab folder (paths.of().lab), named for the client size."""
    return Path(lab) / f"poses{geo.suffix}.npz"


def subset_of(pid: int, n: int, k: int = SUBSET) -> np.ndarray:
    """The pixels (indices into her pose's, row by row) a candidate is first checked by: k of the n, seeded by its id
    (the prototype's: default_rng(id).choice, sorted)."""
    rng = np.random.default_rng(pid)
    return np.sort(rng.choice(n, min(k, n), replace=False))


@dataclass
class Pose:
    """One pose of hers: her pixels (`picture` RGB, `mask`) with the box's top-left at (dx0, dy0) from her feet (dx0 a
    half px: her feet's x is a pixel's edge)."""
    id: int
    picture: np.ndarray
    mask: np.ndarray
    dx0: float
    dy0: int
    n: int = 0                # frames it was seen in (0: mirrored)
    n_src: int = 0            # a mirrored one's: its source's
    places: int = 0           # places (map px) it was seen at
    facing: str = "right"
    mirrored: bool = False
    grounded: float | None = None   # the share of its frames she stood on a platform's line
    subset: np.ndarray | None = None
    anchors: list = field(default_factory=list)   # [(anchor, column from floor(feet x), row from feet y)]

    def __post_init__(self):
        ys, xs = np.nonzero(self.mask)
        self.ys = (ys + self.dy0).astype(np.int32)                 # rows from her feet's
        self.xs = (xs + self.dx0 + 0.5).astype(np.int32)           # columns from floor(feet x)
        self.colour = self.picture[ys, xs].astype(np.float32)
        if self.subset is None:
            self.subset = subset_of(self.id, len(ys))
        self.h, self.w = self.mask.shape
        self.centre = (self.dx0 + 0.5 + xs.mean(), self.dy0 + ys.mean())   # her mass's, from her feet

    def box(self, fx: float, fy: float) -> list[float]:
        """Her box on screen with her feet at (fx, fy): x0, y0, x1, y1."""
        return [fx + self.dx0, fy + self.dy0, fx + self.dx0 + self.w, fy + self.dy0 + self.h]


@dataclass
class Anchor:
    """A small part of her drawn the same in nearly every pose (her glasses; the top of her rope pose), as drawn or
    mirrored."""
    picture: np.ndarray
    mask: np.ndarray
    name: str

    def __post_init__(self):
        self.gray = cv2.cvtColor(np.ascontiguousarray(self.picture), cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.mf = self.mask.astype(np.float32)
        self.mt = self.gray * self.mf
        self.tt = float((self.gray * self.gray * self.mf).sum())
        self.n = int(self.mask.sum())


class PoseBank:
    """Her poses and anchors (`Pose`, `Anchor`), saved in one file."""

    def __init__(self, poses: list[Pose], anchors: list[Anchor], meta: dict | None = None):
        self.poses, self.anchors, self.meta = poses, anchors, dict(meta or {})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        P = self.poses
        out = dict(
            ids=np.array([p.id for p in P], np.int32), hw=np.array([(p.h, p.w) for p in P], np.int32),
            pictures=np.concatenate([p.picture.reshape(-1) for p in P]),
            masks=np.concatenate([p.mask.reshape(-1) for p in P]),
            dx0=np.array([p.dx0 for p in P], np.float64), dy0=np.array([p.dy0 for p in P], np.int32),
            n=np.array([p.n for p in P], np.int32), n_src=np.array([p.n_src for p in P], np.int32),
            places=np.array([p.places for p in P], np.int32),
            facing=np.array([p.facing == "left" for p in P], bool), mirrored=np.array([p.mirrored for p in P], bool),
            grounded=np.array([np.nan if p.grounded is None else p.grounded for p in P], np.float64),
            subsets=np.concatenate([p.subset for p in P]).astype(np.int32),
            subset_n=np.array([len(p.subset) for p in P], np.int32),
            pose_anchors=np.array([(i, k, ax, ay) for i, p in enumerate(P) for k, ax, ay in p.anchors],
                                  np.int32).reshape(-1, 4),
            anchor_hw=np.array([a.mask.shape for a in self.anchors], np.int32).reshape(-1, 2),
            anchor_pictures=np.concatenate([a.picture.reshape(-1) for a in self.anchors]) if self.anchors
            else np.zeros(0, np.uint8),
            anchor_masks=np.concatenate([a.mask.reshape(-1) for a in self.anchors]) if self.anchors
            else np.zeros(0, bool),
            anchor_names=np.array(json.dumps([a.name for a in self.anchors])),
            meta=np.array(json.dumps(self.meta)))
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path: Path) -> PoseBank:
        z = np.load(Path(path))
        poses, at, mt, st = [], 0, 0, 0
        anchors_of: dict[int, list] = {}
        for i, k, ax, ay in z["pose_anchors"]:
            anchors_of.setdefault(int(i), []).append((int(k), int(ax), int(ay)))
        for i, (pid, (h, w)) in enumerate(zip(z["ids"], z["hw"])):
            pic = z["pictures"][at:at + h * w * 3].reshape(h, w, 3)
            mask = z["masks"][mt:mt + h * w].reshape(h, w)
            sub = z["subsets"][st:st + z["subset_n"][i]]
            at, mt, st = at + h * w * 3, mt + h * w, st + int(z["subset_n"][i])
            g = float(z["grounded"][i])
            poses.append(Pose(int(pid), pic, mask, float(z["dx0"][i]), int(z["dy0"][i]), int(z["n"][i]),
                              int(z["n_src"][i]), int(z["places"][i]), "left" if z["facing"][i] else "right",
                              bool(z["mirrored"][i]), None if math.isnan(g) else g, sub.astype(np.int64),
                              anchors_of.get(i, [])))
        anchors, at, mt = [], 0, 0
        for (h, w), name in zip(z["anchor_hw"], json.loads(str(z["anchor_names"]))):
            anchors.append(Anchor(z["anchor_pictures"][at:at + h * w * 3].reshape(h, w, 3),
                                  z["anchor_masks"][mt:mt + h * w].reshape(h, w), name))
            at, mt = at + h * w * 3, mt + h * w
        return cls(poses, anchors, json.loads(str(z["meta"])))


class Background:
    """The map's art behind her, in map units (one a px here): a picture (RGB), where it's known (`known`), its
    top-left's map place, and how much each px varies (`noise`, 0 for drawn art). `screen` cuts what a view shows."""

    def __init__(self, picture: np.ndarray, origin: tuple[int, int], known: np.ndarray | None = None,
                 noise: np.ndarray | None = None):
        self.picture, self.origin = picture, (int(origin[0]), int(origin[1]))
        self.known = np.ones(picture.shape[:2], bool) if known is None else known
        self.noise = noise

    def screen(self, view: tuple[int, int], x0: int, y0: int, x1: int, y1: int):
        """The art behind screen px (x0, y0)-(x1, y1) with the view's top-left at map place `view`: (picture int16,
        known, noise or None), zero and unknown where the art doesn't reach."""
        h, w = y1 - y0, x1 - x0
        pic = np.zeros((h, w, 3), np.int16)
        known = np.zeros((h, w), bool)
        noise = None if self.noise is None else np.full((h, w), 255, np.uint8)
        ox, oy = view[0] + x0 - self.origin[0], view[1] + y0 - self.origin[1]   # the box's top-left in the picture
        a0, a1 = max(0, -oy), min(h, self.picture.shape[0] - oy)
        b0, b1 = max(0, -ox), min(w, self.picture.shape[1] - ox)
        if a1 > a0 and b1 > b0:
            pic[a0:a1, b0:b1] = self.picture[oy + a0:oy + a1, ox + b0:ox + b1]
            known[a0:a1, b0:b1] = self.known[oy + a0:oy + a1, ox + b0:ox + b1]
            if noise is not None:
                noise[a0:a1, b0:b1] = self.noise[oy + a0:oy + a1, ox + b0:ox + b1]
        return pic, known, noise


class BodyHit(NamedTuple):
    """Her body placed: its agreement, the pose, her feet (screen px; x a pixel's edge), her box there."""
    agree: float
    pose: Pose
    fx: float
    fy: float

    @property
    def box(self) -> list[float]:
        return self.pose.box(self.fx, self.fy)


class _Sets:
    """Pixel sets for lc_agree: ys, xs, colours (packed R | G << 8 | B << 16, as drawn and darkened) end to end, each
    set's [first, n, ymin, ymax, xmin, xmax]."""

    def __init__(self, parts: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        ys, xs, cols, ext, at = [], [], [], [], 0
        for y, x, c in parts:
            ys.append(y.astype(np.int32))
            xs.append(x.astype(np.int32))
            cols.append(c)
            ext.append((at, len(y), int(y.min()), int(y.max()), int(x.min()), int(x.max())))
            at += len(y)
        self.ys = np.ascontiguousarray(np.concatenate(ys))
        self.xs = np.ascontiguousarray(np.concatenate(xs))
        c = np.concatenate(cols).astype(np.float32)
        self.c0 = np.ascontiguousarray(c.astype(np.int16))                       # as drawn
        self.c1 = np.ascontiguousarray(np.round(c * np.float32(DARK)).astype(np.int16))   # as the blink darkens her
        self.p0, self.p1 = _packed(self.c0), _packed(self.c1)
        self.ext = np.ascontiguousarray(np.array(ext, np.int32))
        self.first, self.n = self.ext[:, 0], self.ext[:, 1]


def _packed(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.uint32)
    return np.ascontiguousarray(c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16))


class _Picture:
    """A search's pixels: the frame's box (lc_pack's in C; planes for numpy)."""

    def __init__(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, lib):
        self.x0, self.y0, self.w, self.h = x0, y0, x1 - x0, y1 - y0
        sub = frame[y0:y1, x0:x1]
        self.lib = lib
        if lib is not None and sub.strides[1] == 3 and sub.strides[2] == 1:
            if lib.lc_pack(sub.ctypes.data, self.h, self.w, sub.strides[0]) == 0:
                return
        self.lib = None
        self.planes = [np.ascontiguousarray(sub[..., c]).astype(np.int16).ravel() for c in range(3)]

    def agree(self, sets: _Sets, which: np.ndarray, fx: np.ndarray, fy: np.ndarray, R: int) -> np.ndarray:
        """[C, (2R+1)^2]: candidate c (set which[c], feet column fx[c], row fy[c] of this box) at each offset within
        +-R (rows the slower), the share of its pixels agreeing, the better colour mode."""
        which = np.ascontiguousarray(which, np.int32)
        fx, fy = np.ascontiguousarray(fx, np.int32), np.ascontiguousarray(fy, np.int32)
        side = 2 * R + 1
        out = np.zeros((len(which), side * side), np.float64)
        if not len(which):
            return out
        if self.lib is not None:
            if self.lib.lc_agree(sets.ys.ctypes.data, sets.xs.ctypes.data, sets.p0.ctypes.data, sets.p1.ctypes.data,
                                 TOL, DARK_TOL, sets.ext.ctypes.data, len(which), which.ctypes.data, fx.ctypes.data,
                                 fy.ctypes.data, R, out.ctypes.data) == 0:
                return out
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        oy, ox = oy.ravel(), ox.ravel()
        for n in np.unique(sets.n[which]):   # candidates of one set size at a time: [C, offsets, n] gathers
            sel = np.nonzero(sets.n[which] == n)[0]
            idx = sets.first[which[sel]][:, None] + np.arange(n)[None, :]
            ys = sets.ys[idx][:, None, :] + (fy[sel][:, None] + oy[None, :])[:, :, None]
            xs = sets.xs[idx][:, None, :] + (fx[sel][:, None] + ox[None, :])[:, :, None]
            cos = [([sets.c0[idx][:, None, :, c] for c in range(3)], TOL),
                   ([sets.c1[idx][:, None, :, c] for c in range(3)], DARK_TOL)]
            out[sel] = _agree_flat(self.planes, self.w, self.h, ys, xs, cos)
        return out


THREADS = 4             # the anchors' (and the lean groups') correlations on this many threads at once (OpenCV lets go
                        # of Python's lock: the same calls, the same answers); 1: one after another
_pool: list = []


def _each(fn, items) -> list:
    """[fn(item) for item in items], on THREADS threads of a pool of its own (made on first use)."""
    items = list(items)
    if THREADS <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    if not _pool:
        from concurrent.futures import ThreadPoolExecutor
        _pool.append(ThreadPoolExecutor(THREADS, "her-poses"))
    return list(_pool[0].map(fn, items))


def _agree_flat(pl, W, H, ys, xs, cos):
    """The share of pixels agreeing (the last axis a pose's pixels), the better colour mode (the prototype's)."""
    inside = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    flat = np.clip(ys, 0, H - 1) * W + np.clip(xs, 0, W - 1)
    a = [p.take(flat) for p in pl]
    best = None
    for (c0, c1, c2), t in cos:
        d = np.maximum(np.maximum(np.abs(a[0] - c0), np.abs(a[1] - c1)), np.abs(a[2] - c2))
        sc = ((d <= t) & inside).mean(axis=-1)
        best = sc if best is None else np.maximum(best, sc)
    return best


class BodyFinder:
    """Her body's search (`find`) with a bank of her poses; `background` the map's art behind her (`Background`: the
    mass path's foreground; None: every pixel foreground). `feet_x`: the screen columns her feet can be in (the map's
    platforms' ends; None: anywhere)."""

    def __init__(self, bank: PoseBank, background: Background | None = None,
                 feet_x: tuple[int, int] | None = None):
        self.bank, self.background, self.feet_x = bank, background, feet_x
        self.poses = bank.poses
        self.anchors = bank.anchors
        self.by_anchor: dict[int, list[tuple[int, int, int]]] = {}   # anchor -> [(pose, ax, ay)]
        for i, p in enumerate(self.poses):
            for k, ax, ay in p.anchors:
                self.by_anchor.setdefault(k, []).append((i, ax, ay))
        self._lut()
        self._groups()
        # the pixel sets: every pose's SUBSET px, its SCAN px (the first of those) and all of them
        P = self.poses
        self.sub = _Sets([(p.ys[p.subset], p.xs[p.subset], p.colour[p.subset]) for p in P])
        scan = [np.resize(p.subset, SCAN) for p in P]
        self.scan = _Sets([(p.ys[s], p.xs[s], p.colour[s]) for p, s in zip(P, scan)])
        self.full = _Sets([(p.ys, p.xs, p.colour) for p in P])
        self.cen = np.array([p.centre for p in P], np.float32)
        self.paths: list[str] = []

    def _lut(self):
        """Her colours (64 levels a channel), as drawn and darkened, from poses seen at 5 places or more."""
        lut = np.zeros(64 * 64 * 64, np.uint8)
        cols = np.concatenate([p.colour for p in self.poses if not p.mirrored and p.places >= 5]
                              or [np.zeros((0, 3), np.float32)])
        for k in LUT_GAINS:
            c = np.clip(np.round(cols * k), 0, 255).astype(np.int64) >> 2
            lut[(c[:, 0] << 12) | (c[:, 1] << 6) | c[:, 2]] = 1
        self.lut = lut

    def her_colours(self, rgb: np.ndarray) -> np.ndarray:
        q = (rgb >> 2).astype(np.int32)
        return self.lut[(q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]]

    def _groups(self):
        """The lean groups' templates: each group's masks summed with each pose's mass centre at one place, the mean,
        its parts under 0.4 dropped."""
        g: dict[int, list[int]] = {}
        for i, p in enumerate(self.poses):
            cx = p.centre[0]
            g.setdefault(0 if abs(cx) <= LEAN else (1 if cx < 0 else 2), []).append(i)
        self.groups = []
        for key, members in sorted(g.items()):
            CX, CY = 100, 100
            T = np.zeros((220, 220), np.float32)
            for i in members:
                p = self.poses[i]
                ox = int(round(CX + (p.dx0 + 0.5) - p.centre[0]))
                oy = int(round(CY + p.dy0 - p.centre[1]))
                T[oy:oy + p.h, ox:ox + p.w] += p.mask
            T /= len(members)
            T[T < 0.4] = 0
            ys, xs = np.nonzero(T)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            self.groups.append({"members": members, "T": np.ascontiguousarray(T[y0:y1, x0:x1]), "cx": CX - x0,
                                "cy": CY - y0})

    # --- the search ---------------------------------------------------------------------------------------------------
    def find(self, frame: np.ndarray, box: tuple[float, float, float, float], view: tuple[int, int] | None = None,
             pred: tuple[float, float] | None = None, slack: float = 6.0) -> BodyHit | None:
        """Her best pose with her feet in `box` (x lo, x hi, y lo, y hi; screen) of an RGB frame, or None. `view`: the
        view's top-left on the map (the background's), `pred`: where her feet are predicted (the motion's weight,
        `slack` px free). `paths` says how the last search went ("head", "mass(h=...)"): one finder can serve
        several threads but for that."""
        H, W = frame.shape[:2]
        hud = int(round(H * NATIVE.hud_top))
        fx0, fx1, fy0, fy1 = box
        if self.feet_x is not None:
            fx0, fx1 = max(self.feet_x[0], fx0), min(self.feet_x[1], fx1)
        fy0, fy1 = max(FEET_ROWS[0], fy0), min(hud + FEET_ROWS[1], fy1)
        self.paths = []
        if fx1 < fx0 or fy1 < fy0:
            return None
        X0, X1 = max(0, int(fx0 - WINDOW[0])), min(W, int(fx1 + WINDOW[1]))
        Y0, Y1 = max(0, int(fy0 - WINDOW[2])), min(H, int(fy1 + WINDOW[3]))
        if X1 - X0 < MIN_WINDOW or Y1 - Y0 < MIN_WINDOW:
            return None
        RX0, RY0 = max(0, X0 - REACH[0]), max(0, Y0 - REACH[1])
        RX1, RY1 = min(W, X1 + REACH[0]), min(H, Y1 + REACH[1])
        lib = native.locate_lib() if USE_C and frame.dtype == np.uint8 else None
        pic = _Picture(frame, RX0, RY0, RX1, RY1, lib)
        # 1. her anchors
        g = cv2.cvtColor(np.ascontiguousarray(frame[Y0:Y1, X0:X1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        hits = self.anchor_hits(g, g * g, X0, Y0)
        which, fx, fy = [], [], []
        for mn, k, hx, hy in hits:
            for i, ax, ay in self.by_anchor.get(k, ()):
                which.append(i)
                fx.append(hx - ax)
                fy.append(hy - ay)
        best = self._verify(pic, which, fx, fy, box, pred, slack)
        if best is not None and best.agree >= HEAD_OK:
            self.paths.append("head")
            return best
        self.paths.append("mass(h=%s)" % ("none" if best is None else "%.2f" % best.agree))
        # 2. her colours' mass
        sc = self.mass_picture(frame, X0, Y0, X1, Y1, view, lib)

        def group_peaks(gi):
            G = self.groups[gi]
            T = G["T"]
            if T.shape[0] > sc.shape[0] or T.shape[1] > sc.shape[1]:
                return []
            res = cv2.matchTemplate(sc, T, cv2.TM_CCORR)
            res /= max(1.0, T.sum())
            out = []
            for _ in range(MASS_PEAKS):
                _, mx, _, (lx, ly) = cv2.minMaxLoc(res)
                if mx <= MASS_LEAST:
                    break
                out.append((mx, gi, X0 + lx + G["cx"], Y0 + ly + G["cy"]))
                res[max(0, ly - MASS_APART):ly + MASS_APART + 1, max(0, lx - MASS_APART):lx + MASS_APART + 1] = -1
            return out

        cands = [c for part in _each(group_peaks, range(len(self.groups))) for c in part]
        cands.sort(key=lambda c: -c[0])
        centres = []
        for mx, gi, cxm, cym in cands:
            if all(abs(cxm - c[0]) > 3 or abs(cym - c[1]) > 3 for c in centres):
                centres.append((cxm, cym, mx, gi))
            if len(centres) >= CENTRES:
                break
        fine = []
        P = len(self.poses)
        allp = np.arange(P, dtype=np.int32)
        oy, ox = np.mgrid[-FINE_R:FINE_R + 1, -FINE_R:FINE_R + 1]
        oy, ox = oy.ravel(), ox.ravel()
        for cxm, cym, mx, gi in centres:
            fxi = np.floor((cxm - RX0) - self.cen[:, 0]).astype(np.int32)
            fyi = np.round((cym - RY0) - self.cen[:, 1]).astype(np.int32)
            s = pic.agree(self.scan, allp, fxi, fyi, FINE_R)          # [P, offsets]
            FX = fxi[:, None] + ox[None, :]
            FY = fyi[:, None] + oy[None, :]
            for f in np.argsort(-s.ravel())[:6]:
                pi, j = np.unravel_index(int(f), s.shape)
                fine.append((float(s[pi, j]), int(pi), int(FX[pi, j]) + RX0, int(FY[pi, j]) + RY0))
        fine = sorted(fine, key=lambda c: -c[0])[:24]
        b2 = self._verify(pic, [c[1] for c in fine], [c[2] for c in fine], [c[3] for c in fine], box, pred, slack)
        if b2 is not None and (best is None or b2.agree > best.agree):
            best = b2
        return best

    def anchor_hits(self, g: np.ndarray, g2: np.ndarray, X0: int, Y0: int) -> list[tuple[float, int, int, int]]:
        """Her anchors' best places in the gray window g (g2 = g squared): the masked squared difference from two
        plain correlations (sum M I^2 - 2 sum (M T) I + sum M T^2), as drawn and as the blink darkens it (I x 2),
        the better of the two; up to HEAD_PLACES an anchor under HEAD_SQ: (mean square, anchor, x, y)."""
        def one(k):
            a = self.anchors[k]
            if a.gray.shape[0] > g.shape[0] or a.gray.shape[1] > g.shape[1]:
                return []
            A = cv2.matchTemplate(g2, a.mf, cv2.TM_CCORR)
            B = cv2.matchTemplate(g, a.mt, cv2.TM_CCORR)
            r = np.minimum(A - 2 * B, 4 * A - 4 * B)
            r += a.tt
            r /= a.n
            out = []
            for _ in range(HEAD_PLACES):
                mn, _, (lx, ly), _ = cv2.minMaxLoc(r)
                if mn > HEAD_SQ:
                    break
                out.append((mn, k, X0 + lx, Y0 + ly))
                r[max(0, ly - 6):ly + 7, max(0, lx - 6):lx + 7] = 1e9
            return out

        return [h for part in _each(one, range(len(self.anchors))) for h in part]

    def mass_picture(self, frame: np.ndarray, X0: int, Y0: int, X1: int, Y1: int,
                     view: tuple[int, int] | None, lib=None) -> np.ndarray:
        """The window's picture for the mass's correlation: +1 her colour in the foreground, -0.5 other foreground, 0
        the map's art; without the art (no background or view) +1 her colour, -0.25 the rest. In C with `lib`
        (lc_mass: the same values)."""
        win = frame[Y0:Y1, X0:X1]
        art = view is not None and self.background is not None
        if art:
            bgw, ok, _ = self.background.screen(view, X0, Y0, X1, Y1)
        if lib is not None and win.strides[1] == 3 and win.strides[2] == 1:
            out = np.empty(win.shape[:2], np.float32)
            if art:
                known = np.ascontiguousarray(ok, np.uint8)
                done = lib.lc_mass(win.ctypes.data, win.shape[0], win.shape[1], win.strides[0], bgw.ctypes.data,
                                   bgw.strides[0] // 2, known.ctypes.data, known.strides[0], self.lut.ctypes.data, FG,
                                   out.ctypes.data)
            else:
                done = lib.lc_mass(win.ctypes.data, win.shape[0], win.shape[1], win.strides[0], None, 0, None, 0,
                                   self.lut.ctypes.data, FG, out.ctypes.data)
            if done == 0:
                return out
        her = self.her_colours(win).astype(np.float32)
        if art:
            fg = (np.abs(win.astype(np.int16) - bgw).max(axis=2) > FG) | ~ok
            return np.where(fg, np.where(her > 0, FG_SCORE[0], FG_SCORE[1]), 0.0).astype(np.float32)
        return np.where(her > 0, FG_SCORE[0], FG_SCORE[2]).astype(np.float32)

    def _verify(self, pic: _Picture, which, fx, fy, box, pred, slack) -> BodyHit | None:
        """Candidates (pose, feet column, feet row; screen px) with their feet in the box (4 px of slack): each one's
        SUBSET px +-1 (the best offset kept), the best VERIFY checked with all their pixels; the best, the motion
        weighed in with a prediction."""
        bx0, bx1, by0, by1 = box
        keep, seen = [], set()
        for i, x, y in zip(which, fx, fy):
            if bx0 - 4 <= x + 0.5 <= bx1 + 4 and by0 - 4 <= y <= by1 + 4:
                key = (self.poses[i].id, int(x), int(y))
                if key not in seen:   # (the first of each, in order)
                    seen.add(key)
                    keep.append((i, int(x), int(y)))
        if not keep:
            return None
        w = np.array([c[0] for c in keep], np.int32)
        X = np.array([c[1] for c in keep], np.int32) - pic.x0
        Y = np.array([c[2] for c in keep], np.int32) - pic.y0
        r = pic.agree(self.sub, w, X, Y, 1)                           # [C, 9]
        j = r.argmax(axis=1)
        oy, ox = np.mgrid[-1:2, -1:2]
        oy, ox = oy.ravel(), ox.ravel()
        fine = [(float(r[c, j[c]]), keep[c][0], keep[c][1] + int(ox[j[c]]), keep[c][2] + int(oy[j[c]]))
                for c in range(len(keep))]
        fine.sort(key=lambda c: -c[0])
        done, seen = [], set()
        top = []
        for s0, i, x, y in fine[:VERIFY]:
            if (self.poses[i].id, x, y) in seen:
                continue
            seen.add((self.poses[i].id, x, y))
            top.append((i, x, y))
        r = pic.agree(self.full, np.array([t[0] for t in top], np.int32),
                      np.array([t[1] for t in top], np.int32) - pic.x0,
                      np.array([t[2] for t in top], np.int32) - pic.y0, 0)[:, 0]
        done = [BodyHit(float(r[c]), self.poses[i], x + 0.5, float(y)) for c, (i, x, y) in enumerate(top)]
        if not done:
            return None
        if pred is not None:
            px, py = pred
            return max(done, key=lambda d: d.agree - MOTION_W * max(0.0, math.hypot(d.fx - px, d.fy - py) - slack))
        return max(done, key=lambda d: d.agree)


def tracked_box(feet: tuple[float, float], sx: float, sy: float, grounded: bool, speed: float,
                rx_least: int = 12, up_ground: int = 40) -> tuple[tuple[float, float, float, float], float]:
    """The box her feet may be in while tracked (the combined localizer's): from the filter's sd across and down, tight
    across (a cast's twin origin 25 px away left out), reaching up_ground higher over her platform (a jump's first
    frame, ~35) or 12 in the air; and the motion's slack. -> ((x lo, x hi, y lo, y hi), slack)."""
    rx = int(np.clip(3 * sx + 10, rx_least, 90))
    dn = int(np.clip(3 * sy + 12, 24, 90))
    up = dn + (up_ground if grounded else 12)
    return (feet[0] - rx, feet[0] + rx, feet[1] - up, feet[1] + dn), 6.0 + 1.5 * speed * 0.066


# --- the bank, made at calibration --------------------------------------------------------------------------------
# From frames where her tag is sure (her feet: its centre + the plate model's offset), the view each was seen from and
# the map's art behind her (the prototype's camera-sprite bank2 and bank3, and its anchors, in one): her part of each
# frame, the foreground's parts touching the column over her feet (`instance`); frames of one pose together (`_same`:
# their masks overlapping, one colour within a gain); each pose's core, the pixels in her mask at least half the time
# with one colour in most of them; then her own palette (the colours of the poses seen at 5 places or more, in 3 of
# them: what touches her, her pet, a monster, an effect, varies), each pose cut to it and to its large parts, the equal
# ones merged, the mirror images made for her other facing, those she couldn't have (a flash's white, one seen only
# darkened) left out, and her anchors placed in each.
BUILD_WINDOW = (-60, 61, -100, 5)   # the window a frame gives around her feet: x from fx-60.5, y from fy-100
CORE_BAND = (-60, -4, 12)           # her core's rows from her feet and its half-width: what touches it is her
FG_LEAST = 14                       # the foreground: this far from the art on a channel (or twice its noise + 6)
SAME_POSE, SURE_POSE = 0.97, 0.995  # two frames are one pose: this share of their overlap one colour (a gain allowed)
OVERLAP = 0.85                      # ... their masks overlapping this much of the smaller
CORE_SHARE, CORE_COLOUR = 0.5, 0.7  # a pose's core: in her mask this share of its frames, one colour in this share
BRIGHT = 0.85                       # its frames no darker than this share of its brightest (the blink's left out)
PALETTE_PLACES, PALETTE_POSES = 5, 3
MIN_PART, MIN_CORE = 40, 600        # a core's parts kept: its largest and those this big; a core at least this big
MERGE = (0.9, 0.98)                 # equal poses (and a mirror equal to one seen): overlap, colour agreement
PLAUSIBLE = (60, 140, 800, 2)       # kept: her mean colour within these, this many px, seen at this many places
MIN_SEEN = 2                        # ... and in this many frames (a mirrored one: its source)
LENS = (120, 40, 20)                # her glasses' lenses: blue over this, over red by this, red and green this close


def _pack(px: np.ndarray) -> np.ndarray:
    p = px.astype(np.int64)
    return (p[..., 0] << 16) | (p[..., 1] << 8) | p[..., 2]


def instance(frame: np.ndarray, background: Background, view: tuple[int, int], fx: float, fy: float):
    """Her part of a frame with her feet at (fx, fy): (the window's pixels, her rough mask, its top-left from her feet),
    or None where the window leaves the frame or the art (the HUD's top down is the HUD's)."""
    hud = int(round(frame.shape[0] * NATIVE.hud_top))
    X0, Y0 = int(round(fx + BUILD_WINDOW[0] - 0.5)), int(fy + BUILD_WINDOW[2])
    X1, Y1 = X0 + BUILD_WINDOW[1] - BUILD_WINDOW[0], Y0 + BUILD_WINDOW[3] - BUILD_WINDOW[2]
    if X0 < 0 or Y0 < 0 or X1 > frame.shape[1] or Y1 > hud:
        return None
    bg, known, noise = background.screen(view, X0, Y0, X1, Y1)
    if not known.all():
        return None
    f = frame[Y0:Y1, X0:X1].astype(np.int16)
    d = np.abs(f - bg).max(axis=2)
    least = FG_LEAST if noise is None else np.maximum(FG_LEAST, 2 * noise.astype(np.int16) + 6)
    m = (d > least) & known
    mm = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
    cx, cyf = int(round(fx - 0.5)) - X0, int(fy) - Y0
    core = lab[cyf + CORE_BAND[0]:cyf + CORE_BAND[1], cx - CORE_BAND[2]:cx + CORE_BAND[2] + 1]
    keep = [int(k) for k in np.unique(core) if k]
    if not keep:
        return None
    return frame[Y0:Y1, X0:X1].copy(), np.isin(lab, keep) & m, (X0 - fx, Y0 - fy)


def _same(A, mA, Bp, mB, tol=6):
    """Two frames' windows as one pose: (the share of their masks' overlap one colour, a gain allowed; the gain), or
    (0, 1) where they overlap under OVERLAP of the smaller."""
    I = mA & mB
    if int(I.sum()) < OVERLAP * min(mA.sum(), mB.sum()):
        return 0.0, 1.0
    a = A[I].astype(np.float32)
    b = Bp[I].astype(np.float32)
    k = float((a * b).sum() / max(1.0, (a * a).sum()))
    return float((np.abs(b - k * a).max(axis=1) <= tol).mean()), k


def _core(inst):
    """inst: [(px, mask, map place, brightness)] -> (its picture, core mask, places, the core's box) or None."""
    top = max(x[3] for x in inst)
    bri = [x for x in inst if x[3] >= BRIGHT * top]
    fg_share = np.stack([x[1] for x in inst]).mean(axis=0)
    stack = np.stack([x[0] for x in bri]).astype(np.int16)
    rep = np.median(stack, axis=0).astype(np.int16)
    mb = np.stack([x[1] for x in bri])
    same = (np.abs(stack - rep[None]).max(axis=3) <= 6) & mb
    cons = same.sum(0) / np.maximum(1, mb.sum(0))
    places = len(set((round(x[2][0]), round(x[2][1])) for x in inst))
    if places >= 2 and len(bri) >= 2:
        core = (fg_share >= CORE_SHARE) & (cons >= CORE_COLOUR)
    else:
        core = fg_share >= CORE_SHARE
    ys, xs = np.nonzero(core)
    if len(xs) == 0:
        return None
    return rep.astype(np.uint8), core, places, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _overlap_agree(a, b, tol=6):
    """Two poses' cores compared where they overlap (as drawn from her feet): (the overlap's share of the smaller,
    the share of it within tol)."""
    ax0, ay0, bx0, by0 = a["dx0"], a["dy0"], b["dx0"], b["dy0"]
    X0, Y0 = max(ax0, bx0), max(ay0, by0)
    X1 = min(ax0 + a["px"].shape[1], bx0 + b["px"].shape[1])
    Y1 = min(ay0 + a["px"].shape[0], by0 + b["px"].shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return 0.0, 0.0
    ia = (slice(int(Y0 - ay0), int(Y1 - ay0)), slice(int(X0 - ax0), int(X1 - ax0)))
    ib = (slice(int(Y0 - by0), int(Y1 - by0)), slice(int(X0 - bx0), int(X1 - bx0)))
    I = a["mask"][ia] & b["mask"][ib]
    ni = I.sum()
    if ni == 0:
        return 0.0, 0.0
    d = np.abs(a["px"][ia][I].astype(int) - b["px"][ib][I].astype(int)).max(axis=1)
    return ni / min(a["mask"].sum(), b["mask"].sum()), float((d <= tol).mean())


def _clean(e, pal):
    """A pose cut to her palette and its large parts: its core's largest part and those MIN_PART or more."""
    core = e["mask"] & np.isin(_pack(e["px"]), pal)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    big = max(range(1, n), key=lambda k: st[k, 4])
    keep = np.zeros(n, bool)
    for k in range(1, n):
        keep[k] = k == big or st[k, 4] >= MIN_PART
    core = keep[lab]
    ys, xs = np.nonzero(core)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    out = dict(e)
    out.update(px=e["px"][y0:y1, x0:x1].copy(), mask=core[y0:y1, x0:x1].copy(), dx0=e["dx0"] + x0, dy0=e["dy0"] + y0)
    return out


def _mirror(e):
    w = e["px"].shape[1]
    return dict(e, px=e["px"][:, ::-1].copy(), mask=e["mask"][:, ::-1].copy(), dx0=-e["dx0"] - w + 1,
                facing="left" if e["facing"] == "right" else "right", mirrored=True, n_src=e["n"], n=0)


def build_bank(samples, background: Background, on_line=None, log=None) -> PoseBank:
    """Her pose bank from (frame (RGB), her feet (fx, fy) from her sure tag, the view's top-left on the map) samples,
    in time order; `background`: the map's art (`Background`); `on_line(map x, map y)`: whether her feet are on a
    platform's line there (her poses' `grounded`; None: unknown)."""
    say = log or (lambda s: None)
    poses, stats = [], {"frames": 0, "no instance": 0}
    for frame, (fx, fy), view in samples:
        stats["frames"] += 1
        c = instance(frame, background, view, fx, fy)
        if c is None:
            stats["no instance"] += 1
            continue
        px, hm, off = c
        mxy = (fx + view[0], fy + view[1])
        bright = float(px[hm].mean()) if hm.any() else 0.0
        best = None
        for p in poses:
            a, _ = _same(p["px"], p["mask"], px, hm)
            if a >= SAME_POSE and (best is None or a > best[0]):
                best = (a, p)
                if a >= SURE_POSE:
                    break
        if best is None:
            poses.append({"px": px, "mask": hm, "inst": [], "ground": []})
            p = poses[-1]
        else:
            p = best[1]
        p["inst"].append((px, hm, mxy, bright))
        p["ground"].append(None if on_line is None else bool(on_line(*mxy)))
    say(f"poses from {stats['frames']} frames: {len(poses)} ({stats['no instance']} frames gave none)")
    seen = []
    for p in poses:
        c = _core(p["inst"])
        if c is None:
            continue
        rep, core, places, (bx0, by0, bx1, by1) = c
        g = [x for x in p["ground"] if x is not None]
        e = {"px": rep[by0:by1, bx0:bx1].copy(), "mask": core[by0:by1, bx0:bx1].copy(),
             "dx0": float(BUILD_WINDOW[0] - 0.5 + bx0), "dy0": int(BUILD_WINDOW[2] + by0), "n": len(p["inst"]),
             "places": places, "grounded": float(np.mean(g)) if g else None, "mirrored": False, "n_src": 0}
        e["facing"] = "right" if e["dx0"] + e["px"].shape[1] / 2 > 0 else "left"
        seen.append(e)
    seen.sort(key=lambda e: -e["n"])
    # her palette: the colours of the poses seen at PALETTE_PLACES places or more, in PALETTE_POSES of them
    count: dict[int, int] = {}
    for e in seen:
        if e["places"] >= PALETTE_PLACES:
            for c in set(_pack(e["px"][e["mask"]]).tolist()):
                count[c] = count.get(c, 0) + 1
    pal = np.array(sorted(c for c, k in count.items() if k >= PALETTE_POSES), np.int64)
    cleaned = [c for c in (_clean(e, pal) for e in seen) if c is not None and c["mask"].sum() >= MIN_CORE]
    cleaned.sort(key=lambda e: -e["n"])
    merged = []
    for e in cleaned:
        for m in merged:
            ov, ag = _overlap_agree(m, e)
            if ov >= MERGE[0] and ag >= MERGE[1]:
                m["n"] += e["n"]
                m["places"] += e["places"]
                break
        else:
            merged.append(dict(e))
    mirrors = []
    for e in merged:
        m = _mirror(e)
        if not any(o["facing"] == m["facing"] and all(v >= t for v, t in zip(_overlap_agree(o, m), MERGE))
                   for o in merged):
            mirrors.append(m)
    every = merged + mirrors
    say(f"her palette {len(pal)} colours; poses {len(seen)}, cleaned {len(cleaned)}, merged {len(merged)}, "
        f"mirrors {len(mirrors)}")

    def plausible(e):   # her own colours: not a flash's white, not a pose seen only darkened by the blink
        m = float(e["px"][e["mask"]].mean())
        return PLAUSIBLE[0] <= m <= PLAUSIBLE[1] and e["mask"].sum() >= PLAUSIBLE[2] and e["places"] >= PLAUSIBLE[3]

    kept = [Pose(i, np.ascontiguousarray(e["px"]), e["mask"], float(e["dx0"]), int(e["dy0"]), int(e["n"]),
                 int(e.get("n_src", 0)), int(e["places"]), e["facing"], bool(e["mirrored"]), e["grounded"])
            for i, e in enumerate(every)
            if plausible(e) and (e["n_src"] if e["mirrored"] else e["n"]) >= MIN_SEEN]
    anchors = place_anchors(kept)
    say(f"kept {len(kept)} poses ({sum(p.mirrored for p in kept)} mirrored); anchors "
        f"{[a.name for a in anchors]}, held by {sum(bool(p.anchors) for p in kept)} poses")
    return PoseBank(kept, anchors, {"frames": stats["frames"], "instances": stats["frames"] - stats["no instance"],
                                    "palette": len(pal), "poses": len(kept)})


def place_anchors(poses: list[Pose]) -> list[Anchor]:
    """Her anchors (the prototype's make_heads): her glasses (the commonest pose's purple lenses and their rim: the same
    picture standing, walking, jumping, knocked back and casting, where her cap is drawn differently), and the top
    ANCHOR_ROWS of her commonest narrow pose off the ground (her back, on a rope), each as drawn and mirrored; each
    pose's `anchors` set to where each sits in it (ANCHOR_OK of its pixels there in its colours)."""
    anchors: list[Anchor] = []

    def add(px, mk, name):
        for flip in (False, True):
            anchors.append(Anchor(np.ascontiguousarray(px[:, ::-1] if flip else px),
                                  np.ascontiguousarray(mk[:, ::-1] if flip else mk), name + (" mirrored" if flip else "")))

    seen = [p for p in poses if not p.mirrored]
    if not seen:
        return anchors
    p0 = max(seen, key=lambda p: p.n)
    px = p0.picture.astype(int)
    lens = p0.mask & (px[..., 2] > LENS[0]) & (px[..., 2] > px[..., 0] + LENS[1]) & \
        (np.abs(px[..., 0] - px[..., 1]) < LENS[2])
    if lens.any():
        ys, xs = np.nonzero(lens)
        y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
        box = px[y0:y1, x0:x1]
        ln = lens[y0:y1, x0:x1]
        hull = cv2.dilate(cv2.morphologyEx(ln.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
                          np.ones((3, 3), np.uint8)).astype(bool)
        gm = p0.mask[y0:y1, x0:x1] & hull & (ln | (box.min(axis=2) >= 200) | (box.max(axis=2) <= 60))
        n, lab, st, _ = cv2.connectedComponentsWithStats(gm.astype(np.uint8), connectivity=8)
        gm = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))   # the lens: not her eye's glint beside it
        add(p0.picture[y0:y1, x0:x1], gm, "glasses")
    rope = [p for p in seen if p.w <= 48 and (p.grounded or 0) < 0.3]
    if rope:
        r = max(rope, key=lambda p: p.n)
        px, mk = r.picture[:ANCHOR_ROWS], r.mask[:ANCHOR_ROWS]
        cols = np.nonzero(mk.any(axis=0))[0]
        add(px[:, cols.min():cols.max() + 1], mk[:, cols.min():cols.max() + 1], "rope top")
    for p in poses:
        p.anchors = []
        for k, a in enumerate(anchors):
            b = _place(p, a)
            if b is not None and b[0] >= ANCHOR_OK:
                p.anchors.append((k, int(round(p.dx0 + 0.5)) + b[1], p.dy0 + b[2]))
    return anchors


def _place(p: Pose, a: Anchor):
    """Where anchor `a` sits best in pose p: (the share of its pixels there in its colours, x, y of its top-left in the
    pose's box), over its top-left from (-3, -2) to 3 past the box's far side, where 0.6 of it overlaps the pose; the
    first best row by row (the prototype's place(), every place at once)."""
    hh, hw = a.mask.shape
    pad_y, pad_x = hh + 3, hw + 4
    P = np.zeros((p.h + 2 * pad_y, p.w + 2 * pad_x, 3), np.int16)
    M = np.zeros((p.h + 2 * pad_y, p.w + 2 * pad_x), bool)
    P[pad_y:pad_y + p.h, pad_x:pad_x + p.w] = p.picture
    M[pad_y:pad_y + p.h, pad_x:pad_x + p.w] = p.mask
    ys = np.arange(-2, p.h - hh + 3)
    xs = np.arange(-3, p.w - hw + 4)
    if not len(ys) or not len(xs):
        return None
    win_p = np.lib.stride_tricks.sliding_window_view(P, (hh, hw), axis=(0, 1))   # [Y, X, 3, hh, hw]
    win_m = np.lib.stride_tricks.sliding_window_view(M, (hh, hw), axis=(0, 1))
    wp = win_p[pad_y + ys[0]:pad_y + ys[-1] + 1, pad_x + xs[0]:pad_x + xs[-1] + 1]
    wm = win_m[pad_y + ys[0]:pad_y + ys[-1] + 1, pad_x + xs[0]:pad_x + xs[-1] + 1]
    pm = wm & a.mask
    d = np.abs(wp - a.picture.astype(np.int16).transpose(2, 0, 1)[None, None]).max(axis=2)
    share = ((d <= TOL) & pm).sum(axis=(2, 3)) / a.n
    share[pm.sum(axis=(2, 3)) < 0.6 * a.n] = -1.0
    j = int(np.argmax(share))
    iy, ix = np.unravel_index(j, share.shape)
    if share[iy, ix] < 0:
        return None
    return float(share[iy, ix]), int(xs[ix]), int(ys[iy])
