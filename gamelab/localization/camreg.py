"""The camera's view from the scene: each frame's static art registered against the map's own, drawn from the game data
(localization.mapart), so the view is known exactly whether or not her name tag shows, and her place on the map is her
place on screen plus it (screen = map - view at 1366x768, a map unit a px).

`Registration.see(frame)` gives the view of each frame (`View`: the map point at the frame's top-left):
- The frame's sample rows (every ROW_EVERY-th row above the HUD, all columns) as gray, avg(avg(r, b), g) with avg(a, b)
  = (a + b + 1) >> 1 (about (r + 2g + b) / 4: the same for RGB and BGR), against the art's gray at each candidate view:
  the mean of min(|frame - art|, TRUNC) over the pixels in play (not the HUD, the minimap, the quest helper: `exclude`)
  where the map's fixed art alone is drawn (mapart's `static`). What isn't the map's (her, monsters, players, names,
  effects, drops, chat lines, the clouds some maps draw over themselves) costs TRUNC a pixel at most. At the true view
  the art is the frame to the level wherever nothing's drawn over it (Tree Dungeon: 99.98% of those pixels within 2
  levels), so its cost is about TRUNC x the share covered (the clips: 0.5-1.4), and a px off it's ~3 more.
- Searched FIRST px about the view predicted (the last sure one moved on as it last moved, by time with frames' times:
  on the clips it was there or 1-3 px off), then NEAR, then WIDE, then every view the camera can take (`cold`: the first
  frame, or after one it couldn't place; on a map the camera moves across too, within HINT_R of `hint`, her dot's view,
  where given). Where the camera can't move across (or up and down), that way isn't searched (mapart's `fixed`).
- A view is `sure` with its cost at most COST_OK and the best view 2 px or more from it MARGIN_OK dearer (the margin:
  the clips 2.9-3.3; half the play area painted over 0.9-1.7; a capture shifted off the client, a black frame or another
  map's 0.004-0.06), else the view is carried (the prediction, `sure` False, how "carried") and the next frame searches
  as far as it must.

The costs in C where it's built (camreg.c, this module's `lib`: AVX2, the same integers as the numpy here, which runs
without it: a frame 0.2-0.4 ms against 10-20).

    python -m gamelab.localization.camreg MAP_ID PICTURE.png [...]    each picture's view, cost and margin (cold)
"""

import ctypes
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import native
from .mapart import MapArt

SOURCE = Path(native.__file__).with_name("camreg.c")
TRUNC = 24          # gray levels: the most a pixel's difference costs
ROW_EVERY = 6       # every this many rows of the frame compared (from ROW_EVERY // 2)
HUD_TOP = 700       # the chat box's top at 1366x768 (screen.HUD_TOP_GAME): rows from here aren't the map
# Screen boxes (x0, y0, x1, y1) of the game's own windows at 1366x768, left out: the minimap's (top left; Tree Dungeon's
# 240 x 197) and the quest helper's (top right).
EXCLUDE = ((0, 0, 250, 210), (1130, 40, 1366, 78))
FIRST = 3           # px about the predicted view searched first (a sure view inside it, not on its edge: taken) ...
NEAR = 12           # ... then this far (Tree Dungeon's clips at 15 frames a second: the view moved at most 34 px a
                    # frame, at most 8 from where its last move put it) ...
WIDE = 48           # ... then this far
SETTLE = 3          # steps a search's cheapest view may go down to its neighbours' (Registration._settled)
COARSE = 4          # a cold search's first pass: the frame and the art in this many px' blocks, every 2nd row of them
COLD_BEST = 4       # ... its cheapest this many views (2 blocks or more apart) again at full size ...
COLD_EVERY = 4      # ... over every this many of the sample rows
COST_OK = 16.0      # the most a sure view costs (two thirds of TRUNC: Tree Dungeon's clips 0.5-1.4, half the play area
                    # painted over 12-13, West Rocky Mountain II's frames 5.3-5.6 under the clouds it draws over itself)
MARGIN_OK = 0.6     # the least the best view 2 px or more from it costs more (above: 10 times the bad frames' most)
HINT_R = 64         # px about a hint searched on a map the camera moves across too (cold)

_lib: ctypes.CDLL | None = None
_tried = False
_lock = threading.Lock()
why = ""  # why the DLL isn't in use, if it isn't
_P, _I, _L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
SIGNATURES = {
    "cr_simd": (_I, [_I]),
    "cr_gray": (_I, [_P, _I, _I, _L, _I, _P, _I, _P]),
    "cr_costs": (_I, [_P, _P, _I, _I, _P, _P, _P, _I, _I, _L, _I, _I, _P, _P, _I, _I, _P, _P]),
    "cr_shrink": (_I, [_P, _I, _I, _L, _I, _P]),
}


def lib() -> ctypes.CDLL | None:
    """camreg.c's DLL, built (native.build) and loaded on first use; None if it can't be (`why` says why) or with
    native.USE off."""
    global _lib, _tried, why
    if not native.USE:
        why = "turned off (native.USE)"
        return None
    if _tried:
        return _lib
    with _lock:
        if _tried:
            return _lib
        try:
            dll = ctypes.CDLL(str(native.build(SOURCE)))
            for name, (restype, argtypes) in SIGNATURES.items():
                f = getattr(dll, name)
                f.restype, f.argtypes = restype, argtypes
            dll.cr_simd(1)
            _lib, why = dll, ""
        except (OSError, RuntimeError, subprocess.SubprocessError, AttributeError) as e:
            why = str(e).splitlines()[0] if str(e) else type(e).__name__
        _tried = True
    return _lib


def gray_of(rgb: np.ndarray) -> np.ndarray:
    """avg(avg(c0, c2), c1) of each pixel's first three channels, avg(a, b) = (a + b + 1) >> 1 (SSE2's pavgb: camreg.c's
    cr_gray), uint8: the same for RGB and BGR."""
    p = rgb[..., :3].astype(np.uint16)
    return ((((p[..., 0] + p[..., 2] + 1) >> 1) + p[..., 1] + 1) >> 1).astype(np.uint8)


@dataclass(frozen=True)
class View:
    """A frame's view: (x, y) the map point at its top-left (screen = map - (x, y) at 1366x768). `sure`: registered on
    this frame, `how` "near", "wide" or "cold" (where it was found); else "carried" (the view predicted from the last
    sure one) or "none" (none placed yet: (0, 0)). `cost`, `margin`: the registration's (Registration's top), of the
    best view found even where not sure; `share`: of the frame's sampled pixels, those compared (the fixed art in
    play); `ms`: its time."""
    x: int
    y: int
    sure: bool
    how: str
    cost: float | None = None
    margin: float | None = None
    share: float = 0.0
    ms: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The map units on screen at 1366x768: x0, y0, x1, y1 (a scene's `view`)."""
        return self.x, self.y, self.x + 1366, self.y + 768


class Registration:
    """A map's camera, registered frame by frame against its art (the top). `use_c` False: numpy alone (tests); None:
    the DLL where it's built."""

    def __init__(self, art: MapArt, size: tuple[int, int] = (1366, 768), hud_top: int = HUD_TOP,
                 exclude=EXCLUDE, row_every: int = ROW_EVERY, use_c: bool | None = None):
        self.art = art
        self.size = size
        self.gray = np.ascontiguousarray(gray_of(art.rgb))
        self.mask = np.ascontiguousarray(np.where(art.static, 255, 0).astype(np.uint8))
        w, h = size
        self.rows = np.arange(row_every // 2, min(hud_top, h), row_every, dtype=np.int32)
        fm = np.full((h, w), 255, np.uint8)
        fm[min(hud_top, h):] = 0
        for x0, y0, x1, y1 in exclude:
            fm[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
        self.frame_mask = np.ascontiguousarray(fm[self.rows])
        self.full_rows = np.arange(0, min(hud_top, h), dtype=np.int32)   # (a cold search's coarse pass)
        self.full_mask = fm[:min(hud_top, h)]
        self.full_small_mask = shrink(self.full_mask, self.full_mask[:(len(self.full_rows) // COARSE) * COARSE],
                                      COARSE)[1]
        self.small, self.small_mask = shrink(self.gray, self.mask, COARSE)
        self.in_play = int((self.frame_mask > 0).sum())
        self.cold_rows = np.arange(0, len(self.rows), COLD_EVERY)
        self.cold_mask = np.ascontiguousarray(self.frame_mask[self.cold_rows])
        vx0, vx1, vy0, vy1 = art.views
        self.range = (vx0, vx1, vy0, vy1)
        self.lib = lib() if use_c is None else (lib() if use_c else None)
        self.last: View | None = None     # the last view (sure or carried)
        self.sure_at: tuple[int, int, float | None] | None = None  # the last sure view and its time
        self.move = (0.0, 0.0)            # its last move (px a frame, or px a second with times)
        self.lost = 0                     # frames since the last sure one
        self.counts = {"near": 0, "wide": 0, "cold": 0, "carried": 0}
        self._cache: dict[tuple[int, int], tuple[float, int]] = {}  # this frame's views worked out (`_eval`)

    # --- the costs -----------------------------------------------------------------------------------------------

    def sample(self, frame: np.ndarray) -> np.ndarray:
        """The frame's sample rows as gray (len(rows) x width)."""
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"a {w}x{h} frame: registration is for {self.size[0]}x{self.size[1]} (a map unit a px)")
        if self.lib is not None and _plain(frame):
            out = np.empty((len(self.rows), w), np.uint8)
            if self.lib.cr_gray(frame.ctypes.data, h, w, frame.strides[0], frame.shape[2], self.rows.ctypes.data,
                                len(self.rows), out.ctypes.data) != 0:
                raise ValueError("a sample row outside the frame")
            return out
        return gray_of(frame[self.rows])

    def costs(self, g: np.ndarray, vxs, vys, cold: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Each candidate view's (vxs[i], vys[i]) summed truncated difference and pixels compared (the top), over the
        sample rows (`cold`: every COLD_EVERY-th of them)."""
        if cold:
            return self._costs(np.ascontiguousarray(g[self.cold_rows]), self.cold_mask, self.rows[self.cold_rows],
                               self.gray, self.mask, self.art.x0, self.art.y0, vxs, vys)
        return self._costs(g, self.frame_mask, self.rows, self.gray, self.mask, self.art.x0, self.art.y0, vxs, vys)

    # --- the search ----------------------------------------------------------------------------------------------

    def _clamp(self, vx: float, vy: float) -> tuple[int, int]:
        x0, x1, y0, y1 = self.range
        return int(min(max(round(vx), x0), x1)), int(min(max(round(vy), y0), y1))

    def _around(self, vx: int, vy: int, r: int) -> tuple[np.ndarray, np.ndarray]:
        """The views within r of (vx, vy) (square) the camera can take."""
        x0, x1, y0, y1 = self.range
        xs = np.arange(max(x0, vx - r), min(x1, vx + r) + 1)
        ys = np.arange(max(y0, vy - r), min(y1, vy + r) + 1)
        X, Y = np.meshgrid(xs, ys)
        return X.ravel(), Y.ravel()

    def _eval(self, g: np.ndarray, pts) -> None:
        """This frame's costs (over all the sample rows) of the views in `pts` the camera can take and that haven't been
        worked out yet (`_cache`: view -> (cost, pixels compared))."""
        x0, x1, y0, y1 = self.range
        new = [q for q in dict.fromkeys((int(a), int(b)) for a, b in pts)
               if q not in self._cache and x0 <= q[0] <= x1 and y0 <= q[1] <= y1]
        if not new:
            return
        sums, counts = self.costs(g, [q[0] for q in new], [q[1] for q in new])
        for q, sm, n in zip(new, sums, counts):
            self._cache[q] = (float(sm) / n if n > 0 else float("inf"), int(n))

    def _cheapest(self, pts) -> tuple[int, int]:
        """The cheapest of `pts` worked out (the first of equals)."""
        x, y = min((q for q in pts if q in self._cache), key=lambda q: self._cache[q][0])
        return int(x), int(y)

    def _settled(self, g: np.ndarray, x: int, y: int) -> tuple[int, int, float, float, float]:
        """From (x, y) down to a view none of whose neighbours (3x3, where the camera can be) costs less, each one's
        neighbours worked out first, SETTLE steps at most: the cheapest of a window can be its edge, a px short of the
        true view outside it, or (both ways) beside the true view where the window hasn't it. Then (x, y, cost, margin,
        share): the margin is the cheapest view worked out 2 px or more from it, less its cost (inf if none); a view
        still going down after SETTLE steps costs inf (not sure: the next, wider search goes on)."""
        x, y = int(x), int(y)
        for step in range(SETTLE + 1):
            around = [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            self._eval(g, around)
            best = self._cheapest(around)
            if best == (x, y) or self._cache[best][0] >= self._cache[(x, y)][0]:
                break
            if step == SETTLE:
                return x, y, float("inf"), 0.0, 0.0
            x, y = best
        cost, n = self._cache[(x, y)]
        far = [c for (qx, qy), (c, _) in self._cache.items() if max(abs(qx - x), abs(qy - y)) >= 2]
        margin = min(far) - cost if far else float("inf")
        return x, y, cost, margin, n / max(1, self.in_play)

    def search(self, g: np.ndarray, vx: int, vy: int, r: int) -> tuple[int, int, float, float, float]:
        """The best view within r of (vx, vy) (settled: `_settled`): (x, y, cost, margin, share). Where the camera
        moves both ways: within FIRST, the 3x3 about (vx, vy) and the cross through it out to r first, taken where the
        cheapest is in the 3x3; else by y at the given x, then x at that y, then 5x5 about that (a square's every view
        would cost r^2)."""
        x0, x1, y0, y1 = self.range
        if x0 == x1 or y0 == y1:
            pts = list(zip(*self._around(vx, vy, r)))
            self._eval(g, pts)
            return self._settled(g, *self._cheapest(pts))
        if r <= FIRST:  # the 3x3 about it and the cross's arms out to r: 17 views, not 39
            pts = [(x, y) for y in range(vy - 1, vy + 2) for x in range(vx - 1, vx + 2)]
            pts += [(vx + d, vy) for d in range(2, r + 1)] + [(vx - d, vy) for d in range(2, r + 1)]
            pts += [(vx, vy + d) for d in range(2, r + 1)] + [(vx, vy - d) for d in range(2, r + 1)]
            self._eval(g, pts)
            bx, by = self._cheapest(pts)
            if max(abs(bx - vx), abs(by - vy)) <= 1:
                return self._settled(g, bx, by)
        line = [(vx, y) for y in range(max(y0, vy - r), min(y1, vy + r) + 1)]
        self._eval(g, line)
        by = self._cheapest(line)[1]
        line = [(x, by) for x in range(max(x0, vx - r), min(x1, vx + r) + 1)]
        self._eval(g, line)
        bx = self._cheapest(line)[0]
        pts = list(zip(*self._around(bx, by, 2)))
        self._eval(g, pts)
        return self._settled(g, *self._cheapest(pts))

    def cold(self, frame: np.ndarray, g: np.ndarray, hint: tuple[float, float] | None = None
             ) -> tuple[int, int, float, float, float]:
        """The best view of all the camera can take (on a map it moves across too, within HINT_R of `hint` if given):
        first the frame and the art in COARSE x COARSE blocks' means (`shrink`), every view a block apart; then the
        COLD_BEST cheapest of those (each 2 blocks or more from the others) about which every view within COARSE + 1
        over every COLD_EVERY-th sample row; the cheapest of all then settled over all of them (`_settled`). Its
        margin: the least of its own and how much dearer the next-best pick's best is."""
        k = COARSE
        x0, x1, y0, y1 = self.range
        sg, sm = self._frame_small(frame)
        if hint is not None and x0 != x1 and y0 != y1:
            hx, hy = self._clamp(*hint)
            lo_x, hi_x, lo_y, hi_y = hx - HINT_R, hx + HINT_R, hy - HINT_R, hy + HINT_R
        else:
            lo_x, hi_x, lo_y, hi_y = x0, x1, y0, y1
        ax, ay = self.art.x0, self.art.y0
        sxs = np.arange(-(-(max(lo_x, x0) - ax) // k), (min(hi_x, x1) - ax) // k + 1)
        sys_ = np.arange(-(-(max(lo_y, y0) - ay) // k), (min(hi_y, y1) - ay) // k + 1)
        if not len(sxs):
            sxs = np.array([(x0 - ax) // k])
        if not len(sys_):
            sys_ = np.array([(y0 - ay) // k])
        X, Y = np.meshgrid(sxs, sys_)
        xs, ys = X.ravel(), Y.ravel()
        rows = np.arange(0, sg.shape[0], 2, dtype=np.int32)
        sums, counts = self._costs(np.ascontiguousarray(sg[rows]), np.ascontiguousarray(sm[rows]), rows, self.small,
                                   self.small_mask, 0, 0, xs, ys)
        cost = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
        picks: list[int] = []
        for i in np.argsort(cost, kind="stable"):
            if not np.isfinite(cost[i]) or len(picks) >= COLD_BEST:
                break
            if all(max(abs(int(xs[i]) - int(xs[j])), abs(int(ys[i]) - int(ys[j]))) >= 2 for j in picks):
                picks.append(int(i))
        found = []  # (cost over the cold rows, x, y) of each pick's best
        for i in picks:
            fx, fy = self._around(*self._clamp(ax + int(xs[i]) * k, ay + int(ys[i]) * k), k + 1)
            sums, counts = self.costs(g, fx, fy, cold=True)
            c = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
            j = int(np.argmin(c))
            found.append((float(c[j]), int(fx[j]), int(fy[j])))
        if not found:
            return x0, y0, float("inf"), 0.0, 0.0
        found.sort()
        best = self._settled(g, found[0][1], found[0][2])
        others = [f[0] - found[0][0] for f in found[1:] if max(abs(f[1] - best[0]), abs(f[2] - best[1])) >= 2]
        return best[0], best[1], best[2], min([best[3]] + others), best[4]

    def _frame_small(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The frame's play area (rows above the HUD) as gray in COARSE x COARSE blocks' means, and its mask (a block
        in play where all its px are)."""
        h = (len(self.full_rows) // COARSE) * COARSE
        rows = np.ascontiguousarray(self.full_rows[:h])
        if self.lib is not None and _plain(frame):
            g = np.empty((h, frame.shape[1]), np.uint8)
            self.lib.cr_gray(frame.ctypes.data, frame.shape[0], frame.shape[1], frame.strides[0], frame.shape[2],
                             rows.ctypes.data, h, g.ctypes.data)
        else:
            g = gray_of(frame[rows])
        return shrink(g, None, COARSE, self.lib)[0], self.full_small_mask

    def _costs(self, g, gm, rows, art, am, ax0, ay0, vxs, vys) -> tuple[np.ndarray, np.ndarray]:
        """cr_costs (or costs_np) of candidates over given rows of a frame against a given art."""
        vxs = np.ascontiguousarray(vxs, np.int32)
        vys = np.ascontiguousarray(vys, np.int32)
        rows = np.ascontiguousarray(rows, np.int32)
        n = len(vxs)
        if self.lib is None:
            return costs_np(g, gm, rows, art, am, ax0, ay0, vxs, vys, TRUNC)
        sums, counts = np.zeros(n, np.int64), np.zeros(n, np.int64)
        if self.lib.cr_costs(g.ctypes.data, gm.ctypes.data, len(rows), g.shape[1], rows.ctypes.data, art.ctypes.data,
                             am.ctypes.data, art.shape[0], art.shape[1], art.strides[0], ax0, ay0, vxs.ctypes.data,
                             vys.ctypes.data, n, TRUNC, sums.ctypes.data, counts.ctypes.data):
            raise ValueError("cr_costs refused its arguments")
        return sums, counts

    def predict(self, t: float | None = None) -> tuple[int, int] | None:
        """Where the view should be now: the last sure one moved on as it last moved (for the time since, with times),
        clamped to where the camera can be; None before the first."""
        if self.sure_at is None:
            return None
        x, y, at = self.sure_at
        k = (t - at) if (t is not None and at is not None) else (self.lost + 1)
        return self._clamp(x + self.move[0] * k, y + self.move[1] * k)

    def see(self, frame: np.ndarray, t: float | None = None, hint: tuple[float, float] | None = None) -> View:
        """This frame's view (the top), `t` its time (s; None: frames taken as evenly spaced), `hint` a rough view
        (her minimap dot's camera) for a cold search on a map the camera moves across too."""
        t0 = time.perf_counter()
        g = self.sample(frame)
        self._cache = {}
        pred = self.predict(t)
        found, how = None, "cold"
        if pred is not None:
            for how, r in (("near", FIRST), ("near", NEAR), ("wide", WIDE)):
                got = self.search(g, pred[0], pred[1], r)
                inside = r != FIRST or max(abs(got[0] - pred[0]), abs(got[1] - pred[1])) < r
                if got[2] <= COST_OK and got[3] >= MARGIN_OK and inside:
                    found = got
                    break
        if found is None:
            how = "cold"
            got = self.cold(frame, g, hint)
            if got[2] <= COST_OK and got[3] >= MARGIN_OK:
                found = got
        ms = 1e3 * (time.perf_counter() - t0)
        if found is None:
            self.lost += 1
            self.counts["carried"] += 1
            cost, margin, share = round(got[2], 3), round(min(got[3], 99.0), 3), round(got[4], 4)
            view = View(pred[0], pred[1], False, "carried", cost, margin, share, ms) if pred is not None \
                else View(0, 0, False, "none", cost, margin, share, ms)  # (never placed yet)
            self.last = view
            return view
        x, y, cost, margin, share = found
        if self.sure_at is not None:
            px, py, pt = self.sure_at
            k = (t - pt) if (t is not None and pt is not None and t > pt) else (self.lost + 1)
            self.move = ((x - px) / k, (y - py) / k)
        self.sure_at = (x, y, t)
        self.lost = 0
        self.counts[how] += 1
        view = View(x, y, True, how, round(cost, 3), round(min(margin, 99.0), 3), round(share, 4), ms)
        self.last = view
        return view


def for_map(map_id: int, size: tuple[int, int] = (1366, 768)) -> "Registration | None":
    """A map's registration for frames of `size` (the look frames' at the game's own 1366x768, a map unit a px): None at
    any other size (at 1920x1080 a look frame is the stretched view at half size), or where the map's art can't be
    drawn (no game data copy: mapart.load's errors)."""
    from .mapart import VIEW, load
    if tuple(size) != VIEW:
        return None
    try:
        return Registration(load(map_id), size)
    except (OSError, KeyError, ValueError):
        return None


def _plain(frame: np.ndarray) -> bool:
    """Whether cr_gray can read the frame as it is: uint8, 3 or 4 channels a pixel side by side, pixels side by side."""
    return (frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] in (3, 4) and frame.strides[2] == 1
            and frame.strides[1] == frame.shape[2])


def shrink(gray: np.ndarray, mask: np.ndarray | None, k: int, lib: ctypes.CDLL | None = None
           ) -> tuple[np.ndarray, np.ndarray | None]:
    """A gray picture in k x k blocks' means, (sum + k^2 / 2) // k^2, the part past a whole block cut off (`lib`:
    camreg.c's cr_shrink, the same bytes: a frame's play area 0.05 ms, where numpy's sum over two axes took 13-15 on the
    loaded PC), and its mask (255 where all of a block's px are set; None for none)."""
    h, w = (gray.shape[0] // k) * k, (gray.shape[1] // k) * k
    if lib is not None and gray.dtype == np.uint8 and gray.ndim == 2 and gray.strides[1] == 1:
        small = np.empty((h // k, w // k), np.uint8)
        if lib.cr_shrink(gray.ctypes.data, gray.shape[0], gray.shape[1], gray.strides[0], k, small.ctypes.data):
            raise ValueError("cr_shrink refused its arguments")
    else:
        total = gray[:h, :w].reshape(h // k, k, w // k, k).astype(np.int32).sum(axis=(1, 3))
        small = np.ascontiguousarray(((total + k * k // 2) // (k * k)).astype(np.uint8))
    if mask is None:
        return small, None
    least = mask[:h, :w].reshape(h // k, k, w // k, k).min(axis=(1, 3))
    return small, np.ascontiguousarray(np.where(least > 0, 255, 0).astype(np.uint8))


def costs_np(g, gm, rows, art, am, ax0, ay0, vxs, vys, trunc=TRUNC) -> tuple[np.ndarray, np.ndarray]:
    """cr_costs in numpy (camreg.c's top): the same integers."""
    n = len(vxs)
    sums, counts = np.zeros(n, np.int64), np.zeros(n, np.int64)
    ah, aw = art.shape
    gw = g.shape[1]
    rows = np.asarray(rows)
    for i in range(n):
        dx = int(vxs[i]) - ax0
        c0, c1 = max(0, -dx), min(gw, aw - dx)
        ar = int(vys[i]) + rows - ay0
        keep = (ar >= 0) & (ar < ah)
        if c0 >= c1 or not keep.any():
            continue
        a = art[ar[keep], dx + c0:dx + c1].astype(np.int16)
        on = (gm[keep, c0:c1] & am[ar[keep], dx + c0:dx + c1]) != 0
        d = np.minimum(np.abs(g[keep, c0:c1].astype(np.int16) - a), trunc)
        sums[i] = int(d[on].sum())
        counts[i] = int(on.sum())
    return sums, counts


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    from PIL import Image

    from .mapart import load
    reg = Registration(load(int(argv[0])))
    print(f"costs in {'C' if reg.lib is not None else 'numpy (' + why + ')'}")
    for p in argv[1:]:
        frame = np.asarray(Image.open(p).convert("RGB"))
        reg.sure_at = None
        v = reg.see(frame)
        said = f"view ({v.x}, {v.y})" if v.sure else "not placed: no view sure"
        print(f"{p}: {said} (the best's cost {v.cost}, margin {v.margin}, share {v.share}), {v.ms:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
