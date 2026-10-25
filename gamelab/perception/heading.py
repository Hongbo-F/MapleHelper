"""Which way she faces, seen on a look's frame: her picture just above her name tag, matched against pictures of her
facing left and against their mirror images.

The game keeps one picture of each of her poses, facing left, and draws her facing right by mirroring it, so pictures
of her facing left tell both ways: the box above her tag matches them better as it is (she faces left) or mirrored
(she faces right). Her name tag is centred under her whichever way she faces, so it says where to look but not which
way: her body does (her eye patch, her claw, her hair), and the box from her tag up to her head holds it. Under the
minimap (drawn over her) it isn't read; on a rope only her back shows (the hunter doesn't ask there).

Her pictures are her own: a pool of boxes cut from looks whose keys say which way she faced (`learn`: the hunter's,
from looks well after the last arrow sent and with no hit since; behavior.hunter), all turned to face left, and
reduced to at most REFERENCES pose pictures, each the median of a cluster of the pool (standing, walking, in the air,
after a throw, on her chair). They're kept under gamelab/lab per window size (heading.npz, heading_1366x768.npz), as
her name tag is; until there are any, the seed in gamelab/assets stands in (her at 1366x768 in her outfit of
2026-10-24, `python -m gamelab heading --seed`). Looks where the pictures keep disagreeing with keys that can't be
wrong mean they're no longer hers (a new outfit): the pool is dropped and learnt again (OUTFIT).

Measured on 2026-10-24 (README, Her heading; D:/MapleHelper-scratch/heading/eval): on the old recordings (the first map,
1920x1080 looks at half size; her outfit of 2026-10-18 to 21), each recording read with pictures learnt from the others,
1,232 of 1,249 frames told and every one right (the walks, the turns in mid-air, the taps, throws, knockback, chains, the
chair and pick-ups), 17 not told (1.4%); 131 more were of her under the minimap, where 4 had been read the wrong way.
With pictures learnt from walks and jumps alone, 9 of 426 frames of throws told were the wrong way, all barely like
her: while her attack plays her keys go by frames told surely (ATTACK_SCORE, ATTACK_MARGIN) alone, and none of those
was (control.actions.Actions.see). On 21 frames at 1366x768 (her outfit of 2026-10-24:
standing, on her chair, hit, behind the portal's glow) and their mirror images, each read with pictures from the other
20, 41 of 42 right and one not told. A read takes 0.2 ms at half size (p95 0.4) and 0.7 at 1366x768 (at most 1.6).
"""

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..directions import LEFT, RIGHT
from ..screen import Geometry
from .minimap import minimap_box

# Her box, in half-size px of the 1920x1080 window (each geometry converts them): from UP to DOWN above her name tag's
# centre, and HALF either side of it (71 to 11 up and 23 either side at 1366x768). Her head is 50 up (her hat can go
# higher), her feet about 8 up (eyes.TAG_BELOW_FEET); her claw reaches 16 out after a throw.
UP, DOWN, HALF = 50, 8, 16
PAD = 3  # ... looked for this far around where her tag puts it, every way: her picture sways a px or two as she walks,
         # and the tag is found to a px
SURE = 0.08    # her facing is told when the better way matches at least this much better than the other: on the old
               # recordings with REFERENCES pictures the right way led by 0.29 or more on 95% of the frames, by less than
               # this on 1.4%, and never trailed
LOW = 0.3      # ... and at least this well (a picture of someone else in her place, or her behind a window, matches less)
# Sure enough to go over her keys while her attack plays (control.actions.Actions.see: Reading.strong), matching at
# least ATTACK_SCORE and ATTACK_MARGIN ahead: of the old recordings' 461 frames of throws, the chair and pick-ups read
# with pictures learnt from walks and jumps alone (none of a throw), 377 pass and none of the 9 read the wrong way
# (a pose after a throw at 0.31, 0.24 ahead); the right ones matched 0.35 or better on 95% of them.
ATTACK_SCORE, ATTACK_MARGIN = 0.4, 0.2
REFERENCES = 8  # pose pictures at most (k-means clusters of the pool): one picture (the median of the pool) left 15.9%
                # of the old recordings' frames not told and read 10 frames of throws the wrong way; 8, 1.4% and none
DECISIVE = 0.2  # a picture leading the other way by this much decides at once, the rest not matched: 0.2 ms a read at
                # half size where all 8 take 1.4, and the same frames told the same way
POOL = 300      # boxes kept to learn from (the newest), ...
LEARN_EVERY_S = 1.0  # ... one a second at most, so the pool holds minutes of poses, not the last few s of one
FIRST = 6       # boxes before there's a first picture (their median) ...
REBUILD_EVERY = 30  # ... and new boxes before the pictures are made again: k-means on a thread of its own (65 ms on a
                    # full pool at 1366x768), the new pictures used once they're made
# Of the last OUTFIT[1] looks where the keys' facing can't be wrong (learn's), the pictures told the other way
# OUTFIT[0] times: they're not of her as she is now (a new outfit, a hat), so the pool is dropped and learnt again.
OUTFIT = (12, 20)


@dataclass(frozen=True)
class Reading:
    """Her facing as a frame shows it."""
    side: str | None      # LEFT or RIGHT; None when it can't be told (`why`)
    score: float = 0.0    # how well the better way matched (normalised correlation, -1 to 1)
    margin: float = 0.0   # ... its lead over the other way
    why: str = ""         # why it can't be told: "no pictures", "no tag", "off the frame", "minimap", "close" (both
                          # ways alike), "low" (neither like her); "rope" (the hunter's: she's on one)

    @property
    def strong(self) -> bool:
        """Told surely enough to go over her keys while her attack plays (ATTACK_SCORE, ATTACK_MARGIN)."""
        return self.side is not None and self.score >= ATTACK_SCORE and self.margin >= ATTACK_MARGIN

    def record(self) -> dict:
        """As telemetry data (the heading events)."""
        return {"side": self.side, "score": round(self.score, 3), "margin": round(self.margin, 3),
                **({"why": self.why} if self.why else {})}


UNKNOWN = Reading(None, why="no pictures")


def box_px(geo: Geometry) -> tuple[int, int, int, int]:
    """Her box in a look frame of geometry `geo`: (up, down, half, pad) px."""
    return geo.ifpx(UP), geo.ifpx(DOWN), geo.ifpx(HALF), geo.ifpx(PAD)


def her_box(tag: tuple[int, int], geo: Geometry, pad: int = 0) -> tuple[int, int, int, int]:
    """Her box above her name tag's centre `tag` (look-frame px), `pad` px larger every way: (x0, y0, x1, y1)."""
    up, down, half, _ = box_px(geo)
    x, y = tag
    return x - half - pad, y - up - pad, x + half + pad, y - down + pad


def cut(frame: np.ndarray, tag: tuple[int, int], geo: Geometry, pad: int = 0) -> np.ndarray | None:
    """Her box (her_box) cut out of a look frame; None where it's off the frame."""
    x0, y0, x1, y1 = her_box(tag, geo, pad)
    if y0 < 0 or x0 < 0 or y1 > frame.shape[0] or x1 > frame.shape[1]:
        return None
    return frame[y0:y1, x0:x1]


def under_minimap(box: tuple[int, int, int, int], geo: Geometry) -> bool:
    """Whether a box of a look frame (x0, y0, x1, y1) reaches under the minimap, which the game draws over her: on the
    old recordings, the only frames read the wrong way with REFERENCES pictures (4 of 919 key-labelled ones) were of
    her walking under it (walk_snail)."""
    k = geo.look_scale
    mx0, my0, mx1, my1 = (v / k for v in minimap_box(geo))
    return box[0] < mx1 and box[2] > mx0 and box[1] < my1 and box[3] > my0


def canonical(box: np.ndarray, side: str) -> np.ndarray:
    """A box of her facing `side`, turned to face left (the way the game's pictures face)."""
    return np.ascontiguousarray(box if side == LEFT else box[:, ::-1])


def references(pool: list[np.ndarray], k: int = REFERENCES) -> list[np.ndarray]:
    """Up to `k` pose pictures from a pool of boxes facing left: the medians of its k-means clusters (at least 4 boxes
    a cluster: from fewer than 8 boxes, one picture, the median of them all)."""
    if not pool:
        return []
    stack = np.stack(pool).astype(np.float32)
    k = max(1, min(k, len(pool) // 4))
    if k == 1:
        return [np.median(stack, axis=0).astype(np.float32)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(0)  # the same pictures from the same pool, every time
    _, labels, _ = cv2.kmeans(stack.reshape(len(stack), -1), k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    return [np.median(stack[labels == i], axis=0).astype(np.float32) for i in range(k) if (labels == i).any()]


def match(window: np.ndarray, refs: list[np.ndarray], order: list[int] | None = None) -> tuple[float, float, int]:
    """How well a window (her box, padded) matches the pictures as it is and mirrored: the best of each over the
    pictures (tried in `order`), and the picture that matched best. One matching ATTACK_SCORE or better and leading the
    other way by DECISIVE decides."""
    win = window.astype(np.float32)
    mirror = np.ascontiguousarray(win[:, ::-1])
    as_is = flipped = -1.0
    best = 0
    for i in order if order is not None else range(len(refs)):
        a = float(cv2.matchTemplate(win, refs[i], cv2.TM_CCOEFF_NORMED).max())
        b = float(cv2.matchTemplate(mirror, refs[i], cv2.TM_CCOEFF_NORMED).max())
        if max(a, b) > max(as_is, flipped):
            best = i
        as_is, flipped = max(as_is, a), max(flipped, b)
        if abs(a - b) >= DECISIVE and max(a, b) >= ATTACK_SCORE:
            break
    return as_is, flipped, best


def decide(as_is: float, flipped: float) -> Reading:
    """Her facing from how well her box matched as it is (`as_is`: facing left) and mirrored (facing right)."""
    score, margin = max(as_is, flipped), abs(as_is - flipped)
    if margin < SURE:
        return Reading(None, score, margin, "close")
    if score < LOW:
        return Reading(None, score, margin, "low")
    return Reading(LEFT if as_is > flipped else RIGHT, score, margin)


class Heading:
    """Her pictures for look frames of geometry `geo`, kept at `path` (None: nowhere), the `seed` (a file like
    `path`'s, or None) standing in until there are any. `background`: the pictures are made again on a thread of
    their own (False: at once, as tests want)."""

    def __init__(self, geo: Geometry, path: Path | None = None, seed: Path | None = None, background: bool = True):
        self.geo, self.path, self.background = geo, path, background
        self.pool: deque[np.ndarray] = deque(maxlen=POOL)
        # The pictures, and the order they're tried in (the one that matched best last first): one tuple, swapped
        # whole when a thread makes new ones
        self._bank: tuple[list[np.ndarray], list[int]] = ([], [])
        self._new = 0                # boxes learnt since the pictures were last made
        self._learnt_at = -1e18      # when the last box was learnt (its look's time)
        self._making: threading.Thread | None = None
        self.checks: deque[bool] = deque(maxlen=OUTFIT[1])  # the last looks learn could check: whether they agreed
        self.counts = {"read": 0, "told": 0, "learnt": 0, "disagreed": 0, "forgot": 0}
        source = path if path is not None and path.exists() else seed
        if source is not None and source.exists():
            self._load(source)

    @property
    def refs(self) -> list[np.ndarray]:
        """The pictures."""
        return self._bank[0]

    @property
    def ready(self) -> bool:
        """Whether there are pictures to read with."""
        return bool(self.refs)

    def read(self, frame: np.ndarray, tag: tuple[int, int] | None) -> Reading:
        """Her facing on a look frame (of `geo`) whose name tag's centre is `tag` (None: not seen)."""
        refs, order = self._bank  # (as they are now: a thread may make new ones meanwhile)
        if not refs:
            return UNKNOWN
        if tag is None:
            return Reading(None, why="no tag")
        pad = box_px(self.geo)[3]
        window = cut(frame, tag, self.geo, pad)
        if window is None:
            return Reading(None, why="off the frame")
        if under_minimap(her_box(tag, self.geo, pad), self.geo):
            return Reading(None, why="minimap")
        self.counts["read"] += 1
        as_is, flipped, best = match(window, refs, order)
        if order and order[0] != best:
            order.remove(best)
            order.insert(0, best)
        reading = decide(as_is, flipped)
        self.counts["told"] += reading.side is not None
        return reading

    def learn(self, frame: np.ndarray, tag: tuple[int, int], side: str, t: float,
              reading: Reading | None = None) -> bool:
        """A look (shown at `t`) whose keys say beyond doubt that she faces `side`: its box joins the pool (one every
        LEARN_EVERY_S at most), unless the pictures read it the other way (`reading`, this look's), which counts
        against them (OUTFIT). True if the box was learnt."""
        if reading is not None and reading.side is not None:
            agreed = reading.side == side
            self.checks.append(agreed)
            if not agreed:
                self.counts["disagreed"] += 1
                if sum(not ok for ok in self.checks) >= OUTFIT[0]:
                    self.forget()
                return False
        if t - self._learnt_at < LEARN_EVERY_S or not self.add(frame, tag, side):
            return False
        self._learnt_at = t
        self.counts["learnt"] += 1
        if (not self.refs and len(self.pool) >= FIRST) or self._new >= REBUILD_EVERY:
            self.rebuild()
        return True

    def add(self, frame: np.ndarray, tag: tuple[int, int], side: str) -> bool:
        """Her box in a look frame where she faces `side` joins the pool, as it is (the pictures are made again by
        `rebuild`); not where it's off the frame or under the minimap. True if it joined."""
        box = cut(frame, tag, self.geo)
        if box is None or under_minimap(her_box(tag, self.geo), self.geo):
            return False
        self.pool.append(canonical(box, side).copy())
        self._new += 1
        return True

    def rebuild(self) -> None:
        """Make the pictures again from the pool (on a thread of their own if `background`), and keep both (`path`)."""
        if self._making is not None and self._making.is_alive():
            return
        pool = list(self.pool)
        self._new = 0
        if not self.background:
            self._made(pool, references(pool))
            return
        self._making = threading.Thread(target=lambda: self._made(pool, references(pool)), name="heading-pictures",
                                        daemon=True)
        self._making.start()

    def _made(self, pool: list[np.ndarray], refs: list[np.ndarray]) -> None:
        self._bank = (refs, list(range(len(refs))))
        self.save(pool=pool)

    def forget(self) -> None:
        """Drop the pool and the pictures: they're not of her as she is now."""
        self.pool.clear()
        self._bank, self._new = ([], []), 0
        self.checks.clear()
        self.counts["forgot"] += 1
        self.save()

    def save(self, path: Path | None = None, pool: list[np.ndarray] | None = None) -> None:
        """The pool (by default as it is now) and the pictures, at `path` (by default where they're kept; nowhere if
        None)."""
        path = path or self.path
        if path is None:
            return
        pool = list(self.pool) if pool is None else pool
        boxes = np.stack(pool) if pool else np.zeros((0, 0, 0, 3), np.uint8)
        refs = np.stack(self.refs) if self.refs else np.zeros((0, 0, 0, 3), np.float32)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, pool=boxes, refs=refs, client=np.array(self.geo.client))
        except OSError:
            pass

    def wait(self) -> None:
        """Until the pictures being made (if any) are."""
        if self._making is not None:
            self._making.join()

    def _load(self, path: Path) -> None:
        with np.load(path) as z:
            if tuple(z["client"]) != tuple(self.geo.client):
                return
            self.pool.extend(np.ascontiguousarray(b) for b in z["pool"])
            refs = [np.ascontiguousarray(r, np.float32) for r in z["refs"]]
        if not refs and len(self.pool) >= FIRST:
            refs = references(list(self.pool))
        self._bank = (refs, list(range(len(refs))))
