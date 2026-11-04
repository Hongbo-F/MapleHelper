"""The Lie Detector's Violetta game ("Keep your eyes on the Violetta who's wearing make-up"; the countdown form of
perception.lie_detector): where the made-up one is after each of its four rounds of shuffling, for whoever answers it,
and the "You have successfully passed the Lie Detector Test" dialog that follows a pass.

- The Violettas: a YOLO11n model (assets/violetta.pt; ultralytics and torch loaded on first use) on the window's crop
  (CROP), trained only on frames rendered from the game's own pictures (the stage, the sprites, the spotlights blended
  as the game blends them: D:/MapleHelper-scratch/violetta/synth.py), classes 0 a Violetta and 1 the made-up one
  showing its face (before the shuffles). A box's width over SIZE_W is its drawn size (1.0 standing).
- Each round: the client's own shuffle rules (its decompile, 2026-11-03: D:/MapleHelper-scratch/violetta/ghidra/out)
  fitted to the boxes (`solve_round`). The motion lasts D in DS, all four together, in one hop or two (each D/2); a hop
  of length T is a lift over T/4 to its lane's size, a straight travel over T/2, a settle over T/4; the lane by
  direction and distance (LANES: one slot left in front, one right behind, two or more right in front at ~1.5x, two or
  more left behind at ~0.6x); one that stays lunges, in front or behind. Every start, D and hop permutation is scored
  by the boxes near its predicted places and sizes (each prediction its best box a frame), the best kept. The round's
  answer is the server's (the client only animates it).
- The made-up one's slot before the shuffles: the class-1 boxes' votes (`start_slot`).
- A round's answer window: the four buttons lit cyan (`buttons_lit`).
- The pass's dialog: its template (assets/violetta_ok.png) at its place (`ok_dialog`), the OK button's middle returned.
Offline on the 9 recorded games of 2026-10-26 to 11-03 (D:/MapleHelper-scratch/violetta): the made-up one's start slot
on all of them, and every answered round's slot as answered (28 of 28; the model never trained on a real frame).
"""

import itertools
import math
from pathlib import Path

import cv2
import numpy as np

from ..paths import ASSETS

MODEL = ASSETS / "violetta.pt"
OK_TEMPLATE = ASSETS / "violetta_ok.png"
CROP = (465, 285, 905, 515)  # x0, y0, x1, y1 of the game's 1366x768 client: the window around the stage
IMGSZ = 448
CONF = 0.5          # a box kept from this confidence (the rounds came out the same from 0.25 to 0.7)
SIZE_W = 75.7      # a box's width at size 1.0 (a box is the visible pixels: 0.87 of the sprite's 87)
SLOTS = (526.0, 628.0, 730.0, 834.0)  # the slots' x (client px) when the game's own aren't measured
SLOT_NEAR = 30.0    # a standing one this near a slot is at it
DS = (3.8, 4.1, 4.4, 4.7, 5.0)  # a round's motion (s): 5000 - (rand % 5) * 300 ms (FUN_142231970)
REST = 1.0          # a standing one's size
SX, SL = 16.0, 0.14  # a prediction's spread: x px, log-size
SX_STAY = 22.0       # a staying one's x spread (its lunge: within 30 of its slot)
LUNGE = (140.0, 70.0)  # its size at the lunge's height (%): in front (120-159) or behind (50-89)
TWO_MARGIN = 1.03    # two hops must beat one by this much (they fit more)
PERMS = np.array(list(itertools.permutations(range(4))))
# The answer window's cue: the button row lit cyan (its share of the row's pixels); off it's grey.
BUTTONS = (480, 505, 900, 540)  # x0, y0, x1, y1 (client px)
BUTTONS_LIT = 0.3
# The pass's dialog: its template's top-left at OK_AT (the same pixel in all 7 passed games' clips), searched within
# OK_NEAR; the OK button's middle OK_CLICK from it; seen from OK_MATCH (1.000 on those, at most 0.58 on every other
# frame of the 29 alarm clips, the mouse mini-game's and the Truth Booth's among them).
OK_AT = (560, 325)
OK_NEAR = 30
OK_CLICK = (224, 105)
OK_MATCH = 0.8


def lane(i: int, j: int) -> float:
    """The size (%) a Violetta is drawn at going from slot i to slot j (staying: 100, its lunge apart)."""
    d = j - i
    if d == 0:
        return 100.0
    if abs(d) == 1:
        return 116.5 if d < 0 else 82.5
    return 152.5 if d > 0 else 56.5


LANES = np.array([[lane(i, j) for j in range(4)] for i in range(4)])


class Detector:
    """The YOLO model, loaded on its first use (`device`: Ultralytics', None for the GPU if there is one)."""

    def __init__(self, model: Path = MODEL, device=None, conf: float = CONF):
        self.path, self.device, self.conf = Path(model), device, conf
        self._model = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.path))
        return self._model

    def detect(self, frames: list) -> list[list[dict]]:
        """Each frame's (RGB, the game's 1366x768 client) boxes: x, y (the box's middle, client px), size, score, cls,
        w, h."""
        x0, y0, x1, y1 = CROP
        crops = [np.ascontiguousarray(f[y0:y1, x0:x1][..., ::-1]) for f in frames]  # (BGR, as Ultralytics takes)
        out = []
        for res in self.load().predict(crops, imgsz=IMGSZ, conf=self.conf, device=self.device, verbose=False):
            boxes = []
            for (bx0, by0, bx1, by1), score, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                        res.boxes.cls.cpu().numpy()):
                w = float(bx1 - bx0)
                boxes.append({"x": round(float(bx0 + bx1) / 2 + x0, 1), "y": round(float(by0 + by1) / 2 + y0, 1),
                              "size": round(max(0.3, w / SIZE_W), 3), "score": round(float(score), 3),
                              "cls": int(cls), "w": round(w, 1), "h": round(float(by1 - by0), 1)})
            out.append(boxes)
        return out


def slots_of(frames: list[list[dict]]) -> list[float]:
    """The slots' x: on frames where four boxes stand near the default slots (SLOTS, within SLOT_NEAR), each slot's
    median; the default where none does."""
    near = [[] for _ in SLOTS]
    for boxes in frames:
        at = [min((b for b in boxes if abs(b["x"] - s) < SLOT_NEAR), key=lambda b: abs(b["x"] - s), default=None)
              for s in SLOTS]
        if all(a is not None for a in at):
            for k, a in enumerate(at):
                near[k].append(a["x"])
    return [float(np.median(n)) if n else s for n, s in zip(near, SLOTS)]


def start_slot(frames: list[list[dict]], slots) -> int | None:
    """The made-up one's slot (0-3) before the shuffles: the class-1 boxes' votes, by score; None if none."""
    votes = np.zeros(4)
    for boxes in frames:
        for b in boxes:
            if b["cls"] == 1:
                votes[int(np.argmin([abs(b["x"] - s) for s in slots]))] += b["score"]
    return int(np.argmax(votes)) if votes.any() else None


def _predict(slots, t0: float, T: float, ts: np.ndarray):
    """For every permutation: x, the two log-sizes (a staying one's lunge in front and behind; a mover's lane twice)
    and the x spreads (24, F, 4), a hop of length T from t0 (at rest at the slots before, at the targets after)."""
    u = (ts - t0) / T
    lift = np.clip(u / 0.25, 0, 1)
    settle = np.clip((u - 0.75) / 0.25, 0, 1)
    travel = np.clip((u - 0.25) / 0.5, 0, 1)
    sl = np.asarray(slots, dtype=float)
    frm = sl[None, None, :]
    to = sl[PERMS][:, None, :]
    px = frm + (to - frm) * travel[None, :, None]
    stay = PERMS == np.arange(4)[None, :]
    pct = LANES[np.arange(4)[None, :], PERMS]
    up = (lift * (1 - settle))[None, :, None]
    pl_a = np.log(REST * (100 + (np.where(stay, LUNGE[0], pct)[:, None, :] - 100) * up) / 100)
    pl_b = np.log(REST * (100 + (np.where(stay, LUNGE[1], pct)[:, None, :] - 100) * up) / 100)
    active = ((ts > t0) & (ts < t0 + T))[None, :, None]
    sx = np.where(stay[:, None, :] & active, SX_STAY, SX) * np.ones_like(px)
    return px, (pl_a, pl_b), sx


def _support(px, pls, sx, X, L, W) -> np.ndarray:
    """(24,) each permutation's summed support: every prediction its best box a frame (a staying one's at the better of
    its two lunges)."""
    dx = (X[None, :, None, :] - px[..., None]) / sx[..., None]
    best = None
    for pl in pls:
        dl = (L[None, :, None, :] - pl[..., None]) / SL
        k = (W[None, :, None, :] * np.exp(-0.5 * (dx * dx + dl * dl))).max(3)
        best = k if best is None else np.maximum(best, k)
    return best.sum((1, 2))


def solve_round(ts, frames: list[list[dict]], slots, step: int = 2):
    """The round's permutation (slot i's Violetta to slot perm[i]) from its frames' boxes (`ts` their times, s) from
    before its motion to its answer window, and how (`kind`: "one" or "two" hops; `info`: the best of each)."""
    ts = np.asarray(ts, dtype=float)
    m = max(1, max((len(b) for b in frames), default=1))
    X = np.full((len(frames), m), -1e4)
    L = np.zeros((len(frames), m))
    W = np.zeros((len(frames), m))
    for k, boxes in enumerate(frames):
        for q, b in enumerate(boxes):
            if b["score"] >= 0.45 and b["size"] >= 0.43:
                X[k, q], L[k, q], W[k, q] = b["x"], math.log(b["size"]), (b["score"] - 0.4) / 0.6
    best1, best2 = (-1.0, tuple(range(4)), None, None), (-1.0, None, None, None, None)
    for t0 in ts[::step]:
        for D in DS:
            if t0 + D > ts[-1] + 0.4:
                continue
            s1 = _support(*_predict(slots, t0, D, ts), X, L, W)
            k = int(np.argmax(s1))
            if s1[k] > best1[0]:
                best1 = (float(s1[k]), tuple(int(v) for v in PERMS[k]), float(t0), D)
            a = ts <= t0 + D / 2
            if a.all() or not a.any():
                continue
            sa = _support(*_predict(slots, t0, D / 2, ts[a]), X[a], L[a], W[a])
            sb = _support(*_predict(slots, t0 + D / 2, D / 2, ts[~a]), X[~a], L[~a], W[~a])
            ka, kb = int(np.argmax(sa)), int(np.argmax(sb))
            if sa[ka] + sb[kb] > best2[0]:
                best2 = (float(sa[ka] + sb[kb]), tuple(int(v) for v in PERMS[ka]), tuple(int(v) for v in PERMS[kb]),
                         float(t0), D)
    info = {"one": best1, "two": best2}
    if best2[1] is not None and best2[0] > best1[0] * TWO_MARGIN:
        p1, p2 = best2[1], best2[2]
        return tuple(p2[p1[i]] for i in range(4)), "two", info
    return best1[1], "one", info


def buttons_lit(frame: np.ndarray) -> bool:
    """Whether the four answer buttons are lit (the round's answer window), on an RGB 1366x768 client frame."""
    x0, y0, x1, y1 = BUTTONS
    row = frame[y0:y1, x0:x1]
    lit = cv2.inRange(np.ascontiguousarray(row), (0, 131, 151), (89, 255, 255))
    return cv2.countNonZero(lit) / lit.size > BUTTONS_LIT


class OkDialog:
    """The pass's dialog on an RGB 1366x768 client frame: the OK button's middle (client px), or None."""

    def __init__(self, template: Path = OK_TEMPLATE):
        t = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
        if t is None:
            raise FileNotFoundError(template)
        self.template = t

    def __call__(self, frame: np.ndarray):
        th, tw = self.template.shape
        x, y = OK_AT
        y0, x0 = max(0, y - OK_NEAR), max(0, x - OK_NEAR)
        region = frame[y0:y + th + OK_NEAR, x0:x + tw + OK_NEAR]
        if region.shape[0] < th or region.shape[1] < tw:
            return None
        grey = cv2.cvtColor(np.ascontiguousarray(region), cv2.COLOR_RGB2GRAY)
        _, score, _, (lx, ly) = cv2.minMaxLoc(cv2.matchTemplate(grey, self.template, cv2.TM_CCOEFF_NORMED))
        if score < OK_MATCH:
            return None
        return x0 + lx + OK_CLICK[0], y0 + ly + OK_CLICK[1]
