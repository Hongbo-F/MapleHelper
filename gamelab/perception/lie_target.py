"""The Lie Detector mini-game's target ("use the mouse to follow the fading figure") tracked by its outline, no YOLO.

The round, as the recordings show it: a figure (star, square, circle, compound shapes ...) shows white at the board's
centre through the countdown, fades from START (~2 s later it is only an embossed rim, like the decoys printed on the
board: the same kinds of shapes), and moves at a steady ~95 px/s on screen, turning now and then, anywhere on the
board (no wall rule: it turns 22-61 px in from an edge, or in mid-board, and can go half past the bottom edge), while
it spins at a constant 2 rad/s (counter-clockwise on screen) and the board's texture scrolls (~125 px/s, warping as it
goes). The user checked this tracker's work on all 11 recorded rounds (2026-11-02: "small hiccups but overall they
should pass the tests"): 3.5 px off their approved reference tracks on average, 97% of frames within 10 px, none lost.

How:
- Template: the white target's outline grown by RIM px (the embossed rim outside the white fill), from the frame its
  white blob is biggest (before START when the caller knows START), sampled every STEP px with each point's normal.
- Features per frame: gradient orientation in NB bins over 180 deg, strong pixels only, spread over SPREAD px; per
  template bin a response map (LINE-2D like). A pose's outline score s is its points' mean response, the angle known
  from the spin; a point off the board counts as a typical point (the frame's median score: neither a hit nor a miss,
  so a half-off pose is judged on what shows and pulled towards plain texture by what doesn't).
- Particle filter, N particles (x, y, vx, vy, angle offset), weight exp(KAPPA (s + MU flow - LAM d)) and "must move":
  - d, the decoy check: the same outline at the same angle where the board's scroll puts the pose BACK_S earlier
    (a printed decoy rides the board; the target moves on its own); off-board points typical there too;
  - flow: the optical flow off the board's scroll (the cursor masked: its own motion isn't the target's) along the rim;
    the board's scroll itself is that flow's median over the board's own pixels (`flow_shift`);
  - must move: a particle moving with the board loses weight (the target almost never does);
  - turns: each frame TURN of the particles (TURN_EDGE within EDGE_PX of an edge) take a new direction at the
    target's usual speed, pointing away from a near edge, so a turn is followed in a frame or two.
- Estimate: the heaviest particle's neighbourhood (40 px), weighted.

The tracker is fed board crops one by one (`TargetTracker.step`), as the game gives them; `track_clip` runs it on a
recorded alarm clip. Live, the perception service runs it through each round on its own frames (control.lie_tracking)
and records where it is. It suggests where the target is; it moves nothing and sends no input.

Speed (2026-11-02): 43 ms a frame on the 11 clips (below-normal priority, 6 cores), against the 66 of 15 a second; it
took 92. The edge features by a table per bin and one grey dilation (30 -> 8 ms), the template turned once per distinct
particle angle (2 deg steps, ~35 a frame) and its points gathered by flat index, the flow off the scroll as one OpenCV
map sampled at the score's points, the typical score on a 32 px grid. Against the user's references, 11 clips x 2
seeds: 3.48 px off on average, 97.2% of frames within 10 px, 99.0% within 20 (before: 3.54, 96.9%, 99.0%). The optical
flow stays DIS's medium preset (DIS_PRESET, 13-16 ms): fast and ultrafast save 12-14 ms but lose the 10 fps clip
(10-14 px off on average there; the 15 fps clips as good). Then the work beside the flow (the user, 2026-11-02: "let's
match the thread count as the light flow"): only the flow used OpenCV's threads, the rest ran on one; now the edge
features and the masks run in worker threads while the flow is made, the flow off the scroll and the typical scores
while the particles are scored, and the particles' scores in PARTICLE_RUNS runs (`_pool`): 43 -> 30 ms a frame on the
11 clips (the two stepped in turn on the same frames, OpenCV on 6 threads), every answer the same bit for bit.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


NATIVE_FRAME = (1366, 768)  # the game's own client size
BOARD_AT_NATIVE = (308, 180, 750, 500)  # xywh: the playable board there, below its blue header
SPIN = 2.0  # rad/s, constant on every clip measured; in this module's rotation convention the angle goes down
NB = 8  # orientation bins over 180 deg
LUT = np.array([1.0, 0.75, 0.3, 0.0, 0.0])  # similarity by bin distance (0..4)
SPREAD = 3  # px: the disk the orientations are spread over
STEP = 3  # px between template points
RIM = 8  # px: the embossed rim outside the white fill (the rim-only template fits best)
START_S = 4.6  # s from the board's first frame to START (the countdown)

N = 2000  # particles
KAPPA = 25.0  # weight per unit of score
SIG_A = 500.0  # px/s^2: the random change of a particle's speed
SIG_TH = np.radians(8.0)  # rad/sqrt(s): the random change of its angle offset
LAM = 0.3  # the decoy check's weight
BACK_S = 0.2  # s: how far back the decoy check looks (1 frame is too short, past ~0.3 s the board's warp)
MU = 1.0  # the residual flow's weight
GAMMA, SIG_REL = 3.0, 50.0  # must move: weight x e^-GAMMA at the board's velocity, falling off over SIG_REL px/s
TURN, TURN_EDGE, EDGE_PX = 0.1, 0.3, 70.0  # turns: shares a frame, anywhere / within EDGE_PX of an edge
TURN_V, TURN_SD = 95.0, 30.0  # px/s: the target's screen speed (median, sd)
DIS_PRESET = 2  # the optical flow: DIS medium (fast and ultrafast lose the 10 fps clip)
GREEN_LO, GREEN_HI = (41, 121, 121), (84, 255, 255)  # the green crosshair cursor (HSV)

GRID_PX = 32
GY, GX = np.mgrid[GRID_PX // 2:500:GRID_PX, GRID_PX // 2:750:GRID_PX]
GRID = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32)  # the board every 32 px, for a frame's typical score


def board_of(frame: np.ndarray) -> np.ndarray | None:
    """The playable board (750 x 500) out of a 1366 x 768 game frame, or None for any other size."""
    if frame.shape[:2] != (NATIVE_FRAME[1], NATIVE_FRAME[0]):
        return None
    x, y, w, h = BOARD_AT_NATIVE
    return np.ascontiguousarray(frame[y:y + h, x:x + w])


# --- the white target and its template --------------------------------------------------------------------------

def white_blobs(board: np.ndarray) -> list[dict]:
    """White blobs of 800+ px: their mask, centre, area, mean core level and the B-R of the ring just outside."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 40) & (hsv[..., 2] > 215)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 800:
            continue
        blob = (lbl == k).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((15, 15), np.uint8)) - cv2.dilate(blob, np.ones((5, 5), np.uint8))
        core, rim = board[blob > 0].astype(np.float32), board[ring > 0].astype(np.float32)
        out.append({"mask": blob, "c": ce[k], "area": int(st[k, cv2.CC_STAT_AREA]),
                    "core": float(core.mean()), "rim_b_r": float((rim[:, 0] - rim[:, 2]).mean())})
    return out


def target_blob(board: np.ndarray) -> dict | None:
    """The white target, or None: pure white in a tan rim (a countdown digit is faintly blue and its glow bluish:
    measured, the target's rim B-R -59..-76, the digits' -5..+68)."""
    blobs = [o for o in white_blobs(board) if o["rim_b_r"] < -30 and o["core"] > 245]
    return max(blobs, key=lambda o: o["area"]) if blobs else None


def template_from(mask: np.ndarray, rim: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Outline points relative to the mask's centroid and their normal angles (rad, mod pi): the mask's own outline,
    or with rim > 0 the outline of the mask grown by rim px."""
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    mk = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1, 2 * rim + 1))) if rim else mask
    m = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 2.0)
    gy, gx = np.gradient(m)
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    p = max(cs, key=len)[:, 0, :][::STEP]
    return (p - c).astype(np.float32), (np.arctan2(gy[p[:, 1], p[:, 0]], gx[p[:, 1], p[:, 0]]) % np.pi).astype(np.float32)


# --- outline evidence -------------------------------------------------------------------------------------------

DISK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SPREAD + 1, 2 * SPREAD + 1))
# for template bin t, a pixel's value by its own bin b (b = NB: a weak pixel, none): LUT[bin distance] x 255
BIN_TABLES = [np.array([int(LUT[min(abs(t - b), NB - abs(t - b))] * 255) if b < NB else 0 for b in range(256)],
                       np.uint8) for t in range(NB)]


def features(board: np.ndarray, thr_pct: float = 70) -> np.ndarray:
    """Per template bin t, a response map (uint8, x 255): how well the frame's edge orientations near each pixel
    (spread over SPREAD px) agree with bin t (LUT by bin distance). Each pixel's bin's value for t (BIN_TABLES), then
    the disk's max: a grey dilation, the same as the max of each bin's dilated mask x its LUT value in a third of the
    time."""
    g = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    thr = float(np.percentile(mag[::4, ::4], thr_pct))
    q = (ang * np.float32(NB / np.pi)).astype(np.uint8)
    np.bitwise_and(q, NB - 1, out=q)  # the orientation mod 180 deg, in NB bins
    q = cv2.max(q, cv2.bitwise_and(cv2.compare(mag, thr, cv2.CMP_LE), NB))  # weak: NB, no bin
    resp = np.empty((NB,) + g.shape, np.uint8)
    for t in range(NB):
        cv2.dilate(cv2.LUT(q, BIN_TABLES[t]), DISK, dst=resp[t])
    return resp


def turned(pts, nrm, angles):
    """The template turned by each angle (K): its points' offsets x and y (K x points) and their normals' bins."""
    angles = np.atleast_1d(np.asarray(angles, np.float64))
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    rx = (pts[:, 0] * c - pts[:, 1] * s).astype(np.float32)
    ry = (pts[:, 0] * s + pts[:, 1] * c).astype(np.float32)
    bins = np.minimum((((nrm + angles[:, None]) % np.pi) / np.pi * NB).astype(np.int32), NB - 1)
    return rx, ry, bins


def placed(cands, rx, ry):
    """The turned template's points at each candidate centre (N x 2; rx, ry: one row, or a row per candidate): pixel x
    and y (N x points, not clipped)."""
    cands = np.asarray(cands, np.float32)
    return np.rint(cands[:, 0:1] + rx).astype(np.int32), np.rint(cands[:, 1:2] + ry).astype(np.int32)


def flat(xs, ys, H=500, W=750):
    """The points' flat pixel index (clipped onto the board) and whether each is on the board."""
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return np.clip(ys, 0, H - 1) * W + np.clip(xs, 0, W - 1), ok


def _gather(resp, bins, xs, ys, at=None):
    """The responses (as stored: uint8 x 255, or floats) at the points for their bins, and which are on the board
    (at: the points' `flat`, when at hand)."""
    H, W = resp.shape[1:]
    lin, ok = flat(xs, ys, H, W) if at is None else at
    return resp.reshape(-1)[bins * (H * W) + lin], ok


def _scale(resp):
    return 1.0 / 255 if resp.dtype == np.uint8 else 1.0


def _sums(resp, bins, xs, ys, at=None):
    """Each pose's summed response over its on-board points, and how many of its points are on the board."""
    v, ok = _gather(resp, bins, xs, ys, at)
    return (v * ok).sum(1, dtype=np.float64) * _scale(resp), ok.sum(1)


def score_at(resp, bins, xs, ys, p0, at=None) -> np.ndarray:
    """Each pose's outline score from its points (`placed`) and bins: their mean response, an off-board point counting
    as p0 (a typical point)."""
    total, on = _sums(resp, bins, xs, ys, at)
    return (total + (xs.shape[1] - on) * p0) / xs.shape[1]


def typical(resp, pts, nrm, theta) -> float:
    """The frame's typical outline score at this angle: the median over GRID of the on-board points' mean (poses under
    60% on the board left out)."""
    rx, ry, bins = turned(pts, nrm, theta)
    v, ok = _gather(resp, bins, *placed(GRID, rx, ry))
    n = ok.sum(1)
    s = np.where(n > 0.6 * len(pts), (v * ok).sum(1, dtype=np.float64) * _scale(resp) / np.maximum(n, 1), 0.0)
    return float(np.median(s))


def score(resp, pts, nrm, cands, theta, p0) -> np.ndarray:
    """Each candidate centre's outline score at angle theta: its template points' mean response, an off-board point
    counting as p0 (a typical point: `typical`)."""
    rx, ry, bins = turned(pts, nrm, theta)
    xs, ys = placed(cands, rx, ry)
    return score_at(resp, np.broadcast_to(bins, xs.shape), xs, ys, p0)


# --- the board's scroll and what moves off it ---------------------------------------------------------------------

def foreign_mask(board: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """255 where a pixel isn't the rock board: white (the target while white), the green cursor, the blue countdown and
    START text (hsv: the board's, when it's at hand)."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV) if hsv is None else hsv
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bad = (((s < 30) & (v > 230)) | ((s > 80) & (h > 40) & (h < 170))).astype(np.uint8)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(bad, np.ones((11, 11), np.uint8)) * 255


def flow_shift(flow: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """How far the board's texture moved between two frames (px): the median of their optical flow over the board's
    own pixels (`clean`: neither frame's pixel foreign), every 4th pixel each way. On the 11 clips' frames it is
    0.1-0.2 px from phase correlation refined by ECC (0.5 at most), which it replaced (2026-11-02): the flow is the
    residual flow's anyway, and ECC took 23 ms a frame. (The board's path is a fixed ellipse in time since START,
    1 rad/s, 147 x 97 px/s, but the frames' times are their captures': a frame stamped 139 ms after the one before
    can show 18 ms of it, so the board is measured, not predicted.)"""
    f, ok = flow[::4, ::4], clean[::4, ::4]
    if ok.sum() < 100:
        return np.zeros(2)
    return np.array([np.median(f[..., 0][ok]), np.median(f[..., 1][ok])], np.float64)


def off_scroll_map(flow, shift, near_cursor, cap: float = 10.0) -> np.ndarray:
    """Per pixel, how far the frame moved off the board's scroll since the last (the optical flow minus the scroll),
    capped at cap px and scaled to [0, 1]; 0 near the cursor (`near_cursor`: its own motion isn't the target's)."""
    fx, fy = cv2.split(flow)
    r = cv2.magnitude(cv2.subtract(fx, float(shift[0])), cv2.subtract(fy, float(shift[1])))
    r = cv2.min(r, cap) * (1.0 / cap)
    r[near_cursor] = 0
    return r


def off_the_scroll(r, lin) -> np.ndarray:
    """Each pose's mean of the off-scroll map r (`off_scroll_map`) over its points (their `flat` index)."""
    return r.reshape(-1)[lin].mean(1)


# --- the work spread over threads -------------------------------------------------------------------------------

PARTICLE_RUNS = 3  # the particles' scores in this many runs at once: more lose to Python's GIL (on 10-29 20:31, the
#                    filter took 11.7 ms in 1 run, 9.1 in 2, 8.4 in 3, 10.4 in 6)
_POOL: tuple[ThreadPoolExecutor, int] | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> tuple[ThreadPoolExecutor, int]:
    """The trackers' worker threads and how many: as many as OpenCV's own (cv2.getNumThreads() at the first frame, the
    optical flow's: 6 in the live tracker's process), shared by the process's trackers. The frame's other work runs in
    them beside the flow (the features, the masks) and beside the particles' scores (the flow off the scroll, the
    typical scores), and the particles' scores in PARTICLE_RUNS runs (numpy and OpenCV let go of the GIL)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            k = max(1, cv2.getNumThreads())
            _POOL = (ThreadPoolExecutor(k, thread_name_prefix="lie-target"), k)
        return _POOL


def _chunks(n: int, k: int) -> list[tuple[int, int]]:
    """0..n in k nearly equal runs (a, b), the empty ones left out."""
    return [(n * i // k, n * (i + 1) // k) for i in range(k) if n * i // k < n * (i + 1) // k]


def _masks(board, hsv):
    """The board's own pixels (`foreign_mask`'s complement) and the green cursor's."""
    return foreign_mask(board, hsv) == 0, cv2.inRange(hsv, GREEN_LO, GREEN_HI)


def _off_scroll(flow, shift, cursors):
    """`off_scroll_map` with the cursor's pixels (both frames') grown 12 px masked."""
    return off_scroll_map(flow, shift, cv2.dilate(cursors, np.ones((25, 25), np.uint8)) > 0)


def _particle_sums(resp, back_resp, rx, ry, bins, which, q, qb):
    """A run of the particles' outline sums (`_sums`): at their centres q now, and at qb (where the board's scroll puts
    them) on the look-back frame's features back_resp (None: no look-back yet); and their points' flat index now."""
    rx, ry, bins = rx[which], ry[which], bins[which]
    xs, ys = placed(q, rx, ry)
    at = flat(xs, ys)
    back = _sums(back_resp, bins, *placed(qb, rx, ry)) if back_resp is not None else None
    return _sums(resp, bins, xs, ys, at), back, at[0]


# --- the tracker ------------------------------------------------------------------------------------------------

class TargetTracker:
    """Feed it the board's frames in order (`step`); it answers where the target is once it has seen it white.

    fps: the frames' nominal rate (the decoy check looks back round(BACK_S x fps) frames). start_t: START's time, when
    known (the template is taken from the biggest white blob before it; otherwise from the biggest one seen). The
    filter restarts on each bigger blob, so its answers from the template frame on are the same as if it had started
    there.
    """

    def __init__(self, fps: float = 15.0, start_t: float | None = None, seed: int = 0, n: int = N):
        self.kb = max(1, round(BACK_S * fps))
        self.start_t = start_t
        self.seed, self.n = seed, n
        self.history = []  # (t, board, features, shift from the frame before, grey, clean, cursor): the last kb + 1
        self.template = None  # (rim points, normals, t0, blob area)
        self.dis = cv2.DISOpticalFlow_create(DIS_PRESET)

    def _restart(self, blob, t):
        pts, nrm = template_from(blob["mask"], rim=RIM)
        self.template = (pts, nrm, t, blob["area"])
        self.rng = np.random.default_rng(int(self.seed))
        self.P = np.zeros((self.n, 5))
        self.P[:, 0:2] = blob["c"] + self.rng.normal(0, 1.0, (self.n, 2))
        self.w = np.full(self.n, 1.0 / self.n)
        self.prev_t = t

    def step(self, board: np.ndarray, t: float) -> dict | None:
        """One frame (a 750 x 500 board crop, BGR) at time t (s): {"x", "y", "angle" (rad), "score", "white"} or None
        before the target has been seen white."""
        prev = self.history[-1] if self.history else None
        pool, _ = _pool()
        grey = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        feats, masks = pool.submit(features, board), pool.submit(_masks, board, hsv)  # beside the flow
        flow = self.dis.calc(prev[4], grey, None) if prev is not None else None
        clean, cursor = masks.result()
        shift = flow_shift(flow, prev[5] & clean) if prev is not None else np.zeros(2)
        resp = feats.result()
        self.history.append((t, board, resp, shift, grey, clean, cursor))
        del self.history[:-(self.kb + 1)]
        if self.start_t is None or t < self.start_t:
            blob = target_blob(board)
            if blob is not None and (self.template is None or blob["area"] > self.template[3]):
                self._restart(blob, t)
                return {"x": float(blob["c"][0]), "y": float(blob["c"][1]), "angle": 0.0, "score": 1.0,
                        "white": True}
        if self.template is None:
            return None
        return self._filter(board, t, resp, shift, prev, flow)

    def _filter(self, board, t, resp, shift, prev, flow):
        pts, nrm, t0, _ = self.template
        n, rng, P = self.n, self.rng, self.P
        pool, kw = _pool()
        if MU and prev is not None:  # the flow off the scroll, in a thread of its own while the particles move
            off = pool.submit(_off_scroll, flow, shift, prev[6] | self.history[-1][6])
        dt = max(1e-3, t - self.prev_t)
        self.prev_t = t
        vb = shift / dt  # the board's velocity on screen, px/s
        # predict: each particle's own speed, a random change of it, and the turns
        P[:, 2:4] += rng.normal(0, SIG_A * dt, (n, 2))
        near = np.minimum.reduce([P[:, 0], 749 - P[:, 0], P[:, 1], 499 - P[:, 1]]) < EDGE_PX
        tn = rng.random(n) < np.where(near, TURN_EDGE, TURN)
        m = int(tn.sum())
        if m:
            a = rng.uniform(0, 2 * np.pi, m)
            v = np.maximum(30.0, rng.normal(TURN_V, TURN_SD, m))[:, None] * np.c_[np.cos(a), np.sin(a)]
            q = P[tn, 0:2]
            for k, hi in ((0, 749.0), (1, 499.0)):  # near an edge: away from it
                v[:, k] = np.where(q[:, k] < EDGE_PX, np.abs(v[:, k]), np.where(hi - q[:, k] < EDGE_PX, -np.abs(v[:, k]), v[:, k]))
            P[tn, 2:4] = v
        P[:, 0:2] += P[:, 2:4] * dt
        P[:, 4] += rng.normal(0, SIG_TH * np.sqrt(dt), n)
        P[:, 0] = np.clip(P[:, 0], 0, 749)
        P[:, 1] = np.clip(P[:, 1], 0, 499)
        th = -SPIN * (t - t0)
        # weigh: the outline now, the decoy check, the flow off the board, must move
        back = self.history[0] if len(self.history) == self.kb + 1 else None
        p0 = pool.submit(typical, resp, pts, nrm, th)
        if back is not None:
            S = sum(h[3] for h in self.history[1:])  # the board's scroll since the look-back frame
            p0b = pool.submit(typical, back[2], pts, nrm, th)
        # the template turned once per distinct particle angle (to 2 deg), each particle given its angle's; the
        # particles' scores in PARTICLE_RUNS runs, one a thread
        keys, which = np.unique(np.rint(np.degrees(P[:, 4]) / 2.0), return_inverse=True)
        rx, ry, bins = turned(pts, nrm, th + np.radians(2.0 * keys))
        q = P[:, 0:2]
        qb = q - S if back is not None else None
        runs = [pool.submit(_particle_sums, resp, None if back is None else back[2], rx, ry, bins, which[a:b], q[a:b],
                            None if qb is None else qb[a:b]) for a, b in _chunks(n, min(kw, PARTICLE_RUNS))]
        runs = [j.result() for j in runs]
        npts = len(pts)
        total, on = (np.concatenate(v) for v in zip(*(now for now, _, _ in runs)))
        s = (total + (npts - on) * p0.result()) / npts
        if back is not None:
            total, on = (np.concatenate(v) for v in zip(*(b for _, b, _ in runs)))
            d = (total + (npts - on) * p0b.result()) / npts
        else:
            d = np.zeros(n)
        if MU and prev is not None:
            r = off.result()
            mo = np.concatenate(list(pool.map(off_the_scroll, [r] * len(runs), [lin for _, _, lin in runs])))
        else:
            mo = np.zeros(n)
        vrel = np.hypot(*(P[:, 2:4] - vb).T)
        lw = (np.log(self.w + 1e-300) + KAPPA * (s + MU * mo - LAM * d)
              - GAMMA * np.exp(-0.5 * (vrel / SIG_REL) ** 2))
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        top = P[int(np.argmax(w)), 0:2]
        near = np.hypot(*(P[:, 0:2] - top).T) < 40
        est = (w[near, None] * P[near, 0:2]).sum(0) / w[near].sum()
        out = {"x": float(est[0]), "y": float(est[1]), "angle": float(th + (w * P[:, 4]).sum()),
               "score": float((w * s).sum()), "white": False}
        if 1.0 / (w ** 2).sum() < n / 2:  # resample (systematic)
            cs = np.cumsum(w)
            u = (rng.random() + np.arange(n)) / n
            self.P = P[np.minimum(np.searchsorted(cs, u), n - 1)]
            w = np.full(n, 1.0 / n)
        self.w = w
        return out


# --- recorded clips ---------------------------------------------------------------------------------------------

def clip_frames(folder: Path, first: int = 0, last: int | None = None):
    """A recorded alarm clip's board frames in order: (frame number, time, board), frames first..last; frames of
    another size skipped."""
    rows = [json.loads(line) for line in open(Path(folder) / "index.jsonl", encoding="utf-8")]
    for r in rows:
        if "file" not in r or r["i"] < first or (last is not None and r["i"] > last):
            continue
        frame = cv2.imread(str(Path(folder) / r["file"]))
        board = None if frame is None else board_of(frame)
        if board is not None:
            yield r["i"], r["t"], board


def track_clip(folder: Path, first: int, last: int | None = None, fps: float = 15.0, seed: int = 0,
               start_s: float = START_S) -> dict[int, dict]:
    """The tracker run on a recorded clip from its board's first frame `first` (START start_s later): frame number ->
    its answer (frames before the target is seen white left out)."""
    frames = list(clip_frames(folder, first, last))
    start_t = next((t for i, t, _ in frames if i >= first + round(start_s * fps)), None)
    tracker = TargetTracker(fps=fps, start_t=start_t, seed=seed)
    out = {}
    for i, t, board in frames:
        r = tracker.step(board, t)
        if r is not None:
            out[i] = r
    return out


def against_reference(answers: dict[int, dict], ref: dict) -> dict:
    """How far the answers are from a reference track (the user-approved ones: tests/fixtures/lie_detector/
    references), over the round's frames from START: the mean px off, the share within 10 and 20 px, the worst."""
    d = np.array([np.hypot(answers[int(i)]["x"] - x, answers[int(i)]["y"] - y)
                  for i, (x, y) in ref["centre"].items() if int(i) in answers])
    if not len(d):
        return {"frames": 0}
    return {"frames": int(len(d)), "mean": float(d.mean()), "within10": float(np.mean(d <= 10)),
            "within20": float(np.mean(d <= 20)), "worst": float(d.max())}
