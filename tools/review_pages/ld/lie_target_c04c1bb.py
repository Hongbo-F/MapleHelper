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

The tracker is fed board crops one by one (`TargetTracker.step`), as the game would give them; `track_clip` runs it on
a recorded alarm clip. It suggests where the target is; it moves nothing and sends no input.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# (from gamelab.perception.lie_detector_yolo at c04c1bb, a module removed since; this copy of the tracker stands alone)
NATIVE_FRAME = (1366, 768)
BOARD_AT_NATIVE = (308, 180, 750, 500)

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
GREEN_LO, GREEN_HI = (41, 121, 121), (84, 255, 255)  # the green crosshair cursor (HSV)

GY, GX = np.mgrid[8:500:16, 8:750:16]
GRID = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32)  # the board every 16 px, for a frame's typical score


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

def features(board: np.ndarray, thr_pct: float = 70) -> np.ndarray:
    """Per template bin t, a response map in [0, 1]: how well the frame's edge orientations near each pixel (spread over
    SPREAD px) agree with bin t (LUT by bin distance)."""
    g = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    thr = np.percentile(mag[::4, ::4], thr_pct)
    q = (((ang % np.pi) / np.pi) * NB).astype(np.uint8) % NB
    q[mag <= thr] = NB  # weak: no bin
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SPREAD + 1, 2 * SPREAD + 1))
    present = [cv2.dilate((q == k).view(np.uint8), disk) for k in range(NB)]
    lut = (LUT * 255).astype(np.uint8)
    resp = np.empty((NB,) + g.shape, np.float32)
    for t in range(NB):
        r = present[t] * lut[0]
        for d in (1, 2):
            if lut[d]:
                r = cv2.max(r, cv2.max(present[(t + d) % NB], present[(t - d) % NB]) * lut[d])
        resp[t] = r
    resp *= 1.0 / 255
    return resp


def _lookup(resp, pts, nrm, cands, theta):
    """Each candidate pose's template point responses (N x points) and which points are on the board."""
    c, s = np.cos(theta), np.sin(theta)
    rp = pts @ np.array([[c, s], [-s, c]], np.float32)
    bins = np.minimum((((nrm + theta) % np.pi) / np.pi * NB).astype(np.int32), NB - 1)
    xs = np.rint(cands[:, None, 0] + rp[None, :, 0]).astype(np.int32)
    ys = np.rint(cands[:, None, 1] + rp[None, :, 1]).astype(np.int32)
    H, W = resp.shape[1:]
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    v = resp[bins[None, :].repeat(len(cands), 0), np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)]
    return v, ok


def typical(resp, pts, nrm, theta) -> float:
    """The frame's typical outline score at this angle: the median over GRID of the on-board points' mean (poses under
    60% on the board left out)."""
    v, ok = _lookup(resp, pts, nrm, GRID, theta)
    n = ok.sum(1)
    s = np.where(n > 0.6 * len(pts), (v * ok).sum(1) / np.maximum(n, 1), 0.0)
    return float(np.median(s))


def score(resp, pts, nrm, cands, theta, p0) -> np.ndarray:
    """Each candidate centre's outline score at angle theta: its template points' mean response, an off-board point
    counting as p0 (a typical point: `typical`)."""
    v, ok = _lookup(resp, pts, nrm, cands, theta)
    return np.where(ok, v, p0).mean(1)


def along(r, pts, cands, theta) -> np.ndarray:
    """Mean of map r along the template rotated by theta at each candidate centre."""
    c, s = np.cos(theta), np.sin(theta)
    rp = pts @ np.array([[c, s], [-s, c]], np.float32)
    xs = np.clip(np.rint(cands[:, None, 0] + rp[None, :, 0]).astype(np.int32), 0, r.shape[1] - 1)
    ys = np.clip(np.rint(cands[:, None, 1] + rp[None, :, 1]).astype(np.int32), 0, r.shape[0] - 1)
    return r[ys, xs].mean(1)


# --- the board's scroll and what moves off it ---------------------------------------------------------------------

def foreign_mask(board: np.ndarray) -> np.ndarray:
    """255 where a pixel isn't the rock board: white (the target while white), the green cursor, the blue countdown and
    START text."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
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


def residual_flow(flow, shift, prev, cur, cap: float = 10.0) -> np.ndarray:
    """Per pixel, how far the frame moved off the board's scroll since the last (the optical flow minus the scroll),
    capped and scaled to [0, 1]; the cursor (green crosshair, either frame) masked."""
    r = np.hypot(flow[..., 0] - shift[0], flow[..., 1] - shift[1])
    green = (cv2.inRange(cv2.cvtColor(cur, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI)
             | cv2.inRange(cv2.cvtColor(prev, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI))
    r[cv2.dilate(green, np.ones((25, 25), np.uint8)) > 0] = 0
    return np.minimum(r, cap) / cap


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
        self.history = []  # (t, board, features, shift from the frame before, grey, clean) for the last kb + 1
        self.template = None  # (rim points, normals, t0, blob area)
        self.dis = cv2.DISOpticalFlow_create(2)

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
        grey = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        clean = foreign_mask(board) == 0
        flow = self.dis.calc(prev[4], grey, None) if prev is not None else None
        shift = flow_shift(flow, prev[5] & clean) if prev is not None else np.zeros(2)
        resp = features(board)
        self.history.append((t, board, resp, shift, grey, clean))
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
        p0 = typical(resp, pts, nrm, th)
        if back is not None:
            S = sum(h[3] for h in self.history[1:])  # the board's scroll since the look-back frame
            p0b = typical(back[2], pts, nrm, th)
        rflow = residual_flow(flow, shift, prev[1], board) if MU and prev is not None else None
        s, d, mo = np.zeros(n), np.zeros(n), np.zeros(n)
        keys = np.rint(np.degrees(P[:, 4]) / 2.0).astype(int)
        for k in np.unique(keys):
            idx = np.nonzero(keys == k)[0]
            ang = th + np.radians(2.0 * k)
            z = P[idx, 0:2].astype(np.float32)
            s[idx] = score(resp, pts, nrm, z, ang, p0)
            if back is not None:
                d[idx] = score(back[2], pts, nrm, (P[idx, 0:2] - S).astype(np.float32), ang, p0b)
            if rflow is not None:
                mo[idx] = along(rflow, pts, z, ang)
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
