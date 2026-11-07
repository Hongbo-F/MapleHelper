"""pf2 plus, as options (the user, 2026-11-02): bounce off the board edges; off-board template points counted as a
typical point (the frame's median outline score) in s and in d, not left out; the decoy check over several look-backs.

pf2: outline particle filter + a Kalman track on top (no YOLO).

pf_track's filter (rim template from the white phase, known 2 rad/s spin, decoy penalty), plus:
- "must move": the target is almost never still relative to the board (under 40 px/s on 2.1% of frames, median
  ~160 px/s), so a particle whose velocity matches the board's loses weight (decoys ride the board);
- a Kalman track on the particle clusters: the chosen cluster is the one the track's prediction gates, another takes
  over only after it has outweighed it for `hold_s`, or at once when the track has ridden the board (still relative
  to it) for `still_s`: then it's on a decoy;
- re-seeding: each frame a whole-board search at the known angle (outline score minus the decoy penalty) drops a few
  particles on its best spots, so a track that went to a decoy can come back.
"""
import cv2
import numpy as np

import shape_track as st
from scrolls import scrolls

SPIN = 2.0  # rad/s; in shape_track's rotation convention the angle goes down


GREEN_LO, GREEN_HI = (41, 121, 121), (84, 255, 255)  # the crosshair cursor (lib.cursor_of's range)


def residual_flow(prev, cur, shift, dis, cap=10.0, scale=1.0):
    """Per pixel, how far the frame moved off the board's scroll since the last (DIS flow minus the scroll), capped
    and scaled to [0, 1]; the cursor (green crosshair, either frame) masked: its own motion isn't the target's."""
    a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (prev, cur))
    if scale != 1.0:
        size = (a.shape[1], a.shape[0])
        a, b = (cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in (a, b))
        f = cv2.resize(dis.calc(a, b, None), size, interpolation=cv2.INTER_LINEAR) / scale
    else:
        f = dis.calc(a, b, None)
    r = np.hypot(f[..., 0] - shift[0], f[..., 1] - shift[1])
    green = cv2.inRange(cv2.cvtColor(cur, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI) |         cv2.inRange(cv2.cvtColor(prev, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI)
    r[cv2.dilate(green, np.ones((25, 25), np.uint8)) > 0] = 0
    return np.minimum(r, cap) / cap


def along(r, pts, cands, theta):
    """Mean of r along the template rotated by theta at each candidate centre."""
    c, s = np.cos(theta), np.sin(theta)
    rp = pts @ np.array([[c, s], [-s, c]], np.float32)
    xs = np.clip(np.rint(cands[:, None, 0] + rp[None, :, 0]).astype(np.int32), 0, r.shape[1] - 1)
    ys = np.clip(np.rint(cands[:, None, 1] + rp[None, :, 1]).astype(np.int32), 0, r.shape[0] - 1)
    return r[ys, xs].mean(1)


class Kalman:
    """Constant velocity in screen px: state (x, y, vx, vy)."""

    def __init__(self, pos, vel, acc=500.0, meas=6.0):
        self.x = np.r_[pos, vel].astype(float)
        self.P = np.diag([meas ** 2, meas ** 2, 80.0 ** 2, 80.0 ** 2])
        self.acc, self.meas = acc, meas

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.acc ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q
        return self.x[:2].copy()

    def update(self, z):
        H = np.eye(2, 4)
        S = H @ self.P @ H.T + np.eye(2) * self.meas ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P


def clusters(P, w, radius=35.0, most=6, floor=0.02):
    """Greedy modes: the heaviest particle, everything within radius, removed; repeated. [(centre, mass, vel)]."""
    left = np.ones(len(w), bool)
    out = []
    order = np.argsort(-w)
    for k in order:
        if len(out) >= most:
            break
        if not left[k]:
            continue
        near = left & (np.hypot(*(P[:, 0:2] - P[k, 0:2]).T) < radius)
        m = w[near].sum()
        if m < floor:
            left &= ~near
            continue
        c = (w[near, None] * P[near, 0:2]).sum(0) / m
        v = (w[near, None] * P[near, 2:4]).sum(0) / m
        out.append((c, float(m), v))
        left &= ~near
    return out


def grid_candidates(resp, back, shift, pts, nrm, ang, lam, step=8, top=4):
    """The whole board at the known angle: outline score minus the decoy penalty; the best few spots."""
    H, W = resp.shape[1:]
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    cands = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    s = st.score(resp, pts, nrm, cands, ang)
    if back is not None:
        s = s - lam * st.score(back, pts, nrm, (cands - shift).astype(np.float32), ang)
    m = s.reshape(ys.shape)
    import cv2
    peak = (m >= cv2.dilate(m, np.ones((7, 7), np.uint8))) & (m > 0)
    yy, xx = np.nonzero(peak)
    order = np.argsort(-m[yy, xx])[:top]
    return [(np.array([xs[yy[o], xx[o]], ys[yy[o], xx[o]]], float), float(m[yy[o], xx[o]])) for o in order]


def score_mixed(resp, pts, nrm, cands, theta, p0):
    """st.score, but an off-board template point counts as p0 (a typical point) instead of being left out; no 60% rule."""
    c, s = np.cos(theta), np.sin(theta)
    rp = pts @ np.array([[c, s], [-s, c]], np.float32)
    bins = np.minimum((((nrm + theta) % np.pi) / np.pi * st.NB).astype(np.int32), st.NB - 1)
    xs = np.rint(cands[:, None, 0] + rp[None, :, 0]).astype(np.int32)
    ys = np.rint(cands[:, None, 1] + rp[None, :, 1]).astype(np.int32)
    H, W = resp.shape[1:]
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    v = resp[bins[None, :].repeat(len(cands), 0), np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)]
    return np.where(ok, v, p0).mean(1)


GY, GX = np.mgrid[8:500:16, 8:750:16]
GRID = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32)


def typical(resp, pts, nrm, theta):
    """The frame's typical outline score: the median over a 16 px grid of the board at this angle."""
    return float(np.median(st.score(resp, pts, nrm, GRID, theta)))


def run(c, n=2000, kappa=25.0, sig_a=500.0, sig_th=np.radians(8.0), seed=0, lam=0.3, back_s=0.2, rim=8,
        gamma=3.0, sig_rel=50.0, hold_s=0.3, still_s=0.5, still_v=40.0, gate=45.0, reseed=0.02, reseed_min=0.0, cand_lam=1.0, cand_min=0.25, seed_w=0.2,
        snapshots=None, full=None, track=1.0, hinge=0.0, d0=0.55, mu=0.0, flow_scale=1.0, flow_preset=2,
        offboard=0.0, looks=0.0, bounce=-1.0, bounce_spread=0.0,
        turn=0.0, turn_edge=0.0, edge_px=70.0, turn_v=95.0, turn_sd=30.0, smooth=None, lag=10, **_):
    hist = {}  # (smooth: a dict to fill with the fixed-lag smoothed centre per frame)
    rng = np.random.default_rng(int(seed))
    fps = c["fps"]
    shifts = scrolls(c["key"])["s"]
    kb = max(1, round(back_s * fps))
    # look-backs in frames as digits (bench.py passes floats): 234 = 2, 3 and 4 frames back; 0: back_s's one
    lbs = [int(ch) for ch in str(int(looks))] if looks else [kb]
    mixed = bool(offboard)
    feats = {}
    dis = cv2.DISOpticalFlow_create(int(flow_preset)) if mu else None  # 0 ultrafast, 1 fast, 2 medium
    j0, o = st.pick_template(c)
    pts, nrm = st.template_from(o["mask"], rim=int(rim), only_rim=True)
    t0 = c["t"][j0]
    # bounce: the centre kept fill radius + bounce px in from each edge (bounce < 0: off)
    margin = float(np.hypot(*st.template_from(o["mask"])[0].T).max()) + bounce if bounce >= 0 else None
    P = np.zeros((n, 5))
    P[:, 0:2] = o["c"] + rng.normal(0, 1.0, (n, 2))
    w = np.full(n, 1.0 / n)
    kf = Kalman(o["c"], np.zeros(2), acc=sig_a)
    out = {j0: (o["c"].astype(float), 0.0, 1.0)}
    challenger, ch_since, still_for = None, 0.0, 0.0
    prev_t = t0
    for j in range(j0 + 1, len(c["img"])):
        dt = max(1e-3, c["t"][j] - prev_t)
        prev_t = c["t"][j]
        vb = shifts[j] / dt  # the board's velocity on screen, px/s
        # predict
        P[:, 2:4] += rng.normal(0, sig_a * dt, (n, 2))
        # turns: each frame a share of the guesses (turn; turn_edge within edge_px of an edge) take a new direction at
        # the target's usual screen speed (median 95 px/s, cv 0.34), pointing away from a near edge
        if turn or turn_edge:
            near = np.minimum.reduce([P[:, 0], 749 - P[:, 0], P[:, 1], 499 - P[:, 1]]) < edge_px
            tn = rng.random(n) < np.where(near, turn_edge, turn)
            m_ = int(tn.sum())
            if m_:
                a_ = rng.uniform(0, 2 * np.pi, m_)
                v_ = np.maximum(30.0, rng.normal(turn_v, turn_sd, m_))[:, None] * np.c_[np.cos(a_), np.sin(a_)]
                q_ = P[tn, 0:2]
                for a, hi in ((0, 749.0), (1, 499.0)):
                    v_[:, a] = np.where(q_[:, a] < edge_px, np.abs(v_[:, a]),
                                        np.where(hi - q_[:, a] < edge_px, -np.abs(v_[:, a]), v_[:, a]))
                P[tn, 2:4] = v_
        P[:, 0:2] += P[:, 2:4] * dt
        P[:, 4] += rng.normal(0, sig_th * np.sqrt(dt), n)
        if margin is not None:
            for a, hi in ((0, 749.0 - margin), (1, 499.0 - margin)):
                lo_ = P[:, a] < margin
                P[lo_, a] = 2 * margin - P[lo_, a]
                P[lo_, 2 + a] = np.abs(P[lo_, 2 + a])
                hi_ = P[:, a] > hi
                P[hi_, a] = 2 * hi - P[hi_, a]
                P[hi_, 2 + a] = -np.abs(P[hi_, 2 + a])
                if bounce_spread:
                    b = lo_ | hi_
                    P[b, 2:4] += rng.normal(0, bounce_spread, (int(b.sum()), 2))
                P[:, a] = np.clip(P[:, a], margin, hi)
        P[:, 0] = np.clip(P[:, 0], 0, 749)
        P[:, 1] = np.clip(P[:, 1], 0, 499)
        th = -SPIN * (c["t"][j] - t0)
        resp = feats[j] = st.features(c["img"][j])
        for old in [q for q in feats if q < j - max(lbs)]:
            feats.pop(old)
        backs = [j - L for L in lbs if lam > 0 and j - L >= 0]
        for q in backs:
            if q not in feats:
                feats[q] = st.features(c["img"][q])
        Ss = {q: shifts[q + 1:j + 1].sum(0) for q in backs}
        bk = backs[0] if backs else None  # (the re-seed's: the first look-back)
        S = Ss.get(bk)
        p0 = typical(resp, pts, nrm, th) if mixed else None
        p0b = {q: typical(feats[q], pts, nrm, th) for q in backs} if mixed else {}
        # re-seed: a few of the lightest particles onto the board's best spots at the known angle
        if reseed > 0 and c["i"][j] >= c["first"] + round(4.6 * fps) + reseed_min * fps:
            cands = [q for q in grid_candidates(resp, feats.get(bk), S, pts, nrm, th, cand_lam) if q[1] > cand_min]
            m = int(reseed * n)
            if cands and m:
                light = np.argsort(w)[:m]
                pick = rng.integers(0, len(cands), m)
                P[light, 0:2] = np.array([cands[k][0] for k in pick]) + rng.normal(0, 3.0, (m, 2))
                a = rng.uniform(0, 2 * np.pi, m)  # off the board's velocity by ~150 px/s, any direction
                P[light, 2:4] = vb + 150.0 * np.c_[np.cos(a), np.sin(a)] + rng.normal(0, 40.0, (m, 2))
                P[light, 4] = rng.normal(0, np.radians(3), m)
                w[light] = seed_w * np.median(w)
                w /= w.sum()
        # weigh
        rflow = residual_flow(c["img"][j - 1], c["img"][j], shifts[j], dis, scale=flow_scale) if mu else None
        s = np.zeros(n)
        d = np.zeros(n)
        mo = np.zeros(n)
        keys = np.rint(np.degrees(P[:, 4]) / 2.0).astype(int)
        for k in np.unique(keys):
            idx = np.nonzero(keys == k)[0]
            ang = th + np.radians(2.0 * k)
            z = P[idx, 0:2].astype(np.float32)
            s[idx] = score_mixed(resp, pts, nrm, z, ang, p0) if mixed else st.score(resp, pts, nrm, z, ang)
            for q in backs:
                zb = (P[idx, 0:2] - Ss[q]).astype(np.float32)
                d[idx] += (score_mixed(feats[q], pts, nrm, zb, ang, p0b[q]) if mixed else
                           st.score(feats[q], pts, nrm, zb, ang)) / len(backs)
            if mu:
                mo[idx] = along(rflow, pts, P[idx, 0:2].astype(np.float32), ang)
        vrel = np.hypot(*(P[:, 2:4] - vb).T)
        # the decoy penalty: lam d throughout, and a hinge for a clearly persistent outline (texture scores ~0.45)
        lw = (np.log(w + 1e-300) + kappa * (s + mu * mo - lam * d - hinge * np.maximum(0.0, d - d0))
              - gamma * np.exp(-0.5 * (vrel / sig_rel) ** 2))
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        # the track
        pred = kf.predict(dt)
        cl = clusters(P, w)
        if track and cl:
            inside = [q for q in cl if np.hypot(*(q[0] - pred)) < gate]
            cur = max(inside, key=lambda q: q[1]) if inside else None
            best = max(cl, key=lambda q: q[1])
            rel = np.hypot(*(kf.x[2:4] - vb))
            still_for = still_for + dt if rel < still_v else 0.0
            if cur is None:
                choice = best  # nothing where the track goes: the heaviest
                challenger, ch_since = None, 0.0
            elif best is not cur and best[1] > 1.5 * cur[1]:
                if challenger is not None and np.hypot(*(best[0] - challenger)) < gate:
                    ch_since += dt
                else:
                    challenger, ch_since = best[0], dt
                choice = best if ch_since >= hold_s or still_for >= still_s else cur
            else:
                challenger, ch_since = None, 0.0
                choice = cur
            if choice is not cur and cur is not None:  # switched: the track restarts there
                kf = Kalman(choice[0], choice[2], acc=sig_a)
                still_for = 0.0
            else:
                kf.update(choice[0])
            est = choice[0]
        else:
            top = P[int(np.argmax(w)), 0:2]
            near = np.hypot(*(P[:, 0:2] - top).T) < 40
            est = (w[near, None] * P[near, 0:2]).sum(0) / w[near].sum()
        out[j] = (est, th + float((w * P[:, 4]).sum()), float((w * s).sum()))
        if snapshots is not None:
            snapshots[j] = (P[:, 0:2].copy(), w.copy())
        if full is not None:  # positions, velocities, angles; the weight terms
            full[j] = (P.copy(), w.copy(), s.copy(), d.copy(), mo.copy(), vrel.copy())
        if smooth is not None:
            hist[j] = [P[:, 0:2].copy(), w.copy(), None]
        ess = 1.0 / (w ** 2).sum()
        if ess < n / 2:
            cs = np.cumsum(w)
            u = (rng.random() + np.arange(n)) / n
            pick = np.minimum(np.searchsorted(cs, u), n - 1)
            P = P[pick]
            w = np.full(n, 1.0 / n)
            if smooth is not None:
                hist[j][2] = pick
    if smooth is not None:
        # fixed-lag smoothing: frame j's estimate from the particles alive lag frames later, traced back to their
        # ancestors at j (the guesses that turned out right), with the later weights
        last = max(hist)
        for j in sorted(hist):
            q = min(j + int(lag), last)
            wq = hist[q][1]
            anc = np.arange(n)
            for r in range(q - 1, j - 1, -1):
                if r in hist and hist[r][2] is not None:
                    anc = hist[r][2][anc]
            X = hist[j][0][anc]
            top = X[int(np.argmax(wq))]
            near = np.hypot(*(X - top).T) < 40
            smooth[j] = (wq[near, None] * X[near]).sum(0) / wq[near].sum()
    return out, (pts, nrm, o, j0)
