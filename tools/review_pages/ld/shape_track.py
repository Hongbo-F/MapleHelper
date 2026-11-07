"""Track the Lie Detector's target by its outline, no YOLO.

1. Template: the white target before it fades (the white-phase frame where its blob is biggest, not a countdown digit:
   the digits glow blue around them, the target has the board's dark rim). Its outer contour, sampled every ~3 px,
   with each point's normal orientation (mod 180: the embossed rim's edges point either way).
2. Edge features per frame: gradient orientation quantized to 8 bins over 180 deg, strong pixels only, each bin spread
   over a disk (LINE-2D style), then for each template bin a response map: cos-like similarity to the nearest
   present bin.
3. Tracking: pose (x, y, theta), constant velocity and spin; each frame the best-scoring pose within a gate around
   the prediction, a small motion penalty; alpha-beta update.
Scored against the user's cursor: inside the template's box (+15 px), and the distance.
"""
import json
import sys

import cv2
import numpy as np

from lib import CLIPS, load_clip

NB = 8                       # orientation bins over 180 deg
LUT = np.array([1.0, 0.75, 0.3, 0.0, 0.0])  # similarity by bin distance (0..4)
SPREAD = 3                   # px: disk radius the orientations are spread over
STEP = 3                     # px between template points


def white_blobs(b):
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 40) & (hsv[..., 2] > 215)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 800:
            continue
        blob = (lbl == k).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((15, 15), np.uint8)) - cv2.dilate(blob, np.ones((5, 5), np.uint8))
        core, rim = b[blob > 0].astype(np.float32), b[ring > 0].astype(np.float32)
        out.append({"mask": blob, "c": ce[k], "area": int(st[k, cv2.CC_STAT_AREA]),
                    "core": float(core.mean()), "rim_b_r": float((rim[:, 0] - rim[:, 2]).mean())})
    return out


def target_blob(b):
    """The white target, or None: pure white in a tan rim (a countdown digit is faintly blue, its glow bluish:
    measured, the target's rim B-R -59..-76, the digits' -5..+68)."""
    blobs = [o for o in white_blobs(b) if o["rim_b_r"] < -30 and o["core"] > 245]
    return max(blobs, key=lambda o: o["area"]) if blobs else None


def template_from(mask, rim=0, only_rim=False):
    """Contour points relative to the mask's centroid, and their normal angles (rad, mod pi); with rim > 0, also
    the outline of the mask grown by rim px (the embossed rim outside the white fill)."""
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    P, N = [], []
    for g in ([rim] if only_rim and rim else [0, rim] if rim else [0]):
        mk = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) if g else mask
        m = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 2.0)
        gy, gx = np.gradient(m)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = max(cs, key=len)[:, 0, :][::STEP]
        P.append(p - c)
        N.append(np.arctan2(gy[p[:, 1], p[:, 0]], gx[p[:, 1], p[:, 0]]) % np.pi)
    return np.concatenate(P).astype(np.float32), np.concatenate(N).astype(np.float32)


def features(b, thr_pct=70):
    """Per template bin t, a response map in [0, 1]: how well the frame's edge orientations near each pixel (spread
    over SPREAD px) agree with bin t (LUT by bin distance). uint8 work, float32 out."""
    g = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
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


def score(resp, pts, nrm, cands, theta):
    """Mean response of the template rotated by theta at each candidate centre (N x 2): N scores."""
    c, s = np.cos(theta), np.sin(theta)
    rp = pts @ np.array([[c, s], [-s, c]], np.float32)  # rotate points
    bins = np.minimum((((nrm + theta) % np.pi) / np.pi * NB).astype(np.int32), NB - 1)
    xs = np.rint(cands[:, None, 0] + rp[None, :, 0]).astype(np.int32)
    ys = np.rint(cands[:, None, 1] + rp[None, :, 1]).astype(np.int32)
    H, W = resp.shape[1:]
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    v = resp[bins[None, :].repeat(len(cands), 0), np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)] * ok
    n = ok.sum(1)
    return np.where(n > 0.6 * len(pts), v.sum(1) / np.maximum(n, 1), 0.0)


def pick_template(c):
    """The white-phase frame to take the template from: before START, the target's blob biggest."""
    fps, first = c["fps"], c["first"]
    start = first + round(4.6 * fps)
    best = None
    for j, i in enumerate(c["i"]):
        if i >= start:
            break
        o = target_blob(c["img"][j])
        if o is not None and (best is None or o["area"] > best[1]["area"]):
            best = (j, o)
    return best


def track(c, gate_px_s=450.0, ang_deg_s=400.0, penalty=0.002, alpha=0.5, beta=0.3, step_deg=3.0):
    fps = c["fps"]
    j0, o = pick_template(c)
    pts, nrm = template_from(o["mask"])
    pos, vel = o["c"].astype(float), np.zeros(2)
    th, w = 0.0, 0.0
    dt = 1.0 / fps
    gate = max(12.0, gate_px_s * dt)
    span = np.radians(max(6.0, ang_deg_s * dt))
    out = {j0: (pos.copy(), th, 1.0)}
    # forward from the template frame
    for j in range(j0 + 1, len(c["img"])):
        resp = features(c["img"][j])
        pp, pth = pos + vel * dt, th + w * dt
        r = np.arange(-gate, gate + 1, 2.0)
        gx, gy = np.meshgrid(r, r)
        keep = gx ** 2 + gy ** 2 <= gate ** 2
        cands = np.stack([pp[0] + gx[keep], pp[1] + gy[keep]], 1).astype(np.float32)
        best = (-1, None, None)
        for dth in np.arange(-span, span + 1e-9, np.radians(step_deg)):
            sc = score(resp, pts, nrm, cands, pth + dth) - penalty * ((cands - pp) ** 2).sum(1) / gate
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (sc[k], cands[k].astype(float), pth + dth)
        s, z, zth = best
        # alpha-beta update
        rpos, rth = z - pp, zth - pth
        pos = pp + alpha * rpos
        vel = vel + beta * rpos / dt
        th = pth + alpha * rth
        w = w + beta * rth / dt
        out[j] = (z, zth, float(s))
    return out, (pts, nrm, o, j0)


STRICT = 45  # px: the user's cursor sat 3-40 px (p90) off the white target before START


def evaluate_strict(c, out):
    """Frames from START with the cursor within STRICT px of the tracked centre, of those with a cursor."""
    start = c["first"] + round(4.6 * c["fps"])
    hits = n = 0
    for j, i in enumerate(c["i"]):
        if i < start or c["cur"][j] is None or j not in out:
            continue
        n += 1
        hits += float(np.hypot(*(c["cur"][j] - out[j][0]))) <= STRICT
    return hits, n


def evaluate_inside(c, out, tmpl, margin=10.0):
    """Frames from START with the cursor inside the tracked shape (the white fill's outline at the tracked pose and
    angle) or within margin px of it."""
    pts = template_from(tmpl[2]["mask"])[0]
    start = c["first"] + round(4.6 * c["fps"])
    hits = n = 0
    for j, i in enumerate(c["i"]):
        if i < start or c["cur"][j] is None or j not in out:
            continue
        z, th = out[j][0], out[j][1]
        cs, sn = np.cos(th), np.sin(th)
        poly = (pts @ np.array([[cs, sn], [-sn, cs]], np.float32) + z).astype(np.float32).reshape(-1, 1, 2)
        n += 1
        hits += cv2.pointPolygonTest(poly, (float(c["cur"][j][0]), float(c["cur"][j][1])), True) >= -margin
    return hits, n


def evaluate(c, out, tmpl, margin=15):
    pts = template_from(tmpl[2]["mask"])[0]  # the white fill's own outline: the same radius whatever the template
    rmax = float(np.hypot(*pts.T).max())
    start = c["first"] + round(4.6 * c["fps"])
    hits = n = 0
    dist = []
    for j, i in enumerate(c["i"]):
        if i < start or c["cur"][j] is None or j not in out:
            continue
        z = out[j][0]
        d = float(np.hypot(*(c["cur"][j] - z)))
        n += 1
        hits += d <= rmax + margin
        dist.append(d)
    return hits, n, float(np.median(dist)) if dist else None, rmax


if __name__ == "__main__":
    keys = sys.argv[1:] or list(CLIPS)
    tot = [0, 0]
    res = {}
    for key in keys:
        c = load_clip(key)
        out, tmpl = track(c)
        h, n, md, rmax = evaluate(c, out, tmpl)
        tot[0] += h; tot[1] += n
        res[key] = {"hits": h, "n": n, "median_dist": md, "rmax": rmax}
        print(f"{key}: template frame {c['i'][tmpl[3]]}, rmax {rmax:.0f}; on target {h}/{n}, median dist {md:.0f} px", flush=True)
    print("ALL", tot)
