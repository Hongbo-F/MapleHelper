"""Each round solved by the client's own shuffle rules (the decompile: 226c540/228e0f0/2231970, the agent's report):
the motion lasts D in {3.8, 4.1, 4.4, 4.7, 5.0} s, all four together, in 1 hop (each its own slot to its target) or 2
(a random middle arrangement, each hop D/2). A hop of length T: lift over T/4 (to its lane's size), travel over T/2
(x straight from its slot to the target's, size held), settle over T/4 (back to rest size). The lane, by direction
and distance: one slot left = in front (110-123 %), one slot right = behind (76-89 %), two or more right = in front
(146-159 %), two or more left = behind (50-63 %); one staying lunges (front or back, x within 30) and comes back.

Every hypothesis (the start t0, D, the hop permutation(s)) predicts each Violetta's x and size every frame of the
round, at rest before t0 and after t0 + D; it's scored by the detections near its predictions (each prediction its
best detection a frame), the best kept. Given t0 and D, the hops are scored apart (whoever stands where, the slots
are full between them), so every permutation is scored at once.
usage: rules.py [--dets DIR] <clip name> ...   (each round from the answer before it)"""
import itertools
import json
import math
import sys

import numpy as np

DETS = "dets5"
DS = [3.8, 4.1, 4.4, 4.7, 5.0]
REST = 1.0           # a standing Violetta's size in the detector's units (100 %): 0.9-1.05 standing
SX, SL = 16.0, 0.14  # x px, log-size spreads
SX_STAY = 22.0       # a staying one's x spread (its lunge: within 30 of its slot)
LUNGE = (140.0, 70.0)  # its size at the lunge's height: in front (120-159 %) or behind (50-89 %)
PERMS = np.array(list(itertools.permutations(range(4))))  # (24, 4)
TWO_MARGIN = 1.03    # two hops must beat one by this much (they fit more)


def lane(i, j):
    d = j - i
    if d == 0:
        return 100.0
    if abs(d) == 1:
        return 116.5 if d < 0 else 82.5
    return 152.5 if d > 0 else 56.5


LANES = np.array([[lane(i, j) for j in range(4)] for i in range(4)])  # (from, to) -> size %


def predict(slots, t0, T, ts):
    """For every permutation: x, and two log-sizes (a staying one's lunge in front, and behind; a mover's own lane
    twice), their spreads (24, F, 4), over frames ts, a hop of length T from t0 (at rest at the slots before; at the
    targets after)."""
    u = (ts - t0) / T
    lift = np.clip(u / 0.25, 0, 1)
    settle = np.clip((u - 0.75) / 0.25, 0, 1)
    travel = np.clip((u - 0.25) / 0.5, 0, 1)
    sl = np.asarray(slots)
    frm = sl[None, None, :]                       # (1, 1, 4)
    to = sl[PERMS][:, None, :]                    # (24, 1, 4)
    px = frm + (to - frm) * travel[None, :, None]
    stay = (PERMS == np.arange(4)[None, :])       # (24, 4)
    pct = LANES[np.arange(4)[None, :], PERMS]     # (24, 4)
    up = (lift * (1 - settle))[None, :, None]
    pl_a = np.log(REST * (100 + (np.where(stay, LUNGE[0], pct)[:, None, :] - 100) * up) / 100)
    pl_b = np.log(REST * (100 + (np.where(stay, LUNGE[1], pct)[:, None, :] - 100) * up) / 100)
    active = ((ts > t0) & (ts < t0 + T))[None, :, None]
    sx = np.where(stay[:, None, :] & active, SX_STAY, SX) * np.ones_like(px)
    return px, (pl_a, pl_b), sx


def support(px, pls, sx, X, L, W):
    """(24,) each permutation's summed support: every prediction its best detection a frame (a staying one's at the
    better of its two lunges)."""
    dx = (X[None, :, None, :] - px[..., None]) / sx[..., None]    # (24, F, 4, m)
    best = None
    for pl in pls:
        dl = (L[None, :, None, :] - pl[..., None]) / SL
        k = (W[None, :, None, :] * np.exp(-0.5 * (dx * dx + dl * dl))).max(3)
        best = k if best is None else np.maximum(best, k)
    return best.sum((1, 2))


def frame_arrays(rows):
    m = max(1, max(len(r["dets"]) for r in rows))
    X = np.full((len(rows), m), -1e4)
    L = np.zeros((len(rows), m))
    W = np.zeros((len(rows), m))
    for k, r in enumerate(rows):
        for q, d in enumerate(r["dets"]):
            if d["score"] >= 0.45 and d["scale"] >= 0.43:
                X[k, q], L[k, q], W[k, q] = d["x"], math.log(d["scale"]), (d["score"] - 0.4) / 0.6
    return X, L, W


def solve_round(rows, slots, step=2):
    ts = np.array([r["t"] for r in rows])
    X, L, W = frame_arrays(rows)
    best1 = (-1.0,)
    best2 = (-1.0,)
    for t0 in ts[::step]:
        for D in DS:
            if t0 + D > ts[-1] + 0.4:
                continue
            # one hop over D
            s1 = support(*predict(slots, t0, D, ts), X, L, W)
            k = int(np.argmax(s1))
            if s1[k] > best1[0]:
                best1 = (float(s1[k]), tuple(PERMS[k]), float(t0), D)
            # two hops of D/2: the first's frames up to t0 + D/2, the second's after (rest before and after each)
            a = ts <= t0 + D / 2
            sa = support(*predict(slots, t0, D / 2, ts[a]), X[a], L[a], W[a])
            sb = support(*predict(slots, t0 + D / 2, D / 2, ts[~a]), X[~a], L[~a], W[~a])
            ka, kb = int(np.argmax(sa)), int(np.argmax(sb))
            if sa[ka] + sb[kb] > best2[0]:
                best2 = (float(sa[ka] + sb[kb]), tuple(PERMS[ka]), tuple(PERMS[kb]), float(t0), D)
    info = {"one": (round(best1[0], 1), [p + 1 for p in best1[1]], round(best1[2], 2), best1[3]),
            "two": (round(best2[0], 1), [p + 1 for p in best2[1]], [p + 1 for p in best2[2]], round(best2[3], 2),
                    best2[4])}
    if best2[0] > best1[0] * TWO_MARGIN:
        p1, p2 = best2[1], best2[2]
        return tuple(p2[p1[i]] for i in range(4)), "two", info
    return best1[1], "one", info


def run(name):
    res = json.load(open(f"solved/{name}.json"))
    rows = [json.loads(l) for l in open(f"{DETS}/{name}.jsonl")]
    slots = res["slots"]
    first_lit = next(r["t"] for r in rows if r["lit"])
    windows = [r["window_t"] for r in res["rounds"]]
    starts = [first_lit] + windows[:-1]
    out = []
    where = res["made_up_start"] - 1
    for k, (ta, tb) in enumerate(zip(starts, windows)):
        seg = [r for r in rows if ta <= r["t"] <= tb]
        perm, kind, info = solve_round(seg, slots)
        pred = perm[where] + 1
        ans = res["rounds"][k]["answered"]
        out.append((k + 1, pred, ans, perm, kind, info))
        if ans is not None:
            where = ans - 1  # the next round from the answer given
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--dets":
        DETS, args = args[1], args[2:]
    H = N = 0
    for name in args:
        for rnd, pred, ans, perm, kind, info in run(name):
            ok = pred == ans
            H += ok
            N += ans is not None
            print(f"{name[:15]} round {rnd}: {kind} hop(s) -> {[p + 1 for p in perm]}, predicted {pred}, answered {ans}"
                  + ("" if ok else "  <-- differs") + f"   {info}")
    print(f"{H} of {N} rounds as answered")
