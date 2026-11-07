"""Four-Violetta tracker: position and size together, so the one passing in front (drawn big) and the one passing
behind (small) keep who they are through a crossing.

Each track: x, y, log-size, and their velocities (an alpha-beta filter). Every frame the pose-locked detections
(track3.py: the frame's own spin pose only) are assigned to the tracks by the least total cost over every one-to-one
assignment, misses allowed: the distance from each track's prediction in x, y and log-size, each over its spread (the
spread growing while a track goes unseen), less the detection's score. A track unseen coasts on its velocity, slowing.
At each answer window the four stand at the slots: each track snaps to its slot (one to one, nearest), and the made-up
one's slot is the prediction.
usage: mot.py <clip name> [...] (dets3/<clip>.jsonl, solved/<clip>.json)"""
import itertools
import json
import math
import sys

import numpy as np

SX, SY, SS = 16.0, 14.0, 0.16      # spreads: x px, y px, log-size
GROW = 1.25                         # each unseen frame widens the spreads by this
MISS = 6.0                          # a track's cost for no detection this frame
ALPHA, BETA, DAMP = 0.7, 0.35, 0.8  # filter gains; velocity kept a frame unseen
DAMP_S = 0.5                        # ... the size's
Y_RANGE = (320.0, 470.0)            # the stage's floor band (y of a Violetta's middle)
X_RANGE, LS_RANGE = (470.0, 890.0), (math.log(0.25), math.log(2.0))
DETS = "dets3"                      # the detections tracked (--dets)  # the stage, the sizes seen
STAND = None                        # (y, log-size) of the four standing: set by run
MIN_SCORE, MIN_SCALE = 0.5, 0.43    # detections tracked (the tiny weak ones are mostly the cones' leftovers)
PER_ROUND = False                   # each round from the answer before it (scoring rounds apart)
RELAX = 0.25                        # an unseen track's size goes this share of the way back to standing size


class Track:
    def __init__(self, x, y, s):
        self.x, self.y, self.ls = x, y, math.log(s)
        self.stand_ls = self.ls
        self.vx = self.vy = self.vls = 0.0
        self.unseen = 0

    def predict(self, dt):
        return (min(max(self.x + self.vx * dt, X_RANGE[0]), X_RANGE[1]),
                min(max(self.y + self.vy * dt, Y_RANGE[0]), Y_RANGE[1]),
                min(max(self.ls + self.vls * dt, LS_RANGE[0]), LS_RANGE[1]))

    def cost(self, d, dt):
        px, py, pls = self.predict(dt)
        g = GROW ** min(self.unseen, 12)
        return (((d["x"] - px) / (SX * g)) ** 2 + ((d["y"] - py) / (SY * g)) ** 2
                + ((math.log(d["scale"]) - pls) / (SS * g)) ** 2) - 4.0 * d["score"]

    def update(self, d, dt):
        px, py, pls = self.predict(dt)
        rx, ry, rls = d["x"] - px, d["y"] - py, math.log(d["scale"]) - pls
        self.x, self.y, self.ls = px + ALPHA * rx, py + ALPHA * ry, pls + ALPHA * rls
        if dt > 0:
            self.vx += BETA * rx / dt
            self.vy += BETA * ry / dt
            self.vls += BETA * rls / dt
        self.unseen = 0

    def coast(self, dt):
        self.x, self.y, self.ls = self.predict(dt)
        self.ls += RELAX * (self.stand_ls - self.ls)
        self.vx *= DAMP
        self.vy *= DAMP
        self.vls *= DAMP_S
        self.unseen += 1


def assign(tracks, dets, dt):
    """The least-cost one-to-one assignment (track -> detection index or None), misses at MISS each."""
    n = len(tracks)
    C = [[t.cost(d, dt) for d in dets] for t in tracks]
    best = (float("inf"), None)
    options = list(range(len(dets))) + [None] * n
    for choice in set(itertools.permutations(options, n)):
        c = sum(MISS if j is None else C[i][j] for i, j in enumerate(choice))
        if c < best[0]:
            best = (c, choice)
    return best[1]


def snap(tracks, slots):
    """Each track to a slot, one to one, nearest in x overall; its place and motion reset there."""
    best = min(itertools.permutations(range(4)), key=lambda p: sum(abs(tracks[i].x - slots[p[i]]) for i in range(4)))
    for i, t in enumerate(tracks):
        t.x, t.vx, t.vy, t.vls, t.unseen = slots[best[i]], 0.0, 0.0, 0.0, 0
        t.y, t.ls = STAND[best[i]]
    return best


def run(name, verbose=False, log=None):
    """The rounds' (round, predicted slot, answer); `log`, a list: each frame's tracks appended to it (t, i, the
    tracks' x, y, size and frames unseen, the detection each took, the made-up one's track, the round)."""
    res = json.load(open(f"solved/{name}.json"))
    rows = [json.loads(l) for l in open(f"{DETS}/{name}.jsonl")]
    slots = res["slots"]
    first_lit = next(r["t"] for r in rows if r["lit"])
    # the four at their slots just before the cones: y and size
    before = [r for r in rows if first_lit - 2.0 <= r["t"] < first_lit]
    tracks = []
    for s in slots:
        near = [d for r in before for d in r["dets"] if abs(d["x"] - s) < 25 and d["score"] > 0.5]
        y = float(np.median([d["y"] for d in near])) if near else 400.0
        sc = float(np.median([d["scale"] for d in near])) if near else 1.05
        tracks.append(Track(s, y, sc))
    global STAND
    STAND = [(t.y, t.ls) for t in tracks]
    made_up = res["made_up_start"] - 1  # the made-up one's track
    windows = [r["window_t"] for r in res["rounds"]]
    out, w, last_t = [], 0, first_lit
    for r in rows:
        if r["t"] < first_lit:
            continue
        dt = r["t"] - last_t
        last_t = r["t"]
        dets = sorted([d for d in r["dets"] if d["score"] >= MIN_SCORE and d["scale"] >= MIN_SCALE],
                      key=lambda d: -d["score"])[:7]
        choice = assign(tracks, dets, dt)
        for t, j in zip(tracks, choice):
            if j is None:
                t.coast(dt)
            else:
                t.update(dets[j], dt)
        if log is not None:
            log.append({"t": r["t"], "i": r["i"], "round": w + 1, "made_up": made_up,
                        "tracks": [(t.x, t.y, math.exp(t.ls), t.unseen) for t in tracks],
                        "took": [None if j is None else dets[j] for j in choice]})
        if verbose:
            print(round(r["t"], 2), " ".join(f"{t.x:6.0f}/{math.exp(t.ls):.2f}{'*' if t.unseen else ' '}"
                                             for t in tracks))
        if w < len(windows) and r["t"] >= windows[w]:
            where = snap(tracks, slots)
            out.append((w + 1, where[made_up] + 1, res["rounds"][w]["answered"]))
            ans = res["rounds"][w]["answered"]
            if log is not None:
                log[-1]["snapped"] = list(where)
            if PER_ROUND and ans is not None:  # the next round from the answer given (right, in a passed test)
                made_up = where.index(ans - 1)
            w += 1
            if w == len(windows):
                break
    return out


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    PER_ROUND = "--per-round" in sys.argv
    if "--dets" in sys.argv:
        DETS = sys.argv[sys.argv.index("--dets") + 1]
    H = N = 0
    for name in [a for a in sys.argv[1:] if not a.startswith("-") and not a.startswith("dets")]:
        for rnd, pred, ans in run(name, verbose):
            ok = pred == ans
            H += ok
            N += ans is not None
            print(f"{name[:15]} round {rnd}: predicted {pred}, answered {ans}" + ("" if ok else "  <-- differs"))
    print(f"{H} of {N} rounds as answered")
