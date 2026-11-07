"""Track the made-up Violetta through a recorded Violetta game, from the detections (dets/<clip>.jsonl, track2.py):

- the slots: where the four stand (the medians of the strong detections' x in the first still stretch);
- the made-up one at the start: the slot whose Violetta matches the game's made-up sprite (npcSpecial/stand) best
  against the plain one (npc/stand), on the frames before the cones come on;
- the rounds: each answer window, the buttons lit cyan;
- each shuffle (from one still arrangement to the next answer window): the permutation of the 24 whose four paths (each
  Violetta from its slot to its new one, straight in x over the stretch where they move) the detections support most;
- the answers entered: the score box's digits (number1's glyphs) at the end.
usage: solve.py <clip folder> <dets.jsonl>"""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
from common import PICTURES  # noqa: E402

P = f"{PICTURES}/"
BOXES = [645, 666, 687, 708]  # the score box's four digit tiles (native x), y 295-313
DIGITS = {d: np.asarray(Image.open(P + f"number1/{d}.png").convert("RGBA")) for d in range(1, 5)}
SIGMA = 14.0      # px: a detection's support for a path at that distance falls off so
STILL_PX = 18.0   # a detection within this of a slot is at it


def frames(clip):
    rows = [json.loads(l) for l in open(Path(clip) / "index.jsonl")]
    return [r for r in rows if "file" in r]


def load(clip, r):
    return np.asarray(Image.open(Path(clip) / r["file"]).convert("RGB"))


def lit_buttons(f):
    row = f[505:540, 480:900].astype(int)
    return ((row[..., 2] > 150) & (row[..., 1] > 130) & (row[..., 0] < 90)).mean() > 0.3


def read_answers(f):
    out = []
    for x in BOXES:
        tile = f[293:315, x - 2:x + 22].astype(np.float32)
        yellow = ((tile[..., 0] > 180) & (tile[..., 1] > 140) & (tile[..., 2] < 110)).astype(np.float32)
        if yellow.sum() < 10:
            out.append(None)
            continue
        best = None
        for d, g in DIGITS.items():
            gm = ((g[..., 0] > 180) & (g[..., 1] > 140) & (g[..., 2] < 110) & (g[..., 3] > 128)).astype(np.float32)
            r = cv2.matchTemplate(yellow, gm, cv2.TM_CCOEFF_NORMED)
            s = float(np.nan_to_num(r).max())
            if best is None or s > best[1]:
                best = (d, s)
        out.append(best[0] if best[1] > 0.5 else None)
    return out


def slots_of(dets):
    """The four slots' x: the strong detections' x in still frames (4 of them strong), clustered."""
    xs = []
    for row in dets:
        strong = sorted([d for d in row["dets"] if d["score"] > 0.7], key=lambda d: d["x"])
        if len(strong) == 4 and all(b["x"] - a["x"] > 60 for a, b in zip(strong, strong[1:])):
            xs.append([d["x"] for d in strong])
    if not xs:
        raise ValueError("no still frame with the four standing")
    return list(np.median(np.array(xs), 0))


def special_slot(clip, rows, dets, slots):
    """The made-up Violetta's slot before the cones come on: on the first frames where the four stand at the slots,
    npcSpecial/stand against npc/stand around each (both sprites' best over their 8 frames)."""
    sp = [np.asarray(Image.open(P + f"npcSpecial/stand/{i}.png").convert("RGBA")).astype(np.float32) for i in range(8)]
    pl = [np.asarray(Image.open(P + f"npc/stand/{i}.png").convert("RGBA")).astype(np.float32) for i in range(8)]
    first_lit = next((row["t"] for row in dets if row["lit"]), float("inf"))
    still = []
    for row in dets:
        if row["t"] >= first_lit:
            continue
        at = [max((d for d in row["dets"] if abs(d["x"] - sx) < 25 and d["score"] > 0.55), key=lambda d: d["score"],
                  default=None) for sx in slots]
        if all(at):
            still.append((row, at))
    votes = np.zeros(4)

    def best(win, sprites):
        return max(float(np.nan_to_num(cv2.matchTemplate(
            win, s[..., :3], cv2.TM_CCOEFF_NORMED,
            mask=np.repeat((s[..., 3:] > 128).astype(np.float32), 3, 2))).max()) for s in sprites)
    for row, strong in still[:12]:
        r = next(r for r in rows if r["i"] == row["i"])
        f = load(clip, r).astype(np.float32)
        for k, d in enumerate(strong):
            x, y = int(d["x"]), int(d["y"])
            win = f[y - 75:y + 75, x - 70:x + 70]
            votes[k] += best(win, sp) - best(win, pl)
    return int(np.argmax(votes)), votes / max(1, min(12, len(still)))


def solve_round(dets, slots, ta, tb):
    """The permutation (slot i -> perm[i]) between still arrangements at ta and tb, and its support against the
    identity's and the runner-up's: each path straight in x over [m0, m1], the stretch where detections are off the
    slots."""
    seg = [row for row in dets if ta <= row["t"] <= tb]
    off = [row["t"] for row in seg if any(min(abs(d["x"] - s) for s in slots) > 2 * STILL_PX and d["score"] > 0.5
                                         for d in row["dets"])]
    if len(off) < 3:
        return tuple(range(4)), {}, None
    m0, m1 = float(np.percentile(off, 5)), float(np.percentile(off, 95))
    scores = {}
    for perm in itertools.permutations(range(4)):
        total = 0.0
        for row in seg:
            if not (m0 <= row["t"] <= m1):
                continue
            u = (row["t"] - m0) / max(1e-6, m1 - m0)
            for i in range(4):
                px = slots[i] + (slots[perm[i]] - slots[i]) * u
                total += max((d["score"] * np.exp(-0.5 * ((d["x"] - px) / SIGMA) ** 2) for d in row["dets"]),
                             default=0.0)
        scores[perm] = total
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[0], scores, (m0, m1)


def main(clip, dets_path):
    rows = frames(clip)
    dets = [json.loads(l) for l in open(dets_path)]
    slots = slots_of(dets)
    # the answer windows: runs of frames with the buttons lit
    t_first = dets[0]["t"]
    lit, windows = False, []
    for r in rows:
        if r["t"] < t_first:
            continue
        on = lit_buttons(load(clip, r))
        if on and not lit:
            windows.append(r["t"])
        lit = on
    start, votes = special_slot(clip, rows, dets, slots)
    answers, n = [None] * 4, 0
    for r in rows[::-3]:  # the frame showing the most of the score box's digits, from the end
        if windows and r["t"] < windows[-1]:
            break
        a = read_answers(load(clip, r))
        if sum(v is not None for v in a) > n:
            answers, n = a, sum(v is not None for v in a)
        if n == 4:
            break
    # the still arrangement before the first shuffle: the last still frame before the cones come on
    first_lit = next((row["t"] for row in dets if row["lit"]), None)
    ta = first_lit
    where, out = start, []
    for k, tb in enumerate(windows):
        perm, info, _ = solve_steps(dets, slots, ta, tb)
        moved = [(span, p) for span, p in info["steps"] if p != [1, 2, 3, 4]]
        where = perm[where]
        out.append({"round": k + 1, "window_t": round(tb, 2), "steps": moved,
                    "margins": [m for m, (span, p) in zip(info["margins"], info["steps"]) if p != [1, 2, 3, 4]],
                    "perm": [p + 1 for p in perm], "predicted": where + 1,
                    "answered": answers[k] if k < len(answers) else None})
        ta = tb
    return {"clip": Path(clip).name, "slots": [round(s, 1) for s in slots], "made_up_start": start + 1,
            "start_votes": [round(v, 3) for v in votes], "answers": answers, "rounds": out}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=1))


def path_support(seg, ts, xs, ws, slots, i, j, step=0.1, min_s=0.3, max_s=5.0):
    """How well the detections support Violetta i going from slot i to slot j over the segment (the mean, over its
    frames, of the best detection's weight near the path), the path at slot i until a, straight to slot j by b, then
    there: the best (a, b) on a `step` grid. i == j: at slot i all along."""
    n = len(seg)

    def support(path):  # path: (frames,) expected x
        k = ws * np.exp(-0.5 * ((xs - path[:, None]) / SIGMA) ** 2)
        return float(k.max(1).sum() / n)
    if i == j:
        return support(np.full(n, slots[i])), None
    best = (-1.0, None)
    grid = np.arange(ts[0], ts[-1] + 1e-9, step)
    for a in grid:
        for b in grid[(grid >= a + min_s) & (grid <= a + max_s)]:
            u = np.clip((ts - a) / (b - a), 0, 1)
            s = support(slots[i] + (slots[j] - slots[i]) * u)
            if s > best[0]:
                best = (s, (round(float(a), 2), round(float(b), 2)))
    return best


def solve_round2(dets, slots, ta, tb):
    """The permutation whose four paths (each its own start and end) the detections support most, its margin."""
    seg = [row for row in dets if ta <= row["t"] <= tb]
    m = max(len(r["dets"]) for r in seg) if seg else 1
    ts = np.array([r["t"] for r in seg])
    xs = np.full((len(seg), m), -1e4)
    ws = np.zeros((len(seg), m))
    for k, r in enumerate(seg):
        for q, d in enumerate(r["dets"]):
            xs[k, q], ws[k, q] = d["x"], max(0.0, d["score"] - 0.4) / 0.6
    sup = {(i, j): path_support(seg, ts, xs, ws, slots, i, j) for i in range(4) for j in range(4)}
    scores = {perm: sum(sup[(i, perm[i])][0] for i in range(4)) for perm in itertools.permutations(range(4))}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    return best, scores, [sup[(i, best[i])][1] for i in range(4)]


def solve_round3(dets, slots, ta, tb):
    """solve_round2's paths (each pair's own best start and end), each permutation scored with every detection
    counted for at most one Violetta a frame (the best assignment), so two paths can't share the same detections."""
    seg = [row for row in dets if ta <= row["t"] <= tb]
    m = max(len(r["dets"]) for r in seg) if seg else 1
    ts = np.array([r["t"] for r in seg])
    xs = np.full((len(seg), m), -1e4)
    ws = np.zeros((len(seg), m))
    for k, r in enumerate(seg):
        for q, d in enumerate(r["dets"]):
            xs[k, q], ws[k, q] = d["x"], max(0.0, d["score"] - 0.4) / 0.6
    spans = {}
    paths = {}
    for i in range(4):
        for j in range(4):
            _, span = path_support(seg, ts, xs, ws, slots, i, j)
            spans[(i, j)] = span
            if span is None:
                paths[(i, j)] = np.full(len(seg), slots[i])
            else:
                a, b = span
                u = np.clip((ts - a) / (b - a), 0, 1)
                paths[(i, j)] = slots[i] + (slots[j] - slots[i]) * u
    scores = {}
    for perm in itertools.permutations(range(4)):
        P = np.stack([paths[(i, perm[i])] for i in range(4)], 1)  # frames x 4
        total = 0.0
        for k in range(len(seg)):
            K = ws[k][None, :] * np.exp(-0.5 * ((xs[k][None, :] - P[k][:, None]) / SIGMA) ** 2)  # 4 x m
            total += greedy_assign(K)
        scores[perm] = total / max(1, len(seg))
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[0], scores, [spans[(i, ranked[0][i])] for i in range(4)]


def greedy_assign(K):
    """The sum of a greedy one-to-one assignment's weights (the largest pair first, its row and column out)."""
    K = K.copy()
    total = 0.0
    for _ in range(min(K.shape)):
        r, c = np.unravel_index(np.argmax(K), K.shape)
        if K[r, c] <= 0:
            break
        total += float(K[r, c])
        K[r, :] = -1
        K[:, c] = -1
    return total


def still_frames(seg, slots, near=30.0, strong=0.55):
    """Whether each frame has all four standing: a strong detection within `near` of every slot."""
    return [all(any(abs(d["x"] - s) < near and d["score"] >= strong for d in row["dets"]) for s in slots)
            for row in seg]


def solve_steps(dets, slots, ta, tb, min_still=3):
    """A round split into its steps at the moments all four stand at their slots (min_still frames in a row), each
    step solved (solve_round3) from the still frames before it to those after, the permutations chained."""
    seg = [row for row in dets if ta <= row["t"] <= tb]
    still = still_frames(seg, slots)
    runs, k = [], 0
    while k < len(seg):  # the still runs: (first, last) frame indices
        if still[k]:
            j = k
            while j + 1 < len(seg) and still[j + 1]:
                j += 1
            if j - k + 1 >= min_still:
                runs.append((k, j))
            k = j + 1
        else:
            k += 1
    # the steps: between consecutive still runs, where some frame isn't still
    bounds = [(-1, -1)] + runs + [(len(seg), len(seg))]
    total = tuple(range(4))
    steps, margins = [], []
    for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
        if b0 - a1 <= 1:
            continue
        s0 = seg[max(0, a1)]["t"]
        s1 = seg[min(len(seg) - 1, b0)]["t"]
        perm, scores, spans = solve_round3(dets, slots, s0, s1)
        v = sorted(scores.values(), reverse=True)
        margins.append(round((v[0] - v[1]) / v[0], 3) if v[0] else None)
        steps.append(((round(s0, 2), round(s1, 2)), [p + 1 for p in perm]))
        total = tuple(perm[total[i]] for i in range(4))  # slot i -> after all steps so far
    scores = {total: 1.0}
    return total, {"steps": steps, "margins": margins}, None
