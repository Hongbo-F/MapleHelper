"""Side-by-side review sets: the tracker before the speed work (c04c1bb) and after it (the working tree), seed 0, on
each clip: <review root>/ld-speed-compare/<clip>-<fps>fps. Each frame: left before, right after; red = the tracker's
outline, thin green = the user's approved reference (or an unchecked one, worded so); the header: each side's px off
the reference and its time.
Frames numbered as the ld-reference sets (0 = the template frame, the white target), to just before SUCCESS.
problems.json: the frames where either side is over 15 px off the reference.

usage: review_compare.py [clip keys...]
"""
import json
import sys
import time

import cv2
import numpy as np

from lib import clip, common, folder, reference_of  # noqa: E402  (first: it puts the repo on the path)
from gamelab import paths  # noqa: E402
from gamelab.perception import lie_target as after  # noqa: E402
import lie_target_c04c1bb as before  # noqa: E402

ROOT = common.root() / "ld-speed-compare"
REFS = paths.REPO / "tests/fixtures/lie_detector/references"  # the record of what the user approved: its centres
HEAD, OFF = 58, 15.0


def run(mod, data, start_t, fps):
    tr = mod.TargetTracker(fps=fps, start_t=start_t, seed=0)
    out, t0 = {}, time.perf_counter()
    for i, t, b in data:
        r = tr.step(b, t)
        if r is not None:
            out[i] = r
    return out, tr.template, 1000 * (time.perf_counter() - t0) / len(data)


def outline(img, pts, x, y, angle, col, w):
    c, s = np.cos(angle), np.sin(angle)
    poly = (pts @ np.array([[c, s], [-s, c]], np.float32) + (x, y)).astype(np.int32)
    cv2.polylines(img, [poly], True, col, w, cv2.LINE_AA)


def unchecked(key, sref):
    """An unchecked reference in the fixtures' shape."""
    v = clip(key)
    return dict(clip=folder(key).name, fps=v["fps"], first=v["first"], start=v["start"], end=v["end"], fixed_by_hand=[],
                centre={str(f["i"]): [f["x"], f["y"]] for f in sref["frames"].values() if v["start"] <= f["i"] < v["end"]})


def render(key):
    path, approved = reference_of(key)
    sref = json.load(open(path))  # (its angle per frame, and the template frame's number)
    ref = json.loads((REFS / f"{key}.json").read_text(encoding="utf-8")) if approved else unchecked(key, sref)
    green = "your approved reference" if approved else "the new reference (not yet checked)"
    i0 = next(f["i"] - f["k"] for f in sref["frames"].values())
    data = list(after.clip_frames(paths.ALARM_CLIPS / ref["clip"], ref["first"], ref["end"] - 1))
    start_t = next(t for i, t, _ in data if i >= ref["start"])
    a, ta, ma = run(before, data, start_t, ref["fps"])
    b, tb, mb = run(after, data, start_t, ref["fps"])
    pts = tb[0]
    angle_of = {f["i"]: f["th"] for f in sref["frames"].values()}
    dest = ROOT / f"{key}-{ref['fps']}fps"
    (dest / "frames").mkdir(parents=True, exist_ok=False)
    k0 = next(n for n, (i, _, _) in enumerate(data) if i == i0)
    offs = []
    for n, (i, t, board) in enumerate(data[k0:]):
        centre = ref["centre"].get(str(i))
        panels, row = [], []
        for name, res, ms in (("before (c04c1bb)", a, ma), ("after (the speed work)", b, mb)):
            img = board.copy()
            if centre is not None:
                outline(img, pts, centre[0], centre[1], angle_of.get(i, 0.0), (80, 220, 80), 1)
            r = res.get(i)
            off = None
            if r is not None:
                outline(img, pts, r["x"], r["y"], r["angle"], (0, 0, 255), 2)
                if centre is not None:
                    off = float(np.hypot(r["x"] - centre[0], r["y"] - centre[1]))
            row.append(off)
            label = f"{name}  {ms:.0f} ms a frame" + ("" if off is None else f"  {off:.0f} px off" +
                                                     ("  <-- OFF" if off > OFF else ""))
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
            cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(img)
        offs.append((n, row))
        head = np.full((HEAD, 2 * board.shape[1] + 6, 3), 20, np.uint8)
        cv2.putText(head, f"{key}  frame {n}  (clip frame {i})" + ("  white" if i < ref["start"] else ""), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, f"red = the tracker's outline, thin green = {green}; left before, right after",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        frame = np.vstack([head, np.hstack([panels[0], np.full((board.shape[0], 6, 3), 255, np.uint8), panels[1]])])
        cv2.imwrite(str(dest / "frames" / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    runs = []
    for n, row in offs:
        bad = [name for name, off in zip(("before", "after"), row) if off is not None and off > OFF]
        if not bad:
            continue
        if runs and runs[-1][1] == n - 1 and runs[-1][2] == bad:
            runs[-1][1] = n
        else:
            runs.append([n, n, bad])
    (dest / "problems.json").write_text(json.dumps(
        [{"frames": r[:2], "what": f"{' and '.join(r[2])} over {OFF:.0f} px off the reference ({r[1] - r[0] + 1} frames)"}
         for r in runs]))
    da = np.array([o[0] for _, o in offs if o[0] is not None and o[1] is not None])
    db = np.array([o[1] for _, o in offs if o[0] is not None and o[1] is not None])
    made = ("the reference the user approved" if approved
            else "the reference made for this clip, not yet checked by the user")
    (dest / "README.md").write_text(
        f"{key}: the Lie Detector target tracker before the speed work (c04c1bb, left) and after it (right), seed 0. "
        f"Red: the tracker's outline; thin green: {made}. Frame 0 is the white target, the set "
        f"ends just before the SUCCESS banner.\n\nBefore: {da.mean():.1f} px off on average, {ma:.0f} ms a frame. "
        f"After: {db.mean():.1f} px off on average, {mb:.0f} ms a frame.\n", encoding="utf-8")
    print(f"{key}: before {da.mean():.1f} px off ({ma:.0f} ms), after {db.mean():.1f} px off ({mb:.0f} ms); "
          f"{len(runs)} stretches over {OFF:.0f} px", flush=True)


if __name__ == "__main__":
    for key in sys.argv[1:] or sorted(p.stem for p in REFS.glob("*.json")):
        render(key)
