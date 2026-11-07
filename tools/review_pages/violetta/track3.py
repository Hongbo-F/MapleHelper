"""Pose-locked detection: the four Violettas spin in step (every detection in a frame the same pose, or the one half a
turn on, which looks alike), so each frame is matched against its own pose only (the first pass's majority, dets/,
the nearest frame's where it has none), and its twin, at more sizes (down to 0.3: the ones passing behind), with more
candidates kept; the still sprites (maskOn, stand) where the cones are off. -> dets3/<clip>.jsonl
usage: track3.py <clip> <out> [--workers N]"""
import collections, json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import track2

SCALES = [0.3, 0.36, 0.43, 0.5, 0.58, 0.67, 0.77, 0.9, 1.05, 1.2, 1.4, 1.55]
BANKS = None

def banks():
    out = {}
    for k, p in enumerate(track2.POSES):
        a = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
        lst = []
        for s in SCALES:
            w, h = max(6, int(round(a.shape[1] * s / 2))), max(6, int(round(a.shape[0] * s / 2)))
            rgb = cv2.resize(a[..., :3], (w, h), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(a[..., 3], (w, h), interpolation=cv2.INTER_AREA) > 128).astype(np.float32)
            lst.append((s, k, rgb, np.repeat(m[..., None], 3, 2)))
        out[k] = lst
    return out

def work(args):
    global BANKS
    clip, r, poses = args
    if BANKS is None:
        cv2.setNumThreads(1)
        BANKS = banks()
    f = np.asarray(Image.open(clip + "/" + r["file"]).convert("RGB")).astype(np.float32)
    img, lit = track2.stage_fixed(f)
    track2.BANK = [b for k in poses for b in BANKS[k]]
    return {"t": r["t"], "i": r["i"], "lit": lit, "poses": poses, "dets": track2.detect(img, n=8, thresh=0.35)}

def frame_poses(first):
    """Each frame's pose (the majority of its strong spin detections) and its twin; the nearest frame's where none."""
    maj = {}
    for r in first:
        ps = [d["pose"] for d in r["dets"] if d["score"] >= 0.6 and d["pose"] < 8]
        if ps:
            maj[r["i"]] = collections.Counter(p % 4 for p in ps).most_common(1)[0][0]
    out = {}
    keys = sorted(maj)
    for r in first:
        if not r["lit"]:
            out[r["i"]] = [8, 9, 0, 4]  # the cones off: still, masked or not (and the spin's front)
            continue
        if keys:
            near = min(keys, key=lambda k: abs(k - r["i"]))
            p = maj[near]
            out[r["i"]] = [p, p + 4] if abs(near - r["i"]) <= 1 else [p, p + 4, (p + 1) % 4, (p + 1) % 4 + 4,
                                                                         (p - 1) % 4, (p - 1) % 4 + 4]
        else:
            out[r["i"]] = list(range(8))
    return out

if __name__ == "__main__":
    clip, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    first = [json.loads(l) for l in open(Path("dets") / (Path(clip).name + ".jsonl"))]
    poses = frame_poses(first)
    rows = {r["i"]: r for r in (json.loads(l) for l in open(clip + "/index.jsonl")) if "file" in r}
    jobs = [(clip, rows[f["i"]], poses[f["i"]]) for f in first]
    with Pool(workers) as pool, open(out, "w") as fh:
        for res in pool.imap(work, jobs, chunksize=8):
            fh.write(json.dumps(res) + "\n")
