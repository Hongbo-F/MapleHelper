"""track3.py (pose-locked) with the Violettas passing in front matched too: they're drawn up to ~1.5x and run past
the stage's bottom (and at the end slots its sides), so besides each whole sprite, at the bigger sizes its top 80% and
62% (the bottom cut off) and, at the biggest, it with a side cut off (80% kept); larger sizes (1.7, 1.9). A match's
centre is the whole sprite's. -> dets5/<clip>.jsonl (unveil2: the blend fitted against the stage)
usage: track4.py <clip> <out> [--workers N]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import track2, track3, unveil2

SCALES = [0.3, 0.36, 0.43, 0.5, 0.58, 0.67, 0.77, 0.9, 1.05, 1.2, 1.4, 1.55, 1.7, 1.9]
OX, OY = track2.OX, track2.OY
BANKS = None

def banks():
    out = {}
    for k, p in enumerate(track2.POSES):
        a = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
        lst = []
        for s in SCALES:
            W, H = max(6, int(round(a.shape[1] * s / 2))), max(6, int(round(a.shape[0] * s / 2)))
            rgb = cv2.resize(a[..., :3], (W, H), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(a[..., 3], (W, H), interpolation=cv2.INTER_AREA) > 128).astype(np.float32)
            cuts = [(0, H, 0, W)]
            if s >= 0.9:
                cuts += [(0, int(H * 0.8), 0, W), (0, int(H * 0.62), 0, W)]
            if s >= 1.2:
                cuts += [(0, H, int(W * 0.2), W), (0, H, 0, int(W * 0.8))]
            for y0, y1, x0, x1 in cuts:
                lst.append((s, k, np.ascontiguousarray(rgb[y0:y1, x0:x1]),
                            np.repeat(m[y0:y1, x0:x1][..., None], 3, 2), W, H, x0, y0))
        out[k] = lst
    return out

def detect(img, bank, n=8, thresh=0.35):
    half = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cands = []
    for s, k, t, m, W, H, cx0, cy0 in bank:
        if t.shape[0] > half.shape[0] or t.shape[1] > half.shape[1] or m.sum() < 30:
            continue
        r = cv2.matchTemplate(half, t, cv2.TM_CCOEFF_NORMED, mask=m)
        r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
        dil = cv2.dilate(r, np.ones((7, 7), np.uint8))
        ys, xs = np.where((r == dil) & (r > thresh))
        for y, x in zip(ys, xs):
            # the whole sprite's centre: the cut part's top-left less its offset in the sprite
            cands.append((float(r[y, x]), 2 * (x - cx0 + W / 2), 2 * (y - cy0 + H / 2), s, k, 2 * W, 2 * H,
                          (t.shape[1] < W) or (t.shape[0] < H)))
    cands.sort(reverse=True)
    kept = []
    for c in cands:
        if all(abs(c[1] - d[1]) > 0.45 * min(c[5], d[5]) or abs(c[2] - d[2]) > 0.45 * min(c[6], d[6]) for d in kept):
            kept.append(c)
        if len(kept) == n:
            break
    return [{"score": round(c[0], 3), "x": round(c[1] + OX, 1), "y": round(c[2] + OY, 1), "scale": c[3], "pose": c[4],
             "cut": bool(c[7])} for c in kept]

def work(args):
    global BANKS
    clip, r, poses = args
    if BANKS is None:
        cv2.setNumThreads(1)
        BANKS = banks()
    f = np.asarray(Image.open(clip + "/" + r["file"]).convert("RGB")).astype(np.float32)
    img, lit = unveil2.stage(f)
    return {"t": r["t"], "i": r["i"], "lit": lit, "poses": poses,
            "dets": detect(img, [b for k in poses for b in BANKS[k]])}

if __name__ == "__main__":
    clip, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    first = [json.loads(l) for l in open(Path("dets") / (Path(clip).name + ".jsonl"))]
    poses = track3.frame_poses(first)
    rows = {r["i"]: r for r in (json.loads(l) for l in open(clip + "/index.jsonl")) if "file" in r}
    jobs = [(clip, rows[f["i"]], poses[f["i"]]) for f in first]
    with Pool(workers) as pool, open(out, "w") as fh:
        for res in pool.imap(work, jobs, chunksize=8):
            fh.write(json.dumps(res) + "\n")
