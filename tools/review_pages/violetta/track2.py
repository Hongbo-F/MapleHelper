"""track.py at half size (the stage and the sprites halved), for whole games; frames split over `--workers` processes.
usage: track2.py <clip> <t0> <t1> <out.jsonl> [--workers N]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
from common import PICTURES  # noqa: E402

SCALES = [0.45, 0.55, 0.65, 0.75, 0.9, 1.05, 1.2, 1.4, 1.55]
P = f"{PICTURES}/"
POSES = [P + f"npcCommon/move/{i}.png" for i in range(8)] + [P + "npc/maskOn/5.png", P + "npc/stand/0.png"]
OX, OY = 478, 292
BANK = None

def bank():
    out = []
    for s in SCALES:
        for k, p in enumerate(POSES):
            a = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
            w, h = max(8, int(round(a.shape[1] * s / 2))), max(8, int(round(a.shape[0] * s / 2)))
            rgb = cv2.resize(a[..., :3], (w, h), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(a[..., 3], (w, h), interpolation=cv2.INTER_AREA) > 128).astype(np.float32)
            out.append((s, k, rgb, np.repeat(m[..., None], 3, 2)))
    return out

def detect(img, n=6, thresh=0.4):
    half = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cands = []
    for s, k, t, m in BANK:
        if t.shape[0] > half.shape[0] or t.shape[1] > half.shape[1]:
            continue
        r = cv2.matchTemplate(half, t, cv2.TM_CCOEFF_NORMED, mask=m)
        r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
        dil = cv2.dilate(r, np.ones((7, 7), np.uint8))
        ys, xs = np.where((r == dil) & (r > thresh))
        for y, x in zip(ys, xs):
            cands.append((float(r[y, x]), 2 * (x + t.shape[1] / 2), 2 * (y + t.shape[0] / 2), s, k,
                          2 * t.shape[1], 2 * t.shape[0]))
    cands.sort(reverse=True)
    kept = []
    for c in cands:
        if all(abs(c[1] - d[1]) > 0.45 * min(c[5], d[5]) or abs(c[2] - d[2]) > 0.45 * min(c[6], d[6]) for d in kept):
            kept.append(c)
        if len(kept) == n:
            break
    return [{"score": round(c[0], 3), "x": round(c[1] + OX, 1), "y": round(c[2] + OY, 1), "scale": c[3], "pose": c[4]}
            for c in kept]

def stage_fixed(f):
    """The stage with the overlay off, the overlay at its place (OX, OY): its frame the one whose stars fit best there
    (lit if that fits over 0.5)."""
    from overlay import SPARK, sparkles
    from unveil import unveil
    win = cv2.GaussianBlur(sparkles(f[OY:OY + 194, OX:OX + 412]), (0, 0), 1.0)
    best = (-1.0, 0)
    for k, sp in enumerate(SPARK):
        t = cv2.GaussianBlur(sp, (0, 0), 1.0)
        den = float(np.sqrt((win * win).sum() * (t * t).sum()))
        c = float((win * t).sum()) / den if den > 0 else 0.0
        best = max(best, (c, k))
    if best[0] > 0.5:
        u, _, _ = unveil(f, best[1], OX, OY)
        return np.clip(np.where(np.isnan(u), 255, u), 0, 255).astype(np.float32), True
    return f[OY:OY + 194, OX:OX + 412].astype(np.float32), False


def work(args):
    global BANK
    clip, r = args
    if BANK is None:
        cv2.setNumThreads(1)
        BANK = bank()
    f = np.asarray(Image.open(clip + "/" + r["file"]).convert("RGB")).astype(np.float32)
    img, lit = stage_fixed(f)
    return {"t": r["t"], "i": r["i"], "lit": lit, "dets": detect(img)}

if __name__ == "__main__":
    clip, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    rows = [json.loads(l) for l in open(clip + "/index.jsonl")]
    fr = [r for r in rows if "file" in r and t0 <= r["t"] <= t1]
    with Pool(workers) as pool, open(out, "w") as fh:
        for res in pool.imap(work, [(clip, r) for r in fr], chunksize=8):
            fh.write(json.dumps(res) + "\n")
