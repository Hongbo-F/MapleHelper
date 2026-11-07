"""Take the spotlight overlay off a frame: per cone (the overlay's four columns) a colour transform fitted on its opaque
pixels (alpha >= 250), then under = (frame - a * T(overlay)) / (1 - a) where a < AMAX."""
import json, sys
import cv2
import numpy as np
from PIL import Image
from overlay import LE, H, W, locate, paths
AMAX = 0.97
FIT_TOP = 50
COLS = [(0, 103), (103, 206), (206, 309), (309, 412)]

def unveil(f, k=None, x=478, y=292):
    if k is None:
        _, k, x, y = locate(f)
    le = LE[k]
    a = le[..., 3:] / 255
    ov = le[..., :3]
    win = f[y:y + H, x:x + W]
    pred = np.zeros_like(win)
    fits = []
    for c0, c1 in COLS:
        m = le[:, c0:c1, 3] >= 250
        m[:FIT_TOP] = False  # (the score box covers the middle cones' tops)
        X = np.concatenate([ov[:, c0:c1][m], np.ones((m.sum(), 1), np.float32)], 1)
        Y = win[:, c0:c1][m]
        coef, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred[:, c0:c1] = np.concatenate([ov[:, c0:c1], np.ones(ov[:, c0:c1].shape[:2] + (1,), np.float32)], 2) @ coef
        fits.append(float(np.sqrt(((X @ coef - Y) ** 2).mean())))
    under = (win - a * pred) / np.maximum(1 - a, 1 - AMAX)
    under[(a[..., 0] >= AMAX)] = np.nan
    return under, fits, (k, x, y)

if __name__ == "__main__":
    C = f"{paths.ALARM_CLIPS}/20261103-013920-lie-detector-countdown/"
    rows = [json.loads(l) for l in open(C + "index.jsonl")]
    fr = [r for r in rows if "file" in r]
    tiles = []
    for tt in (34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 38.0):
        r = min(fr, key=lambda r: abs(r["t"] - tt))
        f = np.asarray(Image.open(C + r["file"]).convert("RGB")).astype(np.float32)
        u, fits, where = unveil(f)
        print(r["t"], where, [round(v, 1) for v in fits], "under range", np.nanpercentile(u, [1, 50, 99]).round(0))
        tiles.append(np.nan_to_num(np.clip(u, 0, 255), nan=0).astype(np.uint8))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    Image.fromarray(sheet).save("unveil_0139.jpg", quality=88)
