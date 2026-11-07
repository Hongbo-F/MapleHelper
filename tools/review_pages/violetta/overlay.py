"""Fit the spotlight overlay: which lightEffect frame, where, and the colour transform the game applies (per cone, a 3x3
matrix and an offset, fitted on the overlay's opaque pixels, alpha >= 250)."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
from common import PICTURES, paths  # noqa: E402
P = f"{PICTURES}/"
LE = [np.asarray(Image.open(P + f"lightEffect/{i}.png").convert("RGBA")).astype(np.float32) for i in range(8)]
H, W = 194, 412

def sparkles(rgb):  # bright and grey: the overlay's stars, whatever the cone's colour
    mx, mn = rgb.max(2), rgb.min(2)
    return ((mn > 200) & (mx - mn < 60)).astype(np.float32)

SPARK = [sparkles(le[..., :3] * (le[..., 3:] / 255)) for le in LE]

def locate(f, search=(440, 250, 540, 340)):
    """(k, x, y): the overlay frame and top-left whose stars best fit the frame's."""
    x0, y0, x1, y1 = search
    region = sparkles(f[y0:y1 + H, x0:x1 + W])
    region = cv2.GaussianBlur(region, (0, 0), 1.0)
    best = None
    for k, s in enumerate(SPARK):
        r = cv2.matchTemplate(region, cv2.GaussianBlur(s, (0, 0), 1.0), cv2.TM_CCORR_NORMED)
        y, x = np.unravel_index(np.argmax(r), r.shape)
        if best is None or r[y, x] > best[0]:
            best = (float(r[y, x]), k, x0 + int(x), y0 + int(y))
    return best

if __name__ == "__main__":
    C = f"{paths.ALARM_CLIPS}/20261103-013920-lie-detector-countdown/"
    rows = [json.loads(l) for l in open(C + "index.jsonl")]
    fr = [r for r in rows if "file" in r]
    for tt in (34.0, 34.07, 34.13, 34.2, 35.0, 36.0, 37.0):
        r = min(fr, key=lambda r: abs(r["t"] - tt))
        f = np.asarray(Image.open(C + r["file"]).convert("RGB")).astype(np.float32)
        print(r["t"], locate(f))
