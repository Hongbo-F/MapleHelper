"""Foreground of the stage: each frame minus the game's own stage (back.png) and its spotlight overlay as the game
blends it (lightEffect's frame, its alpha, a colour transform per cone fitted robustly: frame - back = a*(M ov + c)
- a*G back), what's left the Violettas (and the score box, masked). The kymograph: its column sums, a row a frame.
usage: fg.py <clip> <t0> <t1> <out prefix>"""
import json, sys
import cv2
import numpy as np
from PIL import Image
from overlay import LE, PICTURES, SPARK, sparkles
OX, OY, W, H = 478, 292, 412, 194
P = f"{PICTURES}/"
_back = np.asarray(Image.open(P + "back.png").convert("RGBA")).astype(np.float32)
B = np.pad(_back[2:2 + H, 2:2 + W, :3], ((0, H - (_back.shape[0] - 2)), (0, 0), (0, 0)), mode="edge")
COLS = [(0, 103), (103, 206), (206, 309), (309, 412)]
SPARK_B = [cv2.GaussianBlur(s, (0, 0), 1.0) for s in SPARK]
MASK = np.ones((H, W), np.float32)
MASK[:50, 140:255] = 0   # the score box
MASK[:30, :] = 0         # the dark band at the top (the curtains' valance)

def foreground(f):
    """|frame - (stage + overlay)| summed over colour, and whether the overlay was on."""
    win = f[OY:OY + H, OX:OX + W]
    w = cv2.GaussianBlur(sparkles(win), (0, 0), 1.0)
    cs = [float((w * s).sum()) / (float(np.sqrt((w * w).sum() * (s * s).sum())) + 1e-9) for s in SPARK_B]
    k = int(np.argmax(cs))
    if cs[k] < 0.5:
        return np.abs(win - B).sum(2) * MASK, False
    le = LE[k]
    a, ov = le[..., 3:] / 255, le[..., :3]
    pred = np.empty_like(win)
    for c0, c1 in COLS:
        sl = slice(c0, c1)
        A, O, Bs, F = a[:, sl], ov[:, sl], B[:, sl], win[:, sl]
        X = np.concatenate([A * O, A, A * Bs], 2).reshape(-1, 7)
        Y = (F - Bs).reshape(-1, 3)
        keep = np.ones(len(X), bool)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
            res = np.abs(X @ coef - Y).sum(1)
            keep = res < np.percentile(res, 65)
        pred[:, sl] = (X @ coef).reshape(F.shape) + Bs
    return np.abs(win - pred).sum(2) * MASK, True

if __name__ == "__main__":
    clip, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    rows = [json.loads(l) for l in open(clip + "/index.jsonl")]
    fr = [r for r in rows if "file" in r and t0 <= r["t"] <= t1]
    K, ts, lit = [], [], []
    for r in fr:
        f = np.asarray(Image.open(clip + "/" + r["file"]).convert("RGB")).astype(np.float32)
        g, on = foreground(f)
        g = np.clip(g - 40, 0, None)  # the residual's noise floor off
        K.append(g.sum(0))
        ts.append(r["t"]); lit.append(on)
    K = np.array(K, np.float32)
    np.savez(out + ".npz", K=K, t=np.array(ts), lit=np.array(lit))
    img = np.clip(K / np.percentile(K, 99) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(255 - img).resize((W * 2, len(ts) * 2), Image.NEAREST).save(out + ".png")
