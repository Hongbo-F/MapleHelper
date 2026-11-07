"""The stage with the spotlights taken off, by the blend fitted against the game's own stage (fg.py's model: per cone,
frame - back = a*(ov M + c) + a*(back G), fitted robustly over all its pixels, the Violettas the outliers), inverted
for what's under: V = (frame - a*(ov M + c)) (I + a G)^-1. Where that can't be told (the overlay nearly opaque, the
inverse blowing up), white. Returns (stage 194x412 float RGB, lit)."""
import cv2
import numpy as np
from fg import B, COLS, LE, SPARK_B, sparkles, OX, OY, W, H

def stage(f):
    win = f[OY:OY + H, OX:OX + W].astype(np.float32)
    w = cv2.GaussianBlur(sparkles(win), (0, 0), 1.0)
    cs = [float((w * s).sum()) / (float(np.sqrt((w * w).sum() * (s * s).sum())) + 1e-9) for s in SPARK_B]
    k = int(np.argmax(cs))
    if cs[k] < 0.5:
        return win, False
    le = LE[k]
    a, ov = le[..., 3:] / 255, le[..., :3]
    out = np.empty_like(win)
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
        M, cvec, G = coef[0:3], coef[3], coef[4:7]
        a1 = A.reshape(-1, 1)
        num = F.reshape(-1, 3) - a1 * (O.reshape(-1, 3) @ M + cvec)
        mats = np.eye(3)[None] + a1[:, :, None] * G[None]
        det = np.linalg.det(mats)
        ok = np.cbrt(np.abs(det)) > 0.06  # (I + aG) ~ (1 - a) I: its cube root is how much shows through
        V = np.full_like(num, 255.0)
        V[ok] = np.einsum("ni,nij->nj", num[ok], np.linalg.inv(mats[ok]))
        out[:, sl] = V.reshape(F.shape)
    return np.clip(out, 0, 255), True
