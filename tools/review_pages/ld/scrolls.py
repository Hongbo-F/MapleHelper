"""Board scroll between consecutive frames of each clip (phase correlation then ECC, foreign pixels masked):
s[j] = shift from frame j-1 to frame j; cached."""
import pickle, sys
import cv2, numpy as np
from lib import load_clip, foreign_mask, phase_shift, ecc_shift, clip, CLIPS, WORK

def scrolls(key):
    path = WORK / "cache" / f"scroll_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    c = load_clip(key)
    g = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in c["img"]]
    m = [(foreign_mask(b) == 0).astype(np.uint8) * 255 for b in c["img"]]
    s, cc = [np.zeros(2)], [1.0]
    for j in range(1, len(g)):
        p, _ = phase_shift(g[j - 1], g[j], m[j - 1])
        e, q = ecc_shift(g[j - 1], g[j], m[j], p)
        s.append(e if e is not None else p); cc.append(q)
    out = {"s": np.array(s), "cc": np.array(cc)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out

if __name__ == "__main__":
    for key in (sys.argv[1:] or CLIPS):
        c = load_clip(key); r = scrolls(key)
        fps, first = clip(key)["fps"], clip(key)["first"]
        start = first + round(4.6 * fps)
        dt = np.diff(c["t"])
        sc = np.hypot(*r["s"][1:].T) / np.maximum(dt, 1e-3)
        # cursor velocity (screen) and relative to board, after START
        rel, scr = [], []
        for j in range(1, len(c["i"])):
            if c["i"][j] < start or c["cur"][j] is None or c["cur"][j - 1] is None:
                continue
            v = (c["cur"][j] - c["cur"][j - 1]) / dt[j - 1]
            scr.append(np.hypot(*v)); rel.append(np.hypot(*(v - r["s"][j] / dt[j - 1])))
        print(f"{key}: scroll {np.median(sc):.0f} px/s (cc med {np.median(r['cc'][1:]):.3f} min {r['cc'][1:].min():.3f}); "
              f"after START cursor on screen {np.median(scr):.0f} px/s, relative to the board {np.median(rel):.0f} px/s")
