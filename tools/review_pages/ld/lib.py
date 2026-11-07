"""Shared bits for the contour / residual experiments on Lie Detector clips (no YOLO)."""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
import clip_range  # noqa: E402
import common  # noqa: E402

X, Y, W, H = 308, 180, 750, 500  # the playable board at 1366x768
HERE = Path(__file__).resolve().parent
LAB = common.paths.LAB
WORK = common.work() / "ld"  # frame caches, worked-out ranges, unchecked references: not in git
APPROVED = HERE / "references"  # the reference tracks the user approved (in git)
UNCHECKED = WORK / "references-unchecked"  # the ones made since, until they do: reference.py writes only here

# the clips from before the live tracker's round files: their board runs, rates and ends, as eval_all and the first
# references found them (the 4 fps watchdog recordings dropped, the user 2026-11-02: "they are noise"; frames deleted)
CLIPS = json.load(open(HERE / "clips.json", encoding="utf-8"))


def clip(key):
    """A clip's fps, first, last, start and end (frame numbers): clips.json's, else worked out once from its live round
    file and SUCCESS banner (clip_range), or None, having said why."""
    if key in CLIPS:
        return CLIPS[key]
    saved = WORK / "ranges" / f"{key}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    [d] = common.paths.ALARM_CLIPS.glob(f"{key}*")
    found = clip_range.clip_range(d, common.paths.LIE_TARGETS)
    if found is not None:
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(found), encoding="utf-8")
    return found


def end_j(c):
    """The index in load_clip's frames of the SUCCESS banner's first frame: where the sets stop."""
    return c["i"].index(clip(c["key"])["end"])


def reference_of(key):
    """A clip's reference track: (its file, whether the user approved it); the approved one first, else the unchecked
    one, which may not be there yet."""
    path = APPROVED / f"{key}.json"
    return (path, True) if path.exists() else (UNCHECKED / f"{key}.json", False)


def folder(key):
    fps = clip(key)["fps"]
    base = LAB / ("lie_detector" if fps == 4 else "alarm_clips")
    [d] = [d for d in base.iterdir() if d.name.startswith(key)]
    return d


def files(key):
    d = folder(key)
    rows = [json.loads(line) for line in open(d / "index.jsonl")]
    return d, {r["i"]: r for r in rows if "file" in r}


def board(d, fr, i):
    a = cv2.imread(str(d / fr[i]["file"]))
    if a is None or a.shape[:2] != (768, 1366):
        return None
    return np.ascontiguousarray(a[Y:Y + H, X:X + W])


def cursor_of(b):
    """The green crosshair's centre on a board crop (the user's mouse: the ground truth), or None."""
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(g)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cents[k] if stats[k, cv2.CC_STAT_AREA] > 80 else None


def foreign_mask(b):
    """255 where the pixel isn't the rock board: white (the target while white), the green cursor, the blue
    countdown/START text."""
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (s < 30) & (v > 230)
    foreign = (s > 80) & (h > 40) & (h < 170)
    bad = (white | foreign).astype(np.uint8)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(bad, np.ones((11, 11), np.uint8)) * 255


def load_clip(key, cache=True):
    """frames (list of board crops, BGR), their numbers, times, cursors."""
    path = WORK / "cache" / f"frames_{key}.pkl"
    if cache and path.exists():
        return pickle.loads(path.read_bytes())
    r = clip(key)
    fps, first, last = r["fps"], r["first"], r["last"]
    d, fr = files(key)
    out = {"key": key, "fps": fps, "first": first, "last": last, "i": [], "t": [], "img": [], "cur": []}
    for i in range(first, last + 1):
        if i not in fr:
            continue
        b = board(d, fr, i)
        if b is None:
            continue
        out["i"].append(i)
        out["t"].append(fr[i]["t"])
        out["img"].append(b)
        out["cur"].append(cursor_of(b))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def ecc_shift(ga, gb, ma, init=(0.0, 0.0)):
    """Translation s with board point x in A at x + s in B (ECC on grey levels, masked)."""
    warp = np.array([[1, 0, -init[0]], [0, 1, -init[1]]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4)
    try:
        cc, warp = cv2.findTransformECC(gb, ga, warp, cv2.MOTION_TRANSLATION, crit, ma, 5)
    except cv2.error:
        return None, 0.0
    return -warp[:, 2].copy(), cc


def phase_shift(ga, gb, ma):
    fa, fb = ga.astype(np.float32), gb.astype(np.float32)
    fa[ma == 0] = fa[ma > 0].mean()
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(fa, fb, win)
    return np.array([dx, dy]), resp
