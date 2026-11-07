"""Reference tracks for the user to check (the user, 2026-11-02: not their crosshair as the truth): per clip, the best
tracker (pf3: off-board points typical, turns, lam 0.3) with fixed-lag smoothing (each frame judged with the 8 frames
after it), 3 seeds, the median, sharpened to the best outline within 5 px; where the seeds disagree by over 15 px the
frame is flagged. Saved to <work>/ld/references-unchecked/<key>.json, the only place written here (approving one is
moving it to ld/references: README.md); a review set per clip in <review root>/ld-reference/.

usage: reference.py <clip keys...>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import pf3  # noqa: E402
import shape_track as st  # noqa: E402
from lib import UNCHECKED, common, end_j, load_clip, reference_of  # noqa: E402

KW = dict(mu=1.0, track=0.0, reseed=0.0, gamma=3.0, lam=0.3, offboard=1.0, turn=0.1, turn_edge=0.3, lag=8)
SEEDS = (0, 1, 2)
FLAG = 15.0  # px: the seeds' spread that flags a frame
REVIEW = common.root() / "ld-reference"
HEAD = 54


def one(args):
    key, seed = args
    c = load_clip(key)
    sm = {}
    out, (pts, nrm, o, j0) = pf3.run(c, seed=seed, smooth=sm, **KW)
    return key, seed, {j: (sm[j].tolist(), float(out[j][1])) for j in sm}, j0


def build(key, runs, page=True):
    """The clip's unchecked reference from its 3 runs and, with `page`, its review set (the reference is needed for the
    other groups even where this group's set is there, and that set is never written over)."""
    if reference_of(key)[1]:
        raise ValueError(f"{key}: the user approved its reference: never rebuilt")
    c = load_clip(key)
    j0, o = st.pick_template(c)
    pts, nrm = st.template_from(o["mask"], rim=8, only_rim=True)
    end = end_j(c)
    start_i = c["first"] + round(4.6 * c["fps"])
    ref = {}
    for j in range(j0 + 1, end):
        Z = np.array([runs[s][j][0] for s in SEEDS])
        th = runs[SEEDS[0]][j][1]
        med = np.median(Z, 0)
        spread = float(np.hypot(*(Z - med).T).max())
        resp = st.features(c["img"][j])
        ys, xs = np.mgrid[-5:6, -5:6]
        cands = (np.stack([xs.ravel(), ys.ravel()], 1) + med).astype(np.float32)
        sc = pf3.score_mixed(resp, pts, nrm, cands, th, pf3.typical(resp, pts, nrm, th))
        z = cands[int(np.argmax(sc))].astype(float)
        ref[j] = dict(k=j - j0, i=c["i"][j], x=float(z[0]), y=float(z[1]), th=th, spread=spread,
                      live=bool(c["i"][j] >= start_i))
    UNCHECKED.mkdir(parents=True, exist_ok=True)
    json.dump({"key": key, "j0": j0, "kw": KW, "seeds": SEEDS, "frames": ref},
              open(UNCHECKED / f"{key}.json", "w"))
    live = [r for r in ref.values() if r["live"]]
    print(f"{key}: {len(live)} round frames, {sum(r['spread'] > FLAG for r in live)} flagged, "
          f"median spread {np.median([r['spread'] for r in live]):.1f} px", flush=True)
    if not page:
        return
    # the review set: frames from the template frame to the SUCCESS banner
    dest = REVIEW / f"{key}-{c['fps']}fps"
    (dest / "frames").mkdir(parents=True, exist_ok=False)
    flagged = []
    for j in range(j0, end):
        img = c["img"][j].copy()
        r = ref.get(j)
        if r is not None:
            bad = r["spread"] > FLAG
            cs, sn = np.cos(r["th"]), np.sin(r["th"])
            poly = (pts @ np.array([[cs, sn], [-sn, cs]], np.float32) + (r["x"], r["y"])).astype(np.int32)
            cv2.polylines(img, [poly], True, (0, 220, 255) if bad else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (int(r["x"]), int(r["y"])), 3, (0, 220, 255) if bad else (0, 0, 255), -1)
            if bad and r["live"]:
                flagged.append(j - j0)
        head = np.full((HEAD, img.shape[1], 3), 20, np.uint8)
        phase = "white" if c["i"][j] < start_i else "after START"
        state = "-" if r is None else (f"runs DISAGREE ({r['spread']:.0f} px): look closely" if r["spread"] > FLAG
                                      else f"runs agree ({r['spread']:.0f} px)")
        cv2.putText(head, f"Reference to check  {key}  frame {j - j0}  ({phase})  {state}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, "red ring = the reference target; yellow = the 3 runs disagree; your crosshair is not used",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        frame = np.vstack([head, img])
        cv2.imwrite(str(dest / "frames" / f"{j - j0:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    runs_ = []
    for k in flagged:
        if runs_ and runs_[-1][1] == k - 1:
            runs_[-1][1] = k
        else:
            runs_.append([k, k])
    (dest / "problems.json").write_text(json.dumps(
        [{"frames": r, "what": f"the 3 runs disagree ({r[1] - r[0] + 1} frames): check the ring"} for r in runs_]))
    (dest / "README.md").write_text(
        f"Reference track for {key} ({c['fps']} fps), to be checked by eye: frame 0 is the white target, START is frame "
        f"{next(r['k'] for r in ref.values() if r['live'])}, the last frame is just before the SUCCESS banner. The red "
        "ring is where the reference puts the target (the best tracker, judged with the 8 frames after each one, 3 runs, "
        "their median); yellow where the 3 runs disagree by over 15 px. Please note every stretch where the ring is not "
        "on the target (e.g. \"frames 195-199 off\"); those get fixed by hand.\n\n"
        f"Settings: {json.dumps(KW)}\n", encoding="utf-8")


if __name__ == "__main__":
    keys = sys.argv[1:]
    runs = {k: {} for k in keys}
    with ProcessPoolExecutor(2) as ex:
        for key, seed, res, j0 in ex.map(one, [(k, s) for k in keys for s in SEEDS]):
            runs[key][seed] = res
            if len(runs[key]) == len(SEEDS):
                build(key, runs[key])
