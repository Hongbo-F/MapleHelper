"""Review sets for pf3 (the live tracker, no smoothing) checked against the user-approved references, not the crosshair
(the user, 2026-11-02): <review root>/<group>/<clip>-<fps>fps, frames from the white target to the SUCCESS banner.

Each frame: the board; the particles (small dots, brighter = heavier); the tracked outline at its known spin (red);
the reference outline (thin green: an unchecked one, worded so, until the user approves it); a header with how far the
track is off the reference. problems.json lists the
stretches after START with the track over 15 px off the reference.

usage: review_pf3.py <group> <k:v+k:v kwargs> [clip keys...]
"""
import json
import sys

import cv2
import numpy as np

import pf3  # noqa: E402
import shape_track as st  # noqa: E402
from lib import CLIPS, common, end_j, load_clip, reference_of  # noqa: E402

ROOT = common.root()
HEAD = 54
OFF = 15.0  # px off the reference that makes a problem stretch


def render(key, group, kw):
    c = load_clip(key)
    snaps = {}
    out, (pts, nrm, o, j0) = pf3.run(c, snapshots=snaps, **kw)
    path, approved = reference_of(key)
    R = json.load(open(path))["frames"]
    green = "the reference you approved" if approved else "the new reference (not yet checked by you)"
    start = c["first"] + round(4.6 * c["fps"])
    dest = ROOT / group / f"{key}-{c['fps']}fps"
    (dest / "frames").mkdir(parents=True, exist_ok=False)
    rot = lambda th: np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]], np.float32)
    offs = []
    for k, j in enumerate(range(j0, end_j(c))):
        img = c["img"][j].copy()
        if j in snaps:
            P, w = snaps[j]
            wn = w / w.max()
            for (x, y), v in zip(P[::4].astype(int), wn[::4]):
                cv2.circle(img, (x, y), 1, (int(255 * v), int(255 * v), 255), -1)
        ref = R.get(str(j))
        if ref is not None:
            poly = (pts @ rot(ref["th"]) + (ref["x"], ref["y"])).astype(np.int32)
            cv2.polylines(img, [poly], True, (80, 220, 80), 1, cv2.LINE_AA)
        z, th, s = out[j]
        cv2.polylines(img, [(pts @ rot(th) + z).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        off = None if ref is None else float(np.hypot(z[0] - ref["x"], z[1] - ref["y"]))
        if off is not None and c["i"][j] >= start:
            offs.append((k, off))
        head = np.full((HEAD, img.shape[1], 3), 20, np.uint8)
        phase = "white" if c["i"][j] < start else "after START"
        cv2.putText(head, f"pf3 (turns 10%)  {key}  frame {k}  ({phase})  "
                    + ("" if off is None else f"{off:.0f} px off the reference" + ("  <-- OFF" if off > OFF else "")),
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, f"red = the live tracker's outline, thin green = {green}, dots = particles",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        frame = np.vstack([head, img])
        cv2.imwrite(str(dest / "frames" / f"{k:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    runs = []
    for k, off in offs:
        if off <= OFF:
            continue
        if runs and runs[-1][1] == k - 1:
            runs[-1][1] = k
            runs[-1][2] = max(runs[-1][2], off)
        else:
            runs.append([k, k, off])
    (dest / "problems.json").write_text(json.dumps(
        [{"frames": r[:2], "what": f"over {OFF:.0f} px off the reference ({r[1] - r[0] + 1} frames, up to {r[2]:.0f} px)"}
         for r in runs]))
    d = np.array([x for _, x in offs])
    made = "track you approved" if approved else "track made for this clip, not yet checked by you"
    (dest / "README.md").write_text(
        f"pf3, the live tracker (no smoothing), on {key} ({c['fps']} fps): the outline from the white target matched at "
        "its known spin; off-board template points count as a typical point; each frame 10% of the particles (30% "
        "within 70 px of an edge) take a new direction at the target's usual speed. Checked against the reference "
        f"{made} (thin green), not your crosshair. Frame 0 is the white target; the set ends just before "
        "the SUCCESS banner.\n\n"
        f"From START: {d.mean():.1f} px off on average, {np.mean(d <= 10):.0%} of frames within 10 px, "
        f"{np.mean(d > OFF):.0%} over {OFF:.0f} px.\n\nSettings: {json.dumps(kw)}\n", encoding="utf-8")
    print(f"{key}: {d.mean():.1f} px off, {len(runs)} stretches over {OFF:.0f} px", flush=True)


if __name__ == "__main__":
    group = sys.argv[1]
    kw = {k: float(v) for k, v in (a.split(":") for a in sys.argv[2].split("+"))}
    for key in sys.argv[3:] or list(CLIPS):
        render(key, group, kw)
