"""Her cues' models made at calibration from saved frames: her plate's (perception.plate.PlateModel) and her pose bank
(perception.poses.PoseBank), kept in her lab folder (lab/characters/<her files>/plate_1366x768.npz, poses_1366x768.npz;
git-ignored), for the cues that find her where her tag's gray match isn't sure (README: Her place by her plate and her
body).

From a folder of lossless frames (PNG, as the golden clips are: their exact colours are what both models are made of,
so not the rolling recorder's JPEG) with its index.jsonl (time order; `t` each frame's time): her tag's sure frames by
the full gray template (her tag as NameTag cuts it), the plate's median picture around them (plate.sure_centres: the
best 0.95 or more, no other peak 0.6, spread over where she was); the camera's view of each frame from the map's art
(localization.camreg on mapart's picture, sure views only), and her pose bank from the frames where her tag is sure
(poses.build_bank: her part of each frame against the art behind her, her feet the plate's centre + its feet offset,
on a platform's line by the map's data). Her feet's offset down from the plate's centre is measured on the way
(`feet_below`: where she stands, the platform's line under her is that far above the plate's centre in most frames;
16 for her orange-ribbon plate, where eyes.tag_below_feet's 11 was the old text tag's), and kept in the plate's model.

    python -m gamelab.localization.her_cues FRAMES_DIR [--map 10002073] [--out DIR] [--every N] [--no-bank]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..perception import plate as plates
from ..perception import poses
from ..screen import NATIVE

SURE = 0.9        # her tag sure for the bank: the gray template's best at least this ...
OTHER = 0.3       # ... no other peak within this of it ...
EDGE = 8          # ... and its centre this far above the HUD's top (camera-sprite's tag_sure)
ON_LINE = 2.0     # her feet on a platform's line: within this (map units) of it, its ends 1 past
FEET_RANGE = (-40, -4)   # her feet's offset down from the plate's centre looked for (screen px)
FEET_SURE = 0.5          # ... taken where this share of the frames with a line in that range agree on it


def frames_of(folder: Path, every: int = 1):
    """(file, t) of a folder's frames in its index's order (or its PNGs by name, 1/15 s apart, without one)."""
    index = folder / "index.jsonl"
    if index.exists():
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = [(folder / r["file"], float(r.get("t", k / 15))) for k, r in enumerate(rows) if "file" in r]
    else:
        out = [(p, k / 15) for k, p in enumerate(sorted(folder.glob("*.png")))]
    return out[::max(1, every)]


def rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"can't read {path}")
    return np.ascontiguousarray(bgr[..., ::-1])


def sure_tag(frame: np.ndarray, template: np.ndarray, hud: int) -> tuple[int, int] | None:
    """Her tag's centre where the full gray template is sure of it (SURE, OTHER, EDGE), else None."""
    peaks = plates.gray_peaks(frame, template, k=4, rows=hud)
    if not peaks or peaks[0][0] < SURE or peaks[0][2] > hud - EDGE:
        return None
    if any(p[0] > peaks[0][0] - OTHER for p in peaks[1:]):
        return None
    return peaks[0][1], peaks[0][2]


def feet_below(seen, platforms, x_offset: float) -> tuple[float | None, int, int]:
    """Her feet's offset down from her plate's centre: of (plate centre, view) pairs, the platform lines under her
    (across at the centre + x_offset) within FEET_RANGE of the centre, the commonest whole offset if FEET_SURE of them
    agree on it: (offset or None, how many agree, how many had one)."""
    counts: dict[int, int] = {}
    for (cx, cy), (vx, vy) in seen:
        mx = cx + x_offset + vx
        for p in platforms:
            if p.left <= mx <= p.right:
                d = int(round(p.y_at(mx) - vy - cy))
                if FEET_RANGE[0] <= d <= FEET_RANGE[1]:
                    counts[d] = counts.get(d, 0) + 1
    n = sum(counts.values())
    if not n:
        return None, 0, 0
    best = max(counts, key=lambda d: (counts[d], -abs(d)))
    return (float(best) if counts[best] >= FEET_SURE * n else None), counts[best], n


def build(folder: Path, map_id: int, out: Path, template: np.ndarray, every: int = 1, bank: bool = True,
          log=print) -> dict:
    """Both models from a folder of frames (the top), saved under `out`: what was made, and from how much."""
    frames = frames_of(folder, every)
    if not frames:
        raise ValueError(f"no frames in {folder}")
    hud = int(round(NATIVE.client[1] * NATIVE.hud_top))
    t0 = time.perf_counter()
    picks = plates.sure_centres(((p, rgb(p)) for p, _ in frames), template, hud)
    model = plates.PlateModel.build((rgb(p), c) for p, c in picks)
    model.save(plates.model_path(out))
    pl = plates.Plate(model)
    said = {"frames": len(frames), "plate": dict(pl.sizes, frames=model.n, path=str(plates.model_path(out)))}
    log(f"her plate from {model.n} sure frames of {len(frames)}: {pl.sizes} ({time.perf_counter() - t0:.1f} s)")
    if not bank:
        return said
    from . import camreg, mapart, mapdata  # (the map's art and data: built or read on first use)
    art = mapart.load(map_id)
    reg = camreg.Registration(art)
    background = poses.Background(art.rgb, (art.x0, art.y0), art.static)
    model_map = mapdata.load(map_id)

    def on_line(mx, my):
        return any(p.left - 1 <= mx <= p.right + 1 and abs(my - p.y_at(mx)) <= ON_LINE for p in model_map.platforms)

    seen = {"views": 0, "sure tags": 0}
    tags = []                       # (frame, her plate's centre, the view) where both are sure
    t0 = time.perf_counter()
    for path, t in frames:
        frame = rgb(path)
        view = reg.see(frame, t)
        if not view.sure:
            continue
        seen["views"] += 1
        tag = sure_tag(frame, template, hud)
        if tag is not None:
            tags.append((path, tag, (view.x, view.y)))
    seen["sure tags"] = len(tags)
    dy, agree, of = feet_below([(c, v) for _, c, v in tags], model_map.platforms, model.feet[0])
    log(f"her feet {dy} px from her plate's centre ({agree} of {of} frames with a line there agree; the model's "
        f"{model.feet[1]}) ({time.perf_counter() - t0:.1f} s)")
    if dy is not None and dy != model.feet[1]:
        model.feet = (model.feet[0], dy)
        model.save(plates.model_path(out))
    said["plate"]["feet"] = list(model.feet)
    said["plate"]["feet_frames"] = [agree, of]

    def samples():
        for path, (cx, cy), view in tags:
            yield rgb(path), (cx + model.feet[0], cy + model.feet[1]), view

    t0 = time.perf_counter()
    b = poses.build_bank(samples(), background, on_line, log=log)
    b.meta.update(made=time.strftime("%Y-%m-%d %H:%M"), folder=str(folder), map=map_id, **seen)
    b.save(poses.bank_path(out))
    said["bank"] = dict(b.meta, path=str(poses.bank_path(out)))
    log(f"her pose bank: {len(b.poses)} poses from {seen['sure tags']} sure frames of {seen['views']} placed "
        f"({time.perf_counter() - t0:.1f} s)")
    return said


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamelab.localization.her_cues", description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", type=Path, help="a folder of lossless frames (PNG) with its index.jsonl")
    ap.add_argument("--map", type=int, default=10002073, help="the map they're of (its art, for the bank)")
    ap.add_argument("--out", type=Path, help="where the models go (default: her lab folder)")
    ap.add_argument("--every", type=int, default=1, help="every Nth frame")
    ap.add_argument("--no-bank", action="store_true", help="her plate's model alone (no camera, no map art)")
    a = ap.parse_args(argv)
    from .. import paths
    mine = paths.of()
    template = np.load(paths.calibrated(mine.tag_full, NATIVE.suffix))
    said = build(a.frames, a.map, a.out or mine.lab, template, a.every, not a.no_bank)
    print(json.dumps(said, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
