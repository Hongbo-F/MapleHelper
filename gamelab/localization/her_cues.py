"""Her cues' models made at calibration from saved frames: her tag's gray templates cut from her plate, her plate's
model (perception.plate.PlateModel) and her pose bank (perception.poses.PoseBank), for the cues that find her where her
tag's gray match isn't sure (README: Her place by her plate and her body). Written to a folder (`--out`; her lab
folder, lab/characters/<her files>/ or lab/ for the thief, only with `--lab`): plate_1366x768.npz, poses_1366x768.npz,
and the tag templates NameTag and the locator read, tag_template_1366x768.npy and tag_template_full_1366x768.npy (the
same picture: at the game's own size the half-size one is full size), which go with them: the models' px are from the
centre of that template's match.

From a folder of lossless frames (PNG, as the golden clips are: their exact colours are what both models are made of,
so not the rolling recorder's JPEG) with its index.jsonl (time order; `t` each frame's time), each step said in the log
and kept in what `build` returns (her_cues.json beside the models):
  1. her tag as she has it (calibration's full gray template) matched on every frame, its sure levels from its own
     scores (plate.sure_levels: the cluster of her clear tags, its spread; never above the prototype's 0.95 and 0.9),
     her plate's first model from its sure frames (plate.pick_centres: the best at the picks' level, no other peak
     0.6; one a 6 x 6 px cell of centres, 400 at most);
  2. her new tag template cut from that model's median picture (plate.tag_template: her letters, 2 px around them)
     and the model re-centred on it; matched on every frame again, its sure levels, the camera's view of each frame
     from the map's art (localization.camreg on mapart's picture, sure views only), and her plate's model made again
     from the new template's sure frames (the box around the new centre);
  3. her feet's offset from the plate's centre: across, the plate's own axis (PlateModel.axis, the middle of its
     stable columns: the game draws her plate centred on her; +1.5 for RiceBaby's, -3.5 from BugBaby's old tag's
     centre, which the build of 10-30 kept at the magician's +1.5 and had to be told), down where she stands
     (`feet_below`: the platform line under her, the commonest offset if FEET_SURE of the frames with a line near agree;
     16 for her orange-ribbon plate, where eyes.tag_below_feet's 11 was the old text tag's);
  4. her pose bank from the frames where her new tag is sure (its bank level, no other peak within OTHER, EDGE above
     the HUD's top) and the view too (poses.build_bank: her part of each frame against the art behind her, her feet
     the plate's centre + its offset, on a platform's line by the map's data), and her mirror axis by her poses seen
     both ways (poses.mirror_axis) said beside the plate's: a disagreement of a px or more is said as a warning.
`--keep-tag`: no new template (steps 1, 3, 4 with hers as she has it; the models then go with it).

BugBaby's of 10-30 (her train frames, 2,401: D:/MapleHelper-scratch/loc2build/bugbaby/v2/models, 276 s): her old text
tag's template 0.882 typical on her plate (the levels 0.857, 0.832), the one cut from her plate 52 x 15, its centre
(-1, 0) from the old one's, 0.996 typical (0.95, 0.9); her feet (-2.5, -16) from it (the old centre's -3.5 + 1); her
poses' mirror axis +0.0 px from her plate's.

    python -m gamelab.localization.her_cues FRAMES_DIR (--out DIR | --lab) [--map 10002073] [--every N] [--no-bank]
                                            [--keep-tag]
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

OTHER = 0.3       # her tag sure for the bank: the gray template's best at its bank level, no other peak within this ...
EDGE = 8          # ... and its centre this far above the HUD's top (camera-sprite's tag_sure)
ON_LINE = 2.0     # her feet on a platform's line: within this (map units) of it, its ends 1 past
FEET_RANGE = (-40, -4)   # her feet's offset down from the plate's centre looked for (screen px)
FEET_SURE = 0.5          # ... taken where this share of the frames with a line in that range agree on it
AXIS_WARN = 1.0          # her poses' mirror axis this far (px) from her plate's: said as a warning
TEMPLATE_NAMES = ("tag_template", "tag_template_full")   # (paths.TAG_TEMPLATE's and TAG_FULL's stems)


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


def sure_of(peaks: list, sure: float, hud: int) -> tuple[int, int] | None:
    """Her tag's centre where the full gray template's peaks (gray_peaks', k=4) are sure of it: the best `sure` or more,
    no other peak within OTHER of it, its centre EDGE above the HUD's top; else None."""
    if not peaks or peaks[0][0] < sure or peaks[0][2] > hud - EDGE:
        return None
    if any(p[0] > peaks[0][0] - OTHER for p in peaks[1:]):
        return None
    return peaks[0][1], peaks[0][2]


def sure_tag(frame: np.ndarray, template: np.ndarray, hud: int,
             sure: float = plates.BANK_MOST) -> tuple[int, int] | None:
    """Her tag's centre in a frame where the full gray template is sure of it (`sure_of`), else None."""
    return sure_of(plates.gray_peaks(frame, template, k=4, rows=hud), sure, hud)


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


def survey(frames, template: np.ndarray, hud: int, reg=None) -> tuple[list, dict]:
    """Every frame's gray peaks of the template (gray_peaks, k=4: [(path, peaks)]) and, with a camera registration
    `reg`, each frame's view where it's sure ({path: (x, y)}), the frames read once, in order."""
    peaks, views = [], {}
    for path, t in frames:
        frame = rgb(path)
        peaks.append((path, plates.gray_peaks(frame, template, k=4, rows=hud)))
        if reg is not None:
            v = reg.see(frame, t)
            if v.sure:
                views[path] = (v.x, v.y)
    return peaks, views


def _levels(peaks) -> tuple[float, float, float]:
    return plates.sure_levels(p[0][0] if p else 0.0 for _, p in peaks)


def _plate(peaks, hud: int, levels, box=plates.BOX, interior=plates.INTERIOR, feet=plates.FEET):
    picks = plates.pick_centres(((k, p[:2]) for k, p in peaks), hud, levels[1], box=box)
    return plates.PlateModel.build(((rgb(k), c) for k, c in picks), box, interior, feet)


def template_files(out: Path) -> list[Path]:
    """Her tag templates' files in a folder (the calibrated names NameTag and calibration read at the game's own
    size)."""
    return [Path(out) / f"{stem}{NATIVE.suffix}.npy" for stem in TEMPLATE_NAMES]


def build(folder: Path, map_id: int, out: Path, template: np.ndarray, every: int = 1, bank: bool = True,
          log=print, cut: bool = True) -> dict:
    """Her tag templates (`cut`) and both models from a folder of frames (the top), saved under `out`: what was made,
    and from how much."""
    frames = frames_of(folder, every)
    if not frames:
        raise ValueError(f"no frames in {folder}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hud = int(round(NATIVE.client[1] * NATIVE.hud_top))
    said: dict = {"frames": len(frames), "folder": str(folder)}
    reg = background = model_map = None
    if bank:
        from . import camreg, mapart, mapdata  # (the map's art and data: built or read on first use)
        art = mapart.load(map_id)
        reg = camreg.Registration(art)
        background = poses.Background(art.rgb, (art.x0, art.y0), art.static)
        model_map = mapdata.load(map_id)
    # 1. her tag as she has it: its levels, her plate's first model (the views too where it's kept)
    t0 = time.perf_counter()
    peaks, views = survey(frames, template, hud, None if cut else reg)
    levels = _levels(peaks)
    model = _plate(peaks, hud, levels)
    said["tag"] = {"template": list(template.shape), "levels": list(levels)}
    log(f"her tag's template {template.shape[1]}x{template.shape[0]}: typical {levels[0]:.3f} on her frames, sure for "
        f"the plate's picks {levels[1]:.3f}, for her bank {levels[2]:.3f}; her plate from {model.n} of its sure frames "
        f"({time.perf_counter() - t0:.1f} s)")
    if cut:
        # 2. her new tag from her plate; the frames again with it
        template, (dx, dy) = plates.tag_template(model)
        model = model.recentred(dx, dy)
        t0 = time.perf_counter()
        peaks, views = survey(frames, template, hud, reg)
        levels = _levels(peaks)
        model = _plate(peaks, hud, levels, model.box, model.interior, model.feet)
        _, again = plates.tag_template(model)
        said["tag"].update(cut=list(template.shape), centre_moved=[dx, dy], levels_cut=list(levels),
                           cut_again=list(again))
        log(f"her new tag template cut from her plate: {template.shape[1]}x{template.shape[0]}, its centre ({dx:+d}, "
            f"{dy:+d}) from the old one's; typical {levels[0]:.3f} on her frames, sure for the picks {levels[1]:.3f}, "
            f"for her bank {levels[2]:.3f}; her plate again from {model.n} of its sure frames (its letters cut again "
            f"centred {tuple(again)}) ({time.perf_counter() - t0:.1f} s)")
    # 3. her feet from the plate's centre
    axis = model.axis()
    fx = model.feet[0] if axis is None else axis
    model.sure = levels
    model.feet = (fx, model.feet[1])
    pl = plates.Plate(model)
    said["plate"] = dict(pl.sizes, frames=model.n, path=str(plates.model_path(out)), axis=axis)
    log(f"her plate: {pl.sizes}; her feet across {fx:+.1f} from its centre (" +
        ("its stable columns' middle" if axis is not None else "no stable columns: the model's") + ")")
    if bank:
        tags = []                   # (frame, her tag's centre, the view) where both are sure
        for path, p in peaks:
            c = sure_of(p, levels[2], hud)
            if c is not None and path in views:
                tags.append((path, c, views[path]))
        seen = {"views": len(views), "sure tags": len(tags)}
        dy, agree, of = feet_below([(c, v) for _, c, v in tags], model_map.platforms, model.feet[0])
        log(f"her feet {dy} px down from her plate's centre ({agree} of {of} frames with a line there agree; the "
            f"model's {model.feet[1]})")
        if dy is not None:
            model.feet = (model.feet[0], dy)
        said["plate"]["feet_frames"] = [agree, of]
    said["plate"]["feet"] = list(model.feet)
    said["plate"]["sure"] = list(model.sure)
    model.save(plates.model_path(out))
    for f in template_files(out):
        np.save(f, template)
    said["templates"] = [str(f) for f in template_files(out)]
    if not bank:
        return said

    def on_line(mx, my):
        return any(p.left - 1 <= mx <= p.right + 1 and abs(my - p.y_at(mx)) <= ON_LINE for p in model_map.platforms)

    def samples():
        for path, (cx, cy), view in tags:
            yield rgb(path), (cx + model.feet[0], cy + model.feet[1]), view

    # 4. her pose bank, her mirror axis by it
    t0 = time.perf_counter()
    b = poses.build_bank(samples(), background, on_line, log=log)
    b.meta.update(made=time.strftime("%Y-%m-%d %H:%M"), folder=str(folder), map=map_id, **seen)
    b.save(poses.bank_path(out))
    said["bank"] = dict(b.meta, path=str(poses.bank_path(out)))
    log(f"her pose bank: {len(b.poses)} poses from {seen['sure tags']} sure frames of {seen['views']} placed "
        f"({time.perf_counter() - t0:.1f} s)")
    corr, votes = poses.mirror_axis(b)
    said["bank"]["mirror"] = [corr, votes]
    if corr is None:
        log("her poses' mirror axis: no pose seen both ways agrees (her feet across by her plate alone)")
    else:
        warn = abs(corr) >= AXIS_WARN
        log(f"her poses' mirror axis {corr:+.1f} px from her feet's x ({votes} frames' votes)" +
            (f": WARNING, her plate's axis and her body's disagree; her feet across may be {model.feet[0] + corr:+.1f}"
             if warn else ": her plate's axis and her body's agree"))
    return said


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamelab.localization.her_cues", description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", type=Path, help="a folder of lossless frames (PNG) with its index.jsonl")
    ap.add_argument("--map", type=int, default=10002073, help="the map they're of (its art, for the bank)")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", type=Path, help="where her tag templates and the models go")
    where.add_argument("--lab", action="store_true", help="... her lab folder (her tag templates there replaced)")
    ap.add_argument("--every", type=int, default=1, help="every Nth frame")
    ap.add_argument("--no-bank", action="store_true", help="her plate's model alone (no camera, no map art)")
    ap.add_argument("--keep-tag", action="store_true", help="her tag's template as she has it (no new one cut)")
    a = ap.parse_args(argv)
    from .. import paths
    mine = paths.of()
    full = paths.calibrated(mine.tag_full, NATIVE.suffix)
    template = np.load(full if full.exists() else paths.calibrated(mine.tag_template, NATIVE.suffix))
    out = mine.lab if a.lab else a.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "her_cues.log").open("a", encoding="utf-8") as f:
        def log(s):
            line = f"{time.strftime('%H:%M:%S')} {s}"
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
        said = build(a.frames, a.map, out, template, a.every, not a.no_bank, log, cut=not a.keep_tag)
    (out / "her_cues.json").write_text(json.dumps(said, indent=1, default=str), encoding="utf-8")
    print(json.dumps(said, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
