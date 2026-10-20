"""Monster templates made from the game's own sprites (gamedata.canvas), drawn the size they appear on
screen, with what the game data says of each frame (mobs.INDEX): its origin (the point that sits on the
monster's feet), its body's box, its head and how long it shows. The screenshot cut-outs (assets/mobs) are
matched against the sprites once, for where their monster's feet are and which way it faces (mobs.CUTOUTS).
They're extracted game data, so they're written next to the other extracts (git-ignored).

A monster's kind is its name in lower case with dashes ("orange-mushroom"; perception.kinds).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import OUT

from ..screen import SCREEN_SCALE
from .kinds import kind_of
from .match import masked_match
from .mobs import ASSETS, CUTOUTS, INDEX, KINDS_WITH_PALETTE, MobDetector


SPRITES = OUT / "sprites"
OPAQUE = 128  # a game picture's pixel is opaque from this alpha up
# A cut-out's best match among its kind's sprites (both ways, masked, normalised) must score this for its
# origin and facing to be kept; below it, the template's centre and bottom stand in. The 11 cut-outs matched
# at 0.86-0.98, the other way round at 0.29-0.52.
CUTOUT_SCORE = 0.85
CUTOUT_PAD = 30  # full-size px around a cut-out a sprite may reach out to: the pig cut-outs are cut off above
                 # the feet (58 px tall, their frame 60)


def screen_size(picture: np.ndarray) -> tuple[int, int]:
    """(width, height) of a game picture as it shows on a 1920x1080 screen."""
    h, w = picture.shape[:2]
    return max(1, round(w * SCREEN_SCALE)), max(1, round(h * SCREEN_SCALE))


def build(names_by_id: dict[int, str], folder: Path = SPRITES) -> list[Path]:
    """Write `<kind>_<action><index>.png` (RGBA, screen size at 1920x1080) for each monster id -> name:
    every frame of gamedata.canvas.mob_frames' poses (stand, move and hit); and each one's frame in the
    game data into the folder's INDEX (origin, lt, rb and head in game px from the picture's top-left and
    from the feet, the picture's size `w` x `h` in game px, the delay in ms)."""
    from gamedata.canvas import mob_frames
    folder.mkdir(parents=True, exist_ok=True)
    index = _read(folder / INDEX)
    written = []
    for mob_id, name in names_by_id.items():
        kind = kind_of(name)
        for frame in mob_frames(mob_id):
            rgba = frame["rgba"]
            size = screen_size(rgba)
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(np.ascontiguousarray(rgba[..., 3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = np.where(alpha >= OPAQUE, 255, 0).astype(np.uint8)
            path = folder / f"{kind}_{frame['action']}{frame['index']}.png"
            Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)
            index[path.name] = {"kind": kind, "action": frame["action"], "index": frame["index"],
                                "frame": f"{frame['action']}/{frame['index']}", "origin": frame["origin"],
                                "lt": frame["lt"], "rb": frame["rb"], "head": frame["head"],
                                "delay": frame["delay"], "w": rgba.shape[1], "h": rgba.shape[0]}
            written.append(path)
    _write(folder / INDEX, index)
    return written


def ensure(names_by_id: dict[int, str], folder: Path = SPRITES, cutouts: Path = ASSETS) -> Path:
    """Build templates for any of these monsters (id -> name) that don't have them yet, or whose pictures
    the folder's INDEX doesn't have (made before it was kept); then match the cut-outs of their kinds that
    the folder's CUTOUTS doesn't have yet. Returns the folder. (loot.ensure, unlike this, rebuilds every
    template when any is missing.)"""
    index = _read(folder / INDEX)
    missing, unindexed = {}, {}
    for mob_id, name in names_by_id.items():
        pictures = list(folder.glob(f"{kind_of(name)}_*.png"))
        if not pictures:
            missing[mob_id] = name
        elif any(p.name not in index for p in pictures):
            unindexed[mob_id] = name
    if missing:
        build(missing, folder)
    if unindexed:
        try:
            build(unindexed, folder)
        except OSError:  # no game data copy here: the pictures still work, without where the feet are
            pass
    kinds = {kind_of(n) for n in names_by_id.values()}
    known = _read(folder / CUTOUTS)
    if any(p.name not in known for p in cutouts.glob("*.png") if p.stem.split("_")[0] in kinds):
        known.update(cutout_origins(kinds, folder, cutouts))
        _write(folder / CUTOUTS, known)
    return folder


def cutout_origins(kinds: set[str], folder: Path = SPRITES, cutouts: Path = ASSETS) -> dict[str, dict]:
    """Where the monster in each cut-out of these kinds stands, by matching every sprite of its kind in
    `folder` (both ways) inside it: by file name, its feet ("origin", full-size px from the cut-out's
    top-left, of its size "w" x "h"), which way it faces as drawn ("faces_right"), the frame that matched
    best and that frame's box ("lt", "rb"), and the match's score. Below CUTOUT_SCORE only the score is kept.
    Matched this way, the four pig cut-outs are all the hop frame (move/2), cut off 9.5 units above the
    feet, and three of the slime cut-outs reach 14-16 units below them (grass)."""
    index = _read(folder / INDEX)
    sprites: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for name, entry in sorted(index.items()):
        if entry["kind"] in kinds and (folder / name).is_file():
            sprites.setdefault(entry["kind"], []).append((entry, np.asarray(Image.open(folder / name).convert("RGBA"))))
    out = {}
    for path in sorted(cutouts.glob("*.png")):
        kind = path.stem.split("_")[0]
        if kind not in sprites:
            continue
        cut = np.asarray(Image.open(path).convert("RGBA"))
        h, w = cut.shape[:2]
        pad = CUTOUT_PAD
        canvas = cv2.copyMakeBorder(np.ascontiguousarray(cut[..., :3]), pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        best = None
        for entry, rgba in sprites[kind]:
            for flip in (False, True):
                t = rgba[:, ::-1] if flip else rgba
                if t.shape[0] > canvas.shape[0] or t.shape[1] > canvas.shape[1]:
                    continue
                mask = np.dstack([(t[..., 3] > 0).astype(np.uint8)] * 3)
                _, score, _, (x, y) = cv2.minMaxLoc(masked_match(canvas, np.ascontiguousarray(t[..., :3]), mask))
                if best is None or score > best[0]:
                    best = (score, entry, flip, x - pad, y - pad, t.shape[1], t.shape[0])
        if best is None:
            continue
        score, entry, flip, x, y, tw, th = best
        if score < CUTOUT_SCORE:
            out[path.name] = {"score": round(score, 3)}
            continue
        ox, oy = entry["origin"]
        ox = entry["w"] - ox if flip else ox  # the sprites face left: flipped, it faces right
        out[path.name] = {"origin": [round(x + ox * tw / entry["w"], 1), round(y + oy * th / entry["h"], 1)],
                          "w": w, "h": h, "faces_right": flip, "frame": entry["frame"], "lt": entry["lt"],
                          "rb": entry["rb"], "score": round(score, 3)}
    return out


def detector_for(model) -> MobDetector:
    """A MobDetector for a map: screenshot cut-outs (good at monsters half hidden in plants) plus the
    game's sprites of every monster that spawns there (every pose, any monster with a colour gate)."""
    names_by_id = {s.mob_id: s.name for s in model.spawns if s.name and kind_of(s.name) in KINDS_WITH_PALETTE}
    return MobDetector([ASSETS, ensure(names_by_id)])


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
