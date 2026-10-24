"""Monster templates made from the game's own sprites (gamedata.canvas), drawn the size they appear on
screen, with what the game data says of each frame (mobs.INDEX): its origin (the point that sits on the
monster's feet), its body's box, its head and how long it shows. The screenshot cut-outs (assets/mobs) are
matched against the sprites once, for where their monster's feet are and which way it faces (mobs.CUTOUTS).
They're extracted game data, so they're written next to the other extracts (git-ignored).

The pictures are drawn as a game window of one size shows them (`stretch`, its px per game px): SPRITES as the
1920x1080 window does (1.40625, linearly, the alpha cut at OPAQUE), and for the game's own 1366x768 a folder of the
game's pictures as they are, pixel for pixel and with their own alpha (`folder_for`; mobs.MobDetector matches only
their opaque pixels there). The cut-outs are of the 1920x1080 window, so they're matched against its sprites, and
what that says of them (CUTOUTS, in their own px) goes for any window that stretches the game; at the game's own
size they aren't used (`detector_for`).

A monster's kind is its name in lower case with dashes ("orange-mushroom"; perception.kinds).

A map's kinds without a hand-made colour test (mobs.PALETTES) are looked for with gates learned against the map's
background colours (perception.gates), which `learn_background` measures on saved frames of the map (`python -m gamelab
learn-colours`): a detector for the map learns them from its own pictures as it's made (`detector_for`).
"""

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import OUT

from ..screen import STRETCHED, Geometry, of_frame
from . import gates
from .kinds import kind_of
from .match import masked_match
from .mobs import ASSETS, CUTOUTS, INDEX, OPAQUE, PALETTES, SCALE_FILE, SEARCH_SCALE, MobDetector, step1_frame


SPRITES = OUT / "sprites"
# What a picture's INDEX entry says of its alpha when it's the game's own, unchanged (at stretch 1): pictures built
# before it was kept had theirs cut at OPAQUE, and are built again.
GAME_ALPHA = {"alpha": "game"}
# A cut-out's best match among its kind's sprites (both ways, masked, normalised) must score this for its
# origin and facing to be kept; below it, the template's centre and bottom stand in. The 11 cut-outs matched
# at 0.86-0.98, the other way round at 0.29-0.52.
CUTOUT_SCORE = 0.85
CUTOUT_PAD = 30  # full-size px around a cut-out a sprite may reach out to: the pig cut-outs are cut off above
                 # the feet (58 px tall, their frame 60)
# learn_background: a monster found on a frame is cut out of the background this many game px around its picture; the
# monsters are found this many times, the first with gates learned against the frames as they are, monsters and all,
# then against the background without those found the time before.
CUT_PAD = 6
CUT_ROUNDS = 2


def screen_size(picture: np.ndarray, stretch: float = STRETCHED.stretch) -> tuple[int, int]:
    """(width, height) of a game picture as a game window shows it `stretch` times larger (by default on a 1920x1080
    screen)."""
    h, w = picture.shape[:2]
    return max(1, round(w * stretch)), max(1, round(h * stretch))


def folder_for(stretch: float) -> Path:
    """Where the sprites drawn `stretch` times larger are kept: SPRITES for the 1920x1080 window's 1.40625, else
    `sprites_x<stretch>` ("sprites_x1": the game's own pictures, for 1366x768)."""
    return SPRITES if stretch == STRETCHED.stretch else OUT / f"sprites_x{stretch:g}"


def build(names_by_id: dict[int, str], folder: Path = SPRITES, stretch: float = STRETCHED.stretch) -> list[Path]:
    """Write `<kind>_<action><index>.png` (RGBA, drawn `stretch` times larger: by default the size they show at on a
    1920x1080 screen) for each monster id -> name:
    every frame of gamedata.canvas.mob_frames' poses (stand, move and hit); and each one's frame in the
    game data into the folder's INDEX (origin, lt, rb and head in game px from the picture's top-left and
    from the feet, the picture's size `w` x `h` in game px, the delay in ms). Another stretch than 1920x1080's is
    written in the folder's mobs.SCALE_FILE. At stretch 1 (the game's own 1366x768) each picture is the game's as it
    is, alpha and all (GAME_ALPHA in its entry): no resize, nothing cut."""
    from gamedata.canvas import mob_frames
    folder.mkdir(parents=True, exist_ok=True)
    if stretch != STRETCHED.stretch:
        _write(folder / SCALE_FILE, {"px_per_game_px": stretch})
    index = _read(folder / INDEX)
    written = []
    for mob_id, name in names_by_id.items():
        kind = kind_of(name)
        for frame in mob_frames(mob_id):
            rgba = frame["rgba"]
            if stretch == 1:
                picture = np.ascontiguousarray(rgba)
            else:
                size = screen_size(rgba, stretch)
                rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_LINEAR)
                alpha = cv2.resize(np.ascontiguousarray(rgba[..., 3]), size, interpolation=cv2.INTER_LINEAR)
                picture = np.dstack([rgb, np.where(alpha >= OPAQUE, 255, 0).astype(np.uint8)])
            path = folder / f"{kind}_{frame['action']}{frame['index']}.png"
            Image.fromarray(picture, "RGBA").save(path)
            index[path.name] = {"kind": kind, "action": frame["action"], "index": frame["index"],
                                "frame": f"{frame['action']}/{frame['index']}", "origin": frame["origin"],
                                "lt": frame["lt"], "rb": frame["rb"], "head": frame["head"],
                                "delay": frame["delay"], "w": rgba.shape[1], "h": rgba.shape[0],
                                **(GAME_ALPHA if stretch == 1 else {})}
            written.append(path)
    _write(folder / INDEX, index)
    return written


def ensure(names_by_id: dict[int, str], folder: Path | None = None, cutouts: Path = ASSETS,
           stretch: float = STRETCHED.stretch, stretched: Path = SPRITES) -> Path:
    """Build templates (drawn `stretch` times larger, in `folder`, by default `folder_for(stretch)`) for any of these
    monsters (id -> name) that don't have them yet, or whose pictures the folder's INDEX doesn't have (made before
    it was kept, or at stretch 1 before the game's alpha was); then match the cut-outs of their kinds that the
    folder's CUTOUTS doesn't have yet. The cut-outs are of the 1920x1080 window, so they're matched against its
    sprites (in `stretched`), and another stretch's folder takes what that says of them; at stretch 1 they aren't
    used (`detector_for`), so nothing is matched. Returns the folder. (loot.ensure, unlike this, rebuilds every
    template when any is missing.)"""
    folder = folder_for(stretch) if folder is None else folder
    _sprites_at(names_by_id, folder, stretch)
    if stretch == 1:
        folder.mkdir(parents=True, exist_ok=True)  # (a map none of whose monsters has a colour test builds none)
        return folder
    if stretch != STRETCHED.stretch:
        folder.mkdir(parents=True, exist_ok=True)  # (a map none of whose monsters has a colour test builds none)
        _write(folder / CUTOUTS, _read(ensure(names_by_id, stretched, cutouts) / CUTOUTS))
        return folder
    kinds = {kind_of(n) for n in names_by_id.values()}
    known = _read(folder / CUTOUTS)
    if any(p.name not in known for p in cutouts.glob("*.png") if p.stem.split("_")[0] in kinds):
        known.update(cutout_origins(kinds, folder, cutouts))
        _write(folder / CUTOUTS, known)
    return folder


def _sprites_at(names_by_id: dict[int, str], folder: Path, stretch: float) -> None:
    """ensure's first part: the pictures and their INDEX."""
    index = _read(folder / INDEX)
    missing, unindexed = {}, {}
    for mob_id, name in names_by_id.items():
        pictures = list(folder.glob(f"{kind_of(name)}_*.png"))
        if not pictures:
            missing[mob_id] = name
        elif any(p.name not in index or stretch == 1 and index[p.name].get("alpha") != GAME_ALPHA["alpha"]
                 for p in pictures):
            unindexed[mob_id] = name
    more = () if stretch == STRETCHED.stretch else (stretch,)  # (build's own default otherwise)
    if missing:
        build(missing, folder, *more)
    if unindexed:
        try:
            build(unindexed, folder, *more)
        except OSError:  # no game data copy here: the pictures still work, without where the feet are
            pass


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


def detector_for(model, geo: Geometry = STRETCHED, background: np.ndarray | None = None) -> MobDetector:
    """A MobDetector for a map, for look frames of geometry `geo`: the game's sprites of every monster that spawns
    there (every pose, any monster with a colour gate), drawn as that window shows them, and where the window
    stretches the game the screenshot cut-outs too (good at monsters half hidden in plants on the 1920x1080 window;
    they and the sprites folder hold every kind built for any map, and all of them with a hand-made colour test are
    looked for, as always). At the game's own 1366x768 the sprites are the game's own pictures, and alone: the
    cut-outs are of the stretched window, and shrunk back they're neither (README, Screen size). There the detector
    holds the map's own kinds only, whatever else the folder holds from other maps, so what it looks for doesn't hang
    on what was built before. The map's kinds without a hand-made colour test get gates learned against its background
    colours (`background`, by default those kept for it: gates.load_background, `learn-colours`); a map with none kept
    gets a detector that doesn't look for them (it finds nothing on a map none of whose monsters has a hand-made test,
    where it had none at all on a checkout that hadn't built the first map's: `no monster templates`)."""
    background = gates.load_background(getattr(model, "id", None)) if background is None else background
    kinds = {kind_of(s.name) for s in model.spawns if s.name}
    learned = {k for k in kinds if k not in PALETTES} if background is not None else set()
    names_by_id = {s.mob_id: s.name for s in model.spawns
                   if s.name and (kind_of(s.name) in PALETTES or kind_of(s.name) in learned)}
    folders = [ensure(names_by_id, stretch=geo.stretch)]
    if geo.stretch == 1:
        return MobDetector(folders, frame_scale=geo.frame_scale, kinds={kind_of(n) for n in names_by_id.values()},
                           background=background, learned=learned)
    return MobDetector([ASSETS] + folders, frame_scale=geo.frame_scale, background=background, learned=learned)


def learn_background(frames: list[np.ndarray], detector: Callable[[Geometry, np.ndarray], MobDetector],
                     rounds: int = CUT_ROUNDS,
                     log: Callable[[str], None] = lambda line: None) -> tuple[np.ndarray, int]:
    """A map's background colours (gates.background), from saved frames of it (full size, RGB, of any window size):
    the play area of each (above the HUD) at step 1's scale, with the monsters found on it cut out. They're found with
    the map's detector for the frame's geometry, its gates learned against the background as learned so far
    (`detector(geo, background)`, as detector_for makes it): first against the frames as they are, monsters and all,
    then `rounds` - 1 times against the background without the monsters found the round before, each found monster's
    picture cut out CUT_PAD game px around. Returns the counts and how many monsters the last round cut out."""
    looks = []  # (geometry, look frame, its play area at step 1's scale)
    for full in frames:
        geo = of_frame(full)
        k = geo.look_scale
        frame = full if k == 1 else cv2.resize(full, (full.shape[1] // k, full.shape[0] // k),
                                               interpolation=cv2.INTER_AREA)
        search = max(1.0, geo.frame_scale / SEARCH_SCALE)
        looks.append((geo, frame, step1_frame(frame[:int(frame.shape[0] * geo.hud_top)], search), search))
    cut: list[list[tuple[int, int, int, int]]] = [[] for _ in looks]
    found = 0
    for r in range(rounds):
        counts = gates.background([picture for _, _, picture, _ in looks], cut)
        detectors = {geo: detector(geo, counts) for geo in {geo for geo, *_ in looks}}
        found, cut = 0, []
        for geo, frame, _, search in looks:
            hits = detectors[geo].detect_wide(frame, (0, int(frame.shape[0] * geo.hud_top)))
            found += len(hits)
            pad = CUT_PAD * geo.frame_scale  # look-frame px; step 1's are `search` of them
            cut.append([(int((m.x - m.w / 2 - pad) / search), int((m.y - m.h / 2 - pad) / search),
                         int(np.ceil((m.x + m.w / 2 + pad) / search)), int(np.ceil((m.y + m.h / 2 + pad) / search)))
                        for m in hits])
        log(f"round {r + 1}: {found} monsters found on {len(looks)} frames, cut out")
    return gates.background([picture for _, _, picture, _ in looks], cut), found


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
