"""Monsters and drops drawn onto saved frames as the game draws them, then looked for (`python -m gamelab
bench-paste`): offline, no game, no keys.

There are no frames of the game at its own 1366x768 with the first map's monsters on them, and the new maps' saved
frames show theirs in a few places and poses only. So the game's own pictures are drawn onto frames of the game
at that size, pixel for pixel and alpha blended as the game draws them (`paste`), at known places, and the search is
asked to find them: at 1366x768 as a look sees it (full size), and the same scene stretched to 1920x1080 as the game's
window stretched it until 2026-10-23 (bilinear, `stretched`), looked at half size as that window was. Every kind of a
map that has a colour gate (what its detector holds: mobs.PALETTES, and the gates learned for it) is drawn, every frame
of its sprites both ways, and every drop (loot.ITEMS and the coins' frames).

What it says, by size and kind: how many were found by the whole screen's search and by a fight's band around each
(its feet as hers), where (their feet against where they were drawn, map units), how well they scored, whether as the
frame and facing drawn (of those the game's sprites matched: a screenshot cut-out, at 1920x1080, says no frame), what
was found that wasn't drawn (on the frames as they are, and in the scenes), and how long each search took. Drops
likewise, each searched in a strip just above where it lies, as Senses searches a platform. And re-finds
(`refinds`): each sprite found, then drawn where it stood as the frame it may show next, and looked for again there as
the feed does each frame (MobDetector.refind).

Where a look frame's px are in the game's own picture: a px's middle, as the stretch and the shrink put it (`_game`).
At half size of the 1920x1080 window that's (x + 0.5) * 1.42 - 0.5, not x * 1.42: by the px's edge, the feet of the
first map's sprites found at 1920x1080 came out 0.27 units left and 0.19 up of where they were drawn on average (160
of them), by its middle 0.06 left and 0.02 down. At the game's own size it's the px itself.
"""

import math
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..perception.loot import ABOVE, BELOW, LootDetector, _coin, _kind
from ..perception.minimap import minimap_box
from ..perception.mobs import INDEX, MobDetector, read_meta
from ..screen import HUD_TOP, NATIVE, STRETCHED, Geometry
from .bodies import is_cutout

PER_SCENE = 6        # monsters (or drops) drawn on one frame, apart from each other
GAP = 12             # game px kept between two drawn
TOP = 40             # game px from the top a picture may start (the window's title strip, the party HP box)
FEET_TOL = 10        # map units across and down a monster found may be from where it was drawn and count
ITEM_TOL = 6         # ... a drop found
FEET_NEAR = 1        # map units across and down a re-found monster's feet may be from where it was drawn, to count as
                     # placed right (a fight aims by them)
# What a monster may show in the next frame of a look, after the frame it was found as (`refinds`)
STEPS = ("unchanged", "its pose's next frame", "standing to walking or back", "a flinch")
STRIP_PAD = 40       # map units either side of a drop its strip reaches
STRETCH_TO = STRETCHED.client


@dataclass(frozen=True)
class Drawn:
    """One picture drawn onto a frame, in game px (the 1366x768 frame's own px)."""
    kind: str
    name: str            # its file's name without ".png" ("pig_move0", "mesos_bronze1")
    facing: int          # monsters: 1 right (drawn flipped: the game's sprites face left), -1 left; drops 0
    left: int
    top: int
    w: int
    h: int
    feet: tuple[float, float] | None = None  # monsters: where its feet are (its origin), game px
    coin: str | None = None


@dataclass
class Tally:
    """What a search made of the pictures drawn, for one size and one kind."""
    drawn: int = 0
    found: int = 0
    found_band: int = 0
    as_drawn: int = 0                  # found as the frame and facing drawn (monsters), the coin drawn (mesos)
    by_cutout: int = 0                 # monsters found by a screenshot cut-out (1920x1080), which says no frame
    errors: list = field(default_factory=list)   # (dx, dy) map units, of each found
    scores: list = field(default_factory=list)
    missed: list = field(default_factory=list)   # names of those not found by the whole screen's search
    extra: int = 0                     # found where none of this kind was drawn (in the scenes)


def paste(frame: np.ndarray, rgba: np.ndarray, left: int, top: int) -> None:
    """Draw a picture (RGBA, the game's own px) onto a frame of the game at its own size, its top-left at (left,
    top), as the game draws it: each pixel its colour where opaque, blended by its alpha where it's see-through."""
    h, w = rgba.shape[:2]
    a = rgba[..., 3:].astype(np.float32) / 255
    region = frame[top:top + h, left:left + w].astype(np.float32)
    frame[top:top + h, left:left + w] = np.clip(region * (1 - a) + rgba[..., :3] * a + 0.5, 0, 255).astype(np.uint8)


def stretched(full: np.ndarray, size: tuple[int, int] = STRETCH_TO) -> np.ndarray:
    """A frame of the game at its own size as the 1920x1080 window showed it: stretched, bilinear."""
    return cv2.resize(full, size, interpolation=cv2.INTER_LINEAR)


def look_frame(full: np.ndarray, geo: Geometry) -> np.ndarray:
    """A full-size frame as a look of geometry `geo` sees it (perception.eyes.Eyes.look_at)."""
    k = geo.look_scale
    return full if k == 1 else cv2.resize(full, (full.shape[1] // k, full.shape[0] // k), interpolation=cv2.INTER_AREA)


def pictures(folder: Path, kinds: set[str] | None = None) -> list[tuple[str, np.ndarray, tuple[float, float] | None]]:
    """The pictures in a folder of the game's own (sprites.build or loot.build at stretch 1): (name, RGBA, origin
    from its INDEX entry if it has one), those of `kinds` only if given (a monster's kind is its file name's first
    part, a drop's all but the last)."""
    meta = read_meta([folder]) if (folder / INDEX).is_file() else {}
    out = []
    for path in sorted(folder.glob("*.png")):
        kind = path.stem.split("_")[0] if meta else _kind(path)
        if kinds is not None and kind not in kinds:
            continue
        entry = meta.get(path.name) or {}
        out.append((path.stem, np.asarray(Image.open(path).convert("RGBA")), entry.get("origin")))
    return out


def scenes(backgrounds: list[np.ndarray], todo: list[tuple], rng: np.random.Generator,
           per_scene: int = PER_SCENE) -> list[tuple[np.ndarray, int, list[Drawn]]]:
    """`todo` (kind, name, RGBA, origin, flip, coin) drawn onto the backgrounds in turn, `per_scene` to a frame at
    random places clear of each other, of the minimap and of the bottom bar (above HUD_TOP, 1920x1080's: in both
    sizes' searches): (the frame drawn on, its background's index, what was drawn where)."""
    out, i = [], 0
    h, w = backgrounds[0].shape[:2]
    bottom = int(h * HUD_TOP)
    mx0, my0, mx1, my1 = minimap_box(NATIVE)
    while i < len(todo):
        b = len(out) % len(backgrounds)
        frame, drawn = backgrounds[b].copy(), []
        for kind, name, rgba, origin, flip, coin in todo[i:i + per_scene]:
            ph, pw = rgba.shape[:2]
            for _ in range(200):
                left, top = int(rng.integers(0, w - pw)), int(rng.integers(TOP, bottom - ph))
                clear = all(left + pw + GAP <= d.left or d.left + d.w + GAP <= left or top + ph + GAP <= d.top
                            or d.top + d.h + GAP <= top for d in drawn)
                if clear and (left >= mx1 or top >= my1 or left + pw <= mx0 or top + ph <= my0):
                    break
            else:
                continue
            picture = rgba[:, ::-1] if flip else rgba
            paste(frame, picture, left, top)
            feet = None
            if origin is not None:
                ox = pw - origin[0] if flip else origin[0]
                feet = (left + ox, top + origin[1])
            drawn.append(Drawn(kind, name, (1 if flip else -1) if origin is not None else 0, left, top, pw, ph, feet,
                               coin))
        out.append((frame, b, drawn))
        i += per_scene
    return out


def _game(v: float, upp: float) -> float:
    """A place in a look frame (its px, a px's middle at its index) in the game's own picture (game px), the frame's
    px `upp` game px each: where the stretch and the shrink put that px's middle (screen.Geometry.to_game)."""
    return (v + 0.5) * upp - 0.5


def _timed(fn: Callable, times: list) -> object:
    t0 = time.perf_counter()
    out = fn()
    times.append(time.perf_counter() - t0)
    return out


def monsters(backgrounds: list[np.ndarray], detectors: dict[str, tuple[MobDetector, Geometry]], folder: Path,
             band: tuple[float, float], each: int = 2, seed: int = 0, kinds: set[str] | None = None) -> dict:
    """Every picture in `folder` (a native sprite folder) of the kinds the detectors know (of `kinds` only, if
    given), both ways, drawn `each` times onto the backgrounds (1366x768 frames), and looked for with each detector
    (name -> (detector, the geometry of the frames it looks at: NATIVE, or STRETCHED for the scene stretched)): by
    kind and name, a Tally; and 'clean' (what each found on the backgrounds as they are) and 'times' (s, by
    search). The detectors look for every kind they know, as a look does."""
    kinds = set.intersection(*(set(d.templates) for d, _ in detectors.values()), *([kinds] if kinds else []))
    todo = [(name.split("_")[0], name, rgba, origin, flip, None)
            for name, rgba, origin in pictures(folder, kinds) if origin is not None for flip in (False, True)] * each
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(todo))
    made = scenes(backgrounds, [todo[i] for i in order], rng)
    out = {"tally": {n: {} for n in detectors}, "clean": {n: [] for n in detectors},
           "times": {n: {"whole screen": [], "band": []} for n in detectors}}
    for name, (det, geo) in detectors.items():
        upp = geo.units_per_px
        for bg in backgrounds:
            frame = look_frame(bg if geo.native else stretched(bg, geo.client), geo)
            out["clean"][name] += det.detect_wide(frame, (0, int(frame.shape[0] * geo.hud_top)))
        tally = out["tally"][name]
        times = out["times"][name]
        for full, _, drawn in made:
            frame = look_frame(full if geo.native else stretched(full, geo.client), geo)
            bottom = int(frame.shape[0] * geo.hud_top)
            found = _timed(lambda: det.detect_wide(frame, (0, bottom)), times["whole screen"])
            used = set()
            for d in drawn:
                t = tally.setdefault(d.kind, Tally())
                t.drawn += 1
                m = _nearest(found, d, upp, used)
                rows = (max(0, int((d.feet[1] - band[0]) / upp)), min(bottom, int((d.feet[1] + band[1]) / upp)))
                in_band = _timed(lambda: det.detect(frame, rows), times["band"])
                if _nearest(in_band, d, upp, set()) is not None:
                    t.found_band += 1
                if m is None:
                    t.missed.append(f"{d.name}{'>' if d.facing == 1 else '<'}")
                    continue
                used.add(id(m))
                t.found += 1
                t.by_cutout += is_cutout(m)
                t.as_drawn += m.template == d.name and m.facing == d.facing
                t.errors.append((_game(m.foot_x, upp) - d.feet[0], _game(m.feet_y, upp) - d.feet[1]))
                t.scores.append(m.score)
            for m in found:
                if id(m) not in used:
                    tally.setdefault(m.kind, Tally()).extra += 1
    return out


def _nearest(found: list, d: Drawn, upp: float, used: set):
    """The monster found nearest where `d` was drawn, of its kind, within FEET_TOL, not already taken; or None."""
    near = [(math.hypot(_game(m.foot_x, upp) - d.feet[0], _game(m.feet_y, upp) - d.feet[1]), i, m)
            for i, m in enumerate(found) if m.kind == d.kind and id(m) not in used
            and abs(_game(m.foot_x, upp) - d.feet[0]) <= FEET_TOL
            and abs(_game(m.feet_y, upp) - d.feet[1]) <= FEET_TOL]
    return min(near)[2] if near else None


def drops(backgrounds: list[np.ndarray], detectors: dict[str, tuple[LootDetector, Geometry]], folder: Path,
          each: int = 2, seed: int = 1) -> dict:
    """Every drop picture in `folder` (a native loot folder) drawn `each` times onto the backgrounds, each looked for
    in a strip just above where it lies (ABOVE and BELOW of its bottom, STRIP_PAD units either side), as Senses
    searches a platform: by kind, a Tally (its errors between the centres); 'clean' (what each found in the same
    strips on the backgrounds as they are) and 'times' (s, a scene's strips searched together)."""
    todo = [(_kind(Path(name)), name, rgba, None, False, _coin(Path(name + ".png")))
            for name, rgba, _ in pictures(folder)] * each
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(todo))
    made = scenes(backgrounds, [todo[i] for i in order], rng)
    out = {"tally": {n: {} for n in detectors}, "clean": {n: [] for n in detectors},
           "times": {n: {"strips": []} for n in detectors}}
    for name, (det, geo) in detectors.items():
        upp = geo.units_per_px
        tally, times = out["tally"][name], out["times"][name]
        above, below = geo.fpx(ABOVE), geo.fpx(BELOW)
        for full, b, drawn in made:
            frame = look_frame(full if geo.native else stretched(full, geo.client), geo)
            clean = look_frame(backgrounds[b] if geo.native else stretched(backgrounds[b], geo.client), geo)
            boxes = [(int((d.left - STRIP_PAD) / upp), int((d.top + d.h) / upp - above),
                      int((d.left + d.w + STRIP_PAD) / upp), int((d.top + d.h) / upp + below)) for d in drawn]
            found = _timed(lambda: det.detect_each(frame, boxes), times["strips"])
            for items in det.detect_each(clean, boxes):
                out["clean"][name] += items
            for d, items in zip(drawn, found):
                t = tally.setdefault(d.kind, Tally())
                t.drawn += 1
                near = [it for it in items if _at(it, d, upp)]
                if not near:
                    t.missed.append(d.name)
                else:
                    it = max(near, key=lambda it: it.score)
                    t.found += 1
                    t.as_drawn += it.coin == d.coin
                    t.errors.append((_game(it.x, upp) - _middle(d)[0], _game(it.y, upp) - _middle(d)[1]))
                    t.scores.append(it.score)
                for it in items:  # (a strip may hold another drop drawn beside it)
                    if not any(_at(it, other, upp) for other in drawn):
                        tally.setdefault(it.kind, Tally()).extra += 1
    return out


def _middle(d: Drawn) -> tuple[float, float]:
    """A drop picture's middle as drawn, game px (a px's middle at its index)."""
    return d.left + (d.w - 1) / 2, d.top + (d.h - 1) / 2


def _at(item, d: Drawn, upp: float) -> bool:
    """Whether a drop found is `d`: its kind, its centre within ITEM_TOL of the picture's."""
    x, y = _middle(d)
    return (item.kind == d.kind and abs(_game(item.x, upp) - x) <= ITEM_TOL
            and abs(_game(item.y, upp) - y) <= ITEM_TOL)


@dataclass
class Refound:
    """What re-finding made of monsters drawn again as a frame they may show next (`refinds`), for one size and step."""
    n: int = 0             # found first, then re-found
    found: int = 0         # found again (with the fallback to the kind's whole set, as a live track is)
    feet: int = 0          # ... with its feet within FEET_NEAR of where it was drawn
    as_drawn: int = 0      # ... as the frame and facing drawn
    fallbacks: int = 0     # re-finds that went on to the kind's whole set
    hidden: int = 0        # found again without the fallback, as a hidden track is
    times: list = field(default_factory=list)  # s, each re-find with the fallback


def _frames_by_pose(names: list[str]) -> dict[tuple[str, str], list[str]]:
    """The game's sprites' names ("<kind>_<action><index>", sprites.build) by kind and action, in their order."""
    out: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for name in names:
        m = re.fullmatch(r"(.+)_(stand|move|hit1)(\d+)", name)
        if m:
            out.setdefault((m.group(1), m.group(2)), []).append((int(m.group(3)), name))
    return {k: [n for _, n in sorted(v)] for k, v in out.items()}


def refinds(backgrounds: list[np.ndarray], detectors: dict[str, tuple[MobDetector, Geometry]], folder: Path,
            band: tuple[float, float], seed: int = 3, kinds: set[str] | None = None) -> dict:
    """Each standing and walking sprite in `folder` (a native sprite folder) of the kinds the detectors know (of
    `kinds` only, if given), both ways, drawn at a random place on one of the backgrounds in turn and found there (a
    fight's band around it), then drawn instead, its feet where they were, as each frame it may show next (STEPS:
    itself, its pose's next, the first of standing or walking, whichever it wasn't, its flinch) and looked for again
    around where it was (MobDetector.refind, within the play area, as the feed does each frame), with the fallback to
    the kind's whole set and without: by detector and step, a Refound."""
    pics = {name: (rgba, origin) for name, rgba, origin in pictures(folder) if origin is not None}
    kinds = set.intersection(*(set(d.templates) for d, _ in detectors.values()), *([kinds] if kinds else []))
    poses = _frames_by_pose(list(pics))
    todo = []  # (kind, the frame found, the frame drawn next, step)
    for (kind, action), names in sorted(poses.items()):
        if kind not in kinds or action == "hit1":
            continue
        other = poses.get((kind, "move" if action == "stand" else "stand"), [])
        flinch = poses.get((kind, "hit1"), [])
        for i, a in enumerate(names):
            nexts = (a, names[(i + 1) % len(names)] if len(names) > 1 else None, other[0] if other else None,
                     flinch[0] if flinch else None)
            todo += [(kind, a, b, step) for step, b in zip(STEPS, nexts) if b is not None]
    rng = np.random.default_rng(seed)
    h, w = backgrounds[0].shape[:2]
    out = {name: {step: Refound() for step in STEPS} for name in detectors}
    for k, (kind, a, b, step) in enumerate(todo):
        for flip in (False, True):
            # (its feet 300 game px from the left: clear of the minimap; room for its body above and the bar below)
            fx, fy = int(rng.integers(300, w - 80)), int(rng.integers(TOP + 110, int(h * HUD_TOP) - 10))
            scenes, drawn = [], []
            for name in (a, b):
                rgba, origin = pics[name]
                pic = rgba[:, ::-1] if flip else rgba
                ox = pic.shape[1] - origin[0] if flip else origin[0]
                scene = backgrounds[k % len(backgrounds)].copy()
                paste(scene, pic, int(fx - ox), int(fy - origin[1]))
                scenes.append(scene)
                drawn.append(Drawn(kind, name, 1 if flip else -1, int(fx - ox), int(fy - origin[1]), pic.shape[1],
                                   pic.shape[0], (fx, fy)))
            for label, (det, geo) in detectors.items():
                upp = geo.units_per_px
                before, after = (look_frame(s if geo.native else stretched(s, geo.client), geo) for s in scenes)
                bottom = int(before.shape[0] * geo.hud_top)
                rows = (max(0, int((fy - band[0]) / upp)), min(bottom, int((fy + band[1]) / upp)))
                m = _nearest(det.detect(before, rows), drawn[0], upp, set())
                if m is None:
                    continue
                r = out[label][step]
                r.n += 1
                box = (m.x - m.w // 2, m.y - m.h // 2, m.x + m.w // 2, m.y + m.h // 2)
                bounds = (0, 0, after.shape[1], bottom)
                fallbacks = det.fallbacks
                again = _timed(lambda: det.refind(after, box, kind, m.template, m.facing, bounds), r.times)
                r.fallbacks += det.fallbacks - fallbacks
                if again is not None:
                    r.found += 1
                    r.feet += (abs(_game(again.foot_x, upp) - fx) <= FEET_NEAR
                               and abs(_game(again.feet_y, upp) - fy) <= FEET_NEAR)
                    r.as_drawn += again.template == b and again.facing == drawn[1].facing
                r.hidden += det.refind(after, box, kind, m.template, m.facing, bounds, fallback=False) is not None
    return out


def refind_lines(result: dict) -> list[str]:
    """A report of `refinds`, one block a detector."""
    out = []
    for name, steps in result.items():
        out.append(f"{name}:")
        for step, r in steps.items():
            if not r.n:
                continue
            ts = sorted(r.times)
            out.append(f"   {step:28s} {r.found} of {r.n} found again ({r.feet} with their feet within {FEET_NEAR} "
                       f"unit, {r.as_drawn} as the frame drawn), {r.fallbacks} going on to the kind's whole set; "
                       f"{1000 * statistics.median(ts):.1f} ms a re-find (median; at 90% "
                       f"{1000 * ts[int(0.9 * (len(ts) - 1))]:.1f}); without the fallback {r.hidden} found again")
    return out


def lines(result: dict, what: str = "monsters") -> list[str]:
    """A report of `monsters` or `drops` (`what`), one block a detector."""
    out = []
    for name, tallies in result["tally"].items():
        times = result["times"][name]
        timing = "; ".join(f"{k} {1000 * statistics.median(v):.1f} ms" for k, v in times.items() if v)
        clean = result["clean"][name]
        out.append(f"{name}: {timing} (medians); on the frames as they are, found {len(clean)}"
                   + (f" ({', '.join(sorted({getattr(m, 'kind', '?') for m in clean}))})" if clean else ""))
        for kind, t in sorted(tallies.items()):
            if not t.drawn:
                out.append(f"   {kind:22s} none drawn, found {t.extra}")
                continue
            dx = [abs(e[0]) for e in t.errors]
            dy = [abs(e[1]) for e in t.errors]
            err = (f"off {statistics.mean(dx):.1f} / {statistics.mean(dy):.1f} units across / down (at most "
                   f"{max(dx):.1f} / {max(dy):.1f})" if t.errors else "")
            sc = f"scored {min(t.scores):.2f}-{max(t.scores):.2f} (median {statistics.median(t.scores):.2f})" \
                if t.scores else ""
            band = f", {t.found_band} in a band around each" if what == "monsters" else ""
            same = "as the coin drawn"
            if what == "monsters":
                same = (f"as the frame and facing drawn (of {t.found - t.by_cutout} found by the game's sprites"
                        + (f"; {t.by_cutout} by a cut-out, which says no frame)" if t.by_cutout else ")"))
            out.append(f"   {kind:22s} {t.found} of {t.drawn} found{band}; {t.as_drawn} {same}; {err}; {sc}; "
                       f"{t.extra} found that weren't drawn"
                       + (f"; missed {', '.join(t.missed[:8])}{' ...' if len(t.missed) > 8 else ''}" if t.missed
                          else ""))
    return out
