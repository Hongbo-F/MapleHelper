"""Monsters and drops drawn onto saved frames as the game draws them, then looked for (`python -m gamelab
bench-paste`): offline, no game, no keys.

There are no frames of the game at its own 1366x768 with the first map's monsters on them, and on West Rocky Mountain
II's frames none of its monsters has a colour gate yet. So the game's own pictures are drawn onto frames of the game
at that size, pixel for pixel and alpha blended as the game draws them (`paste`), at known places, and the search is
asked to find them: at 1366x768 as a look sees it (full size), and the same scene stretched to 1920x1080 as the game's
window stretched it until 2026-10-23 (bilinear, `stretched`), looked at half size as that window was. Every kind of a
map that has a colour gate (what its detector holds: mobs.PALETTES, and any gate added later) is drawn, every frame
of its sprites both ways, and every drop (loot.ITEMS and the coins' frames).

What it says, by size and kind: how many were found by the whole screen's search and by a fight's band around each
(its feet as hers), where (their feet against where they were drawn, map units), how well they scored, whether as the
frame and facing drawn, what was found that wasn't drawn (on the frames as they are, and in the scenes), and how long
each search took. Drops likewise, each searched in a strip just above where it lies, as Senses searches a platform.
"""

import math
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

PER_SCENE = 6        # monsters (or drops) drawn on one frame, apart from each other
GAP = 12             # game px kept between two drawn
TOP = 40             # game px from the top a picture may start (the window's title strip, the party HP box)
FEET_TOL = 10        # map units across and down a monster found may be from where it was drawn and count
ITEM_TOL = 6         # ... a drop found
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
    random places clear of each other, of the minimap and of the bottom bar: (the frame drawn on, its background's
    index, what was drawn where)."""
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
            out["clean"][name] += det.detect_wide(frame, (0, int(frame.shape[0] * HUD_TOP)))
        tally = out["tally"][name]
        times = out["times"][name]
        for full, _, drawn in made:
            frame = look_frame(full if geo.native else stretched(full, geo.client), geo)
            bottom = int(frame.shape[0] * HUD_TOP)
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
                t.as_drawn += m.template == d.name and m.facing == d.facing
                t.errors.append((m.foot_x * upp - d.feet[0], m.feet_y * upp - d.feet[1]))
                t.scores.append(m.score)
            for m in found:
                if id(m) not in used:
                    tally.setdefault(m.kind, Tally()).extra += 1
    return out


def _nearest(found: list, d: Drawn, upp: float, used: set):
    """The monster found nearest where `d` was drawn, of its kind, within FEET_TOL, not already taken; or None."""
    near = [(math.hypot(m.foot_x * upp - d.feet[0], m.feet_y * upp - d.feet[1]), i, m) for i, m in enumerate(found)
            if m.kind == d.kind and id(m) not in used and abs(m.foot_x * upp - d.feet[0]) <= FEET_TOL
            and abs(m.feet_y * upp - d.feet[1]) <= FEET_TOL]
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
                    t.errors.append((it.x * upp - (d.left + d.w / 2), it.y * upp - (d.top + d.h / 2)))
                    t.scores.append(it.score)
                for it in items:  # (a strip may hold another drop drawn beside it)
                    if not any(_at(it, other, upp) for other in drawn):
                        tally.setdefault(it.kind, Tally()).extra += 1
    return out


def _at(item, d: Drawn, upp: float) -> bool:
    """Whether a drop found is `d`: its kind, its centre within ITEM_TOL of the picture's."""
    return item.kind == d.kind and abs(item.x * upp - (d.left + d.w / 2)) <= ITEM_TOL and \
        abs(item.y * upp - (d.top + d.h / 2)) <= ITEM_TOL


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
            same = "as the frame and facing drawn" if what == "monsters" else "as the coin drawn"
            out.append(f"   {kind:22s} {t.found} of {t.drawn} found{band}; {t.as_drawn} {same}; {err}; {sc}; "
                       f"{t.extra} found that weren't drawn"
                       + (f"; missed {', '.join(t.missed[:8])}{' ...' if len(t.missed) > 8 else ''}" if t.missed
                          else ""))
    return out
