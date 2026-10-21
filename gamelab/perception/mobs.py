"""Finding monsters on screen: colour silhouettes first, then masked template matching.

Each monster type has a few templates (sprites cut out with a mask, gamelab/assets/mobs/<kind>_*.png,
RGBA with the alpha as mask) and a colour test for its body. Step 1 matches each template's body-colour
silhouette against the frame's body-colour mask (one fast single-channel pass over the frame). Step 2
checks only the best few spots with the full masked colour template, which rejects scenery that merely
shares the colour (leaves, clouds). Frames are game-window pictures downscaled by SCALE.

The matches run on a few threads (OpenCV lets go of Python's lock while it matches). Which spots step 2
checks doesn't depend on what the checks find, so they're all picked first, and the answers are the same
as one thread's, in the same order.

A monster found says which template matched it, and so what that template's folder knows of it (INDEX,
CUTOUTS): where its feet are in the picture, which way it faces (the game's sprites face left; a flipped
template faces right), and its frame and pose.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from .. import native, paths
from ..screen import SCALE
from .match import masked_match, suppress_overlaps


ASSETS = paths.ASSETS / "mobs"
THRESHOLD = 0.6              # masked colour match a spot needs to count as a monster
SILHOUETTE_THRESHOLD = 0.35  # step 1: a spot worth checking
MAX_CHECKS = 12              # step 2: best spots checked per monster type
MIN_COVER = 0.25             # step 1: a spot's window needs this share of the template's own body colour
MIN_BODY_PIXELS = 20         # a monster type is looked for only where this many pixels have its colour
COARSE = 2                   # step 1 works at 1/COARSE of the frame's size
CHECK_SLACK = 6              # step 2 looks this far (px) around a spot, so spots closer than this are one
WIDE_BAND, WIDE_STEP = 120, 90  # detect_wide: rows per band, and rows from one band's top to the next
THREADS = 6                  # matches run at once: the band 70 -> 32 ms, the whole screen 250 -> 117 (8 are no faster)
# A monster seen in the last frame is looked for again this far (px) around where it should be (`refind`):
# at 30 frames a second a pig walking 125 units/s moves 3 px a frame, and where her minimap dot puts her
# (which places the monsters on the map) can step by a minimap pixel, 11.4 units or 8 px, as she walks.
# The research's prototype re-found 84 of 84 monsters in saved frames with 16 px, 78 with 8.
REFIND_PAD = 16
# What a folder of templates may say about them, by file name: where the monster's feet are in each picture
# ("origin", in the picture's own px, of its size "w" x "h"), which way it faces ("faces_right"; the game's
# sprites face left), its frame in the game data ("frame": "move/2") and that frame's body box ("lt", "rb").
INDEX = "index.json"      # the game's sprites (sprites.build)
CUTOUTS = "cutouts.json"  # the screenshot cut-outs, matched against the sprites (sprites.cutout_origins)


@dataclass(frozen=True)
class Mob:
    """A monster seen on screen, in frame pixels (half size): the template that matched it, and what that
    template knows of it. The template's centre and bottom aren't the monster's feet: a sprite stands on its
    origin, up to 8.5 map units off its picture's centre, and a slime's hop frames end 10 units above its
    feet; the screenshot cut-outs were cropped 9.5 units above a pig's feet and 14 below a slime's. Where the
    template knows its origin (`ox`, `feet`), that is where the monster stands."""
    kind: str
    x: int        # the template's centre
    y: int
    w: int
    h: int
    score: float
    ox: float | None = None    # where its feet are: x ... (None: the template doesn't know)
    feet: float | None = None  # ... and y
    facing: int = 0            # which way it faces: 1 right, -1 left (0: not known)
    pose: str = ""             # what it's doing: "stand", "move" (walking) or "hit1" (flinching); "": not known
    frame: str = ""            # the frame of its animation that matched ("move/2"), if known
    template: str = ""         # the picture that matched (its file's name: "pig_move2", "pig_1")

    @property
    def foot_x(self) -> float:
        """The x it stands at: its origin's, else the template's centre."""
        return self.x if self.ox is None else self.ox

    @property
    def feet_y(self) -> float:
        """The y it stands at: its origin's, else the template's bottom."""
        return self.y + self.h // 2 if self.feet is None else self.feet


class MobTemplate(NamedTuple):
    """One monster picture (one pose, facing one way), at frame size, and what its folder says of it (INDEX,
    CUTOUTS)."""
    rgb: np.ndarray         # its colours
    mask3: np.ndarray       # where the picture is (uint8, one copy per colour channel)
    silhouette: np.ndarray  # its body colour, at 1/COARSE size (step 1)
    origin: tuple[float, float] | None = None  # where the monster's feet are, from its top-left (frame px)
    facing: int = 0         # which way the monster faces in it: 1 right, -1 left, 0 not known
    frame: str = ""         # its frame in the game data ("move/2")
    pose: str = ""          # that frame's action: "stand", "move" or "hit1"
    box: tuple[float, float, float, float] | None = None  # its body (left, top, right, bottom), map units
                                                          # from its feet, y up negative, facing applied
    name: str = ""          # its file's name, without ".png"


def split_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A picture's red, green and blue as int16 arrays, so differences between them can go negative."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return r, g, b


def _pig(r, g, b):
    return (r > 190) & (g > 120) & (b > 110) & (r - g > 18) & (r - b > 15)


def _slime(r, g, b):
    return (g > 170) & (g - r > 30) & (g - b > 90)


def _snail(r, g, b):
    teal = (r < 90) & (g > 80) & (b > 80) & (g < 170) & (np.abs(g - b) < 40)
    lavender = (r > 140) & (b > 190) & (g > 130) & (b - g > 15)
    return teal | lavender


def _orange_mushroom(r, g, b):
    """The orange cap (daffodils are yellower: higher green)."""
    return (r > 215) & (g > 100) & (g < 175) & (b < 90) & (r - g > 55)


# Each monster kind's body colour, as a test on a picture's r, g and b (int16, from split_channels). A kind
# needs one before its templates are used.
PALETTES = {"pig": _pig, "slime": _slime, "snail": _snail, "orange-mushroom": _orange_mushroom}
KINDS_WITH_PALETTE = tuple(PALETTES)


def palette(kind: str, rgb: np.ndarray, channels=None) -> np.ndarray:
    """Pixels that have the colour of this monster's body. `channels` = `split_channels(rgb)`, if already
    split (saves time when testing several kinds on one frame)."""
    r, g, b = channels or split_channels(rgb)
    if kind not in PALETTES:
        raise ValueError(f"unknown monster kind {kind!r}")
    return PALETTES[kind](r, g, b)


def read_meta(folders: list[Path]) -> dict[str, dict]:
    """What these folders say about their templates (INDEX, CUTOUTS), by file name."""
    out: dict[str, dict] = {}
    for folder in folders:
        for name in (INDEX, CUTOUTS):
            if (folder / name).is_file():
                out.update(json.loads((folder / name).read_text(encoding="utf-8")))
    return out


def _known(entry: dict | None, size: tuple[int, int], flip: bool) -> tuple:
    """What a template's entry (INDEX, CUTOUTS) says of it at frame size `size` (height, width), as drawn or
    flipped: MobTemplate's origin, facing, frame, pose and box. Without an origin, nothing is known."""
    if not entry or entry.get("origin") is None:
        return None, 0, "", "", None
    th, tw = size
    ox, oy = entry["origin"][0] * tw / entry["w"], entry["origin"][1] * th / entry["h"]
    right = bool(entry.get("faces_right")) != flip
    box = None
    if entry.get("lt") and entry.get("rb"):  # the game's boxes are for a monster facing left
        (left, top), (r, bottom) = entry["lt"], entry["rb"]
        box = (-r, top, -left, bottom) if right else (left, top, r, bottom)
    frame = entry.get("frame", "")
    return (tw - ox if flip else ox, oy), 1 if right else -1, frame, frame.split("/")[0], box


class MobDetector:
    """Templates come from `folder` (or a list of folders): PNGs named <kind>_<anything>.png, RGBA at full
    screen size. gamelab/assets/mobs has cut-outs from screenshots; sprites.py makes them from the game data.
    What the folders say about their templates (INDEX, CUTOUTS) comes along, and `meta` (by file name) on top.
    `threads`: how many matches run at once (1: all on the caller's thread)."""

    def __init__(self, folder: Path | list[Path] = ASSETS, threads: int = THREADS, meta: dict | None = None):
        self.templates: dict[str, list[MobTemplate]] = {}
        folders = folder if isinstance(folder, list) else [folder]
        meta = {**read_meta(folders), **(meta or {})}
        for path in sorted(p for f in folders for p in f.glob("*.png")):
            kind = path.stem.split("_")[0]
            if kind not in KINDS_WITH_PALETTE:
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            h, w = rgba.shape[:2]
            size = (max(1, w // SCALE), max(1, h // SCALE))
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize((rgba[..., 3] > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            for flip in (False, True):  # monsters face either way
                t, m = (rgb[:, ::-1], mask[:, ::-1]) if flip else (rgb, mask)
                t = np.ascontiguousarray(t)
                silhouette = (palette(kind, t) & (m > 0)).astype(np.float32)
                coarse = cv2.resize(silhouette, (max(1, t.shape[1] // COARSE), max(1, t.shape[0] // COARSE)),
                                    interpolation=cv2.INTER_AREA)
                self.templates.setdefault(kind, []).append(MobTemplate(
                    t, np.dstack([m] * 3), coarse, *_known(meta.get(path.name), t.shape[:2], flip), path.stem))
        if not self.templates:
            raise RuntimeError(f"no monster templates in {folder}")
        self._pool = ThreadPoolExecutor(threads, "monster-search", _one_opencv_thread) if threads > 1 else None
        self.fallbacks = 0  # re-finds that went on to the kind's whole set (refind)
        native.lib()  # built (the first time, a few seconds) and loaded now, not in the first look

    def detect(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Mob]:
        """Monsters in `frame` (RGB, game window / SCALE). `rows` limits the search to a band."""
        return self._detect(frame, rows, self._map)

    def detect_wide(self, frame: np.ndarray, rows: tuple[int, int]) -> list[Mob]:
        """Monsters over a tall stretch of the frame, band by band. Over the whole screen, green
        foliage took all of the slime candidate spots and three slimes in plain view were missed;
        each band gets its own best spots. The bands are searched at once, each on one thread."""
        # A band starts every WIDE_STEP rows until one reaches rows[1] (it's cut short there); always one.
        starts = range(rows[0], max(rows[0] + 1, rows[1] - WIDE_BAND + WIDE_STEP), WIDE_STEP)
        bands = self._map(lambda y0: self._detect(frame, (y0, min(rows[1], y0 + WIDE_BAND)), _serial), starts)
        return suppress_overlaps([hit for band in bands for hit in band])

    def refind(self, frame: np.ndarray, box: tuple[int, int, int, int], kind: str, template: str = "",
               facing: int = 0, bounds: tuple[int, int, int, int] | None = None, fallback: bool = True) -> Mob | None:
        """A monster seen before, looked for again around where it should be: `box` (x0, y0, x1, y1, frame
        px) is where its picture should be, searched REFIND_PAD px around (within `bounds`, by default the
        frame). First the picture that matched it last (`template`, `facing`: Mob.template and Mob.facing; a
        cut-out that doesn't know its facing, both ways), one masked match over the window; if that doesn't
        reach THRESHOLD (and `fallback`), every template of its kind there, as `detect` does. The one found
        nearest the box's centre, or None. On the caller's thread: a window is small, and the pool's
        hand-offs cost more."""
        h, w = frame.shape[:2]
        bx0, by0, bx1, by1 = bounds or (0, 0, w, h)
        x0, y0 = max(bx0, int(box[0]) - REFIND_PAD), max(by0, int(box[1]) - REFIND_PAD)
        x1, y1 = min(bx1, int(box[2]) + REFIND_PAD), min(by1, int(box[3]) + REFIND_PAD)
        if x1 <= x0 or y1 <= y0 or kind not in self.templates:
            return None
        window = frame[y0:y1, x0:x1]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best = None
        for t in self.templates[kind]:
            if not template or t.name != template or (facing and t.facing and t.facing != facing):
                continue
            th, tw = t.rgb.shape[:2]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            _, score, _, (x, y) = cv2.minMaxLoc(masked_match(window, t.rgb, t.mask3))
            if score >= THRESHOLD and (best is None or score > best.score):
                best = _seen(kind, t, x0 + x + tw // 2, y0 + y + th // 2, tw, th, round(float(score), 3))
        if best is not None or not fallback:
            return best
        self.fallbacks += 1
        found = [_moved(m, x0, y0) for m in self._detect(window, None, _serial, (kind,))]
        return min(found, key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2) if found else None

    def _map(self, fn, items) -> list:
        """fn of each item, in order: on the pool, or here with one thread."""
        return list(self._pool.map(fn, items)) if self._pool else [fn(item) for item in items]

    def _detect(self, frame: np.ndarray, rows: tuple[int, int] | None, run, only=None) -> list[Mob]:
        """detect, with `run(fn, items)` doing the matches (on the pool, or one after another); `only`: these
        kinds alone (default: every kind)."""
        y_off = 0
        if rows:
            y_off = max(0, rows[0])
            frame = frame[y_off:rows[1]]
        # A size that's a multiple of COARSE, so step 1's mask is an exact shrink. An odd one (the fight's
        # band was 151 rows) smeared faint colour into blank windows, which then scored a perfect 1.0.
        frame = frame[:frame.shape[0] // COARSE * COARSE, :frame.shape[1] // COARSE * COARSE]
        channels = split_channels(frame)
        coarse_size = (frame.shape[1] // COARSE, frame.shape[0] // COARSE)
        kinds = []  # (kind, its templates, its body colour at 1/COARSE size, that colour's cover)
        for kind, templates in self.templates.items():
            if only is not None and kind not in only:
                continue
            colour = palette(kind, frame, channels)
            if colour.sum() < MIN_BODY_PIXELS:
                continue
            coarse = cv2.resize(colour.astype(np.float32), coarse_size, interpolation=cv2.INTER_AREA)
            kinds.append((kind, templates, coarse, Cover(coarse)))
        # Step 1, every template of every kind; then the spots each kind checks; then step 2, all at once.
        jobs = [(coarse, cover, t, i) for _, templates, coarse, cover in kinds for i, t in enumerate(templates)]
        found = run(lambda job: _spots(*job), jobs)
        checks, start = [], 0
        for kind, templates, _, _ in kinds:
            spots = sorted((s for f in found[start:start + len(templates)] for s in f), reverse=True)
            start += len(templates)
            checks += [(kind, templates[i], cx, cy) for cx, cy, i in _to_check(spots)]
        results = run(lambda c: _check(frame, c[1], c[2], c[3]), checks)
        hits = []
        for (kind, template, *_), result in zip(checks, results):
            if result and result[0] >= THRESHOLD:
                score, x, y, w, h = result
                hits.append(_seen(kind, template, x, y_off + y, w, h, round(score, 3)))
        return suppress_overlaps(hits)


def _seen(kind: str, t: MobTemplate, x: int, y: int, w: int, h: int, score: float) -> Mob:
    """The monster template `t` matched, centred at (x, y): where it stands, if the template knows."""
    ox = feet = None
    if t.origin is not None:  # the window's top-left (the centre is that plus half the size, rounded down)
        ox, feet = round(x - w // 2 + t.origin[0], 2), round(y - h // 2 + t.origin[1], 2)
    return Mob(kind, x, y, w, h, score, ox, feet, t.facing, t.pose, t.frame, t.name)


def _moved(m: Mob, dx: int, dy: int) -> Mob:
    """A monster found in a window of the frame, in the frame's px (the window's top-left at dx, dy)."""
    return replace(m, x=m.x + dx, y=m.y + dy, ox=None if m.ox is None else round(m.ox + dx, 2),
                   feet=None if m.feet is None else round(m.feet + dy, 2))


def _one_opencv_thread() -> None:
    """Each of the pool's threads runs its matches by itself. With OpenCV's own threads as well (24 here),
    the whole screen took 141 ms instead of 117 (and without the pool, 307 instead of 250). OpenCV's thread
    count is the process's, not a thread's, so this sets it for the whole look: the half-size frame takes
    0.8 ms instead of 0.4, and the look without monsters is no slower (bench-look)."""
    cv2.setNumThreads(1)


def _serial(fn, items) -> list:
    return [fn(item) for item in items]


class Cover:
    """How much of each window of the frame has a kind's body colour, from the colour's integral image:
    one pass over the frame, then four lookups a window (a TM_CCORR with a template of ones did the
    same sums, one pass per template size)."""

    def __init__(self, coarse: np.ndarray):
        self.integral = cv2.integral(coarse, sdepth=cv2.CV_64F)
        self._sums: dict[tuple[int, int], np.ndarray] = {}

    def sums(self, h: int, w: int) -> np.ndarray:
        """The body colour in each h x w window, by its top-left corner (the shape of a match's result)."""
        out = self._sums.get((h, w))
        if out is None:  # two threads may both work it out: the same numbers either way
            ii = self.integral
            out = self._sums[(h, w)] = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
        return out


def _spots(coarse: np.ndarray, cover: Cover, template: MobTemplate, i: int) -> list[tuple[float, int, int, int]]:
    """Step 1 for one template (index `i`): where its body-colour silhouette matches the frame's body
    colour (`coarse`, at 1/COARSE size): (score, centre x and y in the frame, i) of each spot."""
    silhouette = template.silhouette
    if coarse.shape[0] < silhouette.shape[0] or coarse.shape[1] < silhouette.shape[1]:
        return []
    res = cv2.matchTemplate(coarse, silhouette, cv2.TM_CCOEFF_NORMED)
    # A window with next to no body colour has next to no variance, and its normalised score
    # is noise that can reach 1.0. In the basin, such windows took all MAX_CHECKS checks,
    # so snails and pigs at her level were never checked.
    least = MIN_COVER * float(silhouette.mean())
    # In C if it's built (native.peaks: the same, but without holding Python's lock), else in numpy.
    found = native.peaks(res, cover.integral, *silhouette.shape, least, SILHOUETTE_THRESHOLD)
    if found is None:
        res[cover.sums(*silhouette.shape) / silhouette.size < least] = 0
        ys, xs = np.nonzero((res >= SILHOUETTE_THRESHOLD) & (res == cv2.dilate(res, np.ones((3, 3), np.uint8))))
        found = ys, xs, res[ys, xs]
    return [(float(v), COARSE * x + COARSE * silhouette.shape[1] // 2,
             COARSE * y + COARSE * silhouette.shape[0] // 2, i) for y, x, v in zip(*found)]


def _to_check(spots: list[tuple[float, int, int, int]]) -> list[tuple[int, int, int]]:
    """Step 2's spots (centre x, y, template), best first: at most MAX_CHECKS, none within CHECK_SLACK of
    one already picked."""
    picked = []
    for _, cx, cy, i in spots:
        if len(picked) >= MAX_CHECKS:
            break
        if any(abs(cx - x) < CHECK_SLACK and abs(cy - y) < CHECK_SLACK for x, y, _ in picked):
            continue
        picked.append((cx, cy, i))
    return picked


def _check(frame: np.ndarray, template: MobTemplate, cx: int, cy: int):
    """Masked colour match of one template centred within CHECK_SLACK of (cx, cy): (score, x, y, w, h),
    or None if it doesn't fit in the frame there."""
    tpl, mask = template.rgb, template.mask3
    th, tw = tpl.shape[:2]
    x0, y0 = max(0, cx - tw // 2 - CHECK_SLACK), max(0, cy - th // 2 - CHECK_SLACK)
    window = frame[y0:y0 + th + 2 * CHECK_SLACK, x0:x0 + tw + 2 * CHECK_SLACK]
    if window.shape[0] < th or window.shape[1] < tw:
        return None
    _, score, _, (x, y) = cv2.minMaxLoc(masked_match(window, tpl, mask))
    return float(score), x0 + x + tw // 2, y0 + y + th // 2, tw, th
