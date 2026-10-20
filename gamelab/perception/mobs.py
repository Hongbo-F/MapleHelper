"""Finding monsters on screen: colour silhouettes first, then masked template matching.

Each monster type has a few templates (sprites cut out with a mask, gamelab/assets/mobs/<kind>_*.png,
RGBA with the alpha as mask) and a colour test for its body. Step 1 matches each template's body-colour
silhouette against the frame's body-colour mask (one fast single-channel pass over the frame). Step 2
checks only the best few spots with the full masked colour template, which rejects scenery that merely
shares the colour (leaves, clouds). Frames are game-window pictures downscaled by SCALE.

The matches run on a few threads (OpenCV lets go of Python's lock while it matches). Which spots step 2
checks doesn't depend on what the checks find, so they're all picked first, and the answers are the same
as one thread's, in the same order.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from .. import paths
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


@dataclass(frozen=True)
class Mob:
    """A monster seen on screen, in frame pixels (half size)."""
    kind: str
    x: int        # centre
    y: int
    w: int
    h: int
    score: float

    @property
    def feet_y(self) -> int:
        return self.y + self.h // 2


class MobTemplate(NamedTuple):
    """One monster picture (one pose, facing one way), at frame size."""
    rgb: np.ndarray         # its colours
    mask3: np.ndarray       # where the picture is (uint8, one copy per colour channel)
    silhouette: np.ndarray  # its body colour, at 1/COARSE size (step 1)


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


class MobDetector:
    """Templates come from `folder` (or a list of folders): PNGs named <kind>_<anything>.png, RGBA at full
    screen size. gamelab/assets/mobs has cut-outs from screenshots; sprites.py makes them from the game data.
    `threads`: how many matches run at once (1: all on the caller's thread)."""

    def __init__(self, folder: Path | list[Path] = ASSETS, threads: int = THREADS):
        self.templates: dict[str, list[MobTemplate]] = {}
        folders = folder if isinstance(folder, list) else [folder]
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
                self.templates.setdefault(kind, []).append(MobTemplate(t, np.dstack([m] * 3), coarse))
        if not self.templates:
            raise RuntimeError(f"no monster templates in {folder}")
        self._pool = ThreadPoolExecutor(threads, "monster-search", _one_opencv_thread) if threads > 1 else None

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

    def _map(self, fn, items) -> list:
        """fn of each item, in order: on the pool, or here with one thread."""
        return list(self._pool.map(fn, items)) if self._pool else [fn(item) for item in items]

    def _detect(self, frame: np.ndarray, rows: tuple[int, int] | None, run) -> list[Mob]:
        """detect, with `run(fn, items)` doing the matches (on the pool, or one after another)."""
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
        for (kind, *_), result in zip(checks, results):
            if result and result[0] >= THRESHOLD:
                score, x, y, w, h = result
                hits.append(Mob(kind, x, y_off + y, w, h, round(score, 3)))
        return suppress_overlaps(hits)


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
    res[cover.sums(*silhouette.shape) / silhouette.size < MIN_COVER * float(silhouette.mean())] = 0
    peaks = (res >= SILHOUETTE_THRESHOLD) & (res == cv2.dilate(res, np.ones((3, 3), np.uint8)))
    return [(float(res[y, x]), COARSE * x + COARSE * silhouette.shape[1] // 2,
             COARSE * y + COARSE * silhouette.shape[0] // 2, i) for y, x in zip(*np.nonzero(peaks))]


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
