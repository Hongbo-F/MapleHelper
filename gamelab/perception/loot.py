"""Seeing loot on the ground: the items this map's monsters drop, and mesos, in a look frame.

Templates are the game's own pictures of the items as they lie on the ground (gamedata.canvas: `iconRaw`
of each item, and the spinning coin frames of mesos), drawn the size they appear on a 1920x1080 screen,
like the monster sprites (sprites.py). They're extracted game data, so they're kept with the other extracts
(git-ignored). Classic World's item ids differ from GMS's, so this map's drops are looked up by name.

A spot counts as an item when the masked colour template matches (THRESHOLD) and its pixels under the
template's mask are close to the template's own colours (PIXEL_SLACK): the normalised match ignores
brightness and tint, and the sea and island backdrop matched the Blue Potion's shape at 0.60-0.63, UI
buttons the White Potion's, but never their colours. Drops lie on the ground, so only strips just above
platforms are searched (localization.scene.find_loot, with ABOVE and BELOW): a whole-screen search took 1.3 s.

The pickup log's full-inventory message (`says_full`) is here too.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import COPY, OUT

from ..screen import SCALE
from .kinds import kind_of
from .match import masked_match, suppress_overlaps
from .sprites import OPAQUE, screen_size


LOOT = OUT / "loot"
# This map's drops, by name: the slime's green bubble, the pig's leather (and its head and ribbon), the
# Orange Mushroom's cap, the snail's shell (snails in her way get shot), and the potions monsters this
# level drop (Use items: their pictures are in Item/Consume).
ITEMS = ("Squishy Liquid", "Slime Bubble", "Leather", "Pig's Head", "Pig's Ribbon", "Orange Mushroom Cap",
         "Snail Shell", "Red Potion", "Orange Potion", "White Potion", "Blue Potion")
MESOS = "mesos"
MESO_KINDS = {0: "bronze", 1: "gold"}  # these monsters drop 10-20 mesos: coins, never bills or sacks
THRESHOLD = 0.6      # masked colour match (the drops in the saved frames: 0.63-0.87)
PIXEL_SLACK = 24     # mean difference from the template, per pixel and channel under its mask (in saved
                     # frames: drops 12-19; scenery and UI that matched a shape 27-57)
THREADS = 4          # templates matched at once (a strip: 30 ms on 4 threads, 90 ms on one)
ABOVE = 30           # half-size px above a platform's top searched (items are up to 22 tall, and bob)
BELOW = 6            # ... and below it
HALF_COVER = 0.5     # a template's mask: where the picture covers at least this much of a pixel
SOLID_COVER = 0.99   # its colours are checked where the picture covers this much (see LootDetector)
MIN_MASK = 20        # fewer pixels in its mask than this, and a template isn't used
MIN_SOLID = 10       # fewer solid pixels than this, and its colours are checked over the whole mask
# The game's message when pick-up fails for want of room: "You can't get anymore items." (seen when her
# Etc tab filled up). It's in the client, not the game data, so this matches it and the usual other wordings.
FULL_MESSAGE = re.compile(r"inventory\W*(?:is\W*)?full|full\W*inventory|(?:can\W?not|can\W?t|unable\W*to)\W*"
                          r"(?:get|pick|loot|hold|carry)|no\W*(?:more\W*)?(?:room|space|slots?)", re.I)


def says_full(text: str) -> bool:
    """Whether a line from the pickup log says her inventory is full."""
    return bool(FULL_MESSAGE.search(text))


@dataclass(frozen=True)
class Item:
    """An item or mesos seen on the ground, in frame pixels (half size)."""
    kind: str    # MESOS, or the item's name as a kind ("squishy-liquid")
    x: int       # centre
    y: int
    w: int
    h: int
    score: float

    @property
    def bottom(self) -> int:
        return self.y + self.h // 2

    @property
    def mesos(self) -> bool:
        return self.kind == MESOS


def resample_opaque(rgba: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A picture at another size (smaller or larger): its colours averaged over its opaque pixels only
    (RGB), and how much of each pixel it covers (0-1). Plain resizing mixed the colour of the transparent
    pixels into the edges."""
    opaque = (rgba[..., 3] >= OPAQUE).astype(np.float32)
    area = cv2.INTER_AREA if size[0] < rgba.shape[1] else cv2.INTER_LINEAR
    cover = cv2.resize(opaque, size, interpolation=area)
    colour = cv2.resize(rgba[..., :3].astype(np.float32) * opaque[..., None], size, interpolation=area)
    rgb = np.where(cover[..., None] > 1e-3, colour / np.maximum(cover[..., None], 1e-3), 0)
    return np.clip(rgb.round(), 0, 255).astype(np.uint8), cover


def resample_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`resample_opaque`, as RGBA with the alpha on where the picture covers at least half the pixel."""
    rgb, cover = resample_opaque(rgba, size)
    return np.dstack([rgb, np.where(cover >= HALF_COVER, 255, 0).astype(np.uint8)])


def item_ids(copy: Path = COPY) -> dict[str, list[int]]:
    """Etc and Use item ids by name (String/Etc.img, Consume.img), lowest first: a name can have several."""
    from gamedata.wz import Package
    strings = Package(copy / "String" / "String_000.wz")
    out: dict[str, list[int]] = {}
    for image in ("Etc.img", "Consume.img"):
        tree = strings.image(image)
        tree = tree.get("Etc", tree)
        for key, entry in tree.items():
            if key.isdigit() and isinstance(entry, dict) and "name" in entry:
                out.setdefault(entry["name"], []).append(int(key))
    return {name: sorted(ids) for name, ids in out.items()}


def build(folder: Path = LOOT) -> list[Path]:
    """Write `<kind>_<n>.png` (RGBA, screen size at 1920x1080) for each of ITEMS, and
    `mesos_<coin><frame>.png` for the coins' spinning frames (the edge-on one, 5 px wide, is left out)."""
    from gamedata.canvas import item_icon, meso_frames
    folder.mkdir(parents=True, exist_ok=True)
    ids, written = item_ids(COPY), []

    def save(rgba: np.ndarray, name: str) -> None:
        path = folder / f"{name}.png"
        Image.fromarray(resample_rgba(rgba, screen_size(rgba)), "RGBA").save(path)
        written.append(path)

    for name in ITEMS:
        for item_id in ids.get(name, []):  # the first id with a picture ("Red Potion" has two, one blank)
            try:
                save(item_icon(item_id, COPY), f"{kind_of(name)}_0")
                break
            except (KeyError, ValueError):
                continue
    for kind, frames in meso_frames(COPY, MESO_KINDS).items():
        for i, rgba in enumerate(frames):
            if rgba.shape[1] >= rgba.shape[0] // 2:  # not edge-on
                save(rgba, f"{MESOS}_{MESO_KINDS[kind]}{i}")
    return written


def _kind(path: Path) -> str:
    """The kind a template is of, from its file name (<kind>_<anything>.png)."""
    return path.stem.rsplit("_", 1)[0]


def ensure(folder: Path = LOOT) -> Path:
    """Build the templates if any of them are missing; returns the folder."""
    have = {_kind(p) for p in folder.glob("*.png")}
    wanted = {kind_of(n) for n in ITEMS} | {MESOS}
    if not wanted <= have:
        build(folder)
    return folder


class LootTemplate(NamedTuple):
    """One drop's picture, at frame size."""
    kind: str
    rgb: np.ndarray    # its colours
    mask3: np.ndarray  # where the picture is (uint8, one copy per colour channel)
    solid: np.ndarray  # where its colours are checked (bool)


class LootDetector:
    """Templates come from `folder` (built from the game data if not given): PNGs named
    <kind>_<anything>.png, RGBA at full screen size."""

    def __init__(self, folder: Path | None = None):
        folder = ensure() if folder is None else folder
        self.templates: list[LootTemplate] = []
        for path in sorted(folder.glob("*.png")):
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            h, w = rgba.shape[:2]
            rgb, cover = resample_opaque(rgba, (max(1, w // SCALE), max(1, h // SCALE)))
            mask = (cover >= HALF_COVER).astype(np.uint8)
            if mask.sum() < MIN_MASK:
                continue
            # Colours are checked where the picture covers the whole pixel: on screen a thin line (the
            # Slime Bubble's string) is half line, half background at half size, and was 41 off.
            solid = cover >= SOLID_COVER
            solid = solid if solid.sum() >= MIN_SOLID else mask > 0
            self.templates.append(LootTemplate(_kind(path), np.ascontiguousarray(rgb), np.dstack([mask] * 3), solid))
        if not self.templates:
            raise RuntimeError(f"no loot templates in {folder}")
        self.height = max(t.rgb.shape[0] for t in self.templates)
        self._pool = ThreadPoolExecutor(THREADS)  # OpenCV lets go of the GIL while matching

    def detect(self, frame: np.ndarray, box: tuple[int, int, int, int] | None = None,
               mesos_only: bool = False) -> list[Item]:
        """Items in `frame` (RGB, game window / SCALE), within `box` (x0, y0, x1, y1) if given. The
        templates are matched on a few threads: 30 ms a platform's strip instead of 90."""
        x0, y0, x1, y1 = box or (0, 0, frame.shape[1], frame.shape[0])
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        area = np.ascontiguousarray(frame[y0:int(y1), x0:int(x1)])

        def wanted(t: LootTemplate) -> bool:
            fits = t.rgb.shape[0] <= area.shape[0] and t.rgb.shape[1] <= area.shape[1]
            return (t.kind == MESOS or not mesos_only) and fits

        found = self._pool.map(lambda t: self._match(area, (x0, y0), t), [t for t in self.templates if wanted(t)])
        return suppress_overlaps([item for hits in found for item in hits])

    def _match(self, area: np.ndarray, origin: tuple[int, int], template: LootTemplate) -> list[Item]:
        """Where one template is in `area`, whose top left is at `origin` in the frame."""
        kind, rgb, mask3, solid = template
        own = rgb[solid].astype(np.float32)
        th, tw = rgb.shape[:2]
        res = masked_match(area, rgb, mask3)
        peaks = (res >= THRESHOLD) & (res == cv2.dilate(res, np.ones((5, 5), np.uint8)))
        hits = []
        for y, x in zip(*np.nonzero(peaks)):
            seen = area[y:y + th, x:x + tw][solid].astype(np.float32)
            if float(np.abs(seen - own).mean()) <= PIXEL_SLACK:
                hits.append(Item(kind, origin[0] + int(x) + tw // 2, origin[1] + int(y) + th // 2, tw, th,
                                 round(float(res[y, x]), 3)))
        return hits
