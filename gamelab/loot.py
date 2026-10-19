"""Seeing loot on the ground: the items this map's monsters drop, and mesos, in a look frame.

Templates are the game's own pictures of the items as they lie on the ground (gamedata.canvas: `iconRaw`
of each item, and the spinning coin frames of mesos), drawn the size they appear on a 1920x1080 screen,
like the monster sprites (sprites.py). They're extracted game data, so they're kept with the other extracts
(git-ignored). Classic World's item ids differ from GMS's, so this map's drops are looked up by name.

A spot counts as an item when the masked colour template matches (THRESHOLD) and its pixels under the
template's mask are close to the template's own colours (PIXEL_SLACK): the normalised match ignores
brightness and tint, and the sea and island backdrop matched the Blue Potion's shape at 0.60-0.63, UI
buttons the White Potion's, but never their colours. Drops lie on the ground, so only strips just above platforms are searched (hunt.Hunter.see_loot):
a whole-screen search took 1.3 s.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import COPY, OUT

from .mobs import SCALE, _suppress
from .mapdata import SCREEN_SCALE
from .sprites import kind_of

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
# The game's message when pick-up fails for want of room. Its wording isn't in the game data (it's in the
# client), and her inventory hasn't been full yet, so this matches the usual wordings.
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


def shrink(rgba: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A picture at another size: its colours averaged over its opaque pixels only (RGB), and how much of
    each pixel it covers (0-1). Plain resizing mixed the colour of the transparent pixels into the edges."""
    opaque = (rgba[..., 3] >= 128).astype(np.float32)
    area = cv2.INTER_AREA if size[0] < rgba.shape[1] else cv2.INTER_LINEAR
    cover = cv2.resize(opaque, size, interpolation=area)
    colour = cv2.resize(rgba[..., :3].astype(np.float32) * opaque[..., None], size, interpolation=area)
    rgb = np.where(cover[..., None] > 1e-3, colour / np.maximum(cover[..., None], 1e-3), 0)
    return np.clip(rgb.round(), 0, 255).astype(np.uint8), cover


def resize(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`shrink`, as RGBA with the alpha on where the picture covers at least half the pixel."""
    rgb, cover = shrink(rgba, size)
    return np.dstack([rgb, np.where(cover >= 0.5, 255, 0).astype(np.uint8)])


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


def build(folder: Path = LOOT, items=ITEMS, copy: Path = COPY) -> list[Path]:
    """Write `<kind>_<n>.png` (RGBA, screen size at 1920x1080) for each item, and `mesos_<coin><frame>.png`
    for the coins' spinning frames (the edge-on one, 5 px wide, is left out)."""
    from gamedata.canvas import item_icon, meso_frames
    folder.mkdir(parents=True, exist_ok=True)
    ids, written = item_ids(copy), []

    def save(rgba: np.ndarray, name: str) -> None:
        path = folder / f"{name}.png"
        h, w = rgba.shape[:2]
        Image.fromarray(resize(rgba, (max(1, round(w * SCREEN_SCALE)), max(1, round(h * SCREEN_SCALE)))),
                        "RGBA").save(path)
        written.append(path)

    for name in items:
        for item_id in ids.get(name, []):  # the first id with a picture ("Red Potion" has two, one blank)
            try:
                save(item_icon(item_id, copy), f"{kind_of(name)}_0")
                break
            except (KeyError, ValueError):
                continue
    for kind, frames in meso_frames(copy, MESO_KINDS).items():
        for i, rgba in enumerate(frames):
            if rgba.shape[1] >= rgba.shape[0] // 2:
                save(rgba, f"{MESOS}_{MESO_KINDS[kind]}{i}")
    return written


def ensure(folder: Path = LOOT, items=ITEMS) -> Path:
    """Build the templates if any of them are missing; returns the folder."""
    have = {p.stem.rsplit("_", 1)[0] for p in folder.glob("*.png")}
    if not {kind_of(n) for n in items} | {MESOS} <= have:
        build(folder, items)
    return folder


class LootDetector:
    """Templates come from `folder`: PNGs named <kind>_<anything>.png, RGBA at full screen size."""

    def __init__(self, folder: Path | None = None, scale: int = SCALE, threshold: float = THRESHOLD,
                 pixel_slack: float = PIXEL_SLACK):
        folder = ensure() if folder is None else folder
        self.threshold, self.pixel_slack = threshold, pixel_slack
        self.templates: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []  # kind, rgb, mask x3, solid
        for path in sorted(folder.glob("*.png")):
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            h, w = rgba.shape[:2]
            size = (max(1, w // scale), max(1, h // scale))
            rgb, cover = shrink(rgba, size)
            mask = (cover >= 0.5).astype(np.uint8)
            if mask.sum() < 20:
                continue
            # Colours are checked where the picture covers the whole pixel: on screen a thin line (the
            # Slime Bubble's string) is half line, half background at half size, and was 41 off.
            solid = cover >= 0.99
            solid = solid if solid.sum() >= 10 else mask > 0
            self.templates.append((path.stem.rsplit("_", 1)[0], np.ascontiguousarray(rgb), np.dstack([mask] * 3), solid))
        if not self.templates:
            raise RuntimeError(f"no loot templates in {folder}")
        self.height = max(t[1].shape[0] for t in self.templates)
        self._pool = ThreadPoolExecutor(THREADS)  # OpenCV lets go of the GIL while matching

    @property
    def kinds(self) -> set[str]:
        return {t[0] for t in self.templates}

    def detect(self, frame: np.ndarray, box: tuple[int, int, int, int] | None = None,
               mesos_only: bool = False) -> list[Item]:
        """Items in `frame` (RGB, game window / SCALE), within `box` (x0, y0, x1, y1) if given. The
        templates are matched on a few threads: 30 ms a platform's strip instead of 90."""
        x0, y0, x1, y1 = box or (0, 0, frame.shape[1], frame.shape[0])
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        area = np.ascontiguousarray(frame[y0:int(y1), x0:int(x1)])
        todo = [t for t in self.templates if (t[0] == MESOS or not mesos_only)
                and t[1].shape[0] <= area.shape[0] and t[1].shape[1] <= area.shape[1]]
        found = self._pool.map(lambda t: self._match(area, t), todo)
        return _suppress([Item(i.kind, x0 + i.x, y0 + i.y, i.w, i.h, i.score) for hits in found for i in hits])

    def _match(self, area: np.ndarray, template) -> list[Item]:
        kind, rgb, mask3, solid = template
        own = rgb[solid].astype(np.float32)
        th, tw = rgb.shape[:2]
        res = np.nan_to_num(cv2.matchTemplate(area, rgb, cv2.TM_CCOEFF_NORMED, mask=mask3), nan=-1.0,
                            posinf=-1.0, neginf=-1.0)
        peaks = (res >= self.threshold) & (res == cv2.dilate(res, np.ones((5, 5), np.uint8)))
        hits = []
        for y, x in zip(*np.nonzero(peaks)):
            seen = area[y:y + th, x:x + tw][solid].astype(np.float32)
            if float(np.abs(seen - own).mean()) <= self.pixel_slack:
                hits.append(Item(kind, int(x) + tw // 2, int(y) + th // 2, tw, th, round(float(res[y, x]), 3)))
        return hits
