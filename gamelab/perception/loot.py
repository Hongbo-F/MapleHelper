"""Seeing loot on the ground: the items a map's monsters drop (perception.drops: `detector_for`; the first map's
ITEMS by default), and mesos, in a look frame.

Templates are the game's own pictures of the items as they lie on the ground (gamedata.canvas: `iconRaw`
of each item, and the spinning coin frames of mesos), drawn the size they appear on a 1920x1080 screen,
like the monster sprites (sprites.py), and for the game's own 1366x768 as they are (`folder_for`); a detector
brings them to its frames' scale (screen.Geometry). They're extracted game data, so they're kept with the other
extracts (git-ignored): one folder a size for every map's drops, a map's detector holding its own (`items`). Classic
World's item ids differ from GMS's, so the drops are looked up by name.

A spot counts as an item when the masked colour template matches (THRESHOLD) and its pixels under the
template's mask are close to the template's own colours (PIXEL_SLACK): the normalised match ignores
brightness and tint, and the sea and island backdrop matched the Blue Potion's shape at 0.60-0.63, UI
buttons the White Potion's, but never their colours. Drops lie on the ground, so only strips just above
platforms are searched (localization.scene.find_loot, with ABOVE and BELOW): a whole-screen search took 1.3 s.

At the game's own 1366x768 the pictures are the game's own, drawn on screen pixel for pixel, so a drop's colours
are its picture's to the pixel: there the shape is matched on brightness alone (one channel instead of three:
`gray`), and the colours are checked as above (PIXEL_SLACK). On the 17 bench frames shrunk to that size it found the
11 drops found in colour (scoring 0.81-0.95, in colour 0.83-0.95), and every platform in view took 78 ms against 112
in colour (73 at 1920x1080 half size); drawn as the game draws them on frames of the game at that size, all 32 were
found, scoring 1.00, and on the frames as they are nothing. At 1920x1080, all of it is as it was. A picture the game
draws partly see-through (of the drops looked for, Tree Dungeon's Fragment of Magic: a crystal in a glow, of its 917
px drawn 404 as they are and 513 partly see-through) is kept as it is at that size (`build`), and matched and checked
on the px drawn as they are only (`_at_size`), as the monster sprites are: on screen the others are partly what lies
behind them. Drawn so on the new maps' frames (200 times), it scored 0.95-0.98 with its half see-through px taken as
drawn (alpha 128 and up, as the 1920x1080 pictures are cut) and its colours were 10.4 off (15.5 at most); now 1.00
and 0.

The pickup log's full-inventory message (`says_full`) is here too.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import COPY, OUT

from ..screen import STRETCHED, Geometry
from . import drops
from .kinds import kind_of
from .match import masked_match, suppress_overlaps
from .mobs import SCALE_FILE, frame_size, picture_scale
from .sprites import OPAQUE, screen_size


LOOT = OUT / "loot"
# The first map's drops, by name (drops.FIRST_ITEMS): the slime's green bubble, the pig's leather (and its head and
# ribbon), the Orange Mushroom's cap, the snail's shell (snails in her way get shot), and the potions monsters this
# level drop (Use items: their pictures are in Item/Consume). Another map's: drops.items_for (`detector_for`).
ITEMS = drops.FIRST_ITEMS
MESOS = "mesos"
# The coins looked for: the first map's monsters drop 10-20 mesos, the new maps' up to 53 (perception.drops), always
# coins (by the v83 client a bronze coin under 50, gold to 99), never bills or sacks
MESO_KINDS = {0: "bronze", 1: "gold"}
THRESHOLD = 0.6      # masked colour match (the drops in the saved frames: 0.63-0.87)
PIXEL_SLACK = 24     # mean difference from the template, per pixel and channel under its mask (in saved
                     # frames: drops 12-19; scenery and UI that matched a shape 27-57)
THREADS = 8          # templates matched at once: her platform's strips 22 -> 14 ms against 4 threads, every
                     # platform's in view 120 -> 88 (back to back on the 17 bench frames, the CPU busy); 12: 13 and 88
ABOVE = 30           # half-size px above a platform's top searched (items are up to 22 tall, and bob)
BELOW = 6            # ... and below it
HALF_COVER = 0.5     # a template's mask: where the picture covers at least this much of a pixel
SOLID_COVER = 0.99   # its colours are checked where the picture covers this much (see LootDetector)
MIN_MASK = 20        # fewer pixels in its mask than this, and a template isn't used
MIN_SOLID = 10       # fewer solid pixels than this, and its colours are checked over the whole mask
# Two kinds whose pictures are one picture recoloured (the same size; the ores, the gems), both found at one place (a px
# apart at most) with shape scores this close: the one whose colours are nearer its own is the drop (`_recoloured`). On
# brightness alone an Iron Ore's picture scores 0.998 where a Silver Ore lies, and its colours are 12.1 off there, under
# PIXEL_SLACK. Drawn 1:1 on West Rocky Mountain II's frames with a fifth of it behind a piece of a monster's sprite, 9
# of 43 Silver Ores were taken for Iron Ore (a third: 14), and none now; the shape scores of the two came within 0.005
# of each other (README: Perception, drops of the new maps).
SAME_SHAPE = 0.02
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
    coin: str | None = None  # mesos: the coin they're drawn as (MESO_KINDS: "bronze", "gold"), which their
                             # amount picks (what they're worth: behavior.looting.worth); None if not known

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


def _at_size(rgba: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A template's colours and cover at frame size (`resample_opaque`); at its own size, where the game draws it
    pixel for pixel, only the px it draws as they are (alpha 255): the others are partly what lies behind them on
    screen. (The game's item pictures are all opaque or clear but the Fragment of Magic's glow: the same either way.)"""
    if size != (rgba.shape[1], rgba.shape[0]):
        return resample_opaque(rgba, size)
    drawn = rgba[..., 3] == 255
    return np.where(drawn[..., None], rgba[..., :3], 0).astype(np.uint8), drawn.astype(np.float32)


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


def folder_for(stretch: float) -> Path:
    """Where the drops drawn `stretch` times larger are kept: LOOT for the 1920x1080 window's 1.40625, else
    `loot_x<stretch>` ("loot_x1": the game's own pictures, for 1366x768)."""
    return LOOT if stretch == STRETCHED.stretch else OUT / f"loot_x{stretch:g}"


def build(folder: Path = LOOT, stretch: float = STRETCHED.stretch, items: tuple[str, ...] = ITEMS) -> list[Path]:
    """Write `<kind>_<n>.png` (RGBA, drawn `stretch` times larger: by default screen size at 1920x1080) for each of
    `items` (by default the first map's), and `mesos_<coin><frame>.png` for the coins' spinning frames (the edge-on
    one, 5 px wide, is left out). At stretch 1 (the game's own 1366x768) each is the game's picture as it is, alpha and
    all. Another stretch than 1920x1080's is written in the folder's mobs.SCALE_FILE. An item whose kind can't be a
    file's name (drops.safe) is refused."""
    from gamedata.canvas import item_icon, meso_frames
    if unsafe := [n for n in items if not drops.safe(n)]:
        raise ValueError(f"drops whose kinds can't be file names: {unsafe}")
    folder.mkdir(parents=True, exist_ok=True)
    if stretch != STRETCHED.stretch:
        (folder / SCALE_FILE).write_text(json.dumps({"px_per_game_px": stretch}), encoding="utf-8")
    ids, written = item_ids(COPY), []

    def save(rgba: np.ndarray, name: str) -> None:
        path = folder / f"{name}.png"
        picture = rgba if stretch == 1 else resample_rgba(rgba, screen_size(rgba, stretch))
        Image.fromarray(np.ascontiguousarray(picture), "RGBA").save(path)
        written.append(path)

    for name in items:
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


def _coin(path: Path) -> str | None:
    """The coin a mesos template is of, from its file name (mesos_<coin><frame>.png), else None."""
    kind, _, rest = path.stem.rpartition("_")
    return (rest.rstrip("0123456789") or None) if kind == MESOS else None


def ensure(folder: Path | None = None, stretch: float = STRETCHED.stretch, items: tuple[str, ...] = ITEMS) -> Path:
    """Build the templates of `items` and the coins (drawn `stretch` times larger, in `folder`: by default
    `folder_for(stretch)`) if any of them are missing; returns the folder."""
    folder = folder_for(stretch) if folder is None else folder
    have = {_kind(p) for p in folder.glob("*.png")}
    if not kinds_of(items) <= have:
        build(folder, stretch, items)
    return folder


def kinds_of(items: tuple[str, ...]) -> set[str]:
    """The kinds a detector for these items holds: theirs, and mesos."""
    return {kind_of(n) for n in items} | {MESOS}


def detector_for(model, geo: Geometry = STRETCHED) -> "LootDetector":
    """A LootDetector for map `model`'s drops (drops.items_for: its monsters' and the potions, and mesos), for look
    frames of geometry `geo`."""
    return LootDetector(geo=geo, items=drops.items_for(model))


class LootTemplate(NamedTuple):
    """One drop's picture, at frame size."""
    kind: str
    rgb: np.ndarray    # its colours
    mask3: np.ndarray  # where the picture is (uint8, one copy per colour channel)
    solid: np.ndarray  # where its colours are checked (bool)
    coin: str | None = None  # mesos: the coin it's a frame of
    gray: tuple[np.ndarray, np.ndarray] | None = None  # its brightness and mask (one channel), when matched on that


class LootDetector:
    """Templates come from `folder` (built from the game data if not given): PNGs named
    <kind>_<anything>.png, RGBA at full screen size (mobs.picture_scale's), brought to look frames of geometry
    `geo` (by default the 1920x1080 window's, at half size). `items`: the drops looked for, by name (a map's:
    drops.items_for, `detector_for`), and mesos; by default the first map's (ITEMS), or with a `folder` given every
    picture in it. `gray`: match the shapes on brightness alone (by default where the pictures are drawn at the
    frames' own scale, the game's own pixels)."""

    def __init__(self, folder: Path | None = None, geo: Geometry = STRETCHED, gray: bool | None = None,
                 items: tuple[str, ...] | None = None):
        if folder is None:
            items = ITEMS if items is None else tuple(items)
            folder = ensure(stretch=geo.stretch, items=items)
        self.items = items
        kinds = None if items is None else kinds_of(items)
        shrink = picture_scale(folder) / geo.frame_scale
        self.gray = abs(shrink - 1) < 1e-9 if gray is None else gray
        self.templates: list[LootTemplate] = []
        for path in sorted(folder.glob("*.png")):
            if kinds is not None and _kind(path) not in kinds:
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            h, w = rgba.shape[:2]
            rgb, cover = _at_size(rgba, frame_size(w, h, shrink))
            mask = (cover >= HALF_COVER).astype(np.uint8)
            if mask.sum() < MIN_MASK:
                continue
            # Colours are checked where the picture covers the whole pixel: on screen a thin line (the
            # Slime Bubble's string) is half line, half background at half size, and was 41 off.
            solid = cover >= SOLID_COVER
            solid = solid if solid.sum() >= MIN_SOLID else mask > 0
            gray = (cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY), mask) if self.gray else None
            self.templates.append(LootTemplate(_kind(path), np.ascontiguousarray(rgb), np.dstack([mask] * 3), solid,
                                               _coin(path), gray))
        if not self.templates:
            raise RuntimeError(f"no loot templates in {folder}")
        self.height = max(t.rgb.shape[0] for t in self.templates)
        self._pool = ThreadPoolExecutor(THREADS)  # OpenCV lets go of the GIL while matching

    def detect(self, frame: np.ndarray, box: tuple[int, int, int, int] | None = None,
               mesos_only: bool = False) -> list[Item]:
        """Items in `frame` (RGB, a look frame of the detector's geometry), within `box` (x0, y0, x1, y1) if given. The
        templates are matched on a few threads: 30 ms a platform's strip instead of 90."""
        return self.detect_each(frame, [box or (0, 0, frame.shape[1], frame.shape[0])], mesos_only)[0]

    def detect_each(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]],
                    mesos_only: bool = False) -> list[list[Item]]:
        """`detect` in each of several boxes, every box's templates on the threads together: the strips of
        every platform in view (13 on the bench frames) took 93 ms one after another, 68 together."""
        areas = []
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            area = np.ascontiguousarray(frame[y0:int(y1), x0:int(x1)])
            areas.append((area, (x0, y0), cv2.cvtColor(area, cv2.COLOR_RGB2GRAY) if self.gray and area.size else None))

        def wanted(t: LootTemplate, area: np.ndarray) -> bool:
            fits = t.rgb.shape[0] <= area.shape[0] and t.rgb.shape[1] <= area.shape[1]
            return (t.kind == MESOS or not mesos_only) and fits

        jobs = [(k, t) for k, (area, _, _) in enumerate(areas) for t in self.templates if wanted(t, area)]
        found = self._pool.map(lambda job: self._match(*areas[job[0]], job[1]), jobs)
        each: list[list[tuple[Item, float]]] = [[] for _ in boxes]
        for (k, _), hits in zip(jobs, found):
            each[k] += hits
        return [suppress_overlaps(_recoloured(hits)) for hits in each]

    def _match(self, area: np.ndarray, origin: tuple[int, int], gray: np.ndarray | None,
               template: LootTemplate) -> list[tuple[Item, float]]:
        """Where one template is in `area`, whose top left is at `origin` in the frame (`gray`: the area's
        brightness, where shapes are matched on that): each with how far its colours are from the template's."""
        kind, rgb, mask3, solid, coin = template[:5]
        own = rgb[solid].astype(np.float32)
        th, tw = rgb.shape[:2]
        res = masked_match(area, rgb, mask3) if gray is None else masked_match(gray, *template.gray)
        peaks = (res >= THRESHOLD) & (res == cv2.dilate(res, np.ones((5, 5), np.uint8)))
        hits = []
        for y, x in zip(*np.nonzero(peaks)):
            seen = area[y:y + th, x:x + tw][solid].astype(np.float32)
            if (off := float(np.abs(seen - own).mean())) <= PIXEL_SLACK:
                hits.append((Item(kind, origin[0] + int(x) + tw // 2, origin[1] + int(y) + th // 2, tw, th,
                                  round(float(res[y, x]), 3), coin), off))
        return hits


def _recoloured(hits: list[tuple[Item, float]]) -> list[Item]:
    """The drops found (each with how far its colours are from its template's), but where two kinds' pictures of one
    size are found at one place with shape scores within SAME_SHAPE: only the one whose colours are nearer."""
    return [item for item, off in hits
            if not any(other.kind != item.kind and (other.w, other.h) == (item.w, item.h)
                       and abs(other.x - item.x) <= 1 and abs(other.y - item.y) <= 1
                       and abs(other.score - item.score) <= SAME_SHAPE and other_off < off
                       for other, other_off in hits)]
