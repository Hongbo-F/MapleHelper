"""Decoding pictures (canvases) and finding a monster's animation frames.

After a canvas's size and format come: a second format byte, 4 zero bytes, the data length (int32,
counting one more zero byte), that byte, then the pixels, zlib-compressed. The game's streams end
without zlib's end marker, so they're inflated with a decompress object rather than zlib.decompress.

Newer clients keep the pictures out of the data images: a monster's frames in Packs/Mob_00000.ms are
empty canvases with a `_hash`, and the pixels are at the same path ("stand/0") in the monster's image
in Mob/_Canvas/_Canvas_000.wz. Frames that repeat another one say so with `_inlink` ("move/2").
Items are the same: Item/Etc/Etc_000.wz 0400.img holds 1x1 placeholders, and the pictures are at the
same path in Item/Etc/_Canvas/_Canvas_000.wz. Mesos on the ground are Item/Special 0900.img.
"""

import struct
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ms import MsPack
from .paths import COPY
from .wz import Canvas, Package


def pixels(canvas: Canvas) -> np.ndarray:
    """The picture as an (height, width, 4) RGBA uint8 array."""
    if canvas.raw is None:
        raise ValueError("this canvas wasn't read from a file")
    data, start, end = canvas.raw
    blob = bytes(data[start:end])
    fmt = canvas.fmt + blob[0]
    length = struct.unpack_from("<i", blob, 5)[0]
    packed = zlib.decompressobj().decompress(blob[10:9 + length])
    w, h = canvas.width, canvas.height
    if fmt == 1:  # BGRA4444
        px = np.frombuffer(packed[: w * h * 2], dtype="<u2").reshape(h, w)
        channels = [(px >> shift) & 0xF for shift in (8, 4, 0, 12)]  # r, g, b, a
        return (np.dstack(channels) * 17).astype(np.uint8)
    if fmt == 2:  # BGRA8888
        bgra = np.frombuffer(packed[: w * h * 4], dtype=np.uint8).reshape(h, w, 4)
        return bgra[..., [2, 1, 0, 3]].copy()
    if fmt == 513:  # RGB565
        px = np.frombuffer(packed[: w * h * 2], dtype="<u2").reshape(h, w).astype(np.uint16)
        r, g, b = (px >> 11) & 0x1F, (px >> 5) & 0x3F, px & 0x1F
        rgb = np.dstack([(r * 255 + 15) // 31, (g * 255 + 31) // 63, (b * 255 + 15) // 31]).astype(np.uint8)
        return np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    raise ValueError(f"picture format {fmt} isn't supported yet")


@lru_cache(maxsize=None)
def _mob_pack(copy: Path = COPY) -> MsPack:
    return MsPack(copy / "Packs" / "Mob_00000.ms")


@lru_cache(maxsize=None)
def _mob_canvases(copy: Path = COPY) -> Package:
    return Package(copy / "Mob" / "_Canvas" / "_Canvas_000.wz")


def mob_frames(mob_id: int, actions=("stand", "move", "hit1"), copy: Path = COPY) -> list[dict]:
    """A monster's frames: [{"action", "index", "rgba", "origin": (x, y), "lt", "rb", "head", "delay"}], one
    per distinct picture. `origin` is the point (in the picture) that sits on the monster's position, i.e. at
    its feet. `lt` and `rb` are the corners of its body's box and `head` its head, from its feet (map units,
    y negative upward; None where the frame has none), and `delay` how long the frame shows (ms, or None).
    The pictures face left."""
    name = f"{mob_id:07d}.img"
    info = _mob_pack(copy).image(f"Mob/{name}")
    pictures = _mob_canvases(copy).image(name)
    out, seen = [], set()
    for action in actions:
        for index, frame in sorted((info.get(action) or {}).items(), key=lambda kv: (len(kv[0]), kv[0])):
            if not (isinstance(frame, Canvas) and index.isdigit()):
                continue
            source = frame.props.get("_inlink") or f"{action}/{index}"
            if source in seen:
                continue
            seen.add(source)
            src_action, src_index = source.split("/")
            picture = (pictures.get(src_action) or {}).get(src_index)
            if not isinstance(picture, Canvas):
                continue
            props = frame.props
            delay = props.get("delay")  # an int, or a string in some monsters' data ("200")
            out.append({"action": action, "index": int(index), "rgba": pixels(picture),
                        "origin": _point(props.get("origin")) or (0, 0), "lt": _point(props.get("lt")),
                        "rb": _point(props.get("rb")), "head": _point(props.get("head")),
                        "delay": int(delay) if delay is not None and str(delay).lstrip("-").isdigit() else None})
    return out


def _point(vector) -> tuple[int, int] | None:
    """A vector property (origin, lt, rb, head) as (x, y)."""
    return None if vector is None else (vector.x, vector.y)


ITEM_FOLDERS = {2: "Consume", 4: "Etc"}  # item id // 1,000,000 -> its folder under Item/


@lru_cache(maxsize=None)
def _item_canvases(folder: str, copy: Path = COPY) -> Package:
    return Package(copy / "Item" / folder / "_Canvas" / "_Canvas_000.wz")


def item_icon(item_id: int, copy: Path = COPY) -> np.ndarray:
    """An item's picture as it lies on the ground (`iconRaw`: the icon without its inventory shadow), RGBA."""
    folder = ITEM_FOLDERS.get(item_id // 1_000_000)
    if folder is None:
        raise ValueError(f"no item pictures read for item {item_id} yet")
    entry = _item_canvases(folder, copy).image(f"{item_id // 10_000:04d}.img").get(f"{item_id:08d}") or {}
    picture = (entry.get("info") or {}).get("iconRaw")
    if not isinstance(picture, Canvas):
        raise KeyError(f"item {item_id} has no picture")
    return pixels(picture)


def meso_frames(copy: Path = COPY, kinds=range(4)) -> dict[int, list[np.ndarray]]:
    """Mesos on the ground, by kind (0 bronze coin, 1 gold coin, 2 bundle of bills, 3 sack; the amount picks
    the kind), each a list of its spinning animation's frames (RGBA)."""
    pictures = Package(copy / "Item" / "Special" / "_Canvas" / "_Canvas_000.wz").image("0900.img")
    out = {}
    for kind in kinds:
        frames = (pictures.get(f"0900000{kind}") or {}).get("iconRaw") or {}
        out[kind] = [pixels(c) for _, c in sorted(frames.items(), key=lambda kv: int(kv[0]))
                     if isinstance(c, Canvas)]
    return out
