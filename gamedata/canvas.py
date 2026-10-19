"""Decoding pictures (canvases) and finding a monster's animation frames.

After a canvas's size and format come: a second format byte, 4 zero bytes, the data length (int32,
counting one more zero byte), that byte, then the pixels, zlib-compressed. The game's streams end
without zlib's end marker, so they're inflated with a decompress object rather than zlib.decompress.

Newer clients keep the pictures out of the data images: a monster's frames in Packs/Mob_00000.ms are
empty canvases with a `_hash`, and the pixels are at the same path ("stand/0") in the monster's image
in Mob/_Canvas/_Canvas_000.wz. Frames that repeat another one say so with `_inlink` ("move/2").
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
    """A monster's frames: [{"action", "index", "rgba", "origin": (x, y)}], one per distinct picture.
    `origin` is the point (in the picture) that sits on the monster's position, i.e. at its feet."""
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
            origin = frame.props.get("origin")
            out.append({"action": action, "index": int(index), "rgba": pixels(picture),
                        "origin": (origin.x, origin.y) if origin is not None else (0, 0)})
    return out
