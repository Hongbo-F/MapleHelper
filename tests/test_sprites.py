"""Picture decoding (gamedata.canvas) and sprite-based monster templates, without game files."""

import struct
import zlib

import numpy as np
import pytest

from gamedata.canvas import pixels
from gamedata.wz import Canvas
from gamelab.perception.mobs import palette
from gamelab.perception.kinds import kind_of


def canvas_bytes(fmt2: int, packed: bytes) -> bytes:
    # second format byte, 4 zero bytes, length (counting one more zero byte), that byte, zlib stream
    stream = zlib.compress(packed)[:-4]  # the game's streams have no zlib end marker
    return bytes([fmt2]) + b"\0" * 4 + struct.pack("<i", len(stream) + 1) + b"\0" + stream


def test_decodes_bgra4444():
    # 2x2: red, green, blue, half-transparent white; each pixel is a little-endian uint16 0xARGB
    px = [0xFF00, 0xF0F0, 0xF00F, 0x8FFF]
    raw = canvas_bytes(0, struct.pack("<4H", *px))
    rgba = pixels(Canvas(2, 2, 1, {}, (raw, 0, len(raw))))
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0].tolist() == [255, 0, 0, 255]
    assert rgba[0, 1].tolist() == [0, 255, 0, 255]
    assert rgba[1, 0].tolist() == [0, 0, 255, 255]
    assert rgba[1, 1].tolist() == [255, 255, 255, 136]


def test_decodes_bgra8888():
    raw = canvas_bytes(0, bytes([10, 20, 30, 40]))  # b, g, r, a
    assert pixels(Canvas(1, 1, 2, {}, (raw, 0, len(raw))))[0, 0].tolist() == [30, 20, 10, 40]


def test_unknown_format_is_refused():
    raw = canvas_bytes(0, b"\0" * 16)
    with pytest.raises(ValueError, match="isn't supported"):
        pixels(Canvas(2, 2, 1026, {}, (raw, 0, len(raw))))


def test_kind_names():
    assert kind_of("Orange Mushroom") == "orange-mushroom"
    assert kind_of(" Pig ") == "pig"


def test_orange_mushroom_cap_colour_but_not_daffodils():
    cap = np.array([[[238, 140, 30]]], np.uint8)       # sampled from the decoded sprite
    daffodil = np.array([[[250, 220, 40]]], np.uint8)  # yellow flowers on this map
    assert palette("orange-mushroom", cap)[0, 0]
    assert not palette("orange-mushroom", daffodil)[0, 0]
