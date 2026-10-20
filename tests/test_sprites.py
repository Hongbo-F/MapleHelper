"""Picture decoding (gamedata.canvas) and sprite-based monster templates, without game files."""

import json
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


def needs_game_data():
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")


def test_sprites_are_indexed_with_their_frames(tmp_path):
    needs_game_data()
    from gamelab.perception import sprites
    from gamelab.perception.mobs import INDEX, MobDetector
    sprites.build({8: "Pig"}, tmp_path)
    index = json.loads((tmp_path / INDEX).read_text(encoding="utf-8"))
    assert set(index) == {p.name for p in tmp_path.glob("pig_*.png")}
    hop = index["pig_move2.png"]  # the hop: its picture ends 4 units above its feet
    assert (hop["frame"], hop["origin"], hop["lt"], hop["rb"], hop["delay"], hop["w"], hop["h"]) == \
        ("move/2", [33, 50], [-33, -47], [33, -4], 300, 66, 43)
    t = next(t for t in MobDetector(tmp_path, threads=1).templates["pig"] if t.name == "pig_move2" and t.facing == -1)
    assert t.pose == "move" and t.origin[1] > t.rgb.shape[0]  # its feet are below the picture


def test_cutout_origins(tmp_path):
    needs_game_data()
    from gamelab.perception import sprites
    sprites.build({8: "Pig", 2: "Snail"}, tmp_path)
    found = sprites.cutout_origins({"pig", "snail"}, tmp_path)
    pig, snail = found["pig_1.png"], found["snail_9.png"]
    assert pig["frame"] == "move/2" and not pig["faces_right"]
    assert abs(pig["origin"][0] - 43.4) <= 1.5 and abs(pig["origin"][1] - 71.3) <= 1.5  # below the cut-out (58)
    assert snail["frame"] == "stand/0" and not snail["faces_right"]
    assert abs(snail["origin"][0] - 54.3) <= 1.5 and abs(snail["origin"][1] - 36.6) <= 1.5
    assert found["pig_0.png"]["faces_right"] and found["snail_8.png"]["faces_right"]
    assert all(v["score"] >= sprites.CUTOUT_SCORE for v in found.values())
    assert not any(name.startswith("slime") for name in found)  # no slime sprites in the folder


def test_sprites_made_before_the_index_are_indexed_again(tmp_path, monkeypatch):
    from gamelab.perception import sprites
    (tmp_path / "pig_move0.png").write_bytes(b"")  # made before INDEX was kept
    built = []
    monkeypatch.setattr(sprites, "build", lambda names, folder: built.append(names))
    monkeypatch.setattr(sprites, "cutout_origins", lambda kinds, folder, cutouts: {})
    sprites.ensure({8: "Pig"}, tmp_path)
    assert built == [{8: "Pig"}]

    def no_data(names, folder):
        raise FileNotFoundError("no game data copy")
    monkeypatch.setattr(sprites, "build", no_data)
    assert sprites.ensure({8: "Pig"}, tmp_path) == tmp_path  # the pictures still do, without the index
