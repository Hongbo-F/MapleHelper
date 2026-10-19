"""gamedata's readers, on small packages built here in the game's formats (no game files needed)."""

import struct
from dataclasses import dataclass

import numpy as np
import pytest

from gamedata import ms, tables
from gamedata.__main__ import copy_data
from gamedata.wz import OFFSET_KEY, Canvas, Package, Reader, Uol, Vector, jsonable

# --- writing .wz data ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Short:
    value: int


@dataclass(frozen=True)
class Long:
    value: int


@dataclass(frozen=True)
class Double:
    value: float


def cint(v: int) -> bytes:
    return struct.pack("<b", v) if -127 <= v <= 127 else b"\x80" + struct.pack("<i", v)


def wz_string(s: str) -> bytes:
    if all(ord(c) < 128 for c in s):
        head = struct.pack("<b", -len(s)) if len(s) < 128 else b"\x80" + struct.pack("<i", len(s))
        return head + bytes(ord(c) ^ ((0xAA + i) & 0xFF) for i, c in enumerate(s))
    head = struct.pack("<b", len(s)) if len(s) < 127 else b"\x7f" + struct.pack("<i", len(s))
    return head + b"".join(struct.pack("<H", ord(c) ^ ((0xAAAA + i) & 0xFFFF)) for i, c in enumerate(s))


def props(tree: dict) -> bytes:
    out = cint(len(tree))
    for name, value in tree.items():
        out += b"\x00" + wz_string(name) + prop(value)
    return out


def prop(v) -> bytes:
    if v is None:
        return b"\x00"
    if isinstance(v, Short):
        return b"\x02" + struct.pack("<h", v.value)
    if isinstance(v, Long):
        return b"\x14" + (struct.pack("<b", v.value) if -127 <= v.value <= 127 else b"\x80" + struct.pack("<q", v.value))
    if isinstance(v, Double):
        return b"\x05" + struct.pack("<d", v.value)
    if isinstance(v, int):
        return b"\x03" + cint(v)
    if isinstance(v, float):
        return b"\x04\x80" + struct.pack("<f", v)
    if isinstance(v, str):
        return b"\x08\x00" + wz_string(v)
    body = extended(v)
    return b"\x09" + struct.pack("<I", len(body)) + body


def extended(v) -> bytes:
    if isinstance(v, dict):
        return b"\x73" + wz_string("Property") + b"\x00\x00" + props(v)
    if isinstance(v, Vector):
        return b"\x73" + wz_string("Shape2D#Vector2D") + cint(v.x) + cint(v.y)
    if isinstance(v, Uol):
        return b"\x73" + wz_string("UOL") + b"\x00\x00" + wz_string(v.path)
    if isinstance(v, Canvas):
        head = b"\x73" + wz_string("Canvas") + b"\x00" + (b"\x01\x00\x00" + props(v.props) if v.props else b"\x00")
        pixels = b"\x78\x9c\x03\x00\x00\x00\x00\x01"
        return (head + cint(v.width) + cint(v.height) + cint(v.fmt) + b"\x00" + bytes(4)
                + struct.pack("<i", len(pixels) + 1) + b"\x00" + pixels)
    raise TypeError(v)


def image(tree: dict) -> bytes:
    return extended(tree)


START = 60


def scramble(p: int, h: int) -> int:
    off = ((p - START) ^ 0xFFFFFFFF) * h & 0xFFFFFFFF
    off = (off - OFFSET_KEY) & 0xFFFFFFFF
    r = off & 0x1F
    return ((off << r) | (off >> (32 - r))) & 0xFFFFFFFF if r else off


def package(tree: dict, version: int = 779) -> bytes:
    """A .wz file holding `tree`: {name: image bytes, or a dict for a folder}."""
    h = Package.version_hash(version)
    folders, blobs = [], []

    def walk(t):  # folders first (listings), then images, in order
        folders.append(t)
        for v in t.values():
            if isinstance(v, dict):
                walk(v)
    walk(tree)

    def listing_size(t):
        return len(cint(len(t))) + sum(1 + len(wz_string(n)) + len(cint(len(v) if isinstance(v, bytes) else 0)) + 1 + 4
                                       for n, v in t.items())
    pos, where = START + 2, {}
    for f in folders:
        where[id(f)] = pos
        pos += listing_size(f)
    for f in folders:
        for v in f.values():
            if isinstance(v, bytes):
                where[id(v)] = pos
                blobs.append(v)
                pos += len(v)
    body = bytearray()
    for f in folders:
        out = bytearray(cint(len(f)))
        for name, v in f.items():
            out += bytes([4 if isinstance(v, bytes) else 3]) + wz_string(name)
            out += cint(len(v) if isinstance(v, bytes) else 0) + cint(0)
            at = where[id(f)] + len(out)
            out += struct.pack("<I", ((where[id(v)] - 2 * START) & 0xFFFFFFFF) ^ scramble(at, h))
        body += out
    for b in blobs:
        body += b
    head = b"PKG1" + struct.pack("<QI", 2 + len(body), START)
    head += b"Package file v1.0 Copyright 2002 Wizet, ZMS\x00".ljust(START - len(head), b"\x00")
    return head + struct.pack("<H", Package.version_check_of(h)) + bytes(body)


# --- .wz -----------------------------------------------------------------------------------------------------

TREE = {
    "info": {"level": 15, "maxHP": 259, "exp": 28, "big": 100_000, "name": "Bubbling", "unicode": "Lügner"},
    "frames": {"0": {"delay": 300}, "1": {"delay": -150}},
    "short": Short(-5), "long": Long(3_000_000_000), "double": Double(0.25), "float": 0.5, "nothing": None,
    "origin": Vector(-4, 30), "link": Uol("../info"),
    "icon": Canvas(32, 32, 1, {"origin": Vector(0, 32), "z": 0}), "plain_icon": Canvas(16, 8, 2),
}


def test_image_tree_round_trips():
    expected = {k: v.value if isinstance(v, (Short, Long, Double)) else v for k, v in TREE.items()}
    assert Reader(image(TREE)).tree() == expected


def test_text_stored_by_offset_is_followed():
    # A name or text can be an offset from the image start to a string written earlier in the image.
    head = b"\x73" + wz_string("Property") + b"\x00\x00" + cint(3) + b"\x00" + wz_string("name") + b"\x08\x00"
    at = len(head)
    data = (head + wz_string("speed")
            + b"\x01" + struct.pack("<i", at) + b"\x03" + cint(5)                            # a name by offset
            + b"\x00" + wz_string("again") + b"\x08\x01" + struct.pack("<i", at))           # a text by offset
    assert Reader(data).tree() == {"name": "speed", "speed": 5, "again": "speed"}


def test_package_lists_images_in_folders_and_parses_them(tmp_path):
    path = tmp_path / "Test_000.wz"
    path.write_bytes(package({"Physics.img": image({"gravityAcc": 2000.0}),
                              "Mob": {"0000002.img": image({"info": {"level": 1}}), "Sub": {"x.img": image({})}}}))
    pkg = Package(path)
    assert pkg.version == 779
    assert pkg.images() == ["Physics.img", "Mob/0000002.img", "Mob/Sub/x.img"]
    assert pkg.image("Mob/0000002.img") == {"info": {"level": 1}}
    assert pkg.image("Physics.img") == {"gravityAcc": 2000.0}


def test_index_package_without_entries(tmp_path):
    path = tmp_path / "String.wz"
    path.write_bytes(package({}))
    pkg = Package(path)
    assert pkg.images() == [] and pkg.version is None


def test_not_a_package(tmp_path):
    path = tmp_path / "x.wz"
    path.write_bytes(b"nope" + bytes(100))
    with pytest.raises(ValueError):
        Package(path)


def test_jsonable():
    assert jsonable({"v": Vector(1, 2), "u": Uol("a/b"), "c": Canvas(2, 3, 1, {"z": 0}), "n": [1]}) == \
        {"v": {"_vec": [1, 2]}, "u": {"_uol": "a/b"}, "c": {"_canvas": [2, 3, 1], "z": 0}, "n": [1]}


# --- .ms -----------------------------------------------------------------------------------------------------

def test_chacha20_matches_rfc_7539():
    # RFC 7539, 2.3.2: the block function test vector.
    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    expected = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    assert ms.chacha20(key, nonce, 1, 1) == expected
    assert ms.chacha20(key, nonce, 0, 3)[64:128] == expected  # several blocks at once, from counter 0


def test_table_reader_restarts_the_counter_when_a_read_ends_on_a_block():
    key = bytes(range(32))
    plain = bytes(range(128))
    block = [ms.chacha20(key, bytes(12), n, 1) for n in (0, 1)]
    restarted = ms.xor(plain[:64], block[0]) + ms.xor(plain[64:], block[0])
    r = ms.TableReader(restarted, 0, key)
    assert r.read(64) == plain[:64] and r.read(64) == plain[64:]
    counted = ms.xor(plain[:64], block[0]) + ms.xor(plain[64:], block[1])
    r = ms.TableReader(counted, 0, key)
    assert r.read(60) == plain[:60] and r.read(68) == plain[60:]  # the first read ended mid-block


def _half(b):
    return ((b - 256 if b > 127 else b) >> 1) & 0xFF


def ms_pack(file_name: str, images: dict[str, bytes], salt: str = "Kx7Qp2Lm9Zr4Tb8Wc3Ny5V") -> bytes:
    """A version 4 .ms pack holding `images` ({path: image bytes}), encrypted the way the game does."""
    name = file_name.lower()
    filler = np.random.default_rng(7).integers(0, 256, sum(map(ord, name)) % 312 + 30, dtype=np.uint8).tobytes()
    half = [_half(b) for b in filler]
    out = bytearray(filler)
    masked_version = 4 ^ half[0]
    hashed_salt_len = 0x5A3C0000 | ((len(salt) ^ half[0]) & 0xFF)
    out += bytes([masked_version]) + struct.pack("<i", hashed_salt_len)
    salt_raw = bytearray()
    for i, ch in enumerate(salt):
        a = next(a for a in range(256) if ((a | 0x4B) << 1) - a - 75 == ord(ch))
        salt_raw += bytes([half[i] ^ a, 0x11])
    out += salt_raw
    header_start = len(out)
    name_salt = name + salt
    check = hashed_salt_len + masked_version + 4 + len(images) + sum(struct.unpack(f"<{len(salt)}H", salt_raw))
    out += ms.xor(struct.pack("<iI", (check + 2**31) % 2**32 - 2**31, len(images)),
                  ms.chacha20(ms.header_key(name_salt), bytes(12), 0, 1))
    out += bytes(header_start + 8 + sum(ord(c) * 3 for c in name) % 212 + 64 - len(out))

    plain, reads, layout, kb = bytearray(), [], [], 0
    for n, (path, blob) in enumerate(images.items()):
        key = bytes((n * 31 + i * 7) & 0xFF for i in range(16))
        plain += struct.pack("<i", len(path)) + path.encode("utf-16-le")
        plain += struct.pack("<7i", 0, 0, kb, len(blob), -(-len(blob) // 1024) * 1024, 0, 0) + key + bytes(8)
        reads += [4, 2 * len(path)] + [4] * 7 + [16, 4, 4]
        layout.append((path, blob, kb, key))
        kb += -(-len(blob) // 1024)
    plain += bytes(128)
    enc = ms.TableReader(bytes(plain), 0, ms.table_key(name_salt))
    table = b"".join(enc.read(n) for n in reads)
    out += table + bytes(enc.pos - len(table))
    out += bytes(-len(out) % 1024)
    data_start = len(out)
    out += bytes(kb * 1024)
    for path, blob, start, key in layout:
        k, nonce, counter = ms.image_cipher(salt, path, key)
        head = min(len(blob), 1024)
        sealed = ms.xor(blob[:head], ms.chacha20(k, nonce, counter, -(-head // 64))) + blob[head:]
        at = data_start + start * 1024
        out[at:at + len(sealed)] = sealed
    return bytes(out)


def test_ms_pack_round_trips(tmp_path):
    long_text = "A huge bubble removed from a Bubbling. " * 60  # past 1 KB: the rest of the image isn't encrypted
    images = {"Mob/0000002.img": image({"info": {"level": 1, "exp": 2}}),
              "Mob/0000014.img": image({"info": {"level": 15, "exp": 28}, "desc": long_text})}
    path = tmp_path / "Mob_00000.ms"
    path.write_bytes(ms_pack(path.name, images))
    pack = ms.MsPack(path)
    assert pack.images() == list(images)
    assert pack.image("Mob/0000002.img") == {"info": {"level": 1, "exp": 2}}
    assert pack.image("Mob/0000014.img")["desc"] == long_text


def test_ms_pack_checks_its_header(tmp_path):
    data = ms_pack("Mob_00000.ms", {"Mob/0000002.img": image({})})
    renamed = tmp_path / "Skill_00000.ms"  # the keys come from the file name
    renamed.write_bytes(data)
    with pytest.raises(ValueError):
        ms.MsPack(renamed)


# --- tables --------------------------------------------------------------------------------------------------

def test_items_leave_out_looks():
    t = tables.items(
        {"Etc": {"4000014": {"name": "Bubbling's Huge Bubble", "desc": "..."}}},
        {"2000000": {"name": "Red Potion"}},
        {"ClassicWorld": {"Weapon": {"1472000": {"name": "Garnier"}}, "Hair": {"30000": {"name": "Toben Hair"}}}})
    assert t == {"Etc": {4000014: "Bubbling's Huge Bubble"}, "Use": {2000000: "Red Potion"},
                 "Equip": {"Weapon": {1472000: "Garnier"}}}


def test_mobs_by_level_with_names():
    rows = tables.mobs({"Mob/0000014.img": {"info": {"level": 15, "exp": 28, "maxHP": 259}},
                        "Mob/0000002.img": {"info": {"level": 1, "exp": 2}}, "Mob/pack_ignore.txt": {}},
                       {2: "Snail", 14: "Bubbling"})
    assert [(r["name"], r["level"], r["exp"]) for r in rows] == [("Snail", 1, 2), ("Bubbling", 15, 28)]


def test_maps_count_spawns_and_exits():
    tree = {"info": {"town": 0, "returnMap": 10000000, "VRLeft": -500, "VRTop": -300, "VRRight": 800, "VRBottom": 200},
            "life": {"0": {"type": "m", "id": "2"}, "1": {"type": "m", "id": "0000002"}, "2": {"type": "n", "id": "1"}},
            "portal": {"0": {"tm": 999999999}, "1": {"tm": 41}, "2": {"tm": 41}, "3": {"tm": 40}}}
    rows = tables.maps({"Map0/000000040.img": tree},
                       {"maple": {"40": {"streetName": "Maple Road", "mapName": "Snail Hunting Ground I"}}}, {2: "Snail"})
    assert rows == [{"id": 40, "street": "Maple Road", "name": "Snail Hunting Ground I", "town": False,
                     "returnMap": 10000000, "bounds": [-500, -300, 800, 200], "mobs": {"Snail": 2}, "npcs": 1,
                     "exits": [41]}]


def test_skills_per_level():
    job = {"skill": {"4001003": {"weapon": 47, "level": {"1": {"hs": "h1", "mpCon": 8, "lt": Vector(-1, -2)},
                                                          "2": {"hs": "h2", "mpCon": 8}}},
                     "4001001": {"action": {"0": "alert2"}, "level": {"1": {"mpCon": 50}}}}}
    out = tables.skills({"Skill/400.img": job}, {"4001003": {"name": "Lucky Seven"}})
    assert list(out) == ["4001001", "4001003"]
    assert out["4001003"] == {"name": "Lucky Seven", "action": None, "weapon": 47,
                              "levels": {1: {"mpCon": 8, "lt": {"_vec": [-1, -2]}}, 2: {"mpCon": 8}}}
    assert out["4001001"]["action"] == ["alert2"]


def test_actions_in_frame_order():
    body = {"info": {}, "swingO1": {"0": {"delay": 300}, "2": {"delay": 350}, "1": {"delay": 150}}, "prone": {"0": {}}}
    assert tables.actions(body) == {"swingO1": [300, 150, 350], "prone": []}


# --- copying -------------------------------------------------------------------------------------------------

def test_copy_skips_sound_and_unchanged_files(tmp_path):
    src, dst = tmp_path / "Data", tmp_path / "copy"
    (src / "String").mkdir(parents=True)
    (src / "Sound").mkdir()
    (src / "String" / "String_000.wz").write_bytes(b"PKG1")
    (src / "Sound" / "Bgm00.wz").write_bytes(b"PKG1")
    assert copy_data(src, dst)[:2] == (1, 0)
    assert not (dst / "Sound").exists()
    assert copy_data(src, dst)[:2] == (0, 1)
    assert copy_data(src, dst, sound=True)[:2] == (1, 1)
