"""Which way she faces, seen on a frame (perception.heading): her picture above her name tag against her pictures and
their mirror images, on frames whose facing is known (tests/fixtures/heading: 21 frames at 1366x768 in her outfit of
2026-10-24, and 23 of the old recordings' at 1920x1080 half size, labelled by the keys sent)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from maplehelper.engine import LEFT, RIGHT

from gamelab import paths
from gamelab.perception import heading
from gamelab.perception.heading import Heading
from gamelab.perception.nametag import NameTag
from gamelab.screen import NATIVE, STRETCHED

FIXTURES = Path(__file__).parent / "fixtures"
LABELS = json.loads((FIXTURES / "heading" / "labels.json").read_text(encoding="utf-8"))
GEOS = {"native": NATIVE, "old": STRETCHED}


def crops(kind: str) -> list[tuple[np.ndarray, str, dict]]:
    """The fixture's crops of her (`kind`: "native" or "old"): each with the way she faces and its labels."""
    meta = LABELS[kind]
    w, h = meta["size"]
    strip = np.asarray(Image.open(FIXTURES / "heading" / f"{kind}.png").convert("RGB"))
    return [(strip[:h, i * w:(i + 1) * w], f["side"], f) for i, f in enumerate(meta["frames"])]


def placed(crop: np.ndarray, kind: str, mirror: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """A look frame of the crop's geometry, blank but for the crop, her name tag's centre in the middle of the play
    area (clear of the minimap); mirrored if asked (she then faces the other way), and where her tag is."""
    geo, (tx, ty) = GEOS[kind], LABELS[kind]["tag"]
    fw, fh = geo.frame_size
    frame = np.zeros((fh, fw, 3), np.uint8)
    x, y = fw // 2, int(fh * 0.6)
    frame[y - ty:y - ty + crop.shape[0], x - tx:x - tx + crop.shape[1]] = crop
    if mirror:
        return np.ascontiguousarray(frame[:, ::-1]), (fw - 1 - x, y)
    return frame, (x, y)


def bank_of(items, kind: str, path=None) -> Heading:
    """Pictures learnt from these crops (each as facing its way)."""
    bank = Heading(GEOS[kind], path, background=False)
    for crop, side, _ in items:
        assert bank.add(*placed(crop, kind), side)
    bank.rebuild()
    return bank


def other(side: str) -> str:
    return LEFT if side == RIGHT else RIGHT


# --- perception -----------------------------------------------------------------------------------------------------

def test_her_box_follows_the_windows_size():
    assert heading.box_px(STRETCHED) == (50, 8, 16, 3)
    assert heading.box_px(NATIVE) == (71, 11, 23, 4)


def test_each_frame_at_1366x768_is_read_with_pictures_of_the_other_20_both_ways_round():
    """Leave one out over the 21 frames and their mirror images: none read the wrong way, one not told (hp0_135329
    mirrored, on her chair hit-tinted: 0.04 between the ways), as measured (README, Her heading)."""
    items = crops("native")
    wrong, untold = [], []
    for i, (crop, side, f) in enumerate(items):
        bank = bank_of(items[:i] + items[i + 1:], "native")
        for mirror, truth in ((False, side), (True, other(side))):
            r = bank.read(*placed(crop, "native", mirror))
            if r.side is None:
                untold.append((f["file"], mirror, r.why))
            elif r.side != truth:
                wrong.append((f["file"], mirror))
    assert not wrong and len(untold) <= 1, (wrong, untold)


def test_the_old_recordings_frames_are_read_with_pictures_of_other_recordings():
    """At 1920x1080 half size, pictures learnt from half the old recordings read the frames of the other half (and
    their mirror images): none the wrong way, nearly all told."""
    items = crops("old")
    names = sorted({f["recording"] for _, _, f in items})
    for learnt in (names[::2], names[1::2]):
        bank = bank_of([c for c in items if c[2]["recording"] in learnt], "old")
        reads = [(bank.read(*placed(crop, "old", mirror)), side if not mirror else other(side))
                 for crop, side, f in items if f["recording"] not in learnt for mirror in (False, True)]
        assert not [r for r, truth in reads if r.side not in (None, truth)]
        assert sum(r.side is not None for r, _ in reads) >= 0.9 * len(reads)


def test_the_seed_tells_her_facing_on_the_saved_frames_and_their_mirror_images():
    """The seed in gamelab/assets (21 frames at 1366x768) on the two full frames of tests/fixtures/native, her tag
    found as a look finds it: she faces left, and right in the frames mirrored (her tag where the mirror puts it: the
    game doesn't mirror her name)."""
    seed = paths.calibrated(paths.HEADING_SEED, NATIVE.suffix)
    bank = Heading(NATIVE, None, seed)
    assert bank.ready and len(bank.refs) <= heading.REFERENCES
    template = FIXTURES / "native" / "tag_template_1366x768.npy"
    for name in ("n28_000.png", "n28_008.png"):
        full = np.asarray(Image.open(FIXTURES / "native" / name).convert("RGB"))
        x, y = NameTag(np.load(template), template, NATIVE).find(full)
        mirrored = np.ascontiguousarray(full[:, ::-1])
        for frame, tag, truth in ((full, (x, y), LEFT), (mirrored, (full.shape[1] - 1 - x, y), RIGHT)):
            r = bank.read(frame, tag)
            assert r.side == truth and r.strong, (name, truth, r)


def test_what_cant_be_told():
    items = crops("native")
    assert Heading(NATIVE).read(*placed(items[0][0], "native")) == heading.UNKNOWN  # no pictures yet
    bank = bank_of(items, "native")
    frame, (x, y) = placed(items[0][0], "native")
    assert bank.read(frame, None).why == "no tag"
    assert bank.read(frame, (5, y)).why == "off the frame"
    assert bank.read(frame, (100, 250)).why == "minimap"  # her box under the minimap (0..213, 68..213 at 1366x768)
    blank = np.zeros_like(frame)
    assert bank.read(blank, (x, y)).side is None  # nobody there: both ways alike


def test_a_decisive_picture_is_the_only_one_matched(monkeypatch):
    """The picture that matched best last is tried first; one leading the other way by DECISIVE decides alone."""
    items = crops("native")
    bank = bank_of(items, "native")
    assert len(bank.refs) > 1
    calls = []
    real = heading.cv2.matchTemplate
    monkeypatch.setattr(heading.cv2, "matchTemplate", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = bank.read(*placed(items[0][0], "native"))
    assert r.margin >= heading.DECISIVE and len(calls) == 2  # as it is and mirrored, one picture


def test_her_pictures_learn_from_looks_whose_keys_leave_no_doubt_and_are_kept(tmp_path):
    items = crops("native")
    path = tmp_path / "heading_1366x768.npz"
    bank = Heading(NATIVE, path, background=False)
    t = 100.0
    for crop, side, _ in items[:heading.FIRST - 1]:
        assert bank.learn(*placed(crop, "native"), side, t)
        t += heading.LEARN_EVERY_S
    assert not bank.ready
    crop, side, _ = items[heading.FIRST - 1]
    assert not bank.learn(*placed(crop, "native"), side, t - heading.LEARN_EVERY_S / 2)  # one a second at most
    assert bank.learn(*placed(crop, "native"), side, t)
    assert bank.ready and path.exists() and len(bank.pool) == heading.FIRST
    again = Heading(NATIVE, path)
    assert len(again.refs) == len(bank.refs) and len(again.pool) == heading.FIRST
    assert not Heading(STRETCHED, path).ready  # pictures of another window size aren't hers there


def test_pictures_that_keep_reading_the_other_way_from_her_keys_are_dropped(tmp_path):
    """A look the pictures read the other way from keys that can't be wrong isn't learnt, and counts against them:
    OUTFIT[0] of the last OUTFIT[1] such looks, and they're no longer hers (a new outfit): dropped."""
    items = crops("native")
    bank = bank_of(items, "native", tmp_path / "h.npz")
    t = 0.0
    for i in range(heading.OUTFIT[0]):
        crop, side, _ = items[i % len(items)]
        frame, tag = placed(crop, "native")
        r = bank.read(frame, tag)
        assert r.side == side
        assert not bank.learn(frame, tag, other(side), t, r)
        t += heading.LEARN_EVERY_S
    assert bank.counts["forgot"] == 1 and not bank.ready and not bank.pool
