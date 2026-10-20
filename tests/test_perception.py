"""Perception without the game: HP/MP bars, the ground under her, monster HP bars, touches, and the
monster detector on pictures."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.analysis import bench
from gamelab.perception import hud
from gamelab.perception.eyes import rgb_of
from gamelab.perception.hud import BAR_ROW, HP_BAR, bars, hp_fill, touch_popup
from gamelab.perception.minimap import MINIMAP_BOX, dot_in_box
from gamelab.perception.mobs import ASSETS, MobDetector
from gamelab.perception.terrain import ground_extent

FIXTURES = Path(__file__).parent / "fixtures"
SKY = (60, 140, 230)
GRASS = (90, 170, 40)


def test_captures_are_rgb_pictures_of_their_own_with_the_same_pixels():
    # A capture was viewed as RGB in place ([:, :, 2::-1]), and every OpenCV call on it copied it first.
    rng = np.random.default_rng(0)
    bgra = rng.integers(0, 256, (1080, 1920, 4), np.uint8)
    bgra[150:156, 100:106] = (40, 230, 240, 255)  # her yellow minimap dot (BGR)
    bgra[1062:1066, 707:800] = (30, 30, 220, 255)  # some HP left
    eyes = bench.SavedEyes(np.zeros((10, 36), np.uint8))
    eyes.show(bgra)
    full = eyes._capture()
    assert full.flags.c_contiguous and np.array_equal(full, bgra[:, :, 2::-1])
    x0, y0, x1, y1 = MINIMAP_BOX
    a, b = int(HP_BAR[0] * 1920), int(HP_BAR[1] * 1920)
    old = dot_in_box(bgra[y0:y1, x0:x1, 2::-1]), hp_fill(bgra[int(BAR_ROW * 1080), a:b, 2::-1].astype(np.int16))
    assert eyes.glance() == old and old[0] is not None and old[1] > 0
    row = bgra[5:6, :50]
    assert np.array_equal(rgb_of(row), row[:, :, 2::-1])


def test_bars_read_the_fill_fraction():
    full = np.zeros((1080, 1920, 3), np.uint8)
    full[1060:1069, 707:707 + 121] = (220, 30, 30)  # HP 121/147 px = 523/642 in game
    full[1060:1069, 859:859 + 74] = (30, 60, 220)   # MP half
    hp, mp = bars(full)
    assert hp == pytest.approx(121 / 147, abs=0.02)
    assert mp == pytest.approx(0.5, abs=0.02)


def test_ground_extent_stops_at_the_platform_edges():
    frame = np.full((300, 600, 3), SKY, np.uint8)
    frame[202:230, 100:250] = GRASS  # platform top at y=202
    assert ground_extent(frame, 180, 200) == pytest.approx((100, 249), abs=1)


def test_hp_bar_reads_the_green_fraction():
    frame = np.full((200, 200, 3), SKY, np.uint8)
    frame[87:93, 68:112] = (165, 165, 165)  # border
    frame[88:92, 70:100] = (0, 196, 0)    # 30 px green
    frame[88:92, 100:105] = (140, 0, 0)   # 5 px red (latest damage)
    frame[88:92, 105:110] = (0, 0, 0)     # 5 px black
    assert hud.hp_bar(frame, 90, 100) == pytest.approx(0.75)
    assert hud.hp_bar(np.full((200, 200, 3), SKY, np.uint8), 90, 100) is None


def test_touches_show_as_purple_text_above_her():
    frame = np.full((300, 400, 3), SKY, np.uint8)
    tag = (200, 200)
    assert touch_popup(frame, tag) == 0
    frame[120:135, 180:220] = (200, 40, 220)  # "MISS" (or damage taken) floating above her head
    assert touch_popup(frame, tag) >= 40


def test_detector_ignores_scenery():
    frame = np.asarray(Image.open(FIXTURES / "game_scene.png").convert("RGB"))  # daffodils, leaves, sky
    half = np.asarray(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))
    assert MobDetector().detect(half) == []


def test_detector_finds_a_pasted_monster():
    frame = np.full((200, 300, 3), SKY, np.uint8)
    frame[150:] = GRASS
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    small = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
    h, w = small.shape[:2]
    region = frame[150 - h:150, 100:100 + w]
    alpha = small[..., 3:] > 0
    frame[150 - h:150, 100:100 + w] = np.where(alpha, small[..., :3], region)
    found = MobDetector().detect(frame)
    assert [m.kind for m in found] == ["pig"]
    assert abs(found[0].x - (100 + w // 2)) <= 3


def test_blank_windows_dont_crowd_out_a_monster_in_an_odd_sized_band():
    # The fight's band was 151 rows. Halving an odd size smeared faint colour into blank windows,
    # whose silhouette scores came out a fake 1.0 and took every check: in the basin, snails and
    # pigs at her level were never seen. Here: pig-coloured specks (like flowers), a pig on the right.
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    pig = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
    ph, pw = pig.shape[:2]
    detector = MobDetector()
    for seed in range(4):
        rng = np.random.default_rng(seed)
        frame = np.full((151, 960, 3), SKY, np.uint8)
        frame[121:] = GRASS
        for _ in range(60):
            y, x = rng.integers(0, 117), rng.integers(0, 700)
            frame[y:y + rng.integers(2, 9), x:x + rng.integers(2, 12)] = (240, 160, 150)
        region = frame[121 - ph:121, 760:760 + pw]
        frame[121 - ph:121, 760:760 + pw] = np.where(pig[..., 3:] > 0, pig[..., :3], region)
        found = detector.detect(frame, rows=(0, 151))
        assert any(m.kind == "pig" and abs(m.x - (760 + pw // 2)) <= 4 for m in found), seed
