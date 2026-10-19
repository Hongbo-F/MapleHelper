"""Hunting controller pieces that don't need the game: bars, platform edges, move choice, detection."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.hunt import BEST_RANGE, TOO_CLOSE, Sight, bars, ground_extent, next_move
from gamelab.mobs import ASSETS, Mob, MobDetector
from maplehelper.engine import LEFT, RIGHT

FIXTURES = Path(__file__).parent / "fixtures"

SKY = (60, 140, 230)
GRASS = (90, 170, 40)


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


def sight(her=(400, 300), mobs=(), ground=(100, 700)):
    return Sight(0.0, np.zeros((540, 960, 3), np.uint8), her, list(mobs), 1.0, 1.0, 0.0, ground)


def mob(dx_game, kind="slime"):
    return Mob(kind, 400 + int(dx_game / 2), 300 - 8 - 15, 30, 30, 0.9)  # feet on her floor


def test_hopping_slime_still_counts_as_on_her_floor():
    hopping = Mob("slime", 300, 300 - 8 - 15 - 30, 30, 30, 0.9)
    above = Mob("slime", 300, 300 - 8 - 15 - 80, 30, 30, 0.9)  # on a platform above
    assert sight(mobs=[hopping, above]).on_floor() == [hopping]


def test_throws_from_good_range_and_faces_the_target():
    target = mob(-200)
    assert next_move(sight(mobs=[target]), target) == ("throw", LEFT)


def test_walks_closer_when_too_far():
    target = mob(600)
    move = next_move(sight(mobs=[target]), target)
    assert move[:2] == ("walk", RIGHT) and 0 < move[2] <= 300
    assert 600 - move[2] >= BEST_RANGE[0]


def test_steps_away_from_a_monster_that_is_too_close():
    target = mob(TOO_CLOSE - 30)
    assert next_move(sight(mobs=[target]), target)[:2] == ("walk", LEFT)


def test_fights_when_cornered_between_monsters():
    target, behind = mob(60), mob(-150)
    assert next_move(sight(mobs=[target, behind]), target) == ("throw", RIGHT)


def test_never_walks_off_the_platform():
    target = mob(800)
    s = sight(mobs=[target], ground=(380, 440))  # 40 half-px of ground to her right = 55 game px of room
    move = next_move(s, target)
    assert move == ("skip", "out of reach from this platform")
    near_edge = mob(340)
    assert next_move(sight(mobs=[near_edge], ground=(380, 410)), near_edge) == ("throw", RIGHT)
    close = mob(40)
    assert next_move(sight(mobs=[close], ground=(395, 700)), close) == ("throw", RIGHT)  # no room to back off


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
