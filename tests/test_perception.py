"""Perception without the game: HP/MP bars, the ground under her, monster HP bars, touches, and the
monster detector on pictures."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.analysis import bench
from gamelab.perception import hud
from gamelab.perception.eyes import rgb_of
from gamelab.perception.hud import BAR_ROW, HP_BAR, bars, hp_fill, touch_popup
from gamelab.perception.minimap import MINIMAP_BOX, dot_in_box
from gamelab.perception.nametag import FULL_EVERY, NameTag
from gamelab.perception.mobs import ASSETS, MobDetector
from gamelab.perception.terrain import ROPE_GAP, ScreenRope, find_ropes, ground_extent, rope_mask

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


def tiled_scene() -> np.ndarray:
    """A 1920x1080 RGB frame tiled from the saved game scene (daffodils, leaves, sky)."""
    scene = np.asarray(Image.open(FIXTURES / "game_scene.png").convert("RGB"))
    return np.ascontiguousarray(np.tile(scene, (5, 4, 1))[:1080, :1920])


def made_up_tag(seed: int = 0) -> np.ndarray:
    """A name tag (grey, full size): white letters on a dark box."""
    rng = np.random.default_rng(seed)
    tag = np.full((20, 72), 40, np.uint8)
    tag[5:15, 4:68] = np.where(rng.random((10, 64)) < 0.45, 235, 40)
    return tag


def with_tags(tag: np.ndarray, *corners: tuple[int, int]) -> np.ndarray:
    """The tiled scene with the tag pasted at these top-left corners (full-size px)."""
    full = tiled_scene()
    for x, y in corners:
        full[y:y + tag.shape[0], x:x + tag.shape[1]] = tag[..., None]
    return full


def name_tag_finder(tmp_path, tag: np.ndarray) -> NameTag:
    np.save(tmp_path / "tag.npy", tag)
    return NameTag(tag[::2, ::2], tmp_path / "tag.npy")


def centre(corner: tuple[int, int], tag: np.ndarray) -> tuple[int, int]:
    return (corner[0] + tag.shape[1] // 2) // 2, (corner[1] + tag.shape[0] // 2) // 2


def test_name_tag_found_near_where_she_was_as_by_the_whole_search(tmp_path):
    tag = made_up_tag()
    finder = name_tag_finder(tmp_path, tag)
    for i, corner in enumerate([(900, 600), (907, 596), (930, 640), (1010, 560)]):  # she walks and falls
        full = with_tags(tag, corner, (1500, 250))  # and her name in someone's list, far away
        whole = finder.search(full)  # what the whole play area gives (the search before)
        assert finder.find(full) == whole == centre(corner, tag)
        assert i == 0 or finder._looks % FULL_EVERY  # these came from the window
    assert finder.search(full, finder._near(full.shape)) == centre((1010, 560), tag)


def test_name_tag_searched_everywhere_once_she_leaves_the_window(tmp_path):
    tag = made_up_tag()
    finder = name_tag_finder(tmp_path, tag)
    assert finder.find(with_tags(tag, (900, 600))) == centre((900, 600), tag)
    assert finder.find(with_tags(tag, (1300, 600))) == centre((1300, 600), tag)  # 400 px away: the fallback
    assert finder.find(with_tags(tag, (300, 800), (900, 600))) == centre((900, 600), tag)  # back: nearest wins
    assert finder.find(tiled_scene()) is None  # hidden
    assert finder._last == centre((900, 600), tag)


def test_name_tag_window_rules_out_the_minimap(tmp_path):
    # Near the top left the window reaches into the minimap, where her name shows too.
    tag = made_up_tag()
    finder = name_tag_finder(tmp_path, tag)
    assert finder.find(with_tags(tag, (400, 360))) == centre((400, 360), tag)
    assert finder.find(with_tags(tag, (200, 250))) is None  # only in the minimap now


def test_name_tag_whole_search_every_so_often(tmp_path, monkeypatch):
    tag = made_up_tag()
    finder = name_tag_finder(tmp_path, tag)
    full = with_tags(tag, (900, 600))
    boxes = []
    search = finder.search
    monkeypatch.setattr(finder, "search", lambda f, box=None: boxes.append(box) or search(f, box))
    for _ in range(2 * FULL_EVERY):
        assert finder.find(full) == centre((900, 600), tag)
    assert boxes.count(None) == 3  # the first look, and every FULL_EVERY-th


def every_colour():
    """Every RGB colour: 16 pictures of 16 reds x 256 greens x 256 blues."""
    g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    for r0 in range(0, 256, 16):
        picture = np.empty((16 * 256, 256, 3), np.uint8)
        picture[..., 0] = np.repeat(np.arange(r0, r0 + 16, dtype=np.uint8), 256)[:, None]
        picture[..., 1] = np.tile(g, (16, 1))
        picture[..., 2] = np.tile(b, (16, 1))
        yield picture


def numpy_rope_mask(rgb):
    """terrain.rope_mask as it was: numpy's tests on int16 copies."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r >= 140) & (r <= 235) & (g >= 125) & (g <= 215) & (b >= 95) & (b <= 180)
            & (r - b >= 20) & (r - b <= 70) & (np.abs(r - g) <= 30))


def numpy_find_ropes(full):
    """terrain.find_ropes as it was, on the numpy mask, over the whole screen."""
    mask = numpy_rope_mask(full[:int(full.shape[0] * 0.9)]).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((31, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    ropes = sorted((ScreenRope(int(x + w // 2), int(y), int(y + rh)) for x, y, w, rh, area in stats[1:count]
                    if rh >= 90 and w <= 12 and area >= rh * 1.5), key=lambda r: (r.x, r.top))
    merged = []
    for rope in ropes:
        last = merged[-1] if merged else None
        if last and abs(last.x - rope.x) <= ROPE_GAP[0] and rope.top - last.bottom <= ROPE_GAP[1]:
            merged[-1] = ScreenRope((last.x + rope.x) // 2, last.top, max(last.bottom, rope.bottom))
        else:
            merged.append(rope)
    return merged


ROPE = (200, 180, 140)  # a rope's light strands


def with_ropes(*ropes: tuple[int, int, int]) -> np.ndarray:
    """The tiled scene with twisted ropes drawn at x from top to bottom (full-size px)."""
    full = tiled_scene()
    for x, top, bottom in ropes:
        for y in range(top, bottom, 8):  # strands crossing: 6 px of rope, 2 of gap, 4 then 3 px wide
            full[y:y + 6, x - 2:x + 2] = ROPE
            full[y + 3:y + 6, x - 1:x + 2] = (150, 130, 100)
    return full


def test_rope_mask_matches_numpys_on_every_colour():
    for picture in every_colour():
        assert np.array_equal(rope_mask(picture), numpy_rope_mask(picture))
    scene = tiled_scene()
    assert np.array_equal(rope_mask(scene[:, 7:900]), numpy_rope_mask(scene[:, 7:900]))  # a strip, not contiguous


def test_ropes_on_the_whole_screen_as_before():
    for full in (tiled_scene(), with_ropes((300, 100, 700), (705, 50, 400), (705, 430, 800), (1245, 200, 900))):
        assert find_ropes(full) == numpy_find_ropes(full)
    assert [r.x for r in find_ropes(full)] == [300, 705, 1245]  # the rope broken at 400-430 is one


def test_ropes_searched_only_in_columns():
    full = with_ropes((300, 100, 700), (705, 50, 400), (705, 430, 800), (1245, 200, 900))
    whole = find_ropes(full)
    near = find_ropes(full, [(1190, 1300)])
    assert near == [r for r in whole if r.x == 1245]
    assert find_ropes(full, [(1190, 1300), (1250, 1350), (650, 760)]) == [r for r in whole if r.x in (705, 1245)]
    assert find_ropes(full, [(-100, -10), (1950, 2000)]) == find_ropes(full, []) == []


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
