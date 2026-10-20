"""Perception without the game: HP/MP bars, the ground under her, monster HP bars, touches, and the
monster detector on pictures."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab import paths
from gamelab.analysis import bench
from gamelab.perception import hud, mobs
from gamelab.perception.eyes import rgb_of
from gamelab.perception.hud import BAR_ROW, HP_BAR, bars, hp_fill, touch_popup
from gamelab.perception.minimap import MINIMAP_BOX, dot_in_box
from gamelab.perception.nametag import FULL_EVERY, NameTag
from gamelab.perception.match import suppress_overlaps
from gamelab.perception.mobs import ASSETS, INDEX, MobDetector, read_meta
from gamelab.perception.sprites import SPRITES
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
        assert i == 0 or finder.looks % FULL_EVERY  # these came from the window
    assert finder.search(full, finder._near(full.shape)) == centre((1010, 560), tag)


def test_name_tag_searched_everywhere_once_she_leaves_the_window(tmp_path):
    tag = made_up_tag()
    finder = name_tag_finder(tmp_path, tag)
    assert finder.find(with_tags(tag, (900, 600))) == centre((900, 600), tag)
    assert finder.find(with_tags(tag, (1300, 600))) == centre((1300, 600), tag)  # 400 px away: the fallback
    assert finder.find(with_tags(tag, (300, 800), (900, 600))) == centre((900, 600), tag)  # back: nearest wins
    assert finder.find(tiled_scene()) is None  # hidden
    assert finder._last == centre((900, 600), tag)
    assert (finder.looks, finder.missed) == (4, 3)


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
    if (SPRITES / "cutouts.json").is_file():  # where the cut-outs' monsters stand, from the game's sprites
        found = MobDetector(meta=read_meta([SPRITES])).detect(frame)
        assert found[0].facing == 1 and found[0].frame == "move/2"  # pig_0 is a pig hopping right


def paste(frame: np.ndarray, rgba: np.ndarray, left: int, top: int) -> None:
    """Draw a half-size RGBA picture onto a frame, its top-left at (left, top)."""
    h, w = rgba.shape[:2]
    region = frame[top:top + h, left:left + w]
    frame[top:top + h, left:left + w] = np.where(rgba[..., 3:] > 0, rgba[..., :3], region)


def test_detector_reports_where_the_feet_are_and_which_way_it_faces(tmp_path):
    # A picture whose feet aren't under its middle (a sprite's origin), facing left as the game's sprites do.
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    Image.fromarray(rgba).save(tmp_path / "pig_move9.png")
    (tmp_path / INDEX).write_text('{"pig_move9.png": {"origin": [20, 60], "w": 104, "h": 64, "frame": "move/9",'
                                  ' "lt": [-30, -40], "rb": [40, 0]}}', encoding="utf-8")
    small = np.asarray(Image.fromarray(rgba).resize((52, 32)))  # half size
    detector = MobDetector(tmp_path, threads=1)
    for flip, facing, feet_x in ((False, -1, 10), (True, 1, 52 - 10)):
        frame = np.full((200, 300, 3), SKY, np.uint8)
        frame[150:] = GRASS
        paste(frame, small[:, ::-1] if flip else small, 120, 120)
        found = detector.detect(frame)
        assert [(m.kind, m.facing, m.pose, m.frame, m.template) for m in found] == [
            ("pig", facing, "move", "move/9", "pig_move9")]
        assert abs(found[0].foot_x - (120 + feet_x)) <= 1 and abs(found[0].feet_y - (120 + 30)) <= 1
        assert abs(found[0].x - (120 + 26)) <= 1  # its centre, as before
    boxes = {t.facing: t.box for t in detector.templates["pig"]}
    assert boxes == {-1: (-30, -40, 40, 0), 1: (-40, -40, 30, 0)}  # the game's boxes face left


def test_a_template_with_nothing_known_stands_on_its_centre_and_bottom():
    found = MobDetector(threads=1).detect(with_monsters(("snail_8", 400, 300, False)))
    assert [(m.ox, m.feet, m.facing, m.pose) for m in found] == [(None, None, 0, "")]
    assert (found[0].foot_x, found[0].feet_y) == (found[0].x, found[0].y + found[0].h // 2)


def test_blank_windows_dont_crowd_out_a_monster_in_an_odd_sized_band():
    # The fight's band was 151 rows. Halving an odd size smeared faint colour into blank windows,
    # whose silhouette scores came out a fake 1.0 and took every check: in the basin, snails and
    # pigs at her level were never seen. Here: pig-coloured specks (like flowers), a pig on the right.
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    pig = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
    ph, pw = pig.shape[:2]
    detector, serial = MobDetector(), MobDetector(threads=1)
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
        assert serial.detect(frame, rows=(0, 151)) == found


def core(found: list) -> list:
    """Detections as the detector gave them before templates said what they knew (box and score)."""
    return [(m.kind, m.x, m.y, m.w, m.h, m.score) for m in found]


def old_detect(detector: MobDetector, frame: np.ndarray, rows=None) -> list:
    """MobDetector.detect as it was: on one thread, the windows' body colour summed by a TM_CCORR with a
    template of ones, and each kind's spots checked as they were picked."""
    y_off = 0
    if rows:
        y_off = max(0, rows[0])
        frame = frame[y_off:rows[1]]
    frame = frame[:frame.shape[0] // mobs.COARSE * mobs.COARSE, :frame.shape[1] // mobs.COARSE * mobs.COARSE]
    channels = mobs.split_channels(frame)
    hits = []
    for kind, templates in detector.templates.items():
        colour = mobs.palette(kind, frame, channels)
        if colour.sum() < mobs.MIN_BODY_PIXELS:
            continue
        coarse = cv2.resize(colour.astype(np.float32), (frame.shape[1] // 2, frame.shape[0] // 2),
                            interpolation=cv2.INTER_AREA)
        spots, cover = [], {}
        for i, t in enumerate(templates):
            sil = t.silhouette
            if coarse.shape[0] < sil.shape[0] or coarse.shape[1] < sil.shape[1]:
                continue
            res = cv2.matchTemplate(coarse, sil, cv2.TM_CCOEFF_NORMED)
            if sil.shape not in cover:
                cover[sil.shape] = cv2.matchTemplate(coarse, np.ones(sil.shape, np.float32), cv2.TM_CCORR) / sil.size
            res[cover[sil.shape] < mobs.MIN_COVER * float(sil.mean())] = 0
            peaks = (res >= mobs.SILHOUETTE_THRESHOLD) & (res == cv2.dilate(res, np.ones((3, 3), np.uint8)))
            spots += [(float(res[y, x]), 2 * x + 2 * sil.shape[1] // 2, 2 * y + 2 * sil.shape[0] // 2, i)
                      for y, x in zip(*np.nonzero(peaks))]
        spots.sort(reverse=True)
        checked = []
        for _, cx, cy, i in spots:
            if len(checked) >= mobs.MAX_CHECKS:
                break
            if any(abs(cx - x) < mobs.CHECK_SLACK and abs(cy - y) < mobs.CHECK_SLACK for x, y in checked):
                continue
            checked.append((cx, cy))
            found = mobs._check(frame, templates[i], cx, cy)
            if found and found[0] >= mobs.THRESHOLD:
                score, x, y, w, h = found
                hits.append(mobs.Mob(kind, x, y_off + y, w, h, round(score, 3)))
    return suppress_overlaps(hits)


def old_detect_wide(detector: MobDetector, frame: np.ndarray, rows) -> list:
    hits = []
    for y0 in range(rows[0], max(rows[0] + 1, rows[1] - mobs.WIDE_BAND + mobs.WIDE_STEP), mobs.WIDE_STEP):
        hits += old_detect(detector, frame, (y0, min(rows[1], y0 + mobs.WIDE_BAND)))
    return suppress_overlaps(hits)


def with_monsters(*placed: tuple[str, int, int, bool]) -> np.ndarray:
    """The tiled scene at half size, with cut-outs (name, left x, feet y, flipped) pasted on it."""
    full = tiled_scene()
    frame = np.ascontiguousarray(cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA))
    for name, x, feet, flip in placed:
        rgba = np.asarray(Image.open(ASSETS / f"{name}.png"))
        small = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
        small = small[:, ::-1] if flip else small
        h, w = small.shape[:2]
        region = frame[feet - h:feet, x:x + w]
        frame[feet - h:feet, x:x + w] = np.where(small[..., 3:] > 0, small[..., :3], region)
    return frame


def test_monsters_searched_on_threads_are_found_as_before():
    frames = [with_monsters(("pig_0", 100, 420, False), ("slime_4", 400, 300, True), ("snail_8", 700, 460, False),
                            ("pig_2", 800, 200, True), ("snail_10", 300, 470, True)),
              with_monsters(), np.full((540, 960, 3), SKY, np.uint8)]
    one, six = MobDetector(threads=1), MobDetector(threads=6)
    found = 0
    for frame in frames:
        for rows in (None, (250, 401), (380, 501)):  # the whole frame, an odd band, another
            old = old_detect(one, frame, rows)
            assert one.detect(frame, rows) == six.detect(frame, rows)
            assert core(six.detect(frame, rows)) == core(old)
            found += len(old)
        old = old_detect_wide(one, frame, (0, 486))
        assert one.detect_wide(frame, (0, 486)) == six.detect_wide(frame, (0, 486))
        assert core(six.detect_wide(frame, (0, 486))) == core(old)
        found += len(old)
    assert found >= 10


@pytest.mark.skipif(not (paths.LAB / "z2_shoot1.png").exists(), reason="needs saved game screenshots (gamelab/lab)")
def test_monsters_in_saved_screenshots_found_as_before():
    folders = [ASSETS] + ([SPRITES] if any(SPRITES.glob("*.png")) else [])
    one, six = MobDetector(folders, threads=1), MobDetector(folders, threads=6)
    for name in ("z2_shoot1.png", "z2_down.png", "after_rest.png", "closeup.png"):
        full = np.asarray(Image.open(paths.LAB / name).convert("RGB"))
        frame = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)
        assert six.detect(frame, (230, 381)) == one.detect(frame, (230, 381))
        assert core(six.detect(frame, (230, 381))) == core(old_detect(one, frame, (230, 381)))
        assert core(six.detect_wide(frame, (0, 486))) == core(old_detect_wide(one, frame, (0, 486)))


def box_of(m) -> tuple[int, int, int, int]:
    return m.x - m.w // 2, m.y - m.h // 2, m.x + m.w // 2, m.y + m.h // 2


def test_a_monster_seen_is_refound_where_it_walked_to():
    # Seen in one frame, then a few px on in the next: the last template finds it in the window around
    # where it was, at the same place the whole search puts it.
    detector = MobDetector(threads=1)
    first = detector.detect(with_monsters(("pig_0", 300, 420, False)))
    assert [m.kind for m in first] == ["pig"]
    for dx, dy in ((0, 0), (7, 0), (-12, 3)):
        frame = with_monsters(("pig_0", 300 + dx, 420 + dy, False))
        again = detector.refind(frame, box_of(first[0]), "pig", first[0].template, first[0].facing)
        whole = detector.detect(frame)
        assert again is not None and (again.kind, again.template) == ("pig", first[0].template)
        assert (again.x, again.y) == (whole[0].x, whole[0].y)
        assert abs(again.x - first[0].x - dx) <= 1 and abs(again.y - first[0].y - dy) <= 1


def test_a_monster_that_turned_is_refound_with_its_kinds_other_templates(tmp_path):
    # The picture that matched it faced left; it turned: the last template doesn't fit any more, and the
    # kind's templates are tried in the window (they find it facing right).
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    Image.fromarray(rgba).save(tmp_path / "pig_move9.png")
    (tmp_path / INDEX).write_text('{"pig_move9.png": {"origin": [20, 60], "w": 104, "h": 64, "frame": "move/9"}}',
                                  encoding="utf-8")
    small = np.asarray(Image.fromarray(rgba).resize((52, 32)))
    detector = MobDetector(tmp_path, threads=1)
    frame = np.full((300, 400, 3), SKY, np.uint8)
    frame[200:] = GRASS
    paste(frame, small, 150, 168)
    seen = detector.detect(frame)[0]
    assert seen.facing == -1
    turned = np.full((300, 400, 3), SKY, np.uint8)
    turned[200:] = GRASS
    paste(turned, small[:, ::-1], 154, 168)
    again = detector.refind(turned, box_of(seen), "pig", seen.template, seen.facing)
    assert again is not None and again.facing == 1 and abs(again.x - (154 + 26)) <= 1
    assert again.foot_x == pytest.approx(154 + 52 - 10, abs=1)  # its feet, in the frame's px, not the window's


def test_a_monster_gone_from_its_window_isnt_refound():
    detector = MobDetector(threads=1)
    seen = detector.detect(with_monsters(("pig_0", 300, 420, False)))[0]
    assert detector.refind(with_monsters(("pig_0", 600, 420, False)), box_of(seen), "pig", seen.template) is None
    assert detector.refind(with_monsters(), box_of(seen), "pig", seen.template) is None
    assert detector.refind(with_monsters(("pig_0", 300, 420, False)), box_of(seen), "slime") is None


def test_a_look_at_a_capture_is_a_look():
    # look() is look_at(its capture): the feed hands it frames it captured itself.
    rng = np.random.default_rng(1)
    bgra = rng.integers(0, 256, (1080, 1920, 4), np.uint8)
    eyes = bench.SavedEyes(np.zeros((10, 36), np.uint8))
    eyes.show(bgra)
    a, b = eyes.look(), eyes.look_at(rgb_of(bgra), t=5.0, ropes=False)
    assert b.t == 5.0 and b.ropes == [] and a.ropes == []  # want_ropes is off without Senses
    for name in ("tag", "hp", "mp", "ground", "dot"):
        assert getattr(a, name) == getattr(b, name), name
    assert np.array_equal(a.frame, b.frame) and np.array_equal(a.exp, b.exp)


def test_window_cover_from_the_integral_image_is_the_tm_ccorr_sum():
    frame = with_monsters(("pig_0", 100, 420, False), ("slime_4", 400, 300, True))
    detector = MobDetector(threads=1)
    for kind, templates in detector.templates.items():
        colour = mobs.palette(kind, frame).astype(np.float32)
        coarse = cv2.resize(colour, (480, 270), interpolation=cv2.INTER_AREA)
        cover = mobs.Cover(coarse)
        for t in templates:
            h, w = t.silhouette.shape
            ccorr = cv2.matchTemplate(coarse, np.ones((h, w), np.float32), cv2.TM_CCORR)
            assert cover.sums(h, w).shape == ccorr.shape
            assert np.abs(cover.sums(h, w) - ccorr).max() / (h * w) < 1e-4


def loop_hp_bar(frame, x, top):
    """hud.hp_bar as it was: the longest run found pixel by pixel in Python."""
    y0, x0 = max(0, top - hud.BAR_ABOVE), max(0, x - hud.BAR_SEARCH)
    box = frame[y0:max(y0, top + 2), x0:x + hud.BAR_SEARCH].astype(np.int16)
    if box.size == 0:
        return None
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    green = (g > 120) & (r < 15) & (b < 15)
    inside = (b < 15) & ((r < 15) | ((r > 90) & (g < 50)))
    best = None
    for y in range(inside.shape[0]):
        run, start = 0, 0
        for i, on in enumerate(inside[y]):
            run = run + 1 if on else 0
            if run == 1:
                start = i
            if run >= hud.BAR_MIN_RUN and (best is None or run > best[0]):
                best = (run, y, start)
    if best is None:
        return None
    run, y, start = best
    return float(green[y, start:start + run].sum()) / run


def test_monster_hp_bars_read_as_before():
    rng = np.random.default_rng(0)
    colours = np.array([(0, 196, 0), (0, 130, 0), (0, 0, 0), (140, 0, 0), SKY, GRASS, (165, 165, 165)], np.uint8)
    seen = 0
    for _ in range(300):
        frame = np.full((120, 160, 3), SKY, np.uint8)
        for _ in range(rng.integers(1, 6)):  # bars and runs of bar colours, some of them as long as each other
            y, x0 = rng.integers(0, 120), rng.integers(0, 150)
            frame[y:y + rng.integers(1, 5), x0:x0 + rng.choice([20, 22, 25, 25, 30, 36, 60])] = colours[rng.integers(0, 4)]
        frame[rng.random((120, 160)) < 0.05] = colours[rng.integers(0, 7)]  # specks
        for x, top in [(rng.integers(-10, 170), rng.integers(-10, 130)) for _ in range(4)] + [(80, 60), (5, 3)]:
            old = loop_hp_bar(frame, x, top)
            assert hud.hp_bar(frame, x, top) == old
            seen += old is not None
    assert seen > 200
