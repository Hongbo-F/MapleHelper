"""Monsters and drops at the game's own 1366x768: the game's own pictures as they are (a monster's opaque pixels
matched, no cut-outs; a drop's shape on brightness, its colours checked), and the monster search coarse to fine (step
1 at the 1920x1080 window's half size, step 2 at full size in small windows, a gate first and several templates a
place). The frames are tests/fixtures/native (West Rocky Mountain II, with none of the first map's monsters on them),
with pictures drawn on them as the game draws them (analysis.pasted)."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.analysis import pasted
from gamelab.perception import mobs, sprites
from gamelab.perception.eyes import Observation, band_rows
from gamelab.perception.loot import LootDetector
from gamelab.perception.mobs import INDEX, SCALE_FILE, MobDetector
from gamelab.screen import HUD_TOP, NATIVE, STRETCHED

from test_pasted import FRAMES, PIG_ORIGIN, frame, game_folder, pig


# --- the pictures as they are ------------------------------------------------------------------------------------

def test_a_picture_at_its_own_size_is_used_as_it_is_and_only_its_opaque_pixels_are_matched():
    rgba = np.zeros((4, 5, 4), np.uint8)
    rgba[..., :3] = np.arange(60).reshape(4, 5, 3)
    rgba[..., 3] = [[0, 100, 128, 200, 255]] * 4  # a see-through edge, as a slime's
    rgb, mask, drawn = mobs._picture(rgba, 1.0)
    assert np.array_equal(rgb, rgba[..., :3])
    assert mask[0].tolist() == [0, 0, 0, 0, 1]   # step 2: the pixels the game draws as they are
    assert drawn[0].tolist() == [0, 0, 1, 1, 1]  # step 1's silhouette: where it's drawn


def test_a_shrunk_picture_is_as_before():
    rgba = pig()
    rgb, mask, drawn = mobs._picture(rgba, 2.0)
    size = (rgba.shape[1] // 2, rgba.shape[0] // 2)
    assert np.array_equal(rgb, cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_AREA))
    assert np.array_equal(mask, cv2.resize((rgba[..., 3] > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST))
    assert mask is drawn


def test_at_1920x1080_half_size_step_1_is_the_look_frame():
    detector = MobDetector(threads=1)
    assert detector.search == 1 and detector.spx == detector.px == 1
    assert all(t.step1 is None for ts in detector.templates.values() for t in ts)


def test_at_1366x768_full_size_step_1_is_the_1920x1080_windows_half_size(tmp_path):
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    assert detector.search == pytest.approx(1366 / 960.5, abs=0.002) and detector.spx == pytest.approx(1)
    t = detector.templates["pig"][0]
    assert t.rgb.shape[:2] == pig().shape[:2]  # the picture as it is
    assert t.step1[0].shape[:2] == (45, 73)    # 64 x 104 at 1/1.42
    assert t.silhouette.shape == (45 // 2, 73 // 2)


# --- the search, coarse to fine ----------------------------------------------------------------------------------

def test_places_hold_every_template_spotted_there_best_first():
    spots = [(0.9, 100, 50, 3), (0.85, 102, 51, 1), (0.8, 300, 50, 2), (0.7, 104, 49, 3), (0.6, 101, 52, 0),
             (0.5, 300, 52, 1)]
    assert mobs._places(spots, 6) == [[(100, 50, 3), (102, 51, 1), (101, 52, 0)], [(300, 50, 2), (300, 52, 1)]]
    first = [place[0] for place in mobs._places(spots, 6)]
    assert first == mobs._to_check(spots, 6)  # each place's first is the spot checked at 1920x1080
    many = [(1 - i / 100, 20 * i, 0, i % 3) for i in range(30)]
    assert len(mobs._places(many, 6)) == mobs.MAX_CHECKS


def test_a_spot_goes_to_the_finer_frame_by_its_windows_corner():
    t = SimpleNamespace(silhouette=np.zeros((11, 18)), rgb=np.zeros((32, 52, 3)))
    cx, cy = 100 + 18, 40 + 11  # a window's top-left (100, 40) plus half the template, at step 1's scale
    assert mobs._finer(t, cx, cy, 1.0, 1.0) == (100 + 26, 40 + 16)
    assert mobs._finer(t, cx, cy, 1.5, 2.0) == (150 + 26, 80 + 16)


def test_a_monster_drawn_at_the_games_own_size_is_found_where_it_stands(tmp_path):
    folder = game_folder(tmp_path, {"pig_move0": pig()})
    coarse = MobDetector(folder, threads=1, frame_scale=NATIVE.frame_scale)
    full = MobDetector(folder, threads=1, frame_scale=NATIVE.frame_scale, search_scale=1.0)
    # (each wholly inside one of the whole screen's bands: 0-171, 128-299, 256-427, 384-555, 512-675)
    for left, top, flip in ((300, 300, False), (900, 150, True), (1101, 520, False)):
        scene = frame()
        pasted.paste(scene, pig()[:, ::-1] if flip else pig(), left, top)
        feet_x = left + (pig().shape[1] - PIG_ORIGIN[0] if flip else PIG_ORIGIN[0])
        for detector in (coarse, full):
            found = detector.detect_wide(scene, (0, int(768 * HUD_TOP)))
            assert [(m.kind, m.foot_x, m.feet_y, m.facing, m.score) for m in found] == [
                ("pig", feet_x, top + PIG_ORIGIN[1], 1 if flip else -1, 1.0)], (left, top, flip)
            band = detector.detect(scene, (top - 60, top + 120))
            assert [(m.foot_x, m.feet_y) for m in band] == [(feet_x, top + PIG_ORIGIN[1])]


def test_monsters_are_looked_for_down_to_the_hud_at_the_games_own_size(tmp_path):
    # The chat box's top is row 700 at 1366x768, where HUD_TOP (0.88) is row 675; at 1920x1080 it's as it was.
    assert int(768 * NATIVE.hud_top) == 700 and int(540 * STRETCHED.hud_top) == int(540 * HUD_TOP) == 475
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    scene = frame()
    pasted.paste(scene, pig(), 600, 619)  # its picture down to row 682 (whole in the band from row 512)
    assert [m.feet_y for m in detector.detect_wide(scene, (0, int(768 * NATIVE.hud_top)))] == [619 + PIG_ORIGIN[1]]
    assert detector.detect_wide(scene, (0, int(768 * HUD_TOP))) == []  # where the search stopped before
    obs = Observation(t=0, frame=scene, tag=(700, 640), hp=1, mp=1, geo=NATIVE)  # her feet at 629
    assert band_rows(obs, 140, 75) == (629 - 140, 700)
    half = Observation(t=0, frame=np.zeros((540, 960, 3), np.uint8), tag=(480, 460), hp=1, mp=1)
    assert band_rows(half, 140, 75) == (452 - 99, 475)  # (1920x1080: 140 units are 99 half-size px)


def test_a_band_with_no_rows_finds_nothing_at_either_size(tmp_path):
    # Her band off the screen's bottom (Eyes.band_rows: its top under the bottom bar): at 1920x1080 half size no colour
    # passed in it; shrunk for step 1 at 1366x768, OpenCV refused the empty picture.
    native = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    stretched = MobDetector(threads=1)
    for detector, scene in ((native, frame()), (stretched, np.zeros((540, 960, 3), np.uint8))):
        for rows in ((675, 675), (700, 675)):
            assert detector.detect(scene, rows) == [] and detector.detect_wide(scene, rows) == []


def test_nothing_is_found_on_frames_of_the_game_at_its_own_size_with_no_monster_of_its_kind(tmp_path):
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), frame_scale=NATIVE.frame_scale)
    for name in FRAMES:
        assert detector.detect_wide(frame(name), (0, int(768 * HUD_TOP))) == [], name


def test_the_frame_checked_first_at_a_place_doesnt_hide_the_monsters_own(tmp_path, monkeypatch):
    # Another frame of the monster drawn: its colours 2 px across inside the same outline (a pose apart),
    # which step 1 ranks first at the place (as it ranked a snail's other frames first in bench-paste). Alone, it's
    # what's found, 2 px off where the monster stands; with the place's other templates, its own frame, exactly.
    right = pig().copy()
    wrong = right.copy()
    wrong[..., :3] = np.roll(right[..., :3], 2, axis=1)
    detector = MobDetector(game_folder(tmp_path, {"pig_right": right, "pig_wrong": wrong}), threads=1,
                           frame_scale=NATIVE.frame_scale)
    names = [t.name for t in detector.templates["pig"]]
    real = mobs._places
    monkeypatch.setattr(mobs, "_places", lambda spots, slack: [sorted(p, key=lambda s: names[s[2]] != "pig_wrong")
                                                               for p in real(spots, slack)])
    scene = frame()
    pasted.paste(scene, right, 400, 300)
    monkeypatch.setattr(mobs, "FINE_TEMPLATES", 1)
    found = detector.detect(scene, (250, 420))
    assert [(m.template, m.foot_x) for m in found] == [("pig_wrong", 400 + PIG_ORIGIN[0] - 2)] and found[0].score < 1
    monkeypatch.setattr(mobs, "FINE_TEMPLATES", 6)
    found = detector.detect(scene, (250, 420))
    assert [(m.template, m.score, m.foot_x) for m in found] == [("pig_right", 1.0, 400 + PIG_ORIGIN[0])]


def test_a_monster_seen_at_the_games_own_size_is_refound(tmp_path):
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    scene = frame()
    pasted.paste(scene, pig(), 500, 300)
    seen = detector.detect(scene, (250, 420))[0]
    box = (seen.x - seen.w // 2, seen.y - seen.h // 2, seen.x + seen.w // 2, seen.y + seen.h // 2)
    moved = frame()
    pasted.paste(moved, pig(), 512, 303)  # 12 and 3 px on: within REFIND_PAD at this size (23 px)
    again = detector.refind(moved, box, "pig", seen.template, seen.facing)
    assert (again.foot_x - seen.foot_x, again.feet_y - seen.feet_y) == (12, 3)
    assert detector.fallbacks == 0


def test_the_frames_a_monster_most_likely_went_to_come_first():
    t = [SimpleNamespace(name=f"slime_{p}{i}", frame=f"{p}/{i}", facing=f) for p, n in (("move", 7), ("stand", 3),
                                                                                      ("hit1", 1))
         for i in range(n) for f in (1, -1)]
    last = next(x for x in t if x.name == "slime_move6" and x.facing == -1)
    names = [x.name for x in mobs._next_frames(t, last, most=20)]
    assert names[:4] == ["slime_move0", "slime_move1", "slime_hit10", "slime_stand0"]  # (an animation loops)
    assert names[4:] == [f"slime_move{i}" for i in range(2, 6)] and len(names) == 8
    assert all(x.facing == -1 for x in mobs._next_frames(t, last, most=20))
    assert [x.name for x in mobs._next_frames(t, last)] == names[:mobs.REFIND_NEXT]
    assert mobs._next_frames(t, SimpleNamespace(name="slime_1", frame="", facing=0)) == []  # a cut-out: no frame


def test_a_monster_whose_animation_stepped_on_is_refound_as_its_new_frame(tmp_path):
    # At the game's own size a monster's frames differ crisply: here its next frame is its colours 4 px across inside
    # the same outline. The frame it was found as still passes there, 4 px off where it stands (0.80); its next frame,
    # tried as that's under REFIND_GOOD, finds it where it stands. With step 1 at full size (as at 1920x1080 half size,
    # where frames were alike) the frame it was found as is all there is.
    right = pig().copy()
    nxt = right.copy()
    nxt[..., :3] = np.roll(right[..., :3], 4, axis=1)
    folder = game_folder(tmp_path, {"pig_move0": right, "pig_move1": nxt})
    index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
    index["pig_move1.png"]["frame"] = "move/1"
    (folder / INDEX).write_text(json.dumps(index), encoding="utf-8")
    scene, later = frame(), frame()
    pasted.paste(scene, right, 500, 300)
    pasted.paste(later, nxt, 500, 300)
    for search_scale, refound in ((mobs.SEARCH_SCALE, ("pig_move1", 1.0, 560)), (1.0, ("pig_move0", 0.796, 564))):
        detector = MobDetector(folder, threads=1, frame_scale=NATIVE.frame_scale, search_scale=search_scale)
        seen = detector.detect(scene, (250, 420))[0]
        box = (seen.x - seen.w // 2, seen.y - seen.h // 2, seen.x + seen.w // 2, seen.y + seen.h // 2)
        assert (seen.template, seen.foot_x) == ("pig_move0", 500 + PIG_ORIGIN[0])
        for fallback in (True, False):
            m = detector.refind(later, box, "pig", seen.template, seen.facing, fallback=fallback)
            assert (m.template, m.score, m.foot_x) == refound and detector.fallbacks == 0


# --- drops ------------------------------------------------------------------------------------------------------

def test_drops_at_the_games_own_size_are_matched_on_brightness_and_their_colours_checked(tmp_path):
    from test_loot import coin, leather
    folder = game_folder(tmp_path, {"mesos_bronze0": coin(), "leather_0": leather()}, origin=None)
    native = LootDetector(folder, NATIVE)
    assert native.gray and all(t.gray[0].ndim == 2 and t.gray[0].shape == t.rgb.shape[:2] for t in native.templates)
    assert not LootDetector(folder, STRETCHED).gray  # the 1920x1080 window's: in colour, as before
    scene = frame()
    pasted.paste(scene, coin(), 500, 400)
    pasted.paste(scene, leather((60, 120, 50)), 700, 395)  # a hide's shape in green: its brightness, not its colours
    box = (450, 380, 800, 440)
    found = native.detect(scene, box)
    assert [(i.kind, i.x, i.y, i.score) for i in found] == [("mesos", 516, 416, 1.0)]
    assert LootDetector(folder, NATIVE, gray=False).detect(scene, box) == found  # the same in colour
    assert native.detect(frame(), box) == []


# --- which pictures, at which size --------------------------------------------------------------------------------

def test_at_the_games_own_size_the_sprites_made_before_its_alpha_was_kept_are_made_again(tmp_path, monkeypatch):
    rgba = np.zeros((4, 4, 4), np.uint8)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "pig_move0.png")
    (tmp_path / INDEX).write_text(json.dumps({"pig_move0.png": {"origin": [1, 1], "w": 4, "h": 4}}), encoding="utf-8")
    built = []
    monkeypatch.setattr(sprites, "build", lambda names, folder, *stretch: built.append((names, stretch)))
    assert sprites.ensure({8: "Pig"}, tmp_path, stretch=1.0) == tmp_path
    assert built == [({8: "Pig"}, (1.0,))]  # its alpha was cut at OPAQUE: made again with the game's
    entry = {"origin": [1, 1], "w": 4, "h": 4, **sprites.GAME_ALPHA}
    (tmp_path / INDEX).write_text(json.dumps({"pig_move0.png": entry}), encoding="utf-8")
    built.clear()
    sprites.ensure({8: "Pig"}, tmp_path, stretch=1.0)
    assert built == [] and not (tmp_path / mobs.CUTOUTS).exists()  # no cut-outs matched at this size


def test_at_the_games_own_size_the_detector_has_the_games_sprites_alone(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, {"pig_move0": pig()})
    monkeypatch.setattr(sprites, "ensure", lambda names, stretch: folder)
    model = SimpleNamespace(spawns=[SimpleNamespace(mob_id=8, name="Pig")])
    native = sprites.detector_for(model, NATIVE)
    assert {t.name for t in native.templates["pig"]} == {"pig_move0"} and native.search > 1
    stretched = sprites.detector_for(model, STRETCHED)  # the cut-outs too where the window stretches the game
    assert {"pig_0", "pig_move0"} <= {t.name for t in stretched.templates["pig"]}


def test_at_the_games_own_size_the_detector_holds_the_maps_own_kinds_alone(tmp_path, monkeypatch):
    # A map none of whose monsters has a colour gate (here, with no background colours learned for it: the model has
    # no id): on a checkout that never built the first map's sprites the folder is empty, and where it did, it holds
    # them. Either way nothing is looked for there, and the first map's pig is still looked for on its own map.
    stumps = SimpleNamespace(spawns=[SimpleNamespace(mob_id=130100, name="Stump")])
    pigs = SimpleNamespace(spawns=[SimpleNamespace(mob_id=8, name="Pig")])
    empty = tmp_path / "empty"
    empty.mkdir()
    built = game_folder(tmp_path, {"pig_move0": pig()})
    scene = frame()
    pasted.paste(scene, pig(), 500, 300)
    for folder in (empty, built):
        monkeypatch.setattr(sprites, "ensure", lambda names, stretch: folder)
        detector = sprites.detector_for(stumps, NATIVE)
        assert detector.templates == {} and detector.detect_wide(scene, (0, int(768 * HUD_TOP))) == []
        assert detector.detect(scene, (250, 420)) == [] and detector.refind(scene, (500, 300, 600, 360), "pig") is None
    assert set(sprites.detector_for(pigs, NATIVE).templates) == {"pig"}
    with pytest.raises(RuntimeError, match="no monster templates"):  # a folder with none, asked for any kind
        MobDetector(empty, frame_scale=NATIVE.frame_scale)


# --- the game's own pictures (needs the game data copy) ----------------------------------------------------------

def needs_game_data():
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")


def test_sprites_at_the_games_own_size_are_its_pictures_with_their_alpha(tmp_path):
    needs_game_data()
    from gamedata.canvas import mob_frames
    sprites.build({7: "Slime"}, tmp_path, stretch=1.0)
    index = json.loads((tmp_path / INDEX).read_text(encoding="utf-8"))
    assert json.loads((tmp_path / SCALE_FILE).read_text(encoding="utf-8")) == {"px_per_game_px": 1.0}
    for f in mob_frames(7):
        name = f"slime_{f['action']}{f['index']}.png"
        assert np.array_equal(np.asarray(Image.open(tmp_path / name).convert("RGBA")), f["rgba"])
        assert index[name]["alpha"] == "game"
    t = MobDetector(tmp_path, threads=1, frame_scale=1.0).templates["slime"][0]
    alpha = np.asarray(Image.open(tmp_path / f"{t.name}.png"))[..., 3]
    assert t.mask3[..., 0].sum() == (alpha == 255).sum() < (alpha >= 128).sum()  # see-through: opaque pixels only


@pytest.fixture(scope="module")
def first_map():
    from gamelab.localization import mapdata
    try:
        return mapdata.load(10000012)
    except (OSError, KeyError):
        pytest.skip("needs Thicket Around the Beach II's game data")


def test_the_first_maps_sprites_drawn_at_the_games_own_size_are_found(first_map):
    needs_game_data()
    from gamelab.perception import loot
    detector = sprites.detector_for(first_map, NATIVE)
    backgrounds = [frame(n) for n in FRAMES]
    result = pasted.monsters(backgrounds, {"native": (detector, NATIVE)}, sprites.folder_for(1.0), (140, 75), 1, 0)
    tallies = result["tally"]["native"]
    assert set(tallies) == {"orange-mushroom", "pig", "slime", "snail"}
    assert sum(t.found for t in tallies.values()) >= 0.9 * sum(t.drawn for t in tallies.values())
    assert all(t.extra == 0 for t in tallies.values()) and result["clean"]["native"] == []
    errors = [e for t in tallies.values() for err in t.errors for e in err]
    assert sum(abs(e) > 1 for e in errors) <= 2  # where they stand, to the px but for a frame or two
    drops = pasted.drops(backgrounds, {"native": (loot.LootDetector(geo=NATIVE), NATIVE)}, loot.folder_for(1.0), 1, 1)
    found = drops["tally"]["native"]
    assert all(t.found == t.drawn and min(t.scores) == 1.0 for t in found.values()), found
    assert drops["clean"]["native"] == []
