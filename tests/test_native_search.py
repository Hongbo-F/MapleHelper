"""Monsters at the game's own 1366x768: the game's own pictures as they are (opaque pixels matched, no cut-outs).
The frames are tests/fixtures/native (West Rocky Mountain II, none of whose monsters has a colour gate yet), with
pictures drawn on them as the game draws them (analysis.pasted)."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.analysis import pasted
from gamelab.perception import mobs, sprites
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


# --- the search -------------------------------------------------------------------------------------------------------

def test_a_monster_drawn_at_the_games_own_size_is_found_where_it_stands(tmp_path):
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    # (each wholly inside one of the whole screen's bands: 0-171, 128-299, 256-427, 384-555, 512-675)
    for left, top, flip in ((300, 300, False), (900, 150, True), (1101, 520, False)):
        scene = frame()
        pasted.paste(scene, pig()[:, ::-1] if flip else pig(), left, top)
        feet_x = left + (pig().shape[1] - PIG_ORIGIN[0] if flip else PIG_ORIGIN[0])
        found = detector.detect_wide(scene, (0, int(768 * HUD_TOP)))
        assert [(m.kind, m.foot_x, m.feet_y, m.facing, m.score) for m in found] == [
            ("pig", feet_x, top + PIG_ORIGIN[1], 1 if flip else -1, 1.0)], (left, top, flip)
        band = detector.detect(scene, (top - 60, top + 120))
        assert [(m.foot_x, m.feet_y) for m in band] == [(feet_x, top + PIG_ORIGIN[1])]


def test_nothing_is_found_on_frames_of_the_game_at_its_own_size_with_no_monster_of_its_kind(tmp_path):
    detector = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), frame_scale=NATIVE.frame_scale)
    for name in FRAMES:
        assert detector.detect_wide(frame(name), (0, int(768 * HUD_TOP))) == [], name


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
    assert {t.name for t in native.templates["pig"]} == {"pig_move0"}
    stretched = sprites.detector_for(model, STRETCHED)  # the cut-outs too where the window stretches the game
    assert {"pig_0", "pig_move0"} <= {t.name for t in stretched.templates["pig"]}


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
