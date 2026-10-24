"""`bench-paste` (analysis/pasted.py): the game's own pictures drawn onto frames of the game at its own 1366x768 as the
game draws them, at known places, and what the monster and drop searches make of them there and on the same scenes
stretched to 1920x1080. The frames are tests/fixtures/native (West Rocky Mountain II, none of whose monsters has a
colour gate yet)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.__main__ import main
from gamelab.analysis import pasted
from gamelab.perception.loot import LootDetector
from gamelab.perception.mobs import ASSETS, INDEX, SCALE_FILE, MobDetector
from gamelab.screen import HUD_TOP, NATIVE, STRETCHED

FIXTURES = Path(__file__).parent / "fixtures" / "native"
FRAMES = ("n28_000.png", "n28_008.png", "native_first.png")
PIG_ORIGIN = (60, 62)  # where a pig cut-out's feet are taken to be, in its own px, for the made-up "game sprites"


def frame(name: str = "n28_000.png") -> np.ndarray:
    return np.array(Image.open(FIXTURES / name).convert("RGB"))


def game_folder(tmp_path: Path, pictures: dict[str, np.ndarray], origin: tuple[int, int] | None = PIG_ORIGIN) -> Path:
    """A folder of made-up game pictures at the game's own size (scale.json 1.0, as sprites.build and loot.build make
    at stretch 1), each standing on `origin` (None: no INDEX, as a folder of drops)."""
    for name, rgba in pictures.items():
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"{name}.png")
    if origin is not None:
        index = {f"{name}.png": {"origin": list(origin), "w": rgba.shape[1], "h": rgba.shape[0], "frame": "move/0"}
                 for name, rgba in pictures.items()}
        (tmp_path / INDEX).write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / SCALE_FILE).write_text(json.dumps({"px_per_game_px": 1.0}), encoding="utf-8")
    return tmp_path


def pig() -> np.ndarray:
    return np.asarray(Image.open(ASSETS / "pig_0.png").convert("RGBA"))


def test_a_picture_is_drawn_as_the_game_draws_it():
    scene = np.full((10, 10, 3), 100, np.uint8)
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., :3] = 200
    rgba[0, :, 3] = (0, 128, 255)
    rgba[1, :, 3] = 255
    pasted.paste(scene, rgba, 4, 5)
    assert scene[5, 4:7, 0].tolist() == [100, 150, 200] and scene[6, 4:7, 0].tolist() == [200] * 3
    assert scene[4].tolist() == [[100] * 3] * 10


def test_pictures_drawn_on_a_frame_keep_apart_and_off_the_minimap_and_the_bottom_bar():
    rgba = np.full((40, 50, 4), 255, np.uint8)
    todo = [("pig", f"pig_{i}", rgba, (25, 40), i % 2 == 1, None) for i in range(12)]
    made = pasted.scenes([frame()], todo, np.random.default_rng(3))
    assert len(made) == 2 and all(len(drawn) == pasted.PER_SCENE for _, _, drawn in made)
    for _, _, drawn in made:
        for a in drawn:
            assert a.top >= pasted.TOP and a.top + a.h <= int(768 * HUD_TOP)
            assert a.left >= 213 or a.top >= 213  # the minimap's box (0, 68)-(213, 213)
            assert a.feet == (a.left + (50 - 25 if a.facing == 1 else 25), a.top + 40)
            for b in drawn:
                if a is not b:
                    assert (a.left + a.w + pasted.GAP <= b.left or b.left + b.w + pasted.GAP <= a.left
                            or a.top + a.h + pasted.GAP <= b.top or b.top + b.h + pasted.GAP <= a.top)


def test_monsters_drawn_are_found_where_they_stand_at_the_games_own_size_and_stretched(tmp_path):
    folder = game_folder(tmp_path, {"pig_move0": pig()})
    native = MobDetector(folder, frame_scale=NATIVE.frame_scale)
    stretched = MobDetector(folder, frame_scale=STRETCHED.frame_scale)  # the picture stretched, as the window did
    result = pasted.monsters([frame(n) for n in FRAMES], {"native": (native, NATIVE), "stretched": (stretched,
                             STRETCHED)}, folder, (140, 75), each=3, seed=2)
    t = result["tally"]["native"]["pig"]
    # One is drawn across two of the whole screen's bands (rows 382-446; the bands 256-427 and 384-555 overlap by 43
    # rows, a pig is 64 tall: the same at 1920x1080, 30 and 32 half-size rows), and found by the band around it only.
    assert (t.drawn, t.found, t.found_band, t.as_drawn, t.extra) == (6, 5, 6, 5, 0) and t.missed == ["pig_move0>"]
    assert max(abs(e) for err in t.errors for e in err) == 0 and min(t.scores) == 1.0
    assert result["clean"]["native"] == []
    s = result["tally"]["stretched"]["pig"]
    assert s.drawn == 6 and s.found >= 5 and max(abs(e) for err in s.errors for e in err) <= 3
    assert pasted.lines(result)[1].split()[:5] == ["pig", "5", "of", "6", "found,"]
    assert "5 as the frame and facing drawn (of 5 found by the game's sprites)" in pasted.lines(result)[1]


def test_a_look_frames_px_are_put_in_the_game_by_their_middles():
    # At 1920x1080 half size a frame px is 1.42 game px: its middle, at x + 0.5, is (x + 0.5) * 1.42 game px from the
    # picture's left edge, so px x's middle is game px (x + 0.5) * 1.42 - 0.5. At the game's own size, the px itself.
    upp = STRETCHED.units_per_px
    assert pasted._game(0, upp) == pytest.approx(0.5 * upp - 0.5) and pasted._game(10, 1.0) == 10
    assert pasted._game(10, upp) == pytest.approx(STRETCHED.to_game(10 * 2 + 0.5))


def test_a_monster_found_by_a_cut_out_isnt_counted_against_its_frame(tmp_path):
    # A screenshot cut-out (named <kind>_<number>, as pig_0: at 1920x1080) says no frame: counted apart, not as found
    # as another frame than the one drawn. Here the "cut-out" is the very picture drawn.
    (tmp_path / "drawn").mkdir()
    (tmp_path / "cutouts").mkdir()
    drawn = game_folder(tmp_path / "drawn", {"pig_move0": pig()})
    detector = MobDetector(game_folder(tmp_path / "cutouts", {"pig_7": pig()}), frame_scale=NATIVE.frame_scale)
    result = pasted.monsters([frame()], {"native": (detector, NATIVE)}, drawn, (140, 75), each=2, seed=2)
    t = result["tally"]["native"]["pig"]
    assert t.found == t.by_cutout >= 3 and t.as_drawn == 0
    said = f"0 as the frame and facing drawn (of 0 found by the game's sprites; {t.found} by a cut-out, which says no"
    assert said + " frame)" in pasted.lines(result)[1]


def test_drops_drawn_are_found_in_the_strip_above_where_they_lie(tmp_path):
    coin = np.zeros((24, 24, 4), np.uint8)
    yy, xx = np.mgrid[:24, :24]
    disc = (yy - 11.5) ** 2 + (xx - 11.5) ** 2 <= 121
    coin[disc] = (205, 140, 60, 255)
    coin[disc & ((yy - 9) ** 2 + (xx - 9) ** 2 <= 16)] = (250, 220, 150, 255)  # its shine
    folder = game_folder(tmp_path, {"mesos_bronze0": coin}, origin=None)
    result = pasted.drops([frame(n) for n in FRAMES], {"native": (LootDetector(folder, NATIVE), NATIVE)}, folder,
                          each=4, seed=1)
    t = result["tally"]["native"]["mesos"]
    assert (t.drawn, t.found, t.as_drawn, t.extra) == (4, 4, 4, 0) and min(t.scores) == 1.0
    assert result["clean"]["native"] == []


@pytest.fixture(scope="module")
def rocky():
    from gamedata.paths import COPY
    from gamelab.localization import mapdata
    if not (COPY / "Item").is_dir():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    try:
        return mapdata.load(10004021)
    except (OSError, KeyError):
        pytest.skip("needs West Rocky Mountain II's game data")


def test_bench_paste_says_when_none_of_a_maps_monsters_has_a_colour_gate(rocky, capsys):
    assert main(["bench-paste", "--map", "10004021", "--each", "1"]) == 0
    out = capsys.readouterr().out
    assert "monsters: none of map 10004021's has a colour gate yet" in out
    assert "drops:" in out and "1366x768, full size" in out


def test_refinds_follow_a_monster_into_its_next_frame(tmp_path):
    right = pig().copy()
    nxt = right.copy()
    nxt[..., :3] = np.roll(right[..., :3], 4, axis=1)  # its next frame: its colours 4 px across in the same outline
    folder = game_folder(tmp_path, {"pig_move0": right, "pig_move1": nxt})
    index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
    index["pig_move1.png"]["frame"] = "move/1"
    (folder / INDEX).write_text(json.dumps(index), encoding="utf-8")
    detector = MobDetector(folder, frame_scale=NATIVE.frame_scale)
    result = pasted.refinds([frame(n) for n in FRAMES], {"native": (detector, NATIVE)}, folder, (140, 75), seed=3)
    for step in ("unchanged", "its pose's next frame"):  # (each of the two frames, both ways)
        r = result["native"][step]
        assert (r.n, r.found, r.feet, r.as_drawn, r.hidden, r.fallbacks) == (4, 4, 4, 4, 4, 0), step
    assert result["native"]["a flinch"].n == 0  # (it has none)
    lines = pasted.refind_lines(result)
    assert lines[0] == "native:" and lines[1].split()[:6] == ["unchanged", "4", "of", "4", "found", "again"]
