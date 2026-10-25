"""The stack at the game's own 1366x768 (since 2026-10-23): what a look reads off frames of that size, drawn pixel
for pixel. The frames are tests/fixtures/native (West Rocky Mountain II, her on stair 28; see its README)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamedata.paths import COPY
from gamelab import screen
from gamelab.perception import exp_number
from gamelab.perception.hud import bars
from gamelab.perception.minimap import her_dot, markers_in_box, minimap_box, other_dots
from gamelab.perception.nametag import NameTag
from gamelab.screen import NATIVE

FIXTURES = Path(__file__).parent / "fixtures" / "native"
TAG = FIXTURES / "tag_template_1366x768.npy"
# Each fixture: the other players' markers on it (kind, core's middle), by eye on the enlarged minimap.
FRAMES = {
    "n28_000.png": [("stranger", 96.0, 115.0), ("stranger", 84.0, 153.0)],
    "n28_008.png": [("stranger", 118.0, 133.0), ("stranger", 84.0, 153.0), ("stranger", 85.0, 161.0)],
    "native_first.png": [("stranger", 146.0, 120.0), ("stranger", 126.0, 144.0), ("stranger", 62.0, 154.0)],
}
HER_DOT = (94.5, 172.5)   # her yellow core, 6 px across, its middle
HER_TAG = (685, 528)      # her name tag's middle
MAP = 10004021


def frame(name: str) -> np.ndarray:
    return np.asarray(Image.open(FIXTURES / name).convert("RGB"))


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_her_dot_and_the_other_players_markers(name):
    full = frame(name)
    assert screen.of_frame(full) == NATIVE
    assert her_dot(full) == HER_DOT
    # The two portals' rings (cyan, a buddy's colour) aren't a buddy: their inner ring, 4 px with a hole, isn't a core
    assert [(m.kind, m.x, m.y) for m in other_dots(full)] == FRAMES[name]


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_hp_mp_and_her_exp_number(name):
    full = frame(name)
    assert bars(full) == (1.0, 1.0)  # 730/730 and 584/584
    x0, y0, x1, y1 = exp_number.read_box(NATIVE)
    r = exp_number.read(full[y0:y1, x0:x1])
    assert (r.exp, r.percent, r.level) == (31_736, 92.47, 23), r
    assert r.fit < 1.0 and r.lead > 20.0  # every glyph its picture, pixel for pixel


@pytest.fixture(scope="module")
def ui_data():
    if not (COPY / "UI" / "UI_000.wz").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")


@pytest.mark.parametrize("value", [(0, 1), (421, 19), (13_549, 20), (31_736, 23), (888_888, 60), (24_401, 21)])
def test_the_readers_own_drawing_reads_back_at_the_games_size(ui_data, value):
    exp, level = value
    drawn = exp_number.draw(exp_number.shown(exp, level), geo=NATIVE)
    r = exp_number.read(drawn)
    assert r.exp == exp and level in r.levels and r.fit < 0.6


def test_a_number_one_game_px_off_its_place_isnt_read(ui_data):
    full = frame("n28_000.png")
    x0, y0, x1, y1 = exp_number.read_box(NATIVE)
    shifted = np.roll(full, 1, axis=1)[y0:y1, x0:x1]
    assert exp_number.read(shifted).exp is None


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_her_name_tag_is_found_with_the_template_made_at_this_size(tmp_path, name):
    tag = NameTag(np.load(TAG), TAG, NATIVE)
    full = frame(name)
    assert tag.find(full) == HER_TAG
    assert tag.find(full) == HER_TAG and tag.missed == 0  # the second time near where she was


def test_eyes_at_this_size_look_at_full_size(monkeypatch):
    from gamelab.analysis import bench
    from gamelab.perception import eyes as eyes_module
    eyes = bench.SavedEyes(np.load(TAG), detector=object(), geo=NATIVE)
    monkeypatch.setattr(eyes.name_tag, "full", np.load(TAG))
    eyes.show(cv2.cvtColor(frame("n28_000.png"), cv2.COLOR_RGB2BGRA))
    obs = eyes.look()
    assert obs.geo == NATIVE and obs.frame.shape == (768, 1366, 3)
    assert obs.tag == HER_TAG and obs.floor_y == HER_TAG[1] - eyes_module.tag_below_feet(NATIVE) == 517
    assert obs.dot == HER_DOT and len(obs.others) == 2 and (obs.hp, obs.mp) == (1.0, 1.0)
    assert obs.exp_box.shape == (12, 99, 3) and exp_number.read(obs.exp_box).exp == 31_736
    with pytest.raises(ValueError, match="1920x1080"):
        bench.SavedEyes(np.load(TAG), detector=object()).look_at(frame("n28_000.png"))


@pytest.fixture(scope="module")
def rocky():
    from gamelab.localization import mapdata
    try:
        return mapdata.load(MAP)
    except (OSError, KeyError):
        pytest.skip("needs West Rocky Mountain II's game data")


def _picture_eyes(rocky, monkeypatch, picture: bool = True):
    """bench's saved-frame eyes on n28_000.png, with this map's minimap picture to find (as `_eyes` makes them in game:
    mapdata.minimap_picture) or without."""
    from gamelab.analysis import bench
    from gamelab.localization import mapdata
    from gamelab.perception.minimap import MinimapPicture
    eyes = bench.SavedEyes(np.load(TAG), detector=object(), geo=NATIVE)
    monkeypatch.setattr(eyes.name_tag, "full", np.load(TAG))
    if picture:
        if mapdata.minimap_picture(MAP) is None:
            pytest.skip("needs the game data copy (gamedata/data/copy)")
        eyes.minimap = MinimapPicture(mapdata.minimap_picture(MAP), NATIVE)
    eyes.show(cv2.cvtColor(frame("n28_000.png"), cv2.COLOR_RGB2BGRA))
    return eyes


def test_her_place_at_this_size(rocky, monkeypatch):
    """Her dot alone puts her 88 units left of stair 28 at 1366x768, as at 1920x1080 on this map (80 there, on step
    24): the minimap's picture starts 5 game px further left than on the first map, where DOT_ORIGIN was fitted. A look
    finds the picture (minimap.MinimapPicture: its corner at game px (42, 72)) and reads her dot as if it were where the
    first map's was: she's on stair 28, and the two ropes in view (at x -536 and -303, seen at screen x 941 and 1174,
    her tag at 685) put her at -792."""
    from gamelab.localization.senses import Senses
    eyes = _picture_eyes(rocky, monkeypatch)
    senses = Senses(eyes, rocky, band=(100, 100))
    s = senses.look(mobs=False)
    assert s.geo == NATIVE and senses.localizer.per_px == 16.0 and eyes.minimap.corner == (42, 72)
    assert s.where.platform.id == 28 and s.how == "rope" and s.where.x == pytest.approx(-792, abs=1)
    assert s.where.y == 2685
    assert [(r.x, r.top) for r in s.ropes] == [(941, 0), (1174, 71)]
    # Without the picture, her dot alone: 88 left, on no platform.
    eyes = _picture_eyes(rocky, monkeypatch, picture=False)
    s = Senses(eyes, rocky, band=(100, 100)).look(mobs=False)
    assert s.where.x == pytest.approx(-880.4, abs=0.1) and s.where.platform is None and s.how == "dot"


@pytest.mark.parametrize("picture", [True, False])
def test_the_map_command_places_her_dot_at_this_size(rocky, monkeypatch, capsys, tmp_path, picture):
    """`map` (not --offline) places her dot as Senses does, through the minimap's picture: with 1920x1080's Localizer
    it put her at (-1319, 1897), and at this size without the picture at (-880, 2697), on no platform. (Its state is
    the perception service's newest: here the step on the eyes' frame, feed.InlineFeed.)"""
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    from gamelab.localization import mapdata
    from gamelab.localization.feed import InlineFeed
    eyes = _picture_eyes(rocky, monkeypatch, picture)
    monkeypatch.setattr(cli, "_perception", lambda model, title, name, acts, throws=False: InlineFeed(
        eyes, model, (100, 100)))
    monkeypatch.setattr(cli, "_game_title", lambda: "MapleStory")
    monkeypatch.setattr(cli.client, "check_alarm", lambda: None)
    monkeypatch.setattr(cli, "LAB", tmp_path)
    drawn = []
    monkeypatch.setattr(mapdata, "render", lambda model, path, marks=(): drawn.extend(marks))
    assert cli.cmd_map(SimpleNamespace(map=MAP, offline=False)) == 0
    there = (-800, 2685) if picture else (-880, 2697)
    on = "platform 28" if picture else "air/rope"
    assert f"she's at map ({there[0]}, {there[1]}) on {on}" in capsys.readouterr().out
    assert [(round(x), round(y), label) for x, y, label in drawn] == [(*there, "her")]


# --- the game's own markers pasted at this size, pixel for pixel -------------------------------------------------

@pytest.fixture(scope="module")
def markers():
    if not (COPY / "Map" / "_Canvas").is_dir():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    from gamedata.canvas import minimap_markers
    return minimap_markers()


ORIGINS = {"another": (5, 5), "friend": (5, 5), "party": (5, 5), "guild": (5, 5), "partymaster": (5, 9),
           "guildmaster": (5, 9), "user": (6, 6), "npc": (5, 6), "portal": (7, 7)}  # (tests/test_minimap.py's)
KIND_OF = {"another": "stranger", "party": "party", "partymaster": "party", "guild": "guild",
           "guildmaster": "guild", "friend": "buddy"}
# Spots on this map's minimap picture (screen px = game px) clear of her dot, the portals and the markers there
SPOTS = [(55, 80), (70, 100), (125, 80), (140, 100), (60, 130), (130, 125), (140, 160), (60, 190), (120, 195),
         (115, 150), (75, 180), (50, 105), (145, 190), (110, 100)]


def paste(box: np.ndarray, rgba: np.ndarray, origin: tuple[int, int], gx: int, gy: int, y0: int) -> np.ndarray:
    """A marker drawn as the game draws it at 1366x768: its picture as it is, its origin at game px (gx, gy), over a
    crop of the minimap box whose top is at screen row y0."""
    out = box.astype(np.float32)
    h, w = rgba.shape[:2]
    x, y = gx - origin[0], gy - origin[1] - y0
    a = rgba[..., 3:].astype(np.float32) / 255
    out[y:y + h, x:x + w] = out[y:y + h, x:x + w] * (1 - a) + rgba[..., :3] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def test_each_kind_of_marker_is_found_where_it_is_drawn_and_nothing_else_is(markers):
    x0, y0, x1, y1 = minimap_box(NATIVE)
    box = frame("n28_000.png")[y0:y1, x0:x1]
    before = markers_in_box(box, NATIVE)
    for sprite, kind in KIND_OF.items():
        for gx, gy in SPOTS:
            found = [m for m in markers_in_box(paste(box, markers[sprite][0], ORIGINS[sprite], gx, gy, y0), NATIVE)
                     if m not in before]
            assert len(found) == 1 and found[0].kind == kind, (sprite, gx, gy, found)
            assert abs(found[0].x - gx) <= 0.2 and abs(found[0].y + y0 - gy) <= 0.2, (sprite, gx, gy, found)
    for sprite in sorted(set(markers) - set(KIND_OF) - {"user"}):  # NPCs, portals, shops, runes, the compass
        for gx, gy in SPOTS:
            drawn = paste(box, markers[sprite][0], ORIGINS.get(sprite, (5, 5)), gx, gy, y0)
            assert markers_in_box(drawn, NATIVE) == before, (sprite, gx, gy)


def test_calibrate_at_this_size_says_the_pictures_it_wrote(monkeypatch, capsys, tmp_path):
    """`calibrate` at 1366x768 writes calibrate_crop_1366x768.png and says so (it said calibrate_crop.png, 1920x1080's,
    which lies in the lab from before, correct-looking); not finding her tag, it names the frame it saved."""
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    from gamelab import paths
    from gamelab.analysis import calibration
    full = frame("n28_000.png")
    monkeypatch.setattr(calibration, "LAB", tmp_path)
    for name in ("CALIBRATION", "TAG_TEMPLATE", "TAG_FULL"):
        monkeypatch.setattr(paths, name, tmp_path / getattr(paths, name).name)
    monkeypatch.setattr(calibration.client, "submit", lambda **kw: {"frames": [full],
                                                                     "client": {"width": 1366, "height": 768}})
    monkeypatch.setattr(calibration.recordings, "screenshot",
                        lambda name, crop, scale: full[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]])
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title: True)
    monkeypatch.setattr(cli, "_game_title", lambda: "MapleStory")
    assert cli.cmd_calibrate(SimpleNamespace(tag=f"{HER_TAG[0]},{HER_TAG[1]}", fresh=False)) == 0
    out = capsys.readouterr().out
    crop = tmp_path / "calibrate_crop_1366x768.png"
    assert f"check {crop}:" in out and crop.is_file() and (tmp_path / "calibrate_1366x768.png").is_file()
    assert (tmp_path / "calibration_1366x768.json").is_file() and not (tmp_path / "calibrate_crop.png").exists()
    monkeypatch.setattr(calibration.client, "submit", lambda **kw: {"frames": [np.zeros_like(full)],
                                                                     "client": {"width": 1366, "height": 768}})
    with pytest.raises(RuntimeError, match="calibrate_1366x768.png"):
        calibration.calibrate(fresh=True)
