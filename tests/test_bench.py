"""`bench-look`: timing a look on saved frames touches no window, and live it refuses unless the game is
in front already (it never brings it forward)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab import paths, window
from gamelab.__main__ import main
from gamelab.analysis import bench

import testdata

FIXTURES = Path(__file__).parent / "fixtures"


def game_frame() -> np.ndarray:
    """A 1920x1080 RGB frame tiled from the saved game scene (daffodils, leaves, sky)."""
    scene = np.asarray(Image.open(FIXTURES / "game_scene.png").convert("RGB"))
    reps = (-(-1080 // scene.shape[0]), -(-1920 // scene.shape[1]), 1)
    return np.ascontiguousarray(np.tile(scene, reps)[:1080, :1920])


def name_tag(seed: int = 0) -> np.ndarray:
    """A made-up name tag (grey, full size): white letters on a dark box."""
    rng = np.random.default_rng(seed)
    tag = np.full((20, 72), 40, np.uint8)
    tag[5:15, 4:68] = np.where(rng.random((10, 64)) < 0.45, 235, 40)
    return tag


def test_saved_frames_go_through_the_capture_path_and_touch_no_window(tmp_path, monkeypatch):
    # conftest makes every window call raise: a look at a saved frame must not need one.
    full = game_frame()
    tag = name_tag()
    full[600:620, 900:972] = tag[..., None]
    np.save(tmp_path / "tag.npy", tag)
    monkeypatch.setattr(paths, "TAG_FULL", tmp_path / "tag.npy")
    eyes = bench.SavedEyes(tag[::2, ::2])
    eyes.show(cv2.cvtColor(full, cv2.COLOR_RGB2BGRA))
    assert np.array_equal(eyes._capture(), full)
    assert eyes.look().tag == ((900 + 36) // 2, (600 + 10) // 2)
    timer = bench.run(eyes, (100, 100), {"a look": eyes.look}, frames=[("frame", eyes.screen.bgra)], reps=2)
    assert len(timer.samples["a look"]) == 2 and len(timer.samples["name tag"]) == 2
    assert timer.table()[0].startswith("stage")


def test_saved_frames_must_be_the_games_size(tmp_path):
    Image.fromarray(np.zeros((540, 960, 3), np.uint8)).save(tmp_path / "half.png")
    with pytest.raises(RuntimeError, match="1920x1080"):
        bench.load([tmp_path / "half.png"])


def test_live_refuses_unless_the_game_is_in_front(monkeypatch):
    # bring_to_front still raises (conftest): refusing must not try it.
    monkeypatch.setattr(window, "foreground_title", lambda: "Notepad")
    monkeypatch.setattr(window, "find_game", lambda title: 1)  # the game there, the size its eyes are for
    monkeypatch.setattr(window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(window, "on_screen", lambda area: True)
    eyes = bench.FrontEyes.__new__(bench.FrontEyes)
    eyes.title = "MapleStory"
    with pytest.raises(RuntimeError, match="isn't in front"):
        eyes._capture()
    assert main(["bench-look", "--live"]) == 1


def test_saved_frames_of_a_map_place_her_by_its_own_minimap_picture():
    """bench-look (and bodies, and the saved feed) read her dot as the game's Eyes do, by the map's own minimap picture:
    on Tree Dungeon's labelled frames (test data: 24 of 2026-10-28, on 8 platforms from 13 to 30, the minimap scrolled
    to each), on the platform under her labelled feet (her_platform) and within a unit of them across (the frame's x
    less the camera's, which this map, narrower than the view, holds still: 0.3) where her name tag places her, 6
    where her dot alone does (td_p21_1 and _2, damage numbers over her tag: 5.4); in the air (td_p24_0, landing) on
    none. Read as
    on the first map's minimap her dot put her off the map (770 wide), which the place check now refuses
    (localization.placecheck), and bench.refused says so of every frame at once but one, which that reading puts
    inside the check's slack. (On the 20 frames of 2026-10-24, all by the portal on 28, her dot put her at x 255-275,
    the watch of then at 261.6-265.3, and at 902 read as the first map's.)"""
    from gamelab.attack import reach
    from gamelab.localization import mapdata
    from gamelab.localization import placecheck
    from gamelab.localization.senses import Senses
    picture = mapdata.minimap_picture(10002073)
    if picture is None:
        pytest.skip("needs Tree Dungeon's game data")
    labels, files = testdata.labelled(testdata.TREE_DUNGEON)
    model = mapdata.load(10002073)
    frames = bench.load(list(files.values()))
    geo = bench.geometry_of(frames)
    template = np.load(FIXTURES / "native" / "tag_template_1366x768.npy")
    eyes = bench.SavedEyes(template, None, geo, picture)
    senses = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN))
    assert bench.refused(eyes, senses, frames) == []
    placed = {}
    for name, bgra in frames:
        eyes.show(bgra)
        s = senses.look(mobs=False)
        e = labels[name]
        placed[name] = (s.platform and s.platform.id, labels[name]["her_platform"], s.how,
                        round(abs(s.where.x - (e["her"][0] - e["camera"][0])), 1))
    assert all(p == want for p, want, _, _ in placed.values()), placed  # (in the air: None)
    assert all(dx <= (1 if how == "camera" else 6) for p, _, how, dx in placed.values() if p is not None), placed
    by_dot = [name for name, (_, _, how, _) in placed.items() if how == "dot"]
    assert by_dot == ["td_p21_1.png", "td_p21_2.png", "td_p24_0.png"]
    eyes = bench.SavedEyes(template, None, geo, None)
    bad = bench.refused(eyes, Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN)), frames)
    # td_p20_0: her dot, 7 minimap px left of the others' on 20, read as the first map's puts her at x 422, within the
    # check's OFF_X of the map's right edge (410): 632 units from her, but not off the map
    assert placecheck.OFF_X == 20 and model.bounds[2] == 410
    assert len(bad) == len(frames) - 1 and all("off the map" in b for b in bad), bad
    assert not any(b.startswith("td_p20_0.png") for b in bad)
