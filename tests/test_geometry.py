"""Screen geometry (gamelab.screen): every screen-space measure follows the game window's client size. At 1920x1080
everything is exactly as it was before the geometry was carried (the saved screenshots, the simulator and the golden
traces depend on it, to the bit); at the game's own 1366x768 the game is drawn pixel for pixel."""

import ast
from pathlib import Path

import numpy as np
import pytest

from gamelab import screen
from gamelab.screen import NATIVE, STRETCHED, Geometry, geometry, of_frame

PKG = Path(__file__).resolve().parents[1] / "gamelab"


def test_the_1920x1080_window_is_what_the_constants_were():
    assert STRETCHED == Geometry((1920, 1080), 2)
    assert screen.GAME_SIZE == (1920, 1080)
    assert screen.SCREEN_SCALE == 1080 / 768 == 1.40625
    assert screen.SCALE == 2
    assert screen.UNITS_PER_HALF_PX == 2 / (1080 / 768)
    assert STRETCHED.stretch_x == 1920 / 1366  # exp_number's SX
    assert STRETCHED.frame_size == (960, 540) and STRETCHED.suffix == "" and not STRETCHED.native


@pytest.mark.parametrize("n", [0, 1, 2.5, 6, 7, 12, 16, 22, 40, 52, 120, 300, 1080, -24, 1.5, 0.703125, 1e-3])
def test_converting_1920x1080_px_to_1920x1080_changes_nothing_to_the_bit(n):
    assert STRETCHED.px(n) == n and STRETCHED.fpx(n) == n
    assert STRETCHED.ipx(n) == round(n) and STRETCHED.ifpx(n) == round(n)
    assert STRETCHED.from_game(STRETCHED.to_game(n)) == pytest.approx(n, abs=1e-12)


def test_the_games_own_size_is_drawn_pixel_for_pixel():
    assert NATIVE == geometry(1366, 768) and NATIVE.native and NATIVE.suffix == "_1366x768"
    assert NATIVE.stretch == NATIVE.stretch_x == 1.0
    assert NATIVE.look_scale == screen.LOOK_SCALES[(1366, 768)]
    assert NATIVE.frame_scale == 1 / NATIVE.look_scale and NATIVE.units_per_px == NATIVE.look_scale
    assert NATIVE.px(1.40625) == 1.0 and NATIVE.px(300) == pytest.approx(213.33, abs=0.01)
    assert NATIVE.to_game(10.0) == NATIVE.from_game(10.0) == 10.0
    half = geometry(1366, 768, 2)
    assert half.frame_size == (683, 384) and half.units_per_px == 2.0 and half.fpx(8) == pytest.approx(5.69, abs=0.01)
    full = geometry(1366, 768, 1)
    assert full.frame_size == (1366, 768) and full.fpx(8) == pytest.approx(11.38, abs=0.01)


def test_a_game_px_is_where_the_stretch_puts_its_middle():
    # At 1920x1080 game px g's middle is at (g + 0.5) * 1.40625 - 0.5 on screen (bilinear filtering): DOT_ORIGIN
    # (65.1, 91.3) is game px (46.15, 64.78).
    assert STRETCHED.to_game(65.1) == pytest.approx(46.149, abs=1e-3)
    assert NATIVE.from_game(STRETCHED.to_game(91.3)) == pytest.approx(64.78, abs=1e-3)


def test_a_frame_says_its_geometry():
    assert of_frame(np.zeros((1080, 1920, 3), np.uint8)) == STRETCHED
    assert of_frame(np.zeros((768, 1366, 3), np.uint8)) == NATIVE
    assert of_frame(np.zeros((768, 1366, 3), np.uint8), 2).frame_size == (683, 384)
    assert geometry(1600, 900).look_scale == 2  # a size the stack doesn't know: half size, stretched 1.17


def test_the_exp_number_box_at_each_size():
    from gamelab.perception import exp_number
    assert exp_number.READ_BOX == exp_number.read_box(STRETCHED) == (1052, 1038, 1191, 1054)
    assert exp_number.read_box(NATIVE) == (748, 738, 847, 750)
    assert exp_number.read(np.zeros((3, 3, 3), np.uint8)).why == "no EXP number box"


def test_the_minimaps_sizes_at_1920x1080_are_its_constants():
    from gamelab.perception import minimap as m
    px = m.sizes(STRETCHED)
    assert px.box == m.MINIMAP_BOX and px.dot_area == m.DOT_AREA and px.dot_size == m.DOT_SIZE
    assert px.dot_full == m.DOT_FULL and px.dot_half == (m.DOT_FULL - 0.5) / 2
    assert px.marker_area == m.MARKER_AREA and px.marker_side == m.MARKER_SIDE and px.marker_fill == 0
    assert px.ring is m._RING and px.other_shift == m.OTHER_SHIFT
    native = m.sizes(NATIVE)
    assert native.box == (0, 68, 213, 213) and native.dot_full == 6 and native.dot_half == 2.5
    assert native.other_shift == 0.5 and native.marker_fill == m.MARKER_FILL


def test_rope_kernels_at_1920x1080_are_as_they_were():
    from gamelab.perception import terrain
    close, open_ = terrain._kernels(STRETCHED)
    assert close.shape == (9, 3) and open_.shape == (31, 1)
    close, open_ = terrain._kernels(NATIVE)
    assert close.shape == (6, 2) and open_.shape == (22, 1)


def test_templates_shrink_by_whole_steps_as_before():
    from gamelab.perception.mobs import frame_size
    assert frame_size(61, 37, 2.0) == (30, 18)  # half size: w // 2, as before
    assert frame_size(61, 37, 1.0) == (61, 37)
    assert frame_size(61, 37, 1.40625) == (43, 26)  # a 1920x1080 cut-out in a 1366x768 frame
    assert frame_size(1, 1, 2.0) == (1, 1)


def test_the_localizer_moves_its_origin_to_the_same_game_px():
    from gamelab.localization import mapdata
    from gamelab.localization.pose import DOT_ORIGIN, Localizer
    tree = {"foothold": {}, "miniMap": {"width": 1948, "height": 2152, "centerX": 479, "centerY": 1387, "mag": 4}}
    model = mapdata.parse(1, tree, "Test", {})
    loc = Localizer(model)
    assert loc.origin == DOT_ORIGIN and loc.per_px == 16 / 1.40625
    at_1920 = loc.to_map((120.0, 200.0))
    loc.use(NATIVE)
    assert loc.per_px == 16.0 and loc.origin == pytest.approx((46.149, 64.78), abs=1e-3)
    # The same game px of the dot at both sizes is the same place on the map
    dot = (STRETCHED.to_game(120.0), STRETCHED.to_game(200.0))
    assert loc.to_map(dot) == pytest.approx(at_1920, abs=1e-9)
    loc.use(STRETCHED)
    assert loc.origin == pytest.approx(DOT_ORIGIN, abs=1e-12)


def test_the_camera_puts_map_units_on_screen_at_the_windows_stretch():
    from gamelab.localization.pose import Camera
    cam = Camera((-1000, -1000, 1000, 1000))
    at = (0.0, 0.0)
    x1920 = cam.to_screen(100.0, 50.0, at)
    xn = cam.to_screen(100.0, 50.0, at, NATIVE)
    assert x1920 == pytest.approx((xn[0] * 1.40625, xn[1] * 1.40625))


# Modules that read the game go by the geometry their frames carry: the 1920x1080 window's numbers (screen.GAME_SIZE,
# SCREEN_SCALE, SCALE, UNITS_PER_HALF_PX) are for the simulator, which draws its world at that size.
ALIASES = {"GAME_SIZE", "SCREEN_SCALE", "SCALE", "UNITS_PER_HALF_PX"}


def _alias_users() -> list[str]:
    out = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG)
        if rel.parts[0] == "sim" or rel.name == "screen.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("screen"):
                used = {a.name for a in node.names} & ALIASES
                if used:
                    out.append(f"{rel}: {sorted(used)}")
    return out


def test_nothing_but_the_simulator_assumes_the_1920x1080_window():
    assert _alias_users() == []


def test_the_trackers_screen_edge_and_the_players_reach_go_by_the_look():
    from gamelab.localization import mapdata
    from gamelab.localization.players import MATCH_PX, Players
    from gamelab.localization.pose import Localizer
    from gamelab.localization.tracking import EDGE_PX, View
    assert View(0, 1, 0, 1, 960, 475).edge == EDGE_PX == STRETCHED.fpx(EDGE_PX)
    assert NATIVE.fpx(EDGE_PX) == pytest.approx(34.13, abs=0.01)
    tree = {"foothold": {}, "miniMap": {"width": 1948, "height": 2152, "centerX": 479, "centerY": 1387, "mag": 4}}
    loc = Localizer(mapdata.parse(1, tree, "Test", {}))
    from gamelab.perception.minimap import Marker
    for geo, step, same in ((STRETCHED, MATCH_PX, True), (STRETCHED, MATCH_PX + 0.5, False),
                            (NATIVE, NATIVE.px(MATCH_PX) - 0.01, True), (NATIVE, NATIVE.px(MATCH_PX) + 0.5, False)):
        loc.use(geo)
        players = Players(loc)
        players.update(0.0, [Marker("stranger", 100.0, 150.0)])
        (p,) = players.update(0.0, [Marker("stranger", 100.0, 150.0)])
        (q,) = players.update(0.0, [Marker("stranger", 100.0 + step, 150.0)])  # the same player, or still the old
        assert q.id == p.id and (q.dot[0] == 100.0 + step) == same, (geo, step)  # one, missed (a new dot waits)


# --- the commands take the game window's size as it is: a minimized one once it's back ----------------------------

def _game_window(monkeypatch, game: dict):
    """window.* for a game window that's minimized while game["iconic"] (its client area then 432x54, as GetClientRect
    gave on 2026-10-24), else 1366x768."""
    from gamelab import __main__ as cli
    monkeypatch.setattr(cli.window, "find_game", lambda title: 7)
    monkeypatch.setattr(cli.window, "minimized", lambda hwnd: game["iconic"])
    monkeypatch.setattr(cli.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 432, "height": 54}
                        if game["iconic"] else {"left": 0, "top": 0, "width": 1366, "height": 768})
    return cli


def test_a_command_that_brings_the_game_forward_restores_it_before_reading_its_size(monkeypatch):
    game = {"iconic": True, "brought": 0}
    cli = _game_window(monkeypatch, game)

    def bring(title, wait=0.0):
        game["brought"] += 1
        game["iconic"] = False
        return True
    monkeypatch.setattr(cli.window, "bring_to_front", bring)
    assert cli._game_geometry("MapleStory", restore=True) == NATIVE and game["brought"] == 1
    assert cli._game_geometry("MapleStory", restore=True) == NATIVE and game["brought"] == 1  # shown: as it is
    game["iconic"] = True
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title, wait=0.0: False)  # Windows wouldn't
    with pytest.raises(RuntimeError, match="minimized and didn't come back"):
        cli._game_geometry("MapleStory", restore=True)


def test_the_hunting_commands_eyes_are_made_for_the_restored_window(monkeypatch):
    from types import SimpleNamespace

    from gamelab.perception import eyes, sprites
    game = {"iconic": True}
    cli = _game_window(monkeypatch, game)
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title, wait=0.0: game.update(iconic=False) or True)
    monkeypatch.setattr(cli, "_calibration", lambda geo: SimpleNamespace(template=np.zeros((15, 55), np.uint8)))
    monkeypatch.setattr(sprites, "detector_for", lambda model, geo: ("detector", geo))
    monkeypatch.setattr(eyes, "Eyes", lambda title, template, detector, geo: SimpleNamespace(detector=detector, geo=geo))
    made = cli._eyes(SimpleNamespace(id=10004021), "MapleStory")
    assert made.geo == NATIVE and made.detector == ("detector", NATIVE)


def test_watch_waits_for_a_minimized_game_and_never_restores_it(monkeypatch, capsys):
    game = {"iconic": True}
    cli = _game_window(monkeypatch, game)
    monkeypatch.setattr(cli.window, "bring_to_front", lambda *a, **k: pytest.fail("brought the game forward"))
    waits = []

    def sleep(s):
        waits.append(s)
        game["iconic"] = len(waits) < 3  # restored by whoever is at the PC after three looks
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli._game_geometry("MapleStory") == NATIVE and waits == [cli.MINIMIZED_POLL_S] * 3
    assert "the game is minimized: waiting" in capsys.readouterr().out


def test_the_three_standing_tests_go_by_the_looks_geometry():
    # Ground under her feet that shows she stands: 6 half-size px of the 1920x1080 window, 8.5 px at 1366x768 full
    # size. looting.standing had kept the 1920x1080 number (6 px), where navigator.standing and Scene.grounded convert.
    from gamelab.behavior import looting
    from gamelab.control import navigator
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    from types import SimpleNamespace
    p = SimpleNamespace(id=1)
    for geo, width, stands in ((STRETCHED, 6, True), (STRETCHED, 5, False), (NATIVE, 9, True), (NATIVE, 8, False),
                               (NATIVE, 6, False)):
        s = Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0, geo=geo,
                  where=Where(0, 0, p, (0, 0)), ground=(100, 100 + width))
        assert looting.standing(s) == navigator.standing(s) == s.grounded == stands, (geo.client, width)
