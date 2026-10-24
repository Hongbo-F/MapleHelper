"""Screen geometry (gamelab.screen): every screen-space measure follows the game window's client size. At 1920x1080
everything is exactly as it was before the geometry was carried (the saved screenshots, the simulator and the golden
traces depend on it, to the bit); at the game's own 1366x768 the game is drawn pixel for pixel."""

import numpy as np
import pytest

from gamelab import screen
from gamelab.screen import NATIVE, STRETCHED, Geometry, geometry, of_frame


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
