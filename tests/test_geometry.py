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
