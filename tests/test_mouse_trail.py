"""Human-like mouse trails and GameLab's playback of them."""

import numpy as np
import pytest

from gamelab import window
from maplehelper.mouse_trail import generate_trail


# The offline fixture replaces window.move_cursor with a guard after collection; keep the real function for its unit
# tests, with every Windows input and clock dependency replaced below.
_move_cursor = window.move_cursor


def test_generate_trail_is_seeded_sampled_and_lands_on_the_target():
    a = generate_trail(100, 120, 900, 650, seed=42)
    b = generate_trail(100, 120, 900, 650, seed=42)

    assert np.array_equal(a, b)
    assert a.shape[0] > 2 and a.shape[1] == 3
    assert a[0].tolist() == [0.0, 100.0, 120.0]
    assert a[-1, 1:].tolist() == [900.0, 650.0]
    assert np.all(np.diff(a[:, 0]) > 0)  # only changed pixels are reported, in time order
    assert np.all(a[:, 1:] == np.rint(a[:, 1:]))
    assert np.all(np.any(np.diff(a[:, 1:], axis=0) != 0, axis=1))


def test_generate_trail_honors_an_explicit_duration():
    trail = generate_trail(20, 30, 400, 300, duration=0.4, seed=7)
    assert trail[-1, 0] == pytest.approx(400.0)
    assert trail[-1, 1:].tolist() == [400.0, 300.0]


def test_generate_trail_for_no_movement_is_just_the_start():
    assert generate_trail(10.4, 20.4, 10.4, 20.4).tolist() == [[0.0, 10.0, 20.0]]


def test_move_cursor_replays_each_timed_trail_sample_then_lands_exactly(monkeypatch):
    asked = []
    moves, exact, sleeps = [], [], []

    def trail(x1, y1, x2, y2):
        asked.append((x1, y1, x2, y2))
        return np.array([[0.0, 100, 120], [10.0, 110, 125], [20.0, 130, 140]])

    def get_cursor(ptr):
        ptr._obj.x, ptr._obj.y = 100, 120
        return True

    clock = [10.0]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(window, "generate_trail", trail)
    monkeypatch.setattr(window.user32, "GetCursorPos", get_cursor)
    monkeypatch.setattr(window.user32, "SetCursorPos", lambda x, y: exact.append((x, y)) or True)
    monkeypatch.setattr(window, "send_mouse_move", lambda x, y: moves.append((x, y)))
    monkeypatch.setattr(window.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(window.time, "sleep", sleep)

    assert _move_cursor(130, 140) is True
    assert asked == [(100, 120, 130, 140)]
    assert moves == [(110, 125), (130, 140)]
    assert sleeps == pytest.approx([0.01, 0.01])
    assert exact == [(130, 140)]


@pytest.mark.parametrize("early", [False, "splice"])
def test_move_cursor_stops_before_another_event_when_the_plan_wait_ends_early(monkeypatch, early):
    moves, exact = [], []
    monkeypatch.setattr(window, "generate_trail", lambda *a: np.array([[0.0, 0, 0], [10.0, 10, 10]]))
    monkeypatch.setattr(window.user32, "GetCursorPos", lambda ptr: True)
    monkeypatch.setattr(window.user32, "SetCursorPos", lambda x, y: exact.append((x, y)) or True)
    monkeypatch.setattr(window, "send_mouse_move", lambda x, y: moves.append((x, y)))

    assert _move_cursor(10, 10, wait=lambda due: early) == early
    assert moves == [] and exact == []


def test_move_cursor_surfaces_a_send_input_failure(monkeypatch):
    def denied(x, y):
        raise OSError(5, "blocked")

    monkeypatch.setattr(window, "generate_trail", lambda *a: np.array([[0.0, 0, 0], [10.0, 10, 10]]))
    monkeypatch.setattr(window.user32, "GetCursorPos", lambda ptr: True)
    monkeypatch.setattr(window, "send_mouse_move", denied)
    with pytest.raises(OSError, match="blocked"):
        _move_cursor(10, 10, wait=lambda due: True)


def test_move_cursor_surfaces_a_final_set_cursor_pos_failure(monkeypatch):
    monkeypatch.setattr(window, "generate_trail", lambda *a: np.array([[0.0, 0, 0]]))
    monkeypatch.setattr(window.user32, "GetCursorPos", lambda ptr: True)
    monkeypatch.setattr(window.user32, "SetCursorPos", lambda x, y: False)
    monkeypatch.setattr(window.ctypes, "get_last_error", lambda: 5)
    with pytest.raises(OSError) as caught:
        _move_cursor(0, 0)
    assert caught.value.winerror == 5
