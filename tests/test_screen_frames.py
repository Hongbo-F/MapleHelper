"""The screen's frames for the perception service (perception.frames.ScreenFrames), from 2026-10-26: each frame's picture
one of a few kept from frame to frame (Pictures), never one somebody still holds; and a frame the capture still holds
from more than HELD_S before it was asked for given with its own time, said once. A fake window and a fake grabber:
nothing touches the game or the screen."""

import time

import numpy as np
import pytest

from gamelab import telemetry, window
from gamelab.analysis import bench
from gamelab.perception import frames as frames_module
from gamelab.perception.frames import Pictures, ScreenFrames


@pytest.fixture
def screen(monkeypatch):
    """The game in front at 1920x1080 at (0, 0), nothing over it; its screen a saved picture (bench.SavedScreen)."""
    monkeypatch.setattr(window, "find_game", lambda title: 7)
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(window, "on_screen", lambda area: True)
    monkeypatch.setattr(window, "foreground_title", lambda: "MapleStory")
    monkeypatch.setattr(window, "shows", lambda hwnd, area, grid=None: True)
    s = bench.SavedScreen()
    s.bgra = np.zeros((1080, 1920, 4), np.uint8)
    return s


def test_a_frame_held_by_anyone_is_never_written_over_and_the_rest_are_used_again(screen):
    frames = ScreenFrames("MapleStory", grabber=screen, log=lambda m: None)
    held = []
    for k in range(5):  # five frames held (the step's, a search's, the recording's ...)
        screen.bgra[...] = (k, 2 * k, 3 * k, 255)
        held.append(frames.next(time.monotonic()).full)
    for k in range(5, 40):  # many more, none of them held
        screen.bgra[...] = (k, 2 * k, 3 * k, 255)
        f = frames.next(time.monotonic())
        assert int(f.full[0, 0, 0]) == 3 * k  # (RGB: BGRA's red)
        assert not any(f.full is h for h in held)
    for k, h in enumerate(held):  # each held picture as it was taken
        assert (h[..., 0] == 3 * k).all() and (h[..., 2] == k).all()
    p = frames.pictures
    assert p.made <= 8 and p.reused >= 30 and p.unkept == 0  # (a new picture only while all kept are held)
    view = held[0][100:200]  # a view of a picture holds it too
    del held
    for k in range(40, 50):
        screen.bgra[...] = (k, 0, 0, 255)
        frames.next(time.monotonic())
    assert (view[..., 2] == 0).all() and (view[..., 0] == 0).all()  # frame 0's, still


def test_with_every_picture_held_a_new_one_is_made_and_not_kept():
    pictures = Pictures(keep=2)
    bgra = np.zeros((4, 5, 4), np.uint8)
    a, b, c = pictures.convert(bgra), pictures.convert(bgra), pictures.convert(bgra)
    assert a is not b and c is not a and c is not b and (pictures.made, pictures.unkept) == (2, 1)
    del a
    assert pictures.convert(bgra) is not b and pictures.reused == 1
    bigger = pictures.convert(np.zeros((6, 5, 4), np.uint8))  # another size: the others go
    assert bigger.shape == (6, 5, 3) and len(pictures._kept) == 1


def test_a_frame_the_capture_holds_from_long_before_the_look_keeps_its_own_time_and_is_said_once(screen, monkeypatch):
    said = []
    frames = ScreenFrames("MapleStory", grabber=screen, log=said.append)
    real = screen.shot
    shown = [None]

    def shot(area, after=None, at=None):
        s = real(area)
        if shown[0] is not None:
            s.t = shown[0]
        return s
    monkeypatch.setattr(screen, "shot", shot)
    # Shown 30 ms before it was asked for: the screen as it still is, stamped as now (as before).
    asked = time.monotonic()
    shown[0] = asked - 0.03
    f = frames.next(asked)
    assert f.visible and f.t >= asked and frames.held == 0 and not said
    # Shown 0.8 s before (a stall short of framecheck.STALE_S): its own time, said once with a `held` event.
    with telemetry.captured() as records:
        for _ in range(3):
            asked = time.monotonic()
            shown[0] = asked - 0.8
            f = frames.next(asked)
            assert f.visible and f.t == pytest.approx(asked - 0.8) and f.t < asked - frames_module.HELD_S
    assert frames.held == 3 and len(said) == 1 and "0.80 s before it was asked for" in said[0]
    assert [(r["topic"], r["kind"]) for r in records] == [("event", "held")]
    # A new frame: as before, and the next stall said again.
    asked = time.monotonic()
    shown[0] = asked + 0.001
    assert frames.next(asked).t == pytest.approx(asked + 0.001)
    shown[0] = time.monotonic() - 1.0
    frames.next(time.monotonic())
    assert len(said) == 2
    # Past STALE_S it's a problem, as before (the duplication made afresh).
    shown[0] = time.monotonic() - frames_module.framecheck.STALE_S - 0.5
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "stale"


def test_with_the_cover_check_a_frame_says_whether_another_window_was_over_the_game_before_or_after_its_shot(
        screen, monkeypatch):
    """window.overs, asked just before the shot and again just after it (the perception service's, for its recording):
    a notification between the grid's points, which window.shows misses, named, and every such window's box as the
    client's (Frame.over_boxes: the recording notes them with the frame, since 2026-11-01); the frame still visible to
    perception.
    Without the check (no recording), not asked at all."""
    over = {"calls": 0, "at": set()}
    box = (500, 400, 700, 460)  # (screen px)

    def window_overs(hwnd, area):
        over["calls"] += 1
        return [(99, box)] if over["calls"] in over["at"] else []
    monkeypatch.setattr(frames_module, "_overs", window_overs)  # (window.overs' walk with the lock kept)
    monkeypatch.setattr(window, "describe", lambda hwnd: f"a window of class 'Toast' ({hwnd})")
    frames = ScreenFrames("MapleStory", grabber=screen, log=lambda m: None)
    f = frames.next(time.monotonic())
    assert f.visible and f.over is None and f.over_boxes == () and over["calls"] == 0  # (not asked)
    frames.cover_check = True
    f = frames.next(time.monotonic())
    assert f.visible and f.over == "" and f.over_boxes == () and over["calls"] == 2  # before and after: nothing
    over["at"] = {4}  # a window come during the shot: seen after it
    f = frames.next(time.monotonic())
    assert f.visible and f.over == "a window of class 'Toast' (99)" and over["calls"] == 4
    assert f.over_boxes == tuple(window.client_boxes(frames.area, [box])) == ((500, 400, 700, 460),)
    over["at"] = {5}  # one there before it (asked after too: a window come meanwhile is noted as well)
    f = frames.next(time.monotonic())
    assert f.visible and f.over.startswith("a window of class") and over["calls"] == 6 and f.over_boxes


def test_frames_held_50_to_500_ms_are_stamped_as_now_and_counted(screen, monkeypatch):
    """Short of HELD_S a held frame is the screen as it still is (stamped as now, as before), but one held from
    HELD_COUNT_S is counted, with the longest, for the service's console line (how often they come live)."""
    frames = ScreenFrames("MapleStory", grabber=screen, log=lambda m: None)
    real, shown = screen.shot, [None]

    def shot(area, after=None, at=None):
        s = real(area)
        s.t = shown[0]
        return s
    monkeypatch.setattr(screen, "shot", shot)
    for held in (0.01, 0.03, 0.08, 0.3, 0.2):
        asked = time.monotonic()
        shown[0] = asked - held
        assert frames.next(asked).t >= asked
    n, longest = frames.holds()
    assert n == 3 and longest == pytest.approx(0.3, abs=0.01) and frames.held == 0
    assert frames.holds() == (0, 0.0)
